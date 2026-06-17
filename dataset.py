"""
src/dataset.py
==============
AirfRANS dataset loading pipeline for Geo-PINO.

Covers:
  - VTU → NPZ preprocessing via pyvista (with scipy-IDW fallback)
  - Signed Distance Function (SDF) generation from binary masks
  - Delaunay-cached barycentric interpolation to a regular grid
  - AirfransGridDataset: unstructured mesh → regular [H, W] tensors
  - PerChannelNormalizer: per-channel Z-score with outlier clipping
"""

from __future__ import annotations

import glob
import hashlib
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import griddata as sci_griddata
from scipy.spatial import Delaunay, cKDTree

import torch
from torch.utils.data import Dataset

log = logging.getLogger("geo_pino.dataset")

# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------

class DomainConfig:
    """Physical domain bounds and grid resolution (chord-normalised units)."""

    X0: float = -0.5
    X1: float = 1.5
    Y0: float = -0.5
    Y1: float = 0.5
    GRID: int = 241        # default FNO grid resolution (H = W = GRID)
    RHO: float = 1.225     # air density [kg/m³] for pressure normalisation


DOMAIN = DomainConfig()


# ---------------------------------------------------------------------------
# SDF helpers
# ---------------------------------------------------------------------------

def sdf_from_mask(mask_hw: np.ndarray,
                  x0: float = DOMAIN.X0, x1: float = DOMAIN.X1,
                  y0: float = DOMAIN.Y0, y1: float = DOMAIN.Y1) -> np.ndarray:
    """
    Compute a signed distance function from a binary body mask.

    Uses Euclidean Distance Transform (EDT).  Positive values are in the
    fluid domain; negative values are inside the body.

    Args:
        mask_hw: Binary array [H, W], 1 = body, 0 = fluid.
        x0, x1, y0, y1: Physical domain extents (used to compute spacing).

    Returns:
        SDF array [H, W], dtype float32.
    """
    H, W = mask_hw.shape
    dx = (x1 - x0) / max(W - 1, 1)
    dy = (y1 - y0) / max(H - 1, 1)
    d_out = ndimage.distance_transform_edt(
        1 - mask_hw.astype(np.uint8), sampling=(dx, dy))
    d_in = ndimage.distance_transform_edt(
        mask_hw.astype(np.uint8), sampling=(dx, dy))
    return (d_out - d_in).astype(np.float32)


# ---------------------------------------------------------------------------
# VTU → NPZ preprocessing
# ---------------------------------------------------------------------------

def _inlet_from_folder(vtu_path: str) -> Tuple[float, float]:
    """
    Parse inlet velocity components from the AirfRANS folder naming convention.

    Expected pattern: ``airFoil2D_SST_{alpha}_{Ux}_{Uy}_...``

    Returns:
        (Ux_in, Uy_in) as floats; defaults to (1.0, 0.0) on parse failure.
    """
    parts = Path(vtu_path).parent.name.split("_")
    try:
        return float(parts[3]), float(parts[4])
    except (IndexError, ValueError):
        return 1.0, 0.0


def _pv_get_field(point_data, candidates: List[str]) -> Optional[np.ndarray]:
    """
    Safely retrieve a named field from pyvista point_data.

    Iterates ``candidates`` in order and returns the first non-empty array,
    or ``None`` if nothing is found.  Using explicit ``None`` checks avoids
    the ``ValueError: truth value of array is ambiguous`` trap.
    """
    for name in candidates:
        if name in point_data:
            arr = np.asarray(point_data[name], dtype=np.float32)
            if arr.size > 0:
                return arr
    return None


def _probe_pyvista(
    mesh,
    xl: np.ndarray,
    yl: np.ndarray,
    z_probe: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Probe an unstructured pyvista mesh at a regular (xl × yl) grid.

    Args:
        mesh: pyvista UnstructuredGrid.
        xl, yl: 1-D coordinate arrays for the probe grid.
        z_probe: Z-coordinate for the probe plane (mid-extrusion for 2-D meshes).

    Returns:
        ``(ux, uy, p, nut, nan_mask)`` flat arrays of length ``len(xl)*len(yl)``,
        or ``None`` if pyvista is unavailable or returns all-zero velocity.
    """
    try:
        import pyvista as pv
    except ImportError:
        return None

    n = len(xl) * len(yl)
    xx, yy = np.meshgrid(xl, yl, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(n, z_probe)])

    try:
        res = pv.PolyData(pts).sample(mesh, tolerance=None, pass_point_data=True)
    except Exception as exc:
        log.debug("pyvista.sample error: %s", exc)
        return None

    U = _pv_get_field(res.point_data, ["U", "Velocity", "velocity"])
    if U is None or U.ndim < 2 or U.shape[1] < 2:
        return None

    ux, uy = U[:, 0], U[:, 1]
    if max(float(np.abs(ux).max()), float(np.abs(uy).max())) < 1e-7:
        log.debug("pyvista returned all-zero velocity — using scipy fallback")
        return None

    _p = _pv_get_field(res.point_data, ["p", "pressure", "Pressure", "p_rgh"])
    p = _p.ravel() if _p is not None else np.zeros(n, np.float32)

    _nt = _pv_get_field(res.point_data,
                        ["nut", "nuTilda", "turbulentViscosity", "k"])
    nut = (np.clip(_nt.ravel(), 0.0, None)
           if _nt is not None else np.zeros(n, np.float32))

    valid = _pv_get_field(res.point_data, ["vtkValidPointMask"])
    nan_mask = (valid == 0) if valid is not None else (
        (np.abs(ux) < 1e-6) & (np.abs(uy) < 1e-6))

    return ux, uy, p, nut, nan_mask


def _probe_scipy_idw(
    pts_2d: np.ndarray,
    ux_p: np.ndarray, uy_p: np.ndarray,
    p_p: np.ndarray, nt_p: np.ndarray,
    xl: np.ndarray, yl: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse-distance weighted (IDW) interpolation fallback using scipy.

    Used when pyvista is unavailable or produces invalid output.

    Args:
        pts_2d: Source mesh 2-D point coordinates [N, 2].
        ux_p, uy_p, p_p, nt_p: Field values at source points [N].
        xl, yl: 1-D target grid coordinate arrays.

    Returns:
        ``(ux, uy, p, nut, nan_mask)`` on the regular grid, each of
        length ``len(xl) * len(yl)``.
    """
    xx, yy = np.meshgrid(xl, yl, indexing="ij")
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

    tree = cKDTree(pts_2d)
    dists, idxs = tree.query(grid_xy, k=4, workers=-1)
    w = 1.0 / np.maximum(dists ** 2, 1e-20)
    w /= w.sum(axis=1, keepdims=True)

    ux = (ux_p[idxs] * w).sum(1).astype(np.float32)
    uy = (uy_p[idxs] * w).sum(1).astype(np.float32)
    p = (p_p[idxs] * w).sum(1).astype(np.float32)
    nut = (nt_p[idxs] * w).sum(1).astype(np.float32)

    if len(pts_2d) <= 600_000:
        ux_lin = sci_griddata(pts_2d, ux_p, grid_xy,
                              method="linear", fill_value=np.nan)
        nan_mask = np.isnan(ux_lin)
    else:
        vel_ref = max(float(np.percentile(np.abs(ux), 95)), 1e-4)
        nan_mask = np.sqrt(ux ** 2 + uy ** 2) < 0.03 * vel_ref

    return ux, uy, p, nut, nan_mask


def vtu_to_npz(vtu_path: str, out_npz: str) -> Tuple[bool, str]:
    """
    Convert a single AirfRANS VTU simulation file to a preprocessed NPZ.

    The NPZ contains fields on a regular ``DOMAIN.GRID × DOMAIN.GRID`` grid:
    ``ux``, ``uy``, ``pressure``, ``nut``, ``mask``, ``sdf``,
    ``inlet_velocity``, ``domain_bounds``.

    Args:
        vtu_path: Path to the source ``internal.vtu`` file.
        out_npz: Destination ``.npz`` file path.

    Returns:
        ``(success, info_string)`` where *info_string* describes the result
        or the error message on failure.
    """
    try:
        import pyvista as pv
    except ImportError:
        return False, "pyvista not installed"
    try:
        mesh = pv.read(vtu_path)
    except Exception as exc:
        return False, f"pv.read: {exc}"

    b = mesh.bounds
    x0 = max(float(b[0]), DOMAIN.X0)
    x1 = min(float(b[1]), DOMAIN.X1)
    y0 = max(float(b[2]), DOMAIN.Y0)
    y1 = min(float(b[3]), DOMAIN.Y1)

    if x1 - x0 < 0.3 or y1 - y0 < 0.3:
        cx = (float(b[0]) + float(b[1])) / 2
        cy = (float(b[2]) + float(b[3])) / 2
        half = min(float(b[1]) - float(b[0]), float(b[3]) - float(b[2])) * 0.4
        x0, x1 = cx - half, cx + half * 2
        y0, y1 = cy - half, cy + half

    GS = DOMAIN.GRID
    xl = np.linspace(x0, x1, GS)
    yl = np.linspace(y0, y1, GS)

    z0, z1 = float(b[4]), float(b[5])
    z_probe = z0 + (z1 - z0) * 0.5001 if abs(z1 - z0) > 1e-12 else z0

    pv_result = _probe_pyvista(mesh, xl, yl, z_probe)

    if pv_result is not None:
        ux, uy, p_arr, nt_arr, nan_mask = pv_result
        method = f"pyvista z={z_probe:.4f}"
    else:
        pts_2d = mesh.points[:, :2].astype(np.float64)
        U_m = _pv_get_field(mesh.point_data, ["U", "Velocity", "velocity"])
        if U_m is None or U_m.ndim < 2:
            return False, "velocity field not found in mesh"
        p_m = _pv_get_field(mesh.point_data, ["p", "pressure", "Pressure"])
        nt_m = _pv_get_field(mesh.point_data, ["nut", "nuTilda", "k"])
        p_pts = (p_m.ravel().astype(np.float32)
                 if p_m is not None else np.zeros(len(pts_2d), np.float32))
        nt_pts = (np.clip(nt_m.ravel(), 0, None).astype(np.float32)
                  if nt_m is not None else np.zeros(len(pts_2d), np.float32))
        ux, uy, p_arr, nt_arr, nan_mask = _probe_scipy_idw(
            pts_2d, U_m[:, 0].astype(np.float32), U_m[:, 1].astype(np.float32),
            p_pts, nt_pts, xl, yl)
        method = "scipy-IDW"

    ux[nan_mask] = 0.0
    uy[nan_mask] = 0.0
    p_arr[nan_mask] = 0.0
    nt_arr[nan_mask] = 0.0

    fluid = ~nan_mask
    if fluid.any():
        p_med = float(np.median(np.abs(p_arr[fluid])))
        if p_med > 1000.0:
            p_arr /= DOMAIN.RHO

    xx_f, _ = np.meshgrid(xl, yl, indexing="ij")
    upstream = fluid & (xx_f.ravel() < x0 + 0.15 * (x1 - x0))
    if int(upstream.sum()) > 5:
        ux_in = float(np.median(ux[upstream]))
        uy_in = float(np.median(uy[upstream]))
    else:
        ux_in, uy_in = _inlet_from_folder(vtu_path)

    mask_hw = nan_mask.reshape(GS, GS).astype(np.float32)
    sdf_hw = sdf_from_mask(mask_hw, x0, x1, y0, y1)

    np.savez_compressed(
        out_npz,
        ux=ux.reshape(GS, GS).astype(np.float32),
        uy=uy.reshape(GS, GS).astype(np.float32),
        pressure=p_arr.reshape(GS, GS).astype(np.float32),
        nut=nt_arr.reshape(GS, GS).astype(np.float32),
        mask=mask_hw,
        sdf=sdf_hw,
        inlet_velocity=np.array([ux_in, uy_in], dtype=np.float32),
        domain_bounds=np.array([x0, x1, y0, y1], dtype=np.float32),
    )
    ux_max = float(np.abs(ux).max())
    body_pct = float(100 * mask_hw.mean())
    info = (f"{method} | [{x0:.2f},{x1:.2f}]×[{y0:.2f},{y1:.2f}] "
            f"| Ux_max={ux_max:.3f} | body={body_pct:.1f}%")
    return True, info


def run_preprocessing(
    vtu_files: List[str],
    out_dir: str,
    max_n: Optional[int] = None,
    clean_old: bool = False,
) -> int:
    """
    Batch-convert VTU simulation files to preprocessed NPZ grids.

    Args:
        vtu_files: List of ``internal.vtu`` file paths.
        out_dir: Output directory for NPZ files.
        max_n: Maximum number of files to process (``None`` = all).
        clean_old: If ``True``, delete and recreate ``out_dir`` before processing.

    Returns:
        Total number of ready NPZ files in ``out_dir`` after processing.
    """
    if clean_old and os.path.exists(out_dir):
        log.info("Cleaning existing output directory: %s", out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if max_n:
        vtu_files = vtu_files[:max_n]

    def _npz(v: str) -> str:
        return os.path.join(out_dir, Path(v).parent.name + ".npz")

    todo = [v for v in vtu_files if not os.path.exists(_npz(v))]
    done = len(vtu_files) - len(todo)
    log.info("VTU→NPZ: %d already done, %d pending of %d total",
             done, len(todo), len(vtu_files))

    ok = fail = 0
    t0 = time.time()
    for i, vtu in enumerate(todo):
        success, info = vtu_to_npz(vtu, _npz(vtu))
        ok += success
        fail += not success
        if (i + 1) % 20 == 0 or i < 4 or not success:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            log.info("  [%d/%d] %s | %.1f files/s | %s",
                     done + i + 1, len(vtu_files),
                     "OK" if success else "FAIL", rate, info[:80])

    total = len(glob.glob(os.path.join(out_dir, "*.npz")))
    log.info("Preprocessing complete: %d NPZ files (%d new, %d errors)",
             total, ok, fail)
    return total


# ---------------------------------------------------------------------------
# Barycentric interpolation helpers (Delaunay-cached)
# ---------------------------------------------------------------------------

def build_bary_weights(
    pts_2d: np.ndarray,
    target_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Delaunay triangulation once and compute barycentric weights.

    Points outside the convex hull of ``pts_2d`` are assigned to their
    nearest neighbour (weight = 1.0 on that vertex).

    Args:
        pts_2d: Source point cloud [N, 2].
        target_xy: Regular grid query points [M, 2].

    Returns:
        ``(vertices [M, 3], weights [M, 3])`` — integer vertex indices and
        float barycentric weights that satisfy ``weights.sum(axis=1) == 1``.
    """
    tri = Delaunay(pts_2d)
    simplex = tri.find_simplex(target_xy)
    M = len(target_xy)

    vertices = np.zeros((M, 3), dtype=np.int64)
    weights = np.zeros((M, 3), dtype=np.float32)

    inside = simplex >= 0
    if inside.any():
        s_in = simplex[inside]
        T = tri.transform[s_in, :2]                      # [N_in, 2, 2]
        r = target_xy[inside] - tri.transform[s_in, 2]   # [N_in, 2]
        b12 = np.einsum("nij,nj->ni", T, r)
        b3 = 1.0 - b12.sum(axis=1, keepdims=True)
        bary = np.clip(np.concatenate([b12, b3], axis=1), 0.0, 1.0)
        bary /= bary.sum(axis=1, keepdims=True) + 1e-12
        vertices[inside] = tri.simplices[s_in]
        weights[inside] = bary.astype(np.float32)

    if (~inside).any():
        tree = cKDTree(pts_2d)
        _, nn = tree.query(target_xy[~inside])
        nn = np.asarray(nn, dtype=np.int64)
        vertices[~inside] = np.stack([nn, nn, nn], axis=1)
        weights[~inside, 0] = 1.0

    return vertices, weights


def apply_bary(
    values: np.ndarray,
    vertices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Apply precomputed barycentric weights in O(M) — no triangulation rebuild.

    Args:
        values: Field values at source vertices [N] or [N, C].
        vertices: Vertex index triples [M, 3].
        weights: Barycentric weight triples [M, 3].

    Returns:
        Interpolated values at target points [M] or [M, C].
    """
    if values.ndim == 1:
        return (values[vertices] * weights).sum(axis=1)
    return (values[vertices] * weights[:, :, np.newaxis]).sum(axis=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AirfransGridDataset(Dataset):
    """
    AirfRANS unstructured mesh dataset mapped to a regular H × W grid.

    Supported input formats:
      - ``.npz`` (preprocessed by :func:`run_preprocessing`)  — fastest path
      - ``.pt``  (PyTorch Geometric graph files)
      - Synthetic fallback for smoke-tests (``allow_synthetic=True``)

    Each sample returns a 3-tuple:
      ``inp``  [4, H, W]  — (mask, Ux_bc, Uy_bc, SDF)  boundary conditions
      ``tgt``  [4, H, W]  — (Ux, Uy, P, nut)           CFD solution fields
      ``pc``   [H, W, 2]  — physical coordinates (shared, read-only)

    Args:
        root_dir: Directory containing preprocessed ``.npz`` or ``.pt`` files.
        grid_size: Resolution of the output regular grid (H = W = grid_size).
        num_samples: Maximum number of samples to load (``None`` = all).
        cache_dir: Directory for Delaunay barycentric weight cache files.
            Defaults to ``<root_dir>/../airfrans_bary_<grid_size>``.
        allow_synthetic: Generate synthetic data when no real files are found.
        vel_thresh: Fractional velocity threshold for airfoil mask detection
            when processing raw point-cloud formats.
    """

    X_MIN, X_MAX = DOMAIN.X0, DOMAIN.X1
    Y_MIN, Y_MAX = DOMAIN.Y0, DOMAIN.Y1

    def __init__(
        self,
        root_dir: str,
        grid_size: int = DOMAIN.GRID,
        num_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        allow_synthetic: bool = False,
        vel_thresh: float = 0.05,
    ) -> None:
        self.grid_size = grid_size
        self.allow_synthetic = allow_synthetic
        self.vel_thresh = vel_thresh

        xl = torch.linspace(self.X_MIN, self.X_MAX, grid_size)
        yl = torch.linspace(self.Y_MIN, self.Y_MAX, grid_size)
        xg, yg = torch.meshgrid(xl, yl, indexing="ij")
        self.target_coords = torch.stack([xg, yg], dim=-1)   # [H, W, 2]
        self._grid_xy = self.target_coords.reshape(-1, 2).numpy()

        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(root_dir)),
            f"airfrans_bary_{grid_size}",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self._records = self._index(root_dir, num_samples)
        if not self._records:
            raise RuntimeError(f"Empty dataset after indexing '{root_dir}'")

        n_cached = sum(
            1 for r in self._records if os.path.exists(self._bary_path(r))
        )
        log.info("Dataset: %d samples | %d bary-cached | cache=%s",
                 len(self._records), n_cached, self.cache_dir)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index(self, root_dir: str, num_samples: Optional[int]) -> List[Dict]:
        if not os.path.exists(root_dir):
            if not self.allow_synthetic:
                raise FileNotFoundError(
                    f"[FATAL] Dataset directory not found: {root_dir}\n"
                    "  Pass allow_synthetic=True for smoke-tests."
                )
            n = num_samples or 8
            log.warning("allow_synthetic=True — generating %d synthetic samples", n)
            return [{"fpath": None, "fmt": "syn", "name": f"syn_{k:04d}", "seed": k}
                    for k in range(n)]

        pts = sorted(glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True))
        if pts:
            idx = pts[:num_samples] if num_samples else pts
            log.info("Found %d .pt files, using %d", len(pts), len(idx))
            return [{"fpath": f, "fmt": "pt", "name": os.path.basename(f)} for f in idx]

        npzs = sorted(glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True))
        if npzs:
            idx = npzs[:num_samples] if num_samples else npzs
            log.info("Found %d .npz files, using %d", len(npzs), len(idx))
            return [{"fpath": f, "fmt": "npz", "name": os.path.basename(f)} for f in idx]

        if not self.allow_synthetic:
            raise FileNotFoundError(
                f"No .pt or .npz files found inside: {root_dir}"
            )
        n = num_samples or 8
        log.warning("No files found — synthetic fallback (%d samples)", n)
        return [{"fpath": None, "fmt": "syn", "name": f"syn_{k:04d}", "seed": k}
                for k in range(n)]

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Barycentric cache
    # ------------------------------------------------------------------

    def _bary_path(self, rec: Dict) -> str:
        key = (hashlib.md5(rec["fpath"].encode()).hexdigest()[:12]
               if rec["fpath"] else rec["name"])
        return os.path.join(self.cache_dir, f"{key}_{self.grid_size}.npz")

    def _get_bary(
        self, rec: Dict, pts_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load cached barycentric weights or compute and save them."""
        bpath = self._bary_path(rec)
        if os.path.exists(bpath):
            d = np.load(bpath)
            return d["vertices"], d["weights"]
        verts, wts = build_bary_weights(pts_2d, self._grid_xy)
        np.savez_compressed(bpath, vertices=verts, weights=wts)
        return verts, wts

    # ------------------------------------------------------------------
    # Raw loaders
    # ------------------------------------------------------------------

    def _load_pt(self, fpath: str) -> Dict:
        """Load a PyTorch Geometric ``.pt`` graph file."""
        d = torch.load(fpath, map_location="cpu", weights_only=False)

        def _g(o, k):
            return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

        pos = _g(d, "pos")
        if pos is None:
            raise ValueError(f"No .pos in {fpath}")
        pos = pos.float().numpy()

        y_v = _g(d, "y")
        x_v = _g(d, "x")

        if y_v is not None and hasattr(y_v, "shape") and y_v.shape[1] >= 4:
            fld = y_v.float().numpy()
        elif x_v is not None and hasattr(x_v, "shape") and x_v.shape[1] >= 4:
            fld = x_v.float().numpy()
        else:
            raise ValueError(f"No usable flow fields in {fpath}")

        ux, uy, p, nt = fld[:, 0], fld[:, 1], fld[:, 2], fld[:, 3]
        ux_in = uy_in = 0.0
        if x_v is not None and hasattr(x_v, "shape") and x_v.shape[1] >= 4:
            xn = x_v.float().numpy()
            ux_in, uy_in = float(xn[:, 2].mean()), float(xn[:, 3].mean())

        return dict(pos=pos, ux=ux, uy=uy, p=p, nt=nt,
                    ux_in=ux_in, uy_in=uy_in)

    def _load_npz(self, fpath: str) -> Dict:
        """
        Load a preprocessed ``.npz`` file.

        Supports both the regular-grid format (produced by :func:`run_preprocessing`)
        and legacy raw point-cloud NPZ files.
        """
        d = np.load(fpath, allow_pickle=True)

        # Regular-grid format (fast path)
        if "sdf" in d and "mask" in d:
            ux = d["ux"].astype(np.float32)
            uy = d["uy"].astype(np.float32)
            p = d["pressure"].astype(np.float32)
            nut = d["nut"].astype(np.float32)
            mask = d["mask"].astype(np.float32)
            sdf = d["sdf"].astype(np.float32)
            iv = d.get("inlet_velocity",
                        np.array([1.0, 0.0])).astype(np.float32)

            GS = ux.shape[0]
            inlet_x = np.full((GS, GS), iv[0], dtype=np.float32)
            inlet_y = np.full((GS, GS), iv[1], dtype=np.float32)
            fluid = 1.0 - mask

            inp = np.stack([mask, inlet_x * fluid, inlet_y * fluid, sdf],
                           axis=0).astype(np.float32)
            tgt = np.stack([ux, uy, p, nut], axis=0).astype(np.float32)
            return {"is_grid": True, "inp": inp, "tgt": tgt}

        # Legacy point-cloud format
        pos = d.get("coords", d.get("pos"))
        if pos is None:
            raise ValueError(f"No coords/pos key in {fpath}")
        pos = np.asarray(pos, np.float32)
        vel = np.asarray(d.get("velocity", np.zeros((len(pos), 2))), np.float32)
        ux = vel[:, 0] if vel.ndim == 2 else np.asarray(d["ux"], np.float32).ravel()
        uy = vel[:, 1] if vel.ndim == 2 else np.asarray(d["uy"], np.float32).ravel()
        p = np.asarray(d.get("pressure", d.get("p", np.zeros(len(pos)))),
                       np.float32).ravel()
        nt = np.clip(np.asarray(d.get("nut", np.zeros(len(pos))),
                                np.float32).ravel(), 0, None)
        iv = np.asarray(d.get("inlet_velocity", [1.0, 0.0]), np.float32)
        return dict(pos=pos, ux=ux, uy=uy, p=p, nt=nt,
                    ux_in=float(iv[0]), uy_in=float(iv[1]))

    def _synthetic(self, seed: int) -> Dict:
        """Generate a synthetic airfoil-like sample for smoke-tests."""
        rng = np.random.default_rng(seed=seed + 42)
        H, W = self.grid_size, self.grid_size
        xl = np.linspace(self.X_MIN, self.X_MAX, W)
        yl = np.linspace(self.Y_MIN, self.Y_MAX, H)
        xx, yy = np.meshgrid(xl, yl, indexing="ij")
        r2 = (xx - 0.5) ** 2 + yy ** 2
        msk = (r2 < 0.07).astype(np.float32)
        sdf = sdf_from_mask(msk, self.X_MIN, self.X_MAX,
                            self.Y_MIN, self.Y_MAX)
        a = math.radians(rng.uniform(-6, 14))
        fl = 1.0 - msk
        dec = (1.0 - np.exp(-np.abs(sdf) / 0.15))
        ux = (math.cos(a) * dec * fl).ravel().astype(np.float32)
        uy = (math.sin(a) * dec * fl).ravel().astype(np.float32)
        p = (0.5 * (1.0 - ux ** 2 - uy ** 2) * fl.ravel()).astype(np.float32)
        nt = (1e-4 * np.clip(1.0 - r2.ravel() / 0.3, 0, None)
              * fl.ravel()).astype(np.float32)
        pos = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        return dict(pos=pos, ux=ux, uy=uy, p=p, nt=nt,
                    ux_in=math.cos(a), uy_in=math.sin(a))

    # ------------------------------------------------------------------
    # Grid interpolation
    # ------------------------------------------------------------------

    def _to_grid(
        self, rec: Dict, raw: Dict
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Interpolate a raw point-cloud sample to the regular grid."""
        pos = raw["pos"].astype(np.float64)
        verts, wts = self._get_bary(rec, pos)

        src = np.column_stack([raw["ux"], raw["uy"], raw["p"], raw["nt"]])
        grid_vals = apply_bary(src, verts, wts)   # [H*W, 4]

        GS = self.grid_size
        grid_4ch = grid_vals.reshape(GS, GS, 4).astype(np.float32)

        vel_mag = np.sqrt(grid_4ch[:, :, 0] ** 2 + grid_4ch[:, :, 1] ** 2)
        vel_max = float(vel_mag.max())
        mask_hw = (vel_mag < self.vel_thresh * max(vel_max, 1e-3)).astype(np.float32)

        for ch in range(2):
            grid_4ch[:, :, ch][mask_hw == 1] = 0.0

        return grid_4ch, mask_hw, raw["ux_in"], raw["uy_in"]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self._records[idx]

        if rec["fmt"] == "pt":
            raw = self._load_pt(rec["fpath"])
        elif rec["fmt"] == "npz":
            raw = self._load_npz(rec["fpath"])
        else:
            raw = self._synthetic(rec["seed"])

        # Fast path: already a preprocessed regular grid
        if raw.get("is_grid"):
            return (
                torch.from_numpy(raw["inp"]),
                torch.from_numpy(raw["tgt"]),
                self.target_coords.clone(),
            )

        # Slow path: interpolate from point cloud
        grid_4ch, mask_hw, ux_in, uy_in = self._to_grid(rec, raw)
        sdf_hw = sdf_from_mask(mask_hw, self.X_MIN, self.X_MAX,
                                self.Y_MIN, self.Y_MAX)
        fluid = 1.0 - mask_hw
        inp = np.stack([mask_hw, ux_in * fluid, uy_in * fluid, sdf_hw],
                       axis=0).astype(np.float32)
        tgt = grid_4ch.transpose(2, 0, 1).astype(np.float32)

        return (
            torch.from_numpy(inp),
            torch.from_numpy(tgt),
            self.target_coords.clone(),
        )


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class PerChannelNormalizer:
    """
    Per-channel Z-score normalisation with outlier clipping.

    Statistics are computed over the full spatial extent of all provided
    samples to avoid batch-size sensitivity.  Values are clipped to
    ``[-clip, clip]`` after standardisation to prevent numerical blow-up
    when any channel has near-zero variance (e.g. a constant turbulent
    viscosity field in synthetic data).

    Args:
        data: Target field tensor [N, C, H, W].
        eps: Minimum standard deviation (prevents division by zero).
        clip: Symmetric clip range applied after standardisation.

    Channel layout (AirfRANS): ``[Ux, Uy, P, nut]``
    """

    CH_NAMES = ["Ux", "Uy", "P", "nut"]

    def __init__(
        self,
        data: torch.Tensor,
        eps: float = 1e-5,
        clip: float = 5.0,
    ) -> None:
        assert data.dim() == 4, f"Expected [N, C, H, W], got {data.shape}"
        N, C, H, W = data.shape
        flat = data.float().permute(1, 0, 2, 3).reshape(C, -1)
        self.mean = flat.mean(dim=1)                      # [C]
        self.std = flat.std(dim=1).clamp(min=eps)         # [C]
        self.clip = clip

        for i in range(C):
            nm = self.CH_NAMES[i] if i < len(self.CH_NAMES) else f"ch{i}"
            log.info("  %4s  min=%9.4f  max=%9.4f  mean=%9.4f  std=%9.5f",
                     nm, flat[i].min(), flat[i].max(),
                     self.mean[i], self.std[i])

        n_zero = int((self.std < eps * 10).sum())
        if n_zero:
            log.warning("%d channel(s) with std≈0 — encode will return zeros", n_zero)

    def _view(self, x: torch.Tensor) -> Tuple[int, ...]:
        return (1, -1, 1, 1) if x.dim() == 4 else (-1, 1, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise ``x`` to zero mean, unit variance, then clip."""
        mu = self.mean.to(x.device).view(self._view(x))
        sigma = self.std.to(x.device).view(self._view(x))
        return ((x - mu) / sigma).clamp(-self.clip, self.clip)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the standardisation to recover physical units."""
        mu = self.mean.to(x.device).view(self._view(x))
        sigma = self.std.to(x.device).view(self._view(x))
        return x * sigma + mu
