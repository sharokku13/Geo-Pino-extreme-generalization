"""
src/losses.py
=============
Loss functions for Geo-PINO training.

Covers:
  - PerChannelRelL2Loss: adaptive per-channel relative L² data loss.
  - PerChannelRelL2LossWeighted: same with configurable per-channel weights.
  - apply_hard_noslip: enforces no-slip BC before PDE residual computation.
  - finite-difference derivative kernels (_fd_kernels).
  - compute_ns_residuals: incompressible RANS residuals with full
    divergence-form turbulent diffusion and dimensionless normalisation.
  - extract_freestream_speed / angle_of_attack_deg: inlet condition recovery.
  - compute_cl_cd: differentiable lift/drag integration via SDF mollifier.
  - cl_cd_aux_loss: auxiliary Cl/Cd relative-error loss for optional use.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data losses
# ---------------------------------------------------------------------------

class PerChannelRelL2Loss(nn.Module):
    """
    Per-channel adaptive relative L² loss.

    For each channel ``c``, the contribution is::

        ||pred_c - tgt_c|| / max(eps_frac * mean(||tgt_c||), min_eps)

    Both ``eps_frac`` and ``min_eps`` guard against near-zero denominators
    when any output channel is nearly constant across the batch.

    Args:
        eps_frac: Denominator = ``eps_frac * mean(||target_c||)``.
        min_eps: Absolute minimum denominator floor.
    """

    def __init__(self, eps_frac: float = 0.05, min_eps: float = 1.0) -> None:
        super().__init__()
        self.eps_frac = eps_frac
        self.min_eps = min_eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"[RelL2] shape mismatch: pred={pred.shape}  target={target.shape}. "
                "Did you forget normaliser.encode(target)?"
            )
        B, C, H, W = pred.shape
        p = pred.reshape(B, C, -1)
        t = target.reshape(B, C, -1)
        num = (p - t).norm(dim=-1)
        t_norm = t.norm(dim=-1)
        eps = (t_norm.detach().mean() * self.eps_frac).clamp(min=self.min_eps)
        return (num / t_norm.clamp(min=eps)).mean()


class PerChannelRelL2LossWeighted(nn.Module):
    """
    Per-channel relative L² loss with configurable channel weights.

    Default weights up-weight the pressure channel (index 2) by 2× to
    compensate for its typically higher relative error.

    Channel layout: ``[Ux, Uy, P, nut]``

    Args:
        channel_weights: Iterable of per-channel multipliers (length = C).
        eps_frac: Relative denominator floor (see :class:`PerChannelRelL2Loss`).
        min_eps: Absolute denominator floor.
    """

    def __init__(
        self,
        channel_weights: Tuple[float, ...] = (1.0, 1.0, 2.0, 1.0),
        eps_frac: float = 0.05,
        min_eps: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("w", torch.tensor(channel_weights, dtype=torch.float32))
        self.eps_frac = eps_frac
        self.min_eps = min_eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred={pred.shape} target={target.shape}. "
                "normaliser.encode(target) missing?"
            )
        B, C, H, W = pred.shape
        p = pred.reshape(B, C, -1)
        t = target.reshape(B, C, -1)
        num = (p - t).norm(dim=-1)
        t_norm = t.norm(dim=-1)
        eps = (t_norm.detach().mean() * self.eps_frac).clamp(min=self.min_eps)
        per_ch = num / t_norm.clamp(min=eps)   # [B, C]
        w = self.w.to(pred.device)[:C]
        return (per_ch * w).sum(dim=1).mean() / w.sum()


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def apply_hard_noslip(
    pred_phys: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce the no-slip boundary condition by zeroing velocity inside the body.

    Applying this before computing NS residuals prevents the trivial
    constant-flow solution from satisfying the BC penalty (which would allow
    the optimiser to collapse to a zero-gradient prediction).

    Args:
        pred_phys: Predictions in physical units ``[B, 4, H, W]``.
        mask: Body mask ``[B, 1, H, W]``, 1 = body cell, 0 = fluid cell.

    Returns:
        Masked predictions ``[B, 4, H, W]`` with zero velocity inside the body.
    """
    out = pred_phys.clone()
    fluid = 1.0 - mask
    out[:, 0:1] *= fluid    # Ux = 0 inside body
    out[:, 1:2] *= fluid    # Uy = 0 inside body
    out[:, 3:4] *= fluid    # nut = 0 inside body
    return out


def _fd_kernels(
    phys_coords: torch.Tensor,
    dev: torch.device,
    dt: torch.dtype,
):
    """
    Build central-difference derivative closures from the physical coordinate grid.

    Returns closures ``(ddx, ddy, lap, dx)`` operating on ``[B, 1, H, W]`` tensors.
    All kernels use replicate padding to avoid boundary artefacts.
    """
    dx = (phys_coords[:, 1:, :, 0] - phys_coords[:, :-1, :, 0]).abs().mean()
    dy = (phys_coords[:, :, 1:, 1] - phys_coords[:, :, :-1, 1]).abs().mean()

    def _pad(f: torch.Tensor) -> torch.Tensor:
        return F.pad(f, [1, 1, 1, 1], mode="replicate")

    def ddx(f: torch.Tensor) -> torch.Tensor:
        k = torch.zeros(1, 1, 3, 3, device=dev, dtype=dt)
        k[0, 0, 0, 1] = -0.5 / dx
        k[0, 0, 2, 1] = 0.5 / dx
        return F.conv2d(_pad(f), k)

    def ddy(f: torch.Tensor) -> torch.Tensor:
        k = torch.zeros(1, 1, 3, 3, device=dev, dtype=dt)
        k[0, 0, 1, 0] = -0.5 / dy
        k[0, 0, 1, 2] = 0.5 / dy
        return F.conv2d(_pad(f), k)

    def lap(f: torch.Tensor) -> torch.Tensor:
        kx = torch.zeros(1, 1, 3, 3, device=dev, dtype=dt)
        ky = torch.zeros(1, 1, 3, 3, device=dev, dtype=dt)
        kx[0, 0, 0, 1] = 1 / dx ** 2
        kx[0, 0, 1, 1] = -2 / dx ** 2
        kx[0, 0, 2, 1] = 1 / dx ** 2
        ky[0, 0, 1, 0] = 1 / dy ** 2
        ky[0, 0, 1, 1] = -2 / dy ** 2
        ky[0, 0, 1, 2] = 1 / dy ** 2
        return F.conv2d(_pad(f), kx) + F.conv2d(_pad(f), ky)

    return ddx, ddy, lap, dx


# ---------------------------------------------------------------------------
# RANS residuals
# ---------------------------------------------------------------------------

def compute_ns_residuals(
    pred_masked: torch.Tensor,
    phys_coords: torch.Tensor,
    mask: torch.Tensor,
    sdf: torch.Tensor,
    reynolds: float,
) -> Dict[str, torch.Tensor]:
    """
    Incompressible RANS residuals with full divergence-form turbulent diffusion.

    The momentum equation for the x-component is::

        u·∂u/∂x + v·∂u/∂y + ∂p/∂x − [ν_eff·∇²u + ∇ν_eff·∇u] = 0

    where ``ν_eff = 1/Re + ν_t``.  Including the non-local ``∇ν_eff·∇u`` term
    is physically correct for spatially varying turbulent viscosity (RANS).

    All residuals are normalised by ``u_ref²`` (95th-percentile velocity
    squared, detached from the computational graph) so that the loss terms
    remain O(1) throughout training.

    Args:
        pred_masked: Predictions with hard no-slip applied ``[B, 4, H, W]``.
        phys_coords: Physical coordinate grid ``[B, H, W, 2]``.
        mask: Body mask ``[B, 1, H, W]``.
        sdf: Signed distance function ``[B, 1, H, W]``.
        reynolds: Reynolds number for the molecular viscosity term ``1/Re``.

    Returns:
        Dictionary with keys ``div``, ``momentum``, ``bc``, ``u_ref_sq``.
    """
    dev, dt = pred_masked.device, pred_masked.dtype
    u, v, p = pred_masked[:, 0:1], pred_masked[:, 1:2], pred_masked[:, 2:3]
    nut = F.softplus(pred_masked[:, 3:4])   # ν_t ≥ 0, C∞

    ddx, ddy, lap, dx = _fd_kernels(phys_coords, dev, dt)

    u_ref_sq = (
        (u.detach() ** 2 + v.detach() ** 2)
        .clamp(min=1e-4).reshape(-1).quantile(0.95).clamp(min=1e-4)
    )

    fluid = (1.0 - mask).clamp(0.0, 1.0)
    nu_eff = 1.0 / reynolds + nut

    # Spatial gradients of effective viscosity (non-zero for variable ν_t)
    dnu_dx = ddx(nu_eff)
    dnu_dy = ddy(nu_eff)

    du_dx, du_dy = ddx(u), ddy(u)
    dv_dx, dv_dy = ddx(v), ddy(v)

    # Full divergence-form diffusion
    diff_u = nu_eff * lap(u) + dnu_dx * du_dx + dnu_dy * du_dy
    diff_v = nu_eff * lap(v) + dnu_dx * dv_dx + dnu_dy * dv_dy

    # Continuity (normalised)
    div_raw = ddx(u) + ddy(v)
    div_norm = div_raw / (u_ref_sq.sqrt() / dx.clamp(min=1e-8) + 1e-8)

    # Momentum (normalised by u_ref²)
    Mx_raw = u * du_dx + v * du_dy + ddx(p) - diff_u
    My_raw = u * dv_dx + v * dv_dy + ddy(p) - diff_v
    Mx = Mx_raw / (u_ref_sq + 1e-8)
    My = My_raw / (u_ref_sq + 1e-8)

    # Soft-BC wall penalty via exponential SDF weight (C∞, no kinks)
    wall_w = torch.exp(-sdf.abs() / (3.0 * dx + 1e-8))
    bc_term = wall_w * (u ** 2 + v ** 2) / (u_ref_sq + 1e-8)

    # Boundary-layer aware residual weighting (higher weight near wall)
    sdf_n = sdf.abs() / (3.0 * dx + 1e-8)
    bl_w = 1.0 + 9.0 * torch.sigmoid(5.0 * (1.0 - sdf_n))

    return {
        "div":      torch.mean(fluid * bl_w * div_norm ** 2),
        "momentum": torch.mean(fluid * bl_w * (Mx ** 2 + My ** 2)),
        "bc":       torch.mean(bc_term),
        "u_ref_sq": u_ref_sq.detach(),
    }


# ---------------------------------------------------------------------------
# Inlet condition recovery
# ---------------------------------------------------------------------------

def extract_freestream_speed(inp: torch.Tensor) -> torch.Tensor:
    """
    Recover the per-sample freestream speed ``V_inf`` from model inputs.

    The inlet velocity channels (1 = Ux_in·fluid, 2 = Uy_in·fluid) are
    constant over fluid cells and zero inside the body.  This function reads
    the first fluid pixel for each batch element.

    Args:
        inp: Model input tensor ``[B, 4, H, W]``.

    Returns:
        ``V_inf`` tensor of shape ``[B]`` (minimum 1e-6 for numerical stability).
    """
    B = inp.shape[0]
    v_inf = torch.zeros(B, device=inp.device, dtype=inp.dtype)
    for b in range(B):
        fluid = (1.0 - inp[b, 0]) > 0.5
        idx = fluid.nonzero(as_tuple=False)
        if len(idx) == 0:
            v_inf[b] = 1.0
            continue
        i, j = idx[0]
        ux_in = inp[b, 1, i, j]
        uy_in = inp[b, 2, i, j]
        v_inf[b] = torch.sqrt(ux_in ** 2 + uy_in ** 2).clamp(min=1e-6)
    return v_inf


def angle_of_attack_deg(inp: torch.Tensor) -> float:
    """
    Recover the angle of attack (degrees) from a single model input sample.

    Args:
        inp: Single-sample input tensor ``[4, H, W]`` (no batch dimension).

    Returns:
        Angle of attack in degrees, computed as ``atan2(Uy_in, Ux_in)``.
    """
    fluid = (1.0 - inp[0]) > 0.5
    idx = fluid.nonzero(as_tuple=False)
    if len(idx) == 0:
        return 0.0
    i, j = idx[0]
    return math.degrees(math.atan2(float(inp[2, i, j]), float(inp[1, i, j])))


# ---------------------------------------------------------------------------
# Aerodynamic force integration
# ---------------------------------------------------------------------------

def compute_cl_cd(
    pred_fields: torch.Tensor,
    sdf: torch.Tensor,
    phys_coords: torch.Tensor,
    v_inf: torch.Tensor,
    chord: float = 1.0,
    eps_factor: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable lift and drag integration via a cosine-cap SDF mollifier.

    The pressure coefficient is integrated against the surface normal on the
    airfoil zero level-set.  The Dirac delta at ``sdf = 0`` is approximated
    by a cosine mollifier of width ``ε = eps_factor * max(dx, dy)``, gated
    by a sigmoid to suppress far-field contributions.

    Per-sample freestream dynamic pressure ``q_inf = 0.5 · V_inf²``
    (kinematic — no density factor, consistent with AirfRANS pressure units).

    Args:
        pred_fields: Flow field tensor ``[B, 4, H, W]`` in physical units.
        sdf: Signed distance function ``[B, 1, H, W]``.
        phys_coords: Physical coordinates ``[B, H, W, 2]``.
        v_inf: Freestream speed per sample ``[B]``.
        chord: Airfoil chord length for normalisation.
        eps_factor: Mollifier half-width in units of grid spacing.

    Returns:
        ``(Cl [B], Cd [B])`` — per-sample lift and drag coefficients.
        These are NOT batch-averaged so that callers can build scatter plots.
    """
    dev, dt = pred_fields.device, pred_fields.dtype
    P = pred_fields[:, 2:3]   # kinematic pressure

    dx = (phys_coords[:, 1:, :, 0] - phys_coords[:, :-1, :, 0]).abs().mean()
    dy = (phys_coords[:, :, 1:, 1] - phys_coords[:, :, :-1, 1]).abs().mean()
    dA = dx * dy

    eps = eps_factor * torch.maximum(dx, dy)
    phi_eps = sdf / (eps + 1e-12)
    delta = (1.0 / (2.0 * eps)) * (1.0 + torch.cos(math.pi * phi_eps))
    gate = torch.sigmoid(20.0 * (1.0 - phi_eps.abs()))
    delta = delta * gate

    phi_pad = F.pad(sdf, [1, 1, 1, 1], mode="replicate")
    kx = torch.zeros(1, 1, 3, 3, device=dev, dtype=dt)
    ky = torch.zeros(1, 1, 3, 3, device=dev, dtype=dt)
    kx[0, 0, 0, 1] = -0.5
    kx[0, 0, 2, 1] = 0.5
    ky[0, 0, 1, 0] = -0.5
    ky[0, 0, 1, 2] = 0.5
    gx = F.conv2d(phi_pad, kx / dx)
    gy = F.conv2d(phi_pad, ky / dy)
    gm = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    nx, ny = gx / gm, gy / gm

    q_inf = 0.5 * v_inf.view(-1, 1, 1, 1) ** 2
    Cp = P / (q_inf + 1e-12)
    w = delta * dA

    Cl = -torch.sum(Cp * ny * w, dim=[1, 2, 3]) / chord
    Cd = torch.sum(Cp * nx * w, dim=[1, 2, 3]) / chord
    return Cl, Cd


# ---------------------------------------------------------------------------
# Auxiliary Cl/Cd loss
# ---------------------------------------------------------------------------

def cl_cd_aux_loss(
    pred_phys: torch.Tensor,
    target: torch.Tensor,
    sdf: torch.Tensor,
    phys_coords: torch.Tensor,
    v_inf: torch.Tensor,
) -> torch.Tensor:
    """
    Relative error between predicted and true aerodynamic coefficients.

    This auxiliary loss provides a coarse integral signal that penalises
    errors in the global lift and drag directly.  Recommended weight: 0.05
    (small — this is a low-rank signal and should not dominate the data loss).

    Args:
        pred_phys: Predicted flow fields in physical units ``[B, 4, H, W]``.
        target: CFD ground-truth fields in physical units ``[B, 4, H, W]``.
        sdf: Signed distance function ``[B, 1, H, W]``.
        phys_coords: Physical coordinates ``[B, H, W, 2]``.
        v_inf: Freestream speed per sample ``[B]``.

    Returns:
        Scalar loss value ``rel_error_Cl + rel_error_Cd``.
    """
    p_cl, p_cd = compute_cl_cd(pred_phys, sdf, phys_coords, v_inf)
    t_cl, t_cd = compute_cl_cd(target, sdf, phys_coords, v_inf)
    eps = 1e-2
    rel_cl = ((p_cl - t_cl).abs() / (t_cl.abs() + eps)).mean()
    rel_cd = ((p_cd - t_cd).abs() / (t_cd.abs() + eps)).mean()
    return rel_cl + rel_cd
