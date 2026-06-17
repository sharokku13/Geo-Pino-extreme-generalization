#sharokku
from __future__ import annotations
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data losses
class PerChannelRelL2Loss(nn.Module):
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
                "Did you forget normaliser.encode(target)?"   #nah i did not
            )
        B, C, H, W = pred.shape
        p = pred.reshape(B, C, -1)
        t = target.reshape(B, C, -1)
        num = (p - t).norm(dim=-1)
        t_norm = t.norm(dim=-1)
        eps = (t_norm.detach().mean() * self.eps_frac).clamp(min=self.min_eps)
        return (num / t_norm.clamp(min=eps)).mean()

class PerChannelRelL2LossWeighted(nn.Module):
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

# Physics helpers
def apply_hard_noslip(
    pred_phys: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
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


# RANS residuals
def compute_ns_residuals(
    pred_masked: torch.Tensor,
    phys_coords: torch.Tensor,
    mask: torch.Tensor,
    sdf: torch.Tensor,
    reynolds: float,
) -> Dict[str, torch.Tensor]:
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


# Inlet condition recovery
def extract_freestream_speed(inp: torch.Tensor) -> torch.Tensor:
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
    fluid = (1.0 - inp[0]) > 0.5
    idx = fluid.nonzero(as_tuple=False)
    if len(idx) == 0:
        return 0.0
    i, j = idx[0]
    return math.degrees(math.atan2(float(inp[2, i, j]), float(inp[1, i, j])))

# Aerodynamic force integration
def compute_cl_cd(
    pred_fields: torch.Tensor,
    sdf: torch.Tensor,
    phys_coords: torch.Tensor,
    v_inf: torch.Tensor,
    chord: float = 1.0,
    eps_factor: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

# Auxiliary Cl/Cd loss
def cl_cd_aux_loss(
    pred_phys: torch.Tensor,
    target: torch.Tensor,
    sdf: torch.Tensor,
    phys_coords: torch.Tensor,
    v_inf: torch.Tensor,
) -> torch.Tensor:
    p_cl, p_cd = compute_cl_cd(pred_phys, sdf, phys_coords, v_inf)
    t_cl, t_cd = compute_cl_cd(target, sdf, phys_coords, v_inf)
    eps = 1e-2
    rel_cl = ((p_cl - t_cl).abs() / (t_cl.abs() + eps)).mean()
    rel_cd = ((p_cd - t_cd).abs() / (t_cd.abs() + eps)).mean()
    return rel_cl + rel_cd
