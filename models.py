#sharokku
from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Spectral convolution (Fourier integral operator)
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.m1, self.m2 = modes1, modes2
        scale = (2.0 / (in_ch + out_ch)) ** 0.5

        def _cparam() -> nn.Parameter:
            w = scale * (
                torch.randn(in_ch, out_ch, modes1, modes2)
                + 1j * torch.randn(in_ch, out_ch, modes1, modes2)
            ) / math.sqrt(2)
            return nn.Parameter(torch.view_as_real(w))

        self.W1 = _cparam()   # lower-frequency modes
        self.W2 = _cparam()   # upper-frequency modes (conjugate symmetry axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        W1 = torch.view_as_complex(self.W1)
        W2 = torch.view_as_complex(self.W2)

        xf = torch.fft.rfft2(x, norm="ortho")
        out_f = torch.zeros(B, W1.shape[1], H, W // 2 + 1,
                            dtype=torch.cfloat, device=x.device)
        out_f[:, :, : self.m1, : self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, : self.m1, : self.m2], W1)
        out_f[:, :, -self.m1 :, : self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -self.m1 :, : self.m2], W2)
        return torch.fft.irfft2(out_f, s=(H, W), norm="ortho")

# Coordinate mapping network (IPHI)
class CoordinateMappingNet(nn.Module):
    def __init__(self, hidden: int = 64, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        flat = x.reshape(-1, 2)
        return (flat + self.alpha * self.net(flat)).reshape(B, H, W, 2)


# Geo-PINO itself
class GeoPINO(nn.Module):
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 4,
        modes: int = 20,
        width: int = 64,
        n_layers: int = 4,
        dropout_p: float = 0.1,
        pad: int = 9,
        iphi_hidden: int = 64,
        use_iphi: bool = True,
    ) -> None:
        super().__init__()
        self.pad = pad
        self.use_iphi = use_iphi
        self.iphi = CoordinateMappingNet(hidden=iphi_hidden)

        self.fc0 = nn.Linear(in_ch + 2, width)

        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.drops = nn.ModuleList(
            [nn.Dropout2d(p=dropout_p) for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.GroupNorm(min(8, width), width) for _ in range(n_layers)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        pc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mapped = self.iphi(pc) if self.use_iphi else pc
        grid = mapped.permute(0, 3, 1, 2)                 # [B, 2, H, W]

        h = torch.cat([x, grid], dim=1)                   # [B, in_ch+2, H, W]
        h = self.fc0(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, W, H, W]

        h = F.pad(h, [0, self.pad, 0, self.pad])
        for conv, w, drop, norm in zip(self.convs, self.ws, self.drops, self.norms):
            h = drop(F.gelu(norm(conv(h) + w(h))))
        h = h[..., : -self.pad, : -self.pad]

        h = F.gelu(self.fc1(h.permute(0, 2, 3, 1)))       # [B, H, W, 128]
        out = self.fc2(h).permute(0, 3, 1, 2)              # [B, out_ch, H, W]
        return out, mapped

# MC-Dropout uncertainty estimation
@torch.no_grad()
def mc_dropout_inference(
    model: nn.Module,
    inp: torch.Tensor,
    pc: torch.Tensor,
    n_runs: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    samples = [model(inp, pc)[0] for _ in range(n_runs)]
    stack = torch.stack(samples, dim=0)   # [n_runs, B, 4, H, W]
    return stack.mean(0), stack.std(0)
