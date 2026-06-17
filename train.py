#sharokku
from __future__ import annotations
import argparse
import glob
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import AirfransGridDataset, PerChannelNormalizer, run_preprocessing
from src.losses import (
    PerChannelRelL2LossWeighted,
    apply_hard_noslip,
    compute_ns_residuals,
)
from src.models import GeoPINO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("geo_pino.train")

# Physics weight scheduling
@dataclass
class PhysicsWeights:
    data: float = 1.0
    momentum: float = 0.0
    div: float = 0.0
    bc: float = 0.0


class PhysicsWeightScheduler:
    def __init__(
        self,
        w_momentum_max: float = 1e-4,
        w_div_max: float = 1e-3,
        w_bc_max: float = 1e-2,
        freeze_epochs: int = 5,
        ramp_epochs: int = 20,
        gamma: float = 2.0,
        adaptive: bool = True,
        ema_alpha: float = 0.95,
    ) -> None:
        self.w_max = {
            "momentum": w_momentum_max,
            "div": w_div_max,
            "bc": w_bc_max,
        }
        self.freeze = freeze_epochs
        self.ramp = ramp_epochs
        self.gamma = gamma
        self.adaptive = adaptive
        self.alpha = ema_alpha
        self._ema: Dict[str, float] = {
            k: 1.0 for k in ("data", "momentum", "div", "bc")
        }
        self._current = PhysicsWeights()

    def get(self, pino_epoch: int) -> PhysicsWeights:
        w = PhysicsWeights(data=1.0)
        if pino_epoch < self.freeze:
            return w
        t = min(pino_epoch - self.freeze, self.ramp) / self.ramp
        for key in ("momentum", "div", "bc"):
            setattr(w, key, self.w_max[key] * (t ** self.gamma))
        self._current = w
        return w

    def adaptive_update(self, grad_norms: Dict[str, float]) -> PhysicsWeights:
        if not self.adaptive:
            return self._current
        for k, v in grad_norms.items():
            if k in self._ema:
                self._ema[k] = self.alpha * self._ema[k] + (1 - self.alpha) * (v + 1e-8)
        ref = self._ema["data"]
        for key in ("momentum", "div", "bc"):
            scale = ref / (self._ema[key] + 1e-8)
            new_w = float(getattr(self._current, key) * scale)
            setattr(self._current, key, min(new_w, self.w_max[key]))
        return self._current

    def state_dict(self) -> dict:
        return {"ema": dict(self._ema), "current": vars(self._current)}

    def load_state_dict(self, d: dict) -> None:
        self._ema = d["ema"]
        self._current = PhysicsWeights(**d["current"])

# Loss computation
def compute_pino_loss(
    pred_enc: torch.Tensor,
    target: torch.Tensor,
    inp: torch.Tensor,
    phys_coords: torch.Tensor,
    normaliser: PerChannelNormalizer,
    weights: PhysicsWeights,
    reynolds: float,
    phase: int,
    criterion: nn.Module,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mask = inp[:, 0:1]
    sdf = inp[:, 3:4]

    tgt_enc = normaliser.encode(target)
    loss_data = criterion(pred_enc, tgt_enc)
    total = weights.data * loss_data

    comps: Dict[str, torch.Tensor] = {"data": loss_data}
    zero = torch.zeros(1, device=pred_enc.device).squeeze()

    need_phys = phase == 2 and max(weights.momentum, weights.div, weights.bc) > 1e-12

    if need_phys:
        pred_phys = normaliser.decode(pred_enc)
        pred_masked = apply_hard_noslip(pred_phys, mask)
        ns = compute_ns_residuals(pred_masked, phys_coords, mask, sdf, reynolds)

        total = (
            total
            + weights.momentum * ns["momentum"]
            + weights.div * ns["div"]
            + weights.bc * ns["bc"]
        )
        comps.update({
            "momentum": ns["momentum"],
            "div": ns["div"],
            "bc": ns["bc"],
            "u_ref_sq": ns["u_ref_sq"],
        })
    else:
        comps.update({"momentum": zero, "div": zero, "bc": zero})

    comps["total"] = total
    return total, comps

# Trainer
class GeoPINOTrainer:
    def __init__(
        self,
        model: nn.Module,
        normaliser: PerChannelNormalizer,
        lr_warmup: float = 1e-3,
        lr_pino_start: float = 5e-5,
        lr_pino_min: float = 1e-6,
        warmup_epochs: int = 100,
        total_epochs: int = 200,
        weight_decay: float = 1e-4,
        reynolds: float = 2e6,
        w_momentum_max: float = 1e-4,
        w_div_max: float = 1e-3,
        w_bc_max: float = 1e-2,
        freeze_epochs: int = 5,
        ramp_epochs: int = 20,
        adaptive_w: bool = True,
        grad_clip: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.norm = normaliser
        self.warm_ep = warmup_epochs
        self.total_ep = total_epochs
        self.pino_ep = total_epochs - warmup_epochs
        self.Re = reynolds
        self.dev = device
        self.lr_w = lr_warmup
        self.lr_p_start = lr_pino_start
        self.lr_p_min = lr_pino_min
        self.wd = weight_decay
        self.grad_clip = grad_clip

        self.criterion = PerChannelRelL2LossWeighted(
            channel_weights=(1.0, 1.0, 2.0, 1.0))
        self.w_sched = PhysicsWeightScheduler(
            w_momentum_max=w_momentum_max,
            w_div_max=w_div_max,
            w_bc_max=w_bc_max,
            freeze_epochs=freeze_epochs,
            ramp_epochs=ramp_epochs,
            adaptive=adaptive_w,
        )
        self.history: List[Dict] = []

    # Optimiser / scheduler factories
    def _opt_warmup(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_w, weight_decay=self.wd)

    def _opt_pino(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_p_start, weight_decay=self.wd)

    def _sched(self, opt: torch.optim.Optimizer, t_max: int, eta_min: float):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=t_max, eta_min=eta_min)

    # Public entry point
    def fit(self, loader: DataLoader) -> List[Dict]:
        o1 = self._opt_warmup()
        s1 = self._sched(o1, self.warm_ep, eta_min=1e-5)

        log.info("Phase 1 - Warmup (data loss only)")
        for ep in range(1, self.warm_ep + 1):
            row = self._epoch(ep, loader, o1, phase=1, pino_ep_idx=0)
            s1.step()
            self.history.append(row)
            if ep % 10 == 0 or ep <= 3:
                self._log(ep, row)

        o2 = self._opt_pino()
        s2 = self._sched(o2, self.pino_ep, eta_min=self.lr_p_min)

        log.info("Phase 2 - Physics-Informed training")
        log.info("    LR start  : %.1e", self.lr_p_start)
        log.info("    Grad clip : %.2f", self.grad_clip)
        log.info("    Freeze    : %d epochs  (w_physics = 0)", self.w_sched.freeze)
        log.info("    Ramp      : %d epochs  (γ = %.1f)", self.w_sched.ramp,
                 self.w_sched.gamma)
        log.info("    Adaptive  : Wang-2021 EMA = %s", self.w_sched.adaptive)

        for ep in range(self.warm_ep + 1, self.total_ep + 1):
            pino_idx = ep - self.warm_ep - 1
            row = self._epoch(ep, loader, o2, phase=2, pino_ep_idx=pino_idx)
            s2.step()
            self.history.append(row)
            if ep % 5 == 0 or ep == self.warm_ep + 1:
                self._log(ep, row)

        log.info("Training complete (!)")
        return self.history

    # Single epoch
    def _epoch(
        self,
        ep: int,
        loader: DataLoader,
        opt: torch.optim.Optimizer,
        phase: int,
        pino_ep_idx: int,
    ) -> Dict:
        self.model.train()
        acc = {k: 0.0 for k in ("data", "momentum", "div", "bc", "total")}
        gn = {k: 0.0 for k in ("data", "momentum", "div", "bc")}
        n = 0
        t0 = time.time()
        w = self.w_sched.get(pino_ep_idx)

        for inp, tgt, pc in loader:
            inp = inp.to(self.dev)
            tgt = tgt.to(self.dev)
            pc = pc.to(self.dev)
            opt.zero_grad(set_to_none=True)

            pred_enc, _ = self.model(inp, pc)

            total_loss, comps = compute_pino_loss(
                pred_enc=pred_enc,
                target=tgt,
                inp=inp,
                phys_coords=pc,
                normaliser=self.norm,
                weights=w,
                reynolds=self.Re,
                phase=phase,
                criterion=self.criterion,
            )

            total_loss.backward()

            gn_val = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip)
            )

            opt.step()

            for key in ("data", "momentum", "div", "bc"):
                gn[key] += gn_val * (
                    1.0 if key == "data"
                    else getattr(w, key) / max(w.data, 1e-9)
                ) + 1e-9

            for key in acc:
                if key in comps:
                    acc[key] += comps[key].detach().item()
            n += 1

        if phase == 2 and pino_ep_idx >= self.w_sched.freeze:
            w = self.w_sched.adaptive_update(
                {k: v / max(n, 1) for k, v in gn.items()})

        nb = max(n, 1)
        spd = n * loader.batch_size / max(time.time() - t0, 1e-3)
        return dict(
            epoch=ep, phase=phase, pino_ep=pino_ep_idx,
            data=acc["data"] / nb,
            momentum=acc["momentum"] / nb,
            div=acc["div"] / nb,
            bc=acc["bc"] / nb,
            total=acc["total"] / nb,
            w_momentum=w.momentum, w_div=w.div, w_bc=w.bc,
            lr=opt.param_groups[0]["lr"],
            samples_per_sec=spd,
        )

    def _log(self, ep: int, row: Dict) -> None:
        if row["phase"] == 1:
            log.info("  Ep %03d/%d  DataL=%.5f  LR=%.2e  %.1f spl/s",
                     ep, self.warm_ep, row["data"], row["lr"],
                     row["samples_per_sec"])
        else:
            log.info(
                "  Ep %03d/%d  DataL=%.5f  NS=%.2e  Div=%.2e  "
                "BC=%.2e  wNS=%.1e  LR=%.2e  %.1f spl/s",
                ep, self.total_ep, row["data"], row["momentum"],
                row["div"], row["bc"], row["w_momentum"],
                row["lr"], row["samples_per_sec"])

    def checkpoint(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "norm_mean": self.norm.mean,
            "norm_std": self.norm.std,
            "history": self.history,
            "w_sched_state": self.w_sched.state_dict(),
        }, path)
        log.info("Checkpoint saved → %s", path)

# Training curves plot
def plot_training_curves(history: List[Dict], out_path: str) -> None:
    ep_all = [r["epoch"] for r in history]
    dl = [r["data"] for r in history]
    tl = [r["total"] for r in history]
    lrs = [r["lr"] for r in history]
    ph2 = [r for r in history if r["phase"] == 2]
    ps = ph2[0]["epoch"] if ph2 else None

    fig, axs = plt.subplots(1, 3, figsize=(17, 4))
    fig.suptitle("Geo-PINO — Training Diagnostics", fontsize=13)

    ax = axs[0]
    ax.semilogy(ep_all, dl, "b-", lw=2, label="Train DataL")
    ax.semilogy(ep_all, tl, "r--", lw=1.2, alpha=0.6, label="Total")
    if ps:
        ax.axvline(ps, color="gray", ls=":", lw=1.5, label="PINO start")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RelL2 Loss")
    ax.set_title("Loss Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axs[1]
    if ph2:
        ep2 = [r["epoch"] for r in ph2]
        ax.semilogy(ep2, [r["momentum"] for r in ph2], lw=1.8, label="Momentum")
        ax.semilogy(ep2, [r["div"] for r in ph2], lw=1.8, label="Continuity")
        ax.semilogy(ep2, [r["bc"] for r in ph2], lw=1.8, label="Soft BC")
        ax.set_title("PINO Physics Residuals")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Phase 2 not reached",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("PINO Physics Residuals")
    ax.set_xlabel("Epoch")

    ax = axs[2]
    ax.plot(ep_all, lrs, "g-", lw=2)
    if ps:
        ax.axvline(ps, color="gray", ls=":", lw=1.5, label="Phase 2 start")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title("Learning-Rate Schedule")
    ax.grid(True, alpha=0.3)
    if ps:
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Training curves → %s", out_path)

# Argument parsing
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "geo_pino_train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir", type=str, default="./data/airfrans_prep",
                   help="Directory with preprocessed .npz or .pt files")
    p.add_argument("--raw_dir", type=str, default=None,
                   help="Directory with raw .vtu files for preprocessing")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grid_size", type=int, default=241)
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of data reserved for validation")
    # Training
    p.add_argument("--warmup_epochs", type=int, default=100)
    p.add_argument("--total_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_pino_start", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=0.5)
    # Model
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_iphi", action="store_true",
                   help="Disable IPHI (ablation mode)")
    # Physics
    p.add_argument("--reynolds", type=float, default=2e6)
    p.add_argument("--w_momentum_max", type=float, default=1e-4)
    p.add_argument("--w_div_max", type=float, default=1e-3)
    p.add_argument("--w_bc_max", type=float, default=1e-2)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument("--ramp_epochs", type=int, default=20)
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args(argv)

# Main
def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    log.info("Device: %s | Output: %s", device, args.output_dir)

    # Optional preprocessing
    if args.raw_dir is not None:
        vtu_files = sorted(glob.glob(
            os.path.join(args.raw_dir, "**", "internal.vtu"), recursive=True))
        if not vtu_files:
            vtu_files = sorted(glob.glob(
                os.path.join(args.raw_dir, "**", "*.vtu"), recursive=True))
        if vtu_files:
            log.info("Preprocessing %d VTU files → %s",
                     len(vtu_files), args.data_dir)
            run_preprocessing(vtu_files, args.data_dir,
                              max_n=args.samples, clean_old=False)

    #Dataset
    log.info("[1/7] Dataset")
    cache_dir = os.path.join(args.output_dir, f"bary_cache_{args.grid_size}")
    try:
        dset = AirfransGridDataset(
            root_dir=args.data_dir,
            grid_size=args.grid_size,
            num_samples=args.samples,
            cache_dir=cache_dir,
            allow_synthetic=False,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        log.warning("Dataset init failed (%s) — falling back to synthetic data", exc)
        dset = AirfransGridDataset(
            root_dir=args.data_dir,
            grid_size=args.grid_size,
            num_samples=16,
            cache_dir=cache_dir,
            allow_synthetic=True,
        )

    n_val = max(1, int(len(dset) * args.val_frac))
    n_train = len(dset) - n_val
    tr_ds, va_ds = random_split(
        dset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    pin = device.type == "cuda"
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin)
    log.info("Train=%d  Val=%d  steps/ep=%d", n_train, n_val, len(tr_ld))

    #Normaliser
    log.info("[2/7] PerChannelNormalizer")
    idx_sample = list(getattr(tr_ds, "indices", range(len(tr_ds))))[:64]
    tgt_tensors = [dset[i][1].unsqueeze(0) for i in idx_sample]
    all_tgt = torch.cat(tgt_tensors, dim=0)

    if float(all_tgt[:, 0].abs().max()) < 1e-6:
        log.error("Ux ≈ 0 across all samples — check preprocessing. "
                  "Switching to synthetic fallback.")
        dset = AirfransGridDataset(
            root_dir=args.data_dir, grid_size=args.grid_size,
            num_samples=16, cache_dir=cache_dir, allow_synthetic=True)
        n_val = max(1, len(dset) // 5)
        tr_ds, va_ds = random_split(
            dset, [len(dset) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed))
        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
        all_tgt = torch.cat([dset[i][1].unsqueeze(0) for i in range(len(tr_ds))], 0)

    norm = PerChannelNormalizer(all_tgt, eps=1e-5, clip=5.0)
    enc_check = norm.encode(all_tgt[:4])
    log.info("encode sanity: mean=%.4f  std=%.4f  max_abs=%.2f  (target: ≈0, ≈1, ≤5)",
             enc_check.mean(), enc_check.std(), enc_check.abs().max())

    #Model
    log.info("[3/7] GeoPINO model")
    model = GeoPINO(
        in_ch=4, out_ch=4,
        modes=args.modes, width=args.width,
        n_layers=args.n_layers, dropout_p=args.dropout,
        use_iphi=not args.no_iphi,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s  (modes=%d, width=%d, layers=%d)",
             f"{n_params:,}", args.modes, args.width, args.n_layers)

    # Epoch-0 sanity check
    crit = PerChannelRelL2LossWeighted()
    with torch.no_grad():
        si, st, sp = next(iter(tr_ld))
        si, st, sp = si.to(device), st.to(device), sp.to(device)
        pr, _ = model(si, sp)
        l_rand = crit(pr, norm.encode(st))
    log.info("Ep-0 loss (random model): %.4f  ← expected 0.5–2.0", l_rand)
    if l_rand.item() > 100:
        log.error("Initial loss > 100 — normaliser or data likely corrupted!")

    #Training
    log.info("[4/7] Curriculum training")
    trainer = GeoPINOTrainer(
        model=model, normaliser=norm,
        lr_warmup=args.lr, lr_pino_start=args.lr_pino_start,
        warmup_epochs=args.warmup_epochs, total_epochs=args.total_epochs,
        weight_decay=args.weight_decay, reynolds=args.reynolds,
        w_momentum_max=args.w_momentum_max, w_div_max=args.w_div_max,
        w_bc_max=args.w_bc_max, freeze_epochs=args.freeze_epochs,
        ramp_epochs=args.ramp_epochs, grad_clip=args.grad_clip,
        device=device,
    )
    history = trainer.fit(tr_ld)

    #Validation
    log.info("[5/7] Validation")
    model.eval()
    val_losses: List[float] = []
    with torch.no_grad():
        for vi, vt, vp in va_ld:
            vi, vt, vp = vi.to(device), vt.to(device), vp.to(device)
            vpr, _ = model(vi, vp)
            val_losses.append(crit(vpr, norm.encode(vt)).item())
    val_l2 = float(np.mean(val_losses))
    best_l2 = min(r["data"] for r in history)
    log.info("Best train RelL2 : %.5f", best_l2)
    log.info("Val   RelL2      : %.5f", val_l2)

    #Checkpoint
    log.info("[6/7] Checkpoint")
    ckpt_path = os.path.join(args.output_dir, "geo_pino.pt")
    trainer.checkpoint(ckpt_path)

    #Training curves
    log.info("[7/7] Training curves")
    plot_training_curves(
        history,
        out_path=os.path.join(args.output_dir, "training_curves.png"),
    )

    log.info("━" * 64)
    log.info("Geo-PINO training complete")
    log.info("  Best train RelL2 : %.5f", best_l2)
    log.info("  Val   RelL2      : %.5f", val_l2)
    log.info("  Checkpoint       : %s", ckpt_path)
    log.info("━" * 64)


if __name__ == "__main__":
    main()
