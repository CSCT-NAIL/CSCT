#!/usr/bin/env python3
"""
CSCT EX1: Waveform Discretization Experiments
=============================================

Tests single-channel waveform discretization comparing SingleGate vs MultiGate.

Purpose: 
  - Verify A2 (Compression) and A4 (Irreversible Anchor) on various waveforms
  - Compare SingleGate (default) vs MultiGate architectures

Hypothesis: 
  - SingleGate: Sufficient for single-source waveforms (peripheral processing)
  - MultiGate: No advantage for single-channel (designed for relational info)

Waveforms:
  - sine: Pure sinusoid (baseline)
  - chirp: Frequency sweep
  - am: Amplitude modulation
  - fm: Frequency modulation
  - ecg: ECG-like periodic spikes
  - saw_bl: Band-limited sawtooth
  - composite: Multi-frequency mixture
  - noisy: Sine + noise
  - burst: Intermittent bursts

Outputs:
  - metrics_history.csv
  - recon_<wave>.png

"""

import argparse
import csv
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from csct_engine import CSCTConfig, CSCT_Engine


# =============================================================================
# Config
# =============================================================================

@dataclass
class EX1Config:
    device: str = "cpu"
    seed: int = 0
    steps: int = 500           # Reduced: dynamics converge fast
    seq_len: int = 200

    # Model
    n_clocks: int = 8
    hidden_dim: int = 64
    z_dim: int = 16
    gate_floor: float = 0.10
    gate_topk: int = 1
    gate_tau: float = 0.7
    use_gumbel: bool = True
    gumbel_noise: float = 0.5
    use_multigate: bool = False  # Default: SingleGate for EX1

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Schedules
    beta: float = 50.0  # τ: Transition penalty coefficient
    beta_warmup_steps: int = 200    # ~40% of steps
    w_sparsity: float = 0.0
    sparsity_warmup_steps: int = 100  # ~20% of steps

    # MultiGate sparsity targets (for comparison runs)
    target_na_sparsity: float = 0.10
    target_nmda_sparsity: float = 0.30

    # Data selection
    wave: str = "sine"
    all_waves: bool = False

    # Saw band-limit
    saw_harmonics: int = 20
    saw_f0: float = 5.0

    # Noise
    snr_db: Optional[float] = None
    noise_std: float = 0.0

    # Anchor mode
    anchor_mode: str = "same"

    # Logging/output
    log_interval: int = 50      # More frequent logging
    anneal_every: int = 25      # More frequent annealing
    output_dir: str = "./results_ex1"


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gate_transition_rate(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 2:
        indices = indices.view(indices.shape[0], -1)
    trans = (indices[:, 1:] != indices[:, :-1]).float()
    return trans.mean()


def save_convergence_curve(hist: Dict[str, List], output_dir: str, prefix: str = "") -> None:
    """Save convergence curve plot showing training dynamics."""
    if not hist or "step" not in hist or len(hist["step"]) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps = hist["step"]
    
    # Loss curve
    ax = axes[0, 0]
    if "loss" in hist:
        ax.plot(steps, hist["loss"], label="train_loss", alpha=0.8)
    if "loss_eval" in hist:
        ax.plot(steps, hist["loss_eval"], label="eval_loss", alpha=0.8)
    if "recon_loss" in hist:
        ax.plot(steps, hist["recon_loss"], label="recon_loss", alpha=0.8, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Discreteness metrics
    ax = axes[0, 1]
    if "unique_codes" in hist:
        ax.plot(steps, hist["unique_codes"], label="unique_codes", marker=".", markersize=3)
    if "code_entropy_norm" in hist:
        ax2 = ax.twinx()
        ax2.plot(steps, hist["code_entropy_norm"], label="entropy_norm", color="orange", alpha=0.7)
        ax2.set_ylabel("Entropy (norm)", color="orange")
        ax2.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Unique Codes")
    ax.set_title("Discretization Quality")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Transition rate
    ax = axes[1, 0]
    if "trans_rate" in hist:
        ax.plot(steps, hist["trans_rate"], label="trans_rate", color="green")
    if "beta" in hist:
        ax2 = ax.twinx()
        ax2.plot(steps, hist["beta"], label="beta (τ)", color="red", alpha=0.5, linestyle="--")
        ax2.set_ylabel("Beta (τ)", color="red")
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Transition Rate")
    ax.set_title("Gate Dynamics")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Summary text
    ax = axes[1, 1]
    ax.axis("off")
    final_idx = -1
    summary_text = "CONVERGENCE SUMMARY\n" + "="*30 + "\n"
    if "loss" in hist:
        summary_text += f"Final loss: {hist['loss'][final_idx]:.4f}\n"
    if "recon_loss" in hist:
        summary_text += f"Final recon: {hist['recon_loss'][final_idx]:.4f}\n"
    if "unique_codes" in hist:
        summary_text += f"Final unique: {hist['unique_codes'][final_idx]:.0f}\n"
    if "trans_rate" in hist:
        summary_text += f"Final trans_rate: {hist['trans_rate'][final_idx]:.3f}\n"
    summary_text += f"\nTotal steps: {steps[final_idx]}\n"
    summary_text += f"Log points: {len(steps)}"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"EX1 Convergence Curve {prefix}", fontsize=12)
    plt.tight_layout()
    
    filename = f"convergence_{prefix}.png" if prefix else "convergence.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def gate_sparsity_loss(actual: torch.Tensor, target: float) -> torch.Tensor:
    tgt = torch.tensor(float(target), device=actual.device, dtype=actual.dtype)
    return (actual - tgt).pow(2)


def codebook_stats(indices, K: int = None) -> dict:
    """Compute code usage stats from discrete indices."""
    if indices is None:
        return {"unique_codes": 0.0, "entropy": 0.0, "entropy_norm": 0.0, "K": float(K or 0)}

    if isinstance(indices, (list, tuple)):
        parts = [t.reshape(-1).to(torch.long) for t in indices if t is not None and isinstance(t, torch.Tensor)]
        if not parts:
            return {"unique_codes": 0.0, "entropy": 0.0, "entropy_norm": 0.0, "K": float(K or 0)}
        flat = torch.cat(parts, dim=0)
    else:
        if not isinstance(indices, torch.Tensor):
            return {"unique_codes": 0.0, "entropy": 0.0, "entropy_norm": 0.0, "K": float(K or 0)}
        flat = indices.reshape(-1).to(torch.long)

    if flat.numel() == 0:
        return {"unique_codes": 0.0, "entropy": 0.0, "entropy_norm": 0.0, "K": float(K or 0)}

    if K is None:
        K = int(flat.max().item()) + 1 if flat.numel() > 0 else 0
    K = max(int(K), 0)

    uniq = int(torch.unique(flat).numel())
    if K <= 1:
        return {"unique_codes": float(uniq), "entropy": 0.0, "entropy_norm": 0.0, "K": float(K)}

    counts = torch.bincount(flat.clamp_min(0), minlength=K).to(torch.float32)
    total = counts.sum().clamp_min(1.0)
    p = counts / total
    p = p[p > 0]
    ent = float((-p * p.log()).sum().item())
    ent_norm = float(ent / math.log(K))
    return {"unique_codes": float(uniq), "entropy": ent, "entropy_norm": ent_norm, "K": float(K)}


def add_noise_by_snr(x: np.ndarray, snr_db: Optional[float], rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to achieve target SNR(dB)."""
    if snr_db is None:
        return x.astype(np.float32)
    sig_p = float(np.mean(x.astype(np.float32) ** 2))
    sig_p = max(sig_p, 1e-8)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    noise_p = sig_p / max(snr_lin, 1e-8)
    noise = rng.standard_normal(size=x.shape).astype(np.float32) * np.sqrt(noise_p)
    return (x.astype(np.float32) + noise).astype(np.float32)


# =============================================================================
# Wave Generators
# =============================================================================

def saw_band_limited(t: np.ndarray, f0: float, harmonics: int) -> np.ndarray:
    """Band-limited saw via truncated Fourier series."""
    N = max(1, int(harmonics))
    x = np.zeros_like(t, dtype=np.float32)
    for k in range(1, N + 1):
        x += (2.0 / np.pi) * ((-1.0) ** (k + 1)) * (1.0 / k) * np.sin(2.0 * np.pi * k * f0 * t)
    x = x / (np.max(np.abs(x)) + 1e-8)
    return x.astype(np.float32)


def generate_wave(wave_type: str, t: np.ndarray, cfg: EX1Config, rng: np.random.Generator) -> np.ndarray:
    """Generate waveform by type."""
    if wave_type == "sine":
        x = np.sin(2 * np.pi * 5 * t).astype(np.float32)
    elif wave_type == "composite":
        x = (np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)).astype(np.float32)
    elif wave_type == "chirp":
        x = np.sin(2 * np.pi * (2 + 10 * t) * t).astype(np.float32)
    elif wave_type == "am":
        x = (np.sin(2 * np.pi * 10 * t) * (1 + 0.5 * np.sin(2 * np.pi * 1 * t))).astype(np.float32)
    elif wave_type == "fm":
        x = (np.sin(2 * np.pi * 8 * t + 2.0 * np.sin(2 * np.pi * 1 * t))).astype(np.float32)
    elif wave_type == "burst":
        x = np.zeros_like(t, dtype=np.float32)
        T = len(t)
        for s in range(0, T, 50):
            end = min(T, s + 20)
            n = end - s
            x[s:end] = (np.sin(np.pi * np.arange(n) / max(1, n)) * np.sin(2 * np.pi * 0.15 * np.arange(n))).astype(np.float32)
    elif wave_type == "noisy":
        x = (np.sin(2 * np.pi * 5 * t) + rng.standard_normal(len(t)).astype(np.float32) * 0.2).astype(np.float32)
    elif wave_type == "saw":
        x = (2 * (t * 5 - np.floor(0.5 + t * 5))).astype(np.float32)
    elif wave_type == "saw_bl":
        x = saw_band_limited(t, cfg.saw_f0, cfg.saw_harmonics)
    elif wave_type == "ecg":
        base = (np.sin(t * 10).astype(np.float32) * 0.2)
        spikes = np.exp(-100 * (np.mod(t, 0.2) - 0.1) ** 2).astype(np.float32)
        x = (base + spikes).astype(np.float32)
    else:
        raise ValueError(f"Unknown wave type: {wave_type}")

    # Normalize
    x = x / (np.max(np.abs(x)) + 1e-8)

    # SNR noise
    x = add_noise_by_snr(x, cfg.snr_db, rng)

    # Additional noise
    if cfg.noise_std > 0:
        x = (x + rng.standard_normal(len(x)).astype(np.float32) * float(cfg.noise_std)).astype(np.float32)

    # Renormalize
    x = x / (np.max(np.abs(x)) + 1e-8)
    return x.astype(np.float32)


# =============================================================================
# Data Preparation
# =============================================================================

def make_xy(cfg: EX1Config, wave_type: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    """Generate (x_target, y_anchor) tensors for training."""
    rng = np.random.default_rng(cfg.seed + 12345)
    aux: Dict[str, np.ndarray] = {}

    T = int(cfg.seq_len)
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    aux['t'] = t.copy()

    # Generate waveform
    x = generate_wave(wave_type, t, cfg, rng)
    aux['x_raw'] = x.copy()

    # Reshape to [1, T, 1] for model
    x_use = x.reshape(1, T, 1).astype(np.float32)

    # Anchor generation
    # EX1: Compare SingleGate vs MultiGate with same anchor as EX0
    if cfg.anchor_mode == "same":
        y = x_use.copy()
    elif cfg.anchor_mode == "slow":
        slow = np.sin(2.0 * np.pi * 0.5 * t).astype(np.float32).reshape(1, T, 1)
        y = slow
    elif cfg.anchor_mode == "absgrad":
        dy = np.zeros_like(x_use)
        dy[:, 1:, :] = np.abs(x_use[:, 1:, :] - x_use[:, :-1, :])
        y = dy
    else:
        # Default: same as input
        y = x_use.copy()

    x_t = torch.from_numpy(x_use).to(cfg.device).float()
    y_t = torch.from_numpy(y).to(cfg.device).float()
    return x_t, y_t, aux


# =============================================================================
# Model Builder
# =============================================================================

def build_model(cfg: EX1Config, input_dim: int = 1) -> torch.nn.Module:
    """Build CSCT_Engine with configuration."""
    mcfg = CSCTConfig(
        n_clocks=cfg.n_clocks,
        hidden_dim=cfg.hidden_dim,
        z_dim=cfg.z_dim,
        input_dim=int(input_dim),
        gate_floor=cfg.gate_floor,
        gate_topk=cfg.gate_topk,
        gate_tau=cfg.gate_tau,
        use_gumbel=cfg.use_gumbel,
        gumbel_noise=cfg.gumbel_noise,
        beta=cfg.beta,
    )
    model = CSCT_Engine(mcfg, use_multigate=cfg.use_multigate).to(cfg.device)
    print(f"[MODEL] use_multigate={cfg.use_multigate}")
    return model


# =============================================================================
# Training
# =============================================================================

def run_one_wave(cfg: EX1Config, wave_type: str) -> Dict[str, List[float]]:
    """Train on one waveform and return metrics history."""
    x, y, aux = make_xy(cfg, wave_type)

    model = build_model(cfg, input_dim=int(x.shape[-1]))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist: Dict[str, List[float]] = {
        "step": [], "loss": [], "loss_eval": [], "recon_loss": [],
        "trans_loss": [], "gate_sup": [], "trans_rate": [],
        "na_sparsity": [], "nmda_sparsity": [], "beta": [],
        "unique_codes": [], "K_tokens": [],
        "code_entropy": [], "code_entropy_norm": [],
    }

    for step in range(cfg.steps + 1):
        beta_eff = cfg.beta * min(1.0, step / max(1, cfg.beta_warmup_steps))
        sparsity_eff = cfg.w_sparsity * min(1.0, step / max(1, cfg.sparsity_warmup_steps))

        model.train()
        out = model(x, y, beta=beta_eff)
        loss_train = out["loss"]

        # MultiGate sparsity (if available)
        na_act, nmda_act = 0.0, 0.0
        if out.get("gate_info", None):
            na_tensors, nmda_tensors = [], []
            for gi in out["gate_info"]:
                if isinstance(gi, dict):
                    if "na_sparsity_tensor" in gi:
                        na_tensors.append(gi["na_sparsity_tensor"])
                    if "nmda_sparsity_tensor" in gi:
                        nmda_tensors.append(gi["nmda_sparsity_tensor"])
            if na_tensors and nmda_tensors:
                na_avg = torch.stack(na_tensors).mean()
                nmda_avg = torch.stack(nmda_tensors).mean()
                na_act = float(na_avg.detach().item())
                nmda_act = float(nmda_avg.detach().item())
                sparsity_loss = gate_sparsity_loss(na_avg, cfg.target_na_sparsity) + \
                                gate_sparsity_loss(nmda_avg, cfg.target_nmda_sparsity)
                loss_train = loss_train + sparsity_eff * sparsity_loss

        opt.zero_grad()
        loss_train.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Temperature annealing
        if step > 0 and step % cfg.anneal_every == 0:
            model.anneal_temperature()

        # Logging
        if step % cfg.log_interval == 0:
            model.eval()
            with torch.no_grad():
                out_eval = model(x, y, beta=beta_eff)
                loss_eval = out_eval["loss"]

            tr = gate_transition_rate(out_eval["indices"])
            cs = codebook_stats(out_eval["indices"], K=cfg.n_clocks)

            hist["step"].append(step)
            hist["loss"].append(float(loss_train.item()))
            hist["loss_eval"].append(float(loss_eval.item()))
            hist["recon_loss"].append(float(out_eval["recon_loss"].item()))
            hist["trans_loss"].append(float(out_eval["trans_loss"].item()))
            hist["gate_sup"].append(float(out_eval.get("gate_supervision", torch.tensor(0.0)).item()))
            hist["trans_rate"].append(float(tr.item()))
            hist["na_sparsity"].append(na_act)
            hist["nmda_sparsity"].append(nmda_act)
            hist["beta"].append(beta_eff)
            hist["unique_codes"].append(cs["unique_codes"])
            hist["K_tokens"].append(cs["K"])
            hist["code_entropy"].append(cs["entropy"])
            hist["code_entropy_norm"].append(cs["entropy_norm"])

            print(f"[{wave_type}] step {step:5d} | loss={loss_train.item():.4f} "
                  f"recon={out_eval['recon_loss'].item():.4f} "
                  f"trans_rate={tr.item():.3f} "
                  f"uniq={cs['unique_codes']:.0f}/{cs['K']:.0f} "
                  f"H={cs['entropy']:.3f}")

    # Save final reconstruction plot
    model.eval()
    with torch.no_grad():
        out_final = model(x, y, beta=cfg.beta)

    x_np = x[0, :, 0].cpu().numpy()
    recon_np = out_final["recon"][0, :, 0].cpu().numpy()
    y_np = y[0, :, 0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_np, label="x_target", alpha=0.7)
    ax.plot(recon_np, label="recon", alpha=0.7)
    ax.plot(y_np, label="y_anchor", alpha=0.5)
    ax.set_title(f"EX1 Recon ({wave_type})")
    ax.legend()
    fig.savefig(os.path.join(cfg.output_dir, f"recon_{wave_type}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save convergence curve
    save_convergence_curve(hist, cfg.output_dir, prefix=wave_type)

    return hist


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="CSCT EX1: Waveform Discretization")
    
    # Basic
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=500, help="Training steps (default: 500)")
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="./results_ex1")

    # Model
    p.add_argument("-K", "--n-clocks", type=int, default=8, help="Codebook size K")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--gate-floor", type=float, default=0.10)
    p.add_argument("--gate-topk", type=int, default=1)
    p.add_argument("--gate-tau", type=float, default=0.7)
    p.add_argument("--use-gumbel", action="store_true", default=True)
    p.add_argument("--no-gumbel", dest="use_gumbel", action="store_false")
    p.add_argument("--gumbel-noise", type=float, default=0.5)
    p.add_argument("--use-multigate", action="store_true", default=False)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--tau", "--beta", type=float, default=50.0, dest="beta", help="Transition penalty tau")
    p.add_argument("--tau-warmup-steps", "--beta-warmup-steps", type=int, default=200, dest="beta_warmup_steps", help="Tau warmup steps (default: 200)")

    # Sparsity
    p.add_argument("--w-sparsity", type=float, default=0.0)
    p.add_argument("--sparsity-warmup-steps", type=int, default=100)
    p.add_argument("--target-na-sparsity", type=float, default=0.10)
    p.add_argument("--target-nmda-sparsity", type=float, default=0.30)

    # Wave selection
    p.add_argument("--wave", type=str, default="sine",
                   choices=["sine", "chirp", "am", "fm", "ecg", "saw_bl", "composite", "noisy", "burst", "saw"])
    p.add_argument("--all-waves", action="store_true")

    # Saw params
    p.add_argument("--saw-harmonics", type=int, default=20)
    p.add_argument("--saw-f0", type=float, default=5.0)

    # Noise
    p.add_argument("--snr-db", type=float, default=None)
    p.add_argument("--noise-std", type=float, default=0.0)

    # Anchor
    p.add_argument("--anchor-mode", type=str, default="same", choices=["same", "slow", "absgrad"])

    # Logging
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--anneal-every", type=int, default=25)

    args = p.parse_args()

    cfg = EX1Config(
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        seq_len=args.seq_len,
        n_clocks=args.n_clocks,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        gate_floor=args.gate_floor,
        gate_topk=args.gate_topk,
        gate_tau=args.gate_tau,
        use_gumbel=args.use_gumbel,
        gumbel_noise=args.gumbel_noise,
        use_multigate=args.use_multigate,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        beta=args.beta,
        beta_warmup_steps=args.beta_warmup_steps,
        w_sparsity=args.w_sparsity,
        sparsity_warmup_steps=args.sparsity_warmup_steps,
        target_na_sparsity=args.target_na_sparsity,
        target_nmda_sparsity=args.target_nmda_sparsity,
        wave=args.wave,
        all_waves=args.all_waves,
        saw_harmonics=args.saw_harmonics,
        saw_f0=args.saw_f0,
        snr_db=args.snr_db,
        noise_std=args.noise_std,
        anchor_mode=args.anchor_mode,
        log_interval=args.log_interval,
        anneal_every=args.anneal_every,
        output_dir=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    # Determine waves to run
    if cfg.all_waves:
        waves = ["sine", "chirp", "am", "fm", "ecg", "saw_bl", "composite", "noisy", "burst"]
    else:
        waves = [cfg.wave]

    all_hist = []
    for w in waves:
        print(f"\n{'='*60}\n[EX1] Running wave: {w}\n{'='*60}")
        h = run_one_wave(cfg, w)
        for i, step in enumerate(h["step"]):
            row = {"wave": w, "step": step}
            for k, v in h.items():
                if k != "step":
                    row[k] = v[i]
            all_hist.append(row)

    # Save metrics
    csv_path = os.path.join(cfg.output_dir, "metrics_history.csv")
    if all_hist:
        keys = list(all_hist[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_hist)
        print(f"\n[EX1] Saved metrics to {csv_path}")

    print("\n[EX1] Done.")


if __name__ == "__main__":
    main()
