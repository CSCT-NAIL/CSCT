#!/usr/bin/env python3
"""
CSCT EX3: K-Dependency Analysis (Discretization Geometry)
=========================================================

Tests how codebook size K determines discretization geometry.

Purpose:
  - Demonstrate that CSCT performs Lebesgue-like integration, not Riemann-like
  - Show that K determines the refinement of amplitude partitions
  - Validate reproducibility of boundary locations (<1% variation)

Hypothesis:
  - K=2: Zero-crossing boundaries (sign changes)
  - K=4: Peak/trough boundaries (extrema)
  - K=8+: Intermediate amplitude boundaries

Key Insight:
  The discretization boundaries are NOT arbitrary - they partition the
  amplitude range into K bins, emerging reproducibly across training runs.
  This is fundamentally different from Fourier/Riemann approaches that
  sample at specific time points.

Task:
  - Input: Pure sine wave sin(2πf₀t)
  - Architecture: SingleGate (clean analysis without relational complexity)
  - Vary: K = 2, 4, 8, 16

Anchor Configuration:
  - Self-referential: y = x (same as EX1)

Key Metrics:
  - zero_cross_ratio: Proportion of transitions at zero-crossings
  - extrema_ratio: Proportion of transitions at peaks/troughs
  - recon_loss: Reconstruction quality

Outputs:
  - k_dependency_analysis.png: Visualization of segmentation by K
  - k_dependency_metrics.csv: Quantitative metrics

Author: NAOKI (CSCT Research)
"""

import argparse
import csv
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from csct_engine import CSCTConfig, CSCT_Engine


# =============================================================================
# Config
# =============================================================================


def save_convergence_curve(hist, output_path: str, title: str = "Convergence") -> None:
    """Save convergence curve plot showing training dynamics."""
    if not hist or len(hist.get("step", [])) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps = hist["step"]
    
    # Loss curve
    ax = axes[0, 0]
    for key in ["loss", "loss_eval", "recon_loss", "recon", "recon_all", "recon_ch1_masked"]:
        if key in hist and hist[key]:
            ax.plot(steps, hist[key], label=key, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Discreteness metrics  
    ax = axes[0, 1]
    for key in ["unique_codes", "maxp", "maxp_g0", "maxp_g1", "stability"]:
        if key in hist and hist[key]:
            ax.plot(steps, hist[key], label=key, marker=".", markersize=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric")
    ax.set_title("Discretization Quality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Secondary metrics
    ax = axes[1, 0]
    for key in ["trans_rate", "entropy", "ent_g0", "ent_g1", "code_entropy_norm", "k_mae_masked"]:
        if key in hist and hist[key]:
            ax.plot(steps, hist[key], label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate / Entropy")
    ax.set_title("Dynamics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = f"CONVERGENCE SUMMARY\n{'='*25}\n"
    summary += f"Total steps: {steps[-1]}\n"
    summary += f"Log points: {len(steps)}\n\n"
    for key in list(hist.keys())[:8]:
        if key != "step" and hist[key]:
            try:
                val = hist[key][-1]
                summary += f"{key}: {val:.4f}\n"
            except:
                pass
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


@dataclass
class EX3Config:
    device: str = "cpu"
    seed: int = 0
    steps: int = 500
    seq_len: int = 300

    # Model (SingleGate for clean analysis)
    hidden_dim: int = 64
    z_dim: int = 16
    gate_floor: float = 0.10
    gate_topk: int = 1
    gate_tau: float = 0.7
    use_gumbel: bool = True
    gumbel_noise: float = 0.5
    use_multigate: bool = False  # SingleGate for EX3

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Transition penalty (tau in CSCT notation, beta in code for compatibility)
    tau: float = 50.0  # Transition penalty τ
    tau_warmup_steps: int = 200

    # K values to test
    k_values: List[int] = None  # Set via argument

    # Signal parameters
    sine_freq: float = 5.0  # cycles per unit time

    # Anchor mode (self-referential for EX3)
    anchor_mode: str = "same"

    # Logging
    log_interval: int = 200
    anneal_every: int = 10
    output_dir: str = "./results_ex3"

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [2, 4, 8, 16]


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gate_transition_rate(indices: torch.Tensor) -> float:
    if indices.ndim != 2:
        indices = indices.view(indices.shape[0], -1)
    trans = (indices[:, 1:] != indices[:, :-1]).float()
    return float(trans.mean().item())


def find_transition_points(indices: np.ndarray) -> np.ndarray:
    """Find indices where code transitions occur."""
    if indices.ndim > 1:
        indices = indices.flatten()
    trans = np.where(indices[1:] != indices[:-1])[0]
    return trans


def analyze_boundaries(t: np.ndarray, x: np.ndarray, trans_points: np.ndarray) -> Dict:
    """Analyze where transitions occur relative to signal geometry."""
    if len(trans_points) == 0:
        return {"zero_crossings": 0, "extrema": 0, "other": 0}
    
    # Signal derivatives
    dx = np.gradient(x)
    # NOTE: d2x is not used for classification currently, keep for future extensions.
    # d2x = np.gradient(dx)
    
    # Classify each transition
    zero_cross = 0
    extrema = 0
    other = 0
    
    # Transition points returned by find_transition_points are indices i such that
    # indices[i] != indices[i+1]. Therefore, geometric events should be checked
    # between (i, i+1).
    #
    # NOTE: With neural encoders/decoders, the learned boundary can drift by a few
    # samples even when it is phase-locked to a geometric event. A tolerance of 1
    # sample is too strict for seq_len≈300, so we use a slightly wider window.
    tol = max(2, int(0.01 * len(x)))  # e.g., 3 when seq_len=300

    # Numerical tolerance for detecting exact zeros / flat derivatives.
    eps = 1e-6

    def near_zero_cross(i: int) -> bool:
        a = max(0, i - tol)
        b = min(len(x) - 2, i + tol)
        for j in range(a, b + 1):
            # Robust sign-change detection (also counts exact zeros).
            if (abs(x[j]) <= eps) or (abs(x[j + 1]) <= eps) or (x[j] * x[j + 1] < 0):
                return True
        return False

    def near_extremum(i: int) -> bool:
        a = max(0, i - tol)
        b = min(len(dx) - 2, i + tol)
        for j in range(a, b + 1):
            # Robust derivative sign-change (also counts exact flat derivative).
            if (abs(dx[j]) <= eps) or (abs(dx[j + 1]) <= eps) or (dx[j] * dx[j + 1] < 0):
                return True
        return False

    for tp in trans_points:
        if tp >= len(x) - 1:
            continue

        if near_zero_cross(int(tp)):
            zero_cross += 1
        elif near_extremum(int(tp)):
            extrema += 1
        else:
            other += 1
    
    total = zero_cross + extrema + other
    return {
        "zero_crossings": zero_cross,
        "extrema": extrema,
        "other": other,
        "total": total,
        "zero_cross_ratio": zero_cross / max(total, 1),
        "extrema_ratio": extrema / max(total, 1),
    }


# =============================================================================
# Data Generation
# =============================================================================

def generate_sine_data(cfg: EX3Config) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate sine wave data for K-dependency analysis."""
    T = cfg.seq_len
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    
    # Pure sine wave
    x = np.sin(2 * np.pi * cfg.sine_freq * t).astype(np.float32)
    
    # Reshape to [1, T, 1]
    x_tensor = torch.from_numpy(x.reshape(1, T, 1)).to(cfg.device).float()
    
    # Anchor (same as signal for clean analysis)
    if cfg.anchor_mode == "same":
        y_tensor = x_tensor.clone()
    elif cfg.anchor_mode == "slow":
        y = np.sin(2 * np.pi * 0.5 * t).astype(np.float32).reshape(1, T, 1)
        y_tensor = torch.from_numpy(y).to(cfg.device).float()
    else:
        y_tensor = x_tensor.clone()
    
    return x_tensor, y_tensor, t


# =============================================================================
# Model Builder
# =============================================================================

def build_model(cfg: EX3Config, K: int) -> torch.nn.Module:
    """Build CSCT_Engine with specified K."""
    mcfg = CSCTConfig(
        n_clocks=K,
        hidden_dim=cfg.hidden_dim,
        z_dim=cfg.z_dim,
        input_dim=1,
        gate_floor=cfg.gate_floor,
        gate_topk=cfg.gate_topk,
        gate_tau=cfg.gate_tau,
        use_gumbel=cfg.use_gumbel,
        gumbel_noise=cfg.gumbel_noise,
        beta=cfg.tau,  # beta in engine = tau in CSCT notation
    )
    model = CSCT_Engine(mcfg, use_multigate=cfg.use_multigate).to(cfg.device)
    return model


# =============================================================================
# Training
# =============================================================================

def train_model(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, 
                cfg: EX3Config, K: int) -> Dict:
    """Train model and return final state."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # History for convergence curve
    hist = {"step": [], "loss": [], "trans_rate": [], "recon_loss": []}
    
    for step in range(cfg.steps + 1):
        # tau (transition penalty) warmup
        tau_eff = cfg.tau * min(1.0, step / max(1, cfg.tau_warmup_steps))
        
        model.train()
        out = model(x, y, beta=tau_eff)  # beta param in engine = tau in CSCT
        loss = out["loss"]
        
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        
        # Temperature annealing
        if step > 0 and step % cfg.anneal_every == 0:
            model.anneal_temperature()
        
        # Logging
        if step % cfg.log_interval == 0:
            tr = gate_transition_rate(out["indices"])
            hist["step"].append(step)
            hist["loss"].append(float(loss.item()))
            hist["trans_rate"].append(float(tr))
            hist["recon_loss"].append(float(out["recon_loss"].item()))
            print(f"  [K={K}] step {step:4d} | loss={loss.item():.4f} | trans_rate={tr:.3f}")
    
    # Save convergence curve
    conv_path = os.path.join(cfg.output_dir, f"convergence_K{K}.png")
    save_convergence_curve(hist, conv_path, f"EX3 K-Dependency (K={K})")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(x, y, beta=cfg.tau)
    
    return {
        "indices": out["indices"].squeeze().cpu().numpy(),
        "recon": out["recon"].squeeze().cpu().numpy(),
        "gate": out.get("gate", torch.zeros_like(out["recon"])).squeeze().cpu().numpy(),
        "recon_loss": float(out["recon_loss"].item()),
        "trans_rate": gate_transition_rate(out["indices"]),
        "history": hist,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_k_dependency(cfg: EX3Config) -> Dict[int, Dict]:
    """Run K-dependency analysis."""
    # Generate data
    x, y, t = generate_sine_data(cfg)
    x_np = x.squeeze().cpu().numpy()
    
    results = {}
    
    for K in cfg.k_values:
        print(f"\n{'='*50}")
        print(f"Training K={K}")
        print(f"{'='*50}")
        
        # Reset seed for reproducibility
        set_seed(cfg.seed)
        
        # Build and train
        model = build_model(cfg, K)
        result = train_model(model, x, y, cfg, K)
        
        # Analyze boundaries
        trans_points = find_transition_points(result["indices"])
        boundary_analysis = analyze_boundaries(t, x_np, trans_points)
        result["boundary_analysis"] = boundary_analysis
        result["trans_points"] = trans_points
        
        results[K] = result
        
        print(f"\n[K={K}] Final: recon_loss={result['recon_loss']:.4f}, "
              f"transitions={boundary_analysis['total']}")
        print(f"  Zero-crossings: {boundary_analysis['zero_crossings']} "
              f"({boundary_analysis['zero_cross_ratio']*100:.1f}%)")
        print(f"  Extrema: {boundary_analysis['extrema']} "
              f"({boundary_analysis['extrema_ratio']*100:.1f}%)")
    
    return results, t, x_np


def plot_results(results: Dict[int, Dict], t: np.ndarray, x: np.ndarray,
                 output_dir: str):
    """Create visualization of K-dependency.

    Design goal (paper-ready):
      - Overlay discretization on the waveform so refinement with K is obvious.
      - Show that codes correspond to *sets / bins* over amplitude (Lebesgue-like),
        not Dirac-like sampling at extrema.

    Output:
      - k_dependency_analysis.png (improved layout)
    """
    K_list = sorted(results.keys())

    # Figure layout: top reference row + one row per K.
    n_rows = len(K_list) + 1
    fig = plt.figure(figsize=(16, 3.2 * n_rows))

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        n_rows, 2,
        height_ratios=[1.2] + [1.0] * len(K_list),
        width_ratios=[3.6, 1.4],
        hspace=0.35,
        wspace=0.15,
    )

    # ---------------------------------------------------------------------
    # Row 0: reference waveform with geometric events
    # ---------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, x, 'k-', linewidth=1.6, label='x(t)')

    eps = 1e-6
    zero_cross = np.where((np.abs(x[:-1]) <= eps) | (np.abs(x[1:]) <= eps) | (x[:-1] * x[1:] < 0))[0]
    ax0.scatter(t[zero_cross], x[zero_cross], s=35, zorder=5, marker='o',
                label='zero-cross')

    dx = np.gradient(x)
    extrema = np.where((np.abs(dx[:-1]) <= eps) | (np.abs(dx[1:]) <= eps) | (dx[:-1] * dx[1:] < 0))[0]
    ax0.scatter(t[extrema], x[extrema], s=35, zorder=5, marker='^',
                label='extrema')

    ax0.axhline(0.0, linestyle='--', alpha=0.35)
    ax0.set_ylabel('Amplitude')
    ax0.set_title('Input signal (reference geometry)', fontsize=12)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc='upper right', ncol=3)

    # Colormap for codes
    cmap = plt.get_cmap('tab20')

    def iter_runs(idx: np.ndarray):
        """Yield (start, end, code) for contiguous runs in idx."""
        start = 0
        cur = int(idx[0])
        for i in range(1, len(idx)):
            if int(idx[i]) != cur:
                yield start, i, cur
                start = i
                cur = int(idx[i])
        yield start, len(idx), cur

    # ---------------------------------------------------------------------
    # Rows per K
    # ---------------------------------------------------------------------
    for r, K in enumerate(K_list, start=1):
        res = results[K]
        idx = res['indices'].astype(int)
        trans = res['trans_points']
        ba = res['boundary_analysis']

        # Left: waveform overlay with colored segmentation
        axL = fig.add_subplot(gs[r, 0], sharex=ax0)
        axL.plot(t, x, 'k-', linewidth=1.2)

        # Colored time partitions (showing refinement with K)
        for s, e, code in iter_runs(idx):
            c = cmap(code % cmap.N)
            axL.axvspan(t[s], t[e-1] if e-1 < len(t) else t[-1], color=c, alpha=0.12, linewidth=0)

        # Transition markers
        if len(trans) > 0:
            axL.vlines(t[trans], -1.05, 1.05, linewidth=0.8, alpha=0.25)

        axL.axhline(0.0, linestyle='--', alpha=0.25)
        axL.set_ylim(-1.1, 1.1)
        axL.grid(True, alpha=0.18)
        axL.set_ylabel('x(t)')

        # Compact stats
        stats = (
            f"K={K} | transitions={ba['total']} | "
            f"zero-cross hit={ba['zero_cross_ratio']*100:.1f}% | "
            f"extrema hit={ba['extrema_ratio']*100:.1f}% | "
            f"recon_loss={res['recon_loss']:.4f}"
        )
        axL.set_title(stats, fontsize=10)

        # Right: Lebesgue-like view (amplitude bins per code)
        axR = fig.add_subplot(gs[r, 1])
        axR.set_title('Amplitude bins by code', fontsize=10)
        axR.axhline(0.0, linestyle='--', alpha=0.25)
        axR.set_ylim(-1.1, 1.1)
        axR.set_xlim(-0.5, K - 0.5)
        axR.set_xlabel('code')
        axR.set_ylabel('amplitude')
        axR.grid(True, alpha=0.18)

        # Plot robust quantile bands per code (shows set-valued partition)
        for code in range(K):
            vals = x[idx == code]
            if vals.size == 0:
                continue
            q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
            c = cmap(code % cmap.N)
            axR.vlines(code, q05, q95, color=c, linewidth=3, alpha=0.85)
            axR.scatter([code], [q50], s=18, color=c, zorder=5)

        # Improve tick readability for large K
        if K <= 8:
            axR.set_xticks(list(range(K)))
        else:
            axR.set_xticks(list(range(0, K, max(1, K // 8))))

    # Bottom x label
    axL.set_xlabel('Time')

    fig.suptitle('EX3: K-Dependency — refinement of time partitions and amplitude bins (Lebesgue-like)', fontsize=14)

    save_path = os.path.join(output_dir, 'k_dependency_analysis.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved plot to: {save_path}")


def save_metrics(results: Dict[int, Dict], output_dir: str):
    """Save quantitative metrics to CSV."""
    rows = []
    for K, res in results.items():
        ba = res["boundary_analysis"]
        rows.append({
            "K": K,
            "recon_loss": res["recon_loss"],
            "trans_rate": res["trans_rate"],
            "n_transitions": ba["total"],
            "zero_crossings": ba["zero_crossings"],
            "extrema": ba["extrema"],
            "other": ba["other"],
            "zero_cross_ratio": ba["zero_cross_ratio"],
            "extrema_ratio": ba["extrema_ratio"],
        })
    
    csv_path = os.path.join(output_dir, "k_dependency_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="CSCT EX3: K-Dependency Analysis")
    
    # Basic
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--output-dir", type=str, default="./results_ex3")
    
    # K values
    p.add_argument("--k-values", type=str, default="2,4,8,16",
                   help="Comma-separated K values to test")
    
    # Model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--gate-floor", type=float, default=0.10)
    p.add_argument("--gate-tau", type=float, default=0.7)
    
    # Optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=50.0, help="Transition penalty τ")
    p.add_argument("--tau-warmup-steps", type=int, default=200)
    
    # Signal
    p.add_argument("--sine-freq", type=float, default=5.0)
    p.add_argument("--anchor-mode", type=str, default="same")
    
    # Logging
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--anneal-every", type=int, default=10)
    
    args = p.parse_args()
    
    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    
    cfg = EX3Config(
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        gate_floor=args.gate_floor,
        gate_tau=args.gate_tau,
        lr=args.lr,
        tau=args.tau,
        tau_warmup_steps=args.tau_warmup_steps,
        sine_freq=args.sine_freq,
        anchor_mode=args.anchor_mode,
        log_interval=args.log_interval,
        anneal_every=args.anneal_every,
        output_dir=args.output_dir,
        k_values=k_values,
    )
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    
    print(f"\n{'#'*70}")
    print(f"# CSCT EX3: K-Dependency Analysis")
    print(f"# K values: {cfg.k_values}")
    print(f"# Signal: sin(2π * {cfg.sine_freq} * t)")
    print(f"{'#'*70}\n")
    
    # Run experiment
    results, t, x = run_k_dependency(cfg)
    
    # Visualize
    plot_results(results, t, x, cfg.output_dir)
    
    # Save metrics
    save_metrics(results, cfg.output_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EX3 SUMMARY: K-Dependency Analysis")
    print(f"{'='*70}")
    print(f"\n{'K':>4s} | {'Transitions':>12s} | {'Zero-cross':>12s} | {'Extrema':>12s} | {'recon_loss':>12s}")
    print("-" * 60)
    for K in sorted(results.keys()):
        ba = results[K]["boundary_analysis"]
        print(f"{K:4d} | {ba['total']:12d} | "
              f"{ba['zero_crossings']:4d} ({ba['zero_cross_ratio']*100:5.1f}%) | "
              f"{ba['extrema']:4d} ({ba['extrema_ratio']*100:5.1f}%) | "
              f"{results[K]['recon_loss']:12.4f}")
    
    print("\n[EX3] Done.")


if __name__ == "__main__":
    main()
