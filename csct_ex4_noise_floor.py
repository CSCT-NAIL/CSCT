#!/usr/bin/env python3
"""
CSCT EX4: Anchor Role — Noise Floor vs Drift
=============================================

Tests the fundamental role of anchors in CSCT: long-term stability vs short-term accuracy.

Purpose:
  - Demonstrate that anchors provide LONG-TERM STABILITY, not immediate accuracy
  - Show the tradeoff between noise import (anchored) and drift (free-running)
  - Validate A4 (Irreversible Anchor) through a simple toy model

Hypothesis:
  - Short-term (t < T_cross): Closed system wins (anchor imports noise)
  - Long-term (t > T_cross): Open system wins (internal integrator drifts)
  - Crossover point T_cross depends on noise level σ

Task Design:
  - Target: Clean sine wave sin(t)
  - Anchor: Noisy sawtooth (intentionally mismatched waveform + Gaussian noise)
  
  Two Modes:
    - Closed (Free-run): Pure internal integrator, ignores anchor
    - Open (Anchored): Continuously corrected by noisy anchor (PLL-style)

Critical Parameters:
  - noise_levels = [0.05, 0.1, 0.2, 0.3]: Range of anchor noise σ
      These span from "clean" (σ=0.05) to "very noisy" (σ=0.3) conditions.
      The experiment tests whether anchors remain beneficial despite noise.
  
  - freq_error = 0.03 (3%): Internal integrator frequency mismatch
      Models realistic imperfection in internal oscillators.
      Without this error, the integrator would be unrealistically perfect.
      3% is biologically plausible (neural oscillator variability).
  
  - pll_alpha = 1.0: Full instantaneous phase lock
      α=1.0 means the anchored system fully imports anchor noise.
      This is the "worst case" for anchor - if it still wins long-term,
      the anchor's role in drift prevention is conclusively demonstrated.
      (α<1.0 would low-pass filter noise but also introduce lag)

Theoretical Significance:
  This demonstrates WHY irreversible anchors are necessary in CSCT:
  - NOT for immediate accuracy (internal model is better short-term)
  - BUT for PREVENTING DRIFT (external reference provides ground truth)
  
  Biological analogy:
  - Sensory systems need external calibration to stay accurate
  - Pure imagination (no anchor) can drift arbitrarily

Key Metrics:
  - short_mse: MSE in early time window (t < 100)
  - long_mse: MSE in late time window (t > 1500)
  - crossover_idx: Step where cumulative MSE of closed exceeds open

Anchor Configuration:
  - External reference: y = noisy sawtooth (different from target)
  - PLL α = 1.0: Full noise import for honest comparison

Outputs:
  - ex4_noise_floor.png: Visualization of drift vs noise floor
  - ex4_metrics.csv: Quantitative results

Author: NAOKI (CSCT Research)
"""

import argparse
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# Config
# =============================================================================

@dataclass  
class EX4Config:
    # Signal parameters
    seq_len: int = 2000      # Extended for drift visibility
    t_max: float = 200.0     # Much longer to see drift

    # Reproducibility
    seed: int = 0
    
    # Noise levels to test (default: span clean to very noisy)
    noise_levels: List[float] = None
    
    # Analysis windows
    short_term_end: int = 100
    long_term_start: int = 1500
    
    # Integrator imperfection: 3% frequency error (biologically plausible)
    freq_error: float = 0.03

    # Anchor interpretation / locking (simple PLL-style)
    # alpha=1.0: perfect instantaneous lock (imports full anchor noise)
    # This is the "worst case" for anchor - honest comparison
    pll_alpha: float = 1.0

    # Treat anchor as a bounded phase observable in [-1, 1]
    anchor_clip: bool = True
    
    # Output
    output_dir: str = "./results_ex4"
    
    def __post_init__(self):
        if self.noise_levels is None:
            # Default: span from clean (0.05) to very noisy (0.3)
            self.noise_levels = [0.05, 0.1, 0.2, 0.3]


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(cfg: EX4Config, noise_std: float, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate target signal and noisy anchor.
    
    Target: Clean sine wave (what we want to reconstruct)
    Anchor: Sawtooth + noise (intentionally different waveform + noise)
    
    This tests whether the model can use a NOISY, MISMATCHED reference
    to stay on track long-term.
    """
    np.random.seed(seed)
    
    # Use a dt that is exactly consistent with the generated timebase.
    # np.linspace(0, t_max, seq_len) implies dt = t_max/(seq_len-1).
    dt = cfg.t_max / max(cfg.seq_len - 1, 1)
    t = np.arange(cfg.seq_len, dtype=np.float64) * dt
    
    # Target: Clean sine wave
    target = np.sin(t).reshape(1, -1, 1).astype(np.float32)
    
    # Anchor: Sawtooth (different waveform) + Gaussian noise
    # Sawtooth has same period as sine, but different shape
    phase = np.mod(t, 2 * np.pi)
    saw = (phase / (2 * np.pi)) * 2 - 1  # Maps to [-1, 1]
    noise = np.random.normal(0, noise_std, saw.shape)
    anchor = (saw + noise).reshape(1, -1, 1).astype(np.float32)
    
    return torch.from_numpy(target), torch.from_numpy(anchor), t


# =============================================================================
# Models
# =============================================================================

class HarmonicIntegrator(nn.Module):
    """Pure internal integrator (harmonic oscillator) with imperfection.
    
    Simulates what happens when the model runs "closed" - 
    using only its internal dynamics without anchor correction.
    
    Physics: z'' = -ω²z (simple harmonic motion)
    Discrete: Symplectic Euler integration
    
    Key: We add a small frequency error to simulate realistic model imperfection.
    Without this, the symplectic integrator is too perfect.
    """
    def __init__(self, dt: float = 0.1, freq_error: float = 0.001):
        super().__init__()
        self.dt = dt
        # Slightly wrong frequency: ω = 1 + error instead of exactly 1
        self.omega_sq = (1.0 + freq_error) ** 2
        
    def forward(self, z0: torch.Tensor, v0: torch.Tensor, steps: int) -> torch.Tensor:
        """Run the integrator for `steps` timesteps.
        
        Args:
            z0: Initial position (should be sin(0) = 0 for sine wave)
            v0: Initial velocity (should be cos(0) = 1 for sine wave)
            steps: Number of steps to integrate
            
        Returns:
            Tensor of shape [1, steps, 1] containing predictions
        """
        preds = []
        z = z0.clone().squeeze()
        v = v0.clone().squeeze()
        
        for _ in range(steps):
            preds.append(z.view(1, 1, 1))
            # Symplectic Euler with slightly wrong frequency
            v_new = v - self.omega_sq * z * self.dt
            z_new = z + v_new * self.dt
            z, v = z_new, v_new
            
        return torch.cat(preds, dim=1)


def _wrap_angle_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap an angle to [-pi, pi]."""
    two_pi = 2.0 * math.pi
    return (x + math.pi) - two_pi * torch.floor((x + math.pi) / two_pi) - math.pi


def anchored_reconstruction(anchor: torch.Tensor, *, pll_alpha: float = 1.0, anchor_clip: bool = True) -> torch.Tensor:
    """Phase-lock to a noisy anchor (PLL-style).

    Anchor is treated as a bounded phase observable in [-1, 1] (sawtooth).
    We map it to phase in [0, 2π] and reconstruct sin(phase).

    pll_alpha controls how strongly we follow instantaneous anchor phase:
      - alpha = 1.0: perfect instantaneous lock (imports full noise)
      - alpha < 1.0: low-pass lock (less noise import, more lag)

    This keeps the experiment honest: "anchored" need not be a magical
    perfect extractor; it is a controllable tradeoff between drift prevention
    and imported noise.
    """
    if anchor_clip:
        anchor = torch.clamp(anchor, -1.0, 1.0)

    # Map anchor value in [-1, 1] to phase in [0, 2π]
    phase_obs = (anchor + 1.0) * math.pi  # [B, T, 1]

    # Fast path: perfect lock
    if pll_alpha >= 1.0 - 1e-9:
        return torch.sin(phase_obs)

    # Recursive lock: phase_hat[t] = phase_hat[t-1] + alpha * wrap(phase_obs[t] - phase_hat[t-1])
    B, T, C = phase_obs.shape
    assert C == 1, "EX4 expects a scalar anchor phase observation"
    phase_hat = torch.empty_like(phase_obs)
    phase_hat[:, 0:1] = phase_obs[:, 0:1]
    for i in range(1, T):
        err = _wrap_angle_pi(phase_obs[:, i:i+1] - phase_hat[:, i-1:i])
        phase_hat[:, i:i+1] = phase_hat[:, i-1:i] + pll_alpha * err

    return torch.sin(phase_hat)


# =============================================================================
# Analysis
# =============================================================================

def analyze_errors(pred_closed: torch.Tensor, pred_open: torch.Tensor, 
                   target: torch.Tensor, cfg: EX4Config) -> Dict:
    """Compute MSE errors over different time windows."""

    T = pred_closed.shape[1]
    # Clamp analysis windows to sequence length to avoid empty slices.
    short_end = min(max(int(cfg.short_term_end), 1), T)
    if cfg.long_term_start >= T:
        long_start = max(T - max(1, T // 4), 0)  # last quarter as fallback
    else:
        long_start = min(max(int(cfg.long_term_start), 0), T - 1)
    
    mse_closed = (pred_closed - target) ** 2
    mse_open = (pred_open - target) ** 2
    
    # Short-term analysis
    short_mse_closed = mse_closed[:, :short_end].mean().item()
    short_mse_open = mse_open[:, :short_end].mean().item()
    
    # Long-term analysis
    long_mse_closed = mse_closed[:, long_start:].mean().item()
    long_mse_open = mse_open[:, long_start:].mean().item()
    
    # Find crossover point (where closed starts losing to open)
    cumulative_mse_closed = mse_closed.squeeze().cumsum(dim=0) / (torch.arange(1, mse_closed.shape[1] + 1).float())
    cumulative_mse_open = mse_open.squeeze().cumsum(dim=0) / (torch.arange(1, mse_open.shape[1] + 1).float())
    
    crossover_mask = cumulative_mse_closed > cumulative_mse_open
    if crossover_mask.any():
        crossover_idx = crossover_mask.float().argmax().item()
        if crossover_idx == 0 and not crossover_mask[0]:
            crossover_idx = T  # Never crosses
    else:
        crossover_idx = T  # Closed always wins
    
    return {
        "short_mse_closed": short_mse_closed,
        "short_mse_open": short_mse_open,
        "long_mse_closed": long_mse_closed,
        "long_mse_open": long_mse_open,
        "short_winner": "closed" if short_mse_closed < short_mse_open else "open",
        "long_winner": "closed" if long_mse_closed < long_mse_open else "open",
        "crossover_idx": crossover_idx,
        "mse_closed": mse_closed.squeeze().numpy(),
        "mse_open": mse_open.squeeze().numpy(),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_single_noise_level(cfg: EX4Config, noise_std: float) -> Dict:
    """Run experiment for a single noise level."""
    
    # Generate data
    target, anchor, t = generate_data(cfg, noise_std, seed=cfg.seed)
    
    # Closed model (internal integrator with small frequency error)
    # dt must match the generated timebase (t_max/(seq_len-1)).
    dt = cfg.t_max / max(cfg.seq_len - 1, 1)
    integrator = HarmonicIntegrator(dt=dt, freq_error=cfg.freq_error)
    z0 = torch.tensor([0.0])  # sin(0) = 0
    v0 = torch.tensor([1.0])  # cos(0) = 1
    pred_closed = integrator(z0, v0, cfg.seq_len)
    
    # Open model (anchored, PLL-style lock)
    pred_open = anchored_reconstruction(anchor, pll_alpha=cfg.pll_alpha, anchor_clip=cfg.anchor_clip)
    
    # Analyze
    results = analyze_errors(pred_closed, pred_open, target, cfg)
    results["noise_std"] = noise_std
    results["pred_closed"] = pred_closed.squeeze().numpy()
    results["pred_open"] = pred_open.squeeze().numpy()
    results["target"] = target.squeeze().numpy()
    results["anchor"] = anchor.squeeze().numpy()
    results["t"] = t
    
    return results


def run_experiment(cfg: EX4Config) -> List[Dict]:
    """Run experiment across all noise levels."""
    all_results = []
    
    for noise in cfg.noise_levels:
        print(f"\nNoise level σ = {noise}")
        result = run_single_noise_level(cfg, noise)
        all_results.append(result)
        
        print(f"  Short-term MSE: Closed={result['short_mse_closed']:.6f}, "
              f"Open={result['short_mse_open']:.6f} → Winner: {result['short_winner'].upper()}")
        print(f"  Long-term MSE:  Closed={result['long_mse_closed']:.6f}, "
              f"Open={result['long_mse_open']:.6f} → Winner: {result['long_winner'].upper()}")
        print(f"  Crossover at step={result['crossover_idx']}")
    
    return all_results


def plot_results(results: List[Dict], cfg: EX4Config):
    """Create comprehensive visualization."""
    
    n_noise = len(results)
    fig, axes = plt.subplots(n_noise + 1, 2, figsize=(14, 4 * (n_noise + 1)))
    
    # Plot each noise level
    for i, res in enumerate(results):
        noise = res["noise_std"]
        t = res["t"]

        # Clamp window markers to avoid index errors on short debug runs.
        idx_short = min(max(int(cfg.short_term_end), 0), len(t) - 1)
        idx_long = min(max(int(cfg.long_term_start), 0), len(t) - 1)
        
        # Left: Waveform comparison
        ax = axes[i, 0]
        ax.plot(t, res["target"], 'k-', alpha=0.5, linewidth=2, label='Target (clean sine)')
        ax.plot(t, res["pred_closed"], 'b-', alpha=0.8, label='Closed (internal integrator)')
        ax.plot(t, res["pred_open"], 'r--', alpha=0.6,
                label=f'Open (anchored PLL α={cfg.pll_alpha:g})')
        ax.axvline(x=t[idx_short], color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=t[idx_long], color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f"Anchor Noise σ = {noise}")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Right: MSE over time
        ax = axes[i, 1]
        window = 20  # Smoothing window
        mse_closed_smooth = np.convolve(res["mse_closed"], np.ones(window)/window, mode='valid')
        mse_open_smooth = np.convolve(res["mse_open"], np.ones(window)/window, mode='valid')
        # Align lengths: valid mode gives len(input) - len(kernel) + 1
        n_smooth = len(mse_closed_smooth)
        t_smooth = np.linspace(t[0], t[-1], n_smooth)
        
        ax.semilogy(t_smooth, mse_closed_smooth, 'b-', label='Closed (drift)', alpha=0.8)
        ax.semilogy(t_smooth, mse_open_smooth, 'r-', label=f'Open (noise floor, PLL α={cfg.pll_alpha:g})', alpha=0.8)
        ax.axvline(x=t[res["crossover_idx"]] if res["crossover_idx"] < len(t) else t[-1], 
                   color='green', linestyle='--', alpha=0.7, label=f'Crossover')
        ax.axvline(x=t[idx_short], color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=t[idx_long], color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f"MSE over Time (Anchor Noise σ = {noise})")
        ax.set_ylabel("MSE (log scale)")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Summary plot
    ax = axes[n_noise, 0]
    noises = [r["noise_std"] for r in results]
    short_closed = [r["short_mse_closed"] for r in results]
    short_open = [r["short_mse_open"] for r in results]
    long_closed = [r["long_mse_closed"] for r in results]
    long_open = [r["long_mse_open"] for r in results]
    
    x = np.arange(len(noises))
    width = 0.35
    
    ax.bar(x - width/2, short_closed, width, label='Closed', color='blue', alpha=0.7)
    ax.bar(x + width/2, short_open, width, label='Open', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ={n}" for n in noises])
    ax.set_ylabel("MSE")
    ax.set_title("Short-Term MSE: Closed WINS (no noise import)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[n_noise, 1]
    ax.bar(x - width/2, long_closed, width, label='Closed', color='blue', alpha=0.7)
    ax.bar(x + width/2, long_open, width, label='Open', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ={n}" for n in noises])
    ax.set_ylabel("MSE")
    ax.set_title("Long-Term MSE: Open WINS (no drift)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    
    save_path = os.path.join(cfg.output_dir, "ex4_noise_floor.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to: {save_path}")


def save_metrics_csv(results: List[Dict], cfg: EX4Config):
    """Save metrics to CSV for reproducibility."""
    import csv
    
    csv_path = os.path.join(cfg.output_dir, "ex4_metrics.csv")
    
    fieldnames = [
        "noise_std", "short_mse_closed", "short_mse_open", "short_winner",
        "long_mse_closed", "long_mse_open", "long_winner", "crossover_idx",
        "freq_error", "pll_alpha"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = {
                "noise_std": res["noise_std"],
                "short_mse_closed": res["short_mse_closed"],
                "short_mse_open": res["short_mse_open"],
                "short_winner": res["short_winner"],
                "long_mse_closed": res["long_mse_closed"],
                "long_mse_open": res["long_mse_open"],
                "long_winner": res["long_winner"],
                "crossover_idx": res["crossover_idx"],
                "freq_error": cfg.freq_error,
                "pll_alpha": cfg.pll_alpha,
            }
            writer.writerow(row)
    
    print(f"Saved metrics to: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="CSCT EX4: Anchor Role - Noise Floor vs Drift Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Critical Parameters:
  --noise-levels: Range of anchor noise σ (default: 0.05 0.1 0.2 0.3)
      Tests whether anchors remain beneficial despite noise.
      
  --freq-error: Internal integrator frequency mismatch (default: 0.03 = 3%)
      Models realistic imperfection. Without this, integrator is too perfect.
      
  --pll-alpha: Phase lock strength (default: 1.0 = full noise import)
      α=1.0 is worst case for anchor - if it still wins long-term,
      drift prevention is conclusively demonstrated.

Example:
  python csct_ex4_noise_floor.py --noise-levels 0.05 0.1 0.2 0.3 --freq-error 0.03 --pll-alpha 1.0
"""
    )
    
    p.add_argument("--seq-len", type=int, default=2000,
                   help="Sequence length (default: 2000)")
    p.add_argument("--t-max", type=float, default=200.0,
                   help="Maximum time (default: 200.0)")
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3],
                   help="Anchor noise levels σ (default: 0.05 0.1 0.2 0.3)")
    p.add_argument("--short-term-end", type=int, default=100,
                   help="End of short-term analysis window (default: 100)")
    p.add_argument("--long-term-start", type=int, default=1500,
                   help="Start of long-term analysis window (default: 1500)")
    p.add_argument("--freq-error", type=float, default=0.03,
                   help="Integrator frequency error, 0.03 = 3%% (default: 0.03)")
    p.add_argument("--pll-alpha", type=float, default=1.0,
                   help="Anchor lock strength. 1.0=full noise import (worst case, default)")
    p.add_argument("--no-anchor-clip", action="store_true",
                   help="Disable clipping anchor to [-1,1] before mapping to phase")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for anchor noise generation (default: 0)")
    p.add_argument("--output-dir", type=str, default="./results_ex4",
                   help="Output directory (default: ./results_ex4)")
    
    args = p.parse_args()
    
    cfg = EX4Config(
        seq_len=args.seq_len,
        t_max=args.t_max,
        seed=args.seed,
        noise_levels=args.noise_levels,
        short_term_end=args.short_term_end,
        long_term_start=args.long_term_start,
        freq_error=args.freq_error,
        pll_alpha=args.pll_alpha,
        anchor_clip=not args.no_anchor_clip,
        output_dir=args.output_dir,
    )

    # Optional but recommended: control torch RNG as well (future-proof)
    torch.manual_seed(cfg.seed)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# CSCT EX4: Anchor Role — Noise Floor vs Drift")
    print(f"{'#'*70}")
    print(f"#")
    print(f"# Target: Clean sine wave sin(t)")
    print(f"# Anchor: Noisy sawtooth (intentionally mismatched waveform)")
    print(f"#")
    print(f"# CRITICAL PARAMETERS:")
    print(f"#   noise_levels = {cfg.noise_levels}")
    print(f"#   seed = {cfg.seed}")
    print(f"#     → Spans clean (σ=0.05) to very noisy (σ=0.3)")
    print(f"#   freq_error = {cfg.freq_error*100:.1f}%")
    print(f"#     → Biologically plausible integrator imperfection")
    print(f"#   pll_alpha = {cfg.pll_alpha:g}")
    print(f"#     → 1.0 = full noise import (worst case for anchor)")
    print(f"#")
    print(f"# Models:")
    print(f"#   Closed: Internal integrator only (ignores anchor)")
    print(f"#   Open:   Anchor-locked reconstruction (PLL α={cfg.pll_alpha:g})")
    print(f"#")
    print(f"# Hypothesis:")
    print(f"#   Short-term: Closed wins (anchor = noise source)")
    print(f"#   Long-term:  Open wins (integrator drifts)")
    print(f"#")
    print(f"# Noise levels: {cfg.noise_levels}")
    print(f"# Time: 0 to {cfg.t_max} ({cfg.seq_len} steps)")
    print(f"{'#'*70}")
    
    # Run experiment
    results = run_experiment(cfg)
    
    # Summary
    print(f"\n{'='*70}")
    print("EX4 SUMMARY: Noise Floor Test")
    print(f"{'='*70}")
    print(f"\nModels:")
    print(f"  Closed: Internal integrator only, no anchor correction")
    print(f"  Open:   Anchor-locked reconstruction (PLL α={cfg.pll_alpha:g}, clip={cfg.anchor_clip})")
    print(f"\n{'Noise σ':>8s} | {'Short-term MSE':^27s} | {'Long-term MSE':^27s} | {'Crossover':>10s}")
    print(f"{'':>8s} | {'Closed':>12s} {'Open':>12s} | {'Closed':>12s} {'Open':>12s} | {'step':>10s}")
    print("-" * 85)
    
    for res in results:
        short_win = "◀" if res["short_winner"] == "closed" else ""
        long_win = "◀" if res["long_winner"] == "open" else ""
        cross = res["crossover_idx"]
        print(f"{res['noise_std']:>8.2f} | {res['short_mse_closed']:>12.6f}{short_win:1s} "
              f"{res['short_mse_open']:>12.6f} | {res['long_mse_closed']:>12.6f} "
              f"{res['long_mse_open']:>12.6f}{long_win:1s} | {cross:>10d}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION (A4 - Irreversible Anchor):")
    print("-" * 70)
    print("  This toy model compares internal free-run vs external reference (anchor).")
    print("")
    print("  SHORT-TERM: Closed is BETTER")
    print("    → Anchor imports noise, degrading immediate accuracy")
    print("    → This is why 1ch experiments showed 'anchor unnecessary'")
    print("")
    print("  LONG-TERM: Open is BETTER")
    print("    → Internal integrator DRIFTS without external correction")
    print("    → Anchor provides GROUND TRUTH despite noise")
    print("")
    print("  CONCLUSION:")
    print("    Irreversible anchor is NOT for immediate accuracy")
    print("    It is for LONG-TERM STABILITY and DRIFT PREVENTION")
    print("    (Like a phase-locked loop or sensory calibration)")
    print(f"{'='*70}")
    
    # Plot
    plot_results(results, cfg)
    
    # Save CSV
    save_metrics_csv(results, cfg)
    
    print("\n[EX4] Done.")


if __name__ == "__main__":
    main()
