#!/usr/bin/env python3
"""
CSCT EX5: Binding Problem — Module Synchronization via Common Anchor
====================================================================

Tests the "binding problem": how do independent modules preserve RELATIONS 
(here: phase difference φ) over long horizons despite individual drift?

Purpose:
  - Demonstrate that shared anchor provides BINDING through common clock
  - Show that independent drift destroys inter-module relations
  - Validate that binding is maintained even with noisy anchor

Hypothesis:
  - Closed (no anchor): Modules drift apart → binding (phase relation) degrades
  - Open (with anchor): Common clock preserves binding despite individual noise

Task Design:
  - Two modules generating sin(t) and sin(t + φ) where φ = 1.5 rad (target relation)
  - Modules have OPPOSITE frequency errors: ch0 = +1.5%, ch1 = -1.5%
  - Without synchronization, the phase difference drifts away from φ
  
  Two Modes:
    - Closed: Independent integrators (no anchor correction)
    - Open: PLL-like sync to common anchor clock; relation stored as internal offset

Critical Parameters:
  - phase_diff = 1.5 rad (≈86°): Target binding relation to preserve
      A clearly identifiable phase difference.
      The experiment tests whether this relation survives module drift.
  
  - freq_error_ch0 = +1.5%, freq_error_ch1 = -1.5%: OPPOSITE drift directions
      This ensures modules drift APART (relative drift = 3%/cycle).
      Same-direction drift would accidentally preserve the relation.
      At t=200: cumulative relative drift ≈ 6 rad ≈ 2π (complete phase wrap).
  
  - anchor_noise_std = 0.1 rad (≈6°): Phase noise on common clock
      Realistic noise level — not too clean, not overwhelming.
      Tests whether binding survives noisy synchronization.
  
  - pll_alpha = 0.1: Soft lock (10% correction per step)
      Balances noise rejection with drift prevention.
      α=1.0 would import all anchor noise (like EX4).
      α=0.0 would be equivalent to Closed (no sync).
      α=0.1 provides smooth tracking with noise filtering.

Theoretical Significance:
  This demonstrates the "binding problem" from cognitive science:
  - How do distributed neural modules maintain coherent representations?
  - Answer: Shared temporal reference (anchor clock) enables binding
  - Without common reference, local computations drift apart

Key Metrics:
  - early_phase_err: Binding error in short-term window (t < 200)
  - late_phase_err: Binding error in long-term window (t > 1600)
  - early_mse / late_mse: Reconstruction quality

Anchor Configuration:
  - Common clock: y = noisy phase observation (not the signal itself)
  - Each module stores its own offset internally
  - Anchor provides synchronization, NOT the relation

Outputs:
  - ex5_binding.png: Visualization of binding preservation
  - ex5_metrics.csv: Quantitative results

"""

import argparse
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Small DSP utilities (SciPy-free)
# =============================================================================

def hilbert_analytic(x: np.ndarray) -> np.ndarray:
    """Return analytic signal of a real 1D array using FFT Hilbert transform.

    SciPy-free implementation.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    Xf = np.fft.fft(x)

    h = np.zeros(n)
    if n % 2 == 0:
        # even
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        # odd
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0

    return np.fft.ifft(Xf * h)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Config
# =============================================================================

@dataclass
class EX5Config:
    # Signal parameters (consistent with EX4)
    seq_len: int = 2000      # Steps (same as EX4)
    t_max: float = 200.0     # Total time (same as EX4)

    # Target phase difference: the "binding information" to preserve
    # 1.5 rad ≈ 86° — clearly identifiable, not a special angle
    phase_diff: float = 1.5

    # Integrator errors: OPPOSITE directions ensure modules drift APART
    # At t=200: relative drift ≈ 6 rad ≈ 2π (complete phase wrap)
    freq_error_ch0: float = 0.015   # +1.5% for module 0
    freq_error_ch1: float = -0.015  # -1.5% for module 1 (opposite!)

    # Anchor noise: phase noise σ = 0.1 rad ≈ 6°
    # Realistic level — tests binding under noisy synchronization
    anchor_noise_std: float = 0.10

    # PLL synchronizer strength: 10% correction per step (soft lock)
    # Balances noise rejection with drift prevention
    pll_alpha: float = 0.10

    # Analysis windows (consistent with EX4 style)
    short_term_end: int = 200
    long_term_start: int = 1600

    # Reproducibility
    seed: int = 0

    # Output
    output_dir: str = "./results_ex5"


# =============================================================================
# Integrator Models
# =============================================================================

class DualHarmonicIntegrator:
    """Two independent harmonic integrators with different frequency errors.

    Closed: free-run integrators (drift apart).
    Open:   PLL-like phase correction to a common anchor clock.

    Note: The relation (+phase_diff) is *internal* to module 1 as a fixed offset.
    The anchor provides only a common phase reference.
    """

    def __init__(self, dt: float, freq_error_0: float, freq_error_1: float, phase_diff: float, pll_alpha: float):
        self.dt = float(dt)
        self.omega_sq_0 = (1.0 + float(freq_error_0)) ** 2
        self.omega_sq_1 = (1.0 + float(freq_error_1)) ** 2
        self.phase_diff = float(phase_diff)
        self.pll_alpha = float(pll_alpha)

    @staticmethod
    def _phase_from_state(z: float, v: float) -> float:
        # For ideal oscillator: z=sin(phi), v=cos(phi)
        return math.atan2(z, v)

    def run_closed(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run both integrators independently (no anchor correction)."""
        z0, v0 = 0.0, 1.0
        z1, v1 = math.sin(self.phase_diff), math.cos(self.phase_diff)

        ch0 = np.zeros(steps)
        ch1 = np.zeros(steps)

        for t in range(steps):
            ch0[t] = z0
            ch1[t] = z1

            # Symplectic Euler with individual frequency errors
            v0_new = v0 - self.omega_sq_0 * z0 * self.dt
            z0_new = z0 + v0_new * self.dt

            v1_new = v1 - self.omega_sq_1 * z1 * self.dt
            z1_new = z1 + v1_new * self.dt

            z0, v0 = z0_new, v0_new
            z1, v1 = z1_new, v1_new

        return ch0, ch1

    def run_open(self, steps: int, anchor_phase_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run both integrators with PLL-like anchor correction.

        anchor_phase_obs: noisy observation of the *common* anchor phase.

        Each module predicts internally, then applies a phase correction:
          phi <- phi + pll_alpha * wrap(phi_desired - phi)

        Module 1 desires an internal offset +phase_diff.
        """
        z0, v0 = 0.0, 1.0
        z1, v1 = math.sin(self.phase_diff), math.cos(self.phase_diff)

        ch0 = np.zeros(steps)
        ch1 = np.zeros(steps)

        a = self.pll_alpha

        for t in range(steps):
            ch0[t] = z0
            ch1[t] = z1

            # Internal prediction (with drift)
            v0_new = v0 - self.omega_sq_0 * z0 * self.dt
            z0_new = z0 + v0_new * self.dt

            v1_new = v1 - self.omega_sq_1 * z1 * self.dt
            z1_new = z1 + v1_new * self.dt

            # Convert predicted state to phase
            phi0 = self._phase_from_state(z0_new, v0_new)
            phi1 = self._phase_from_state(z1_new, v1_new)

            # Desired phases from anchor clock (relation stored as internal offset)
            phi_anchor = float(anchor_phase_obs[t])
            phi0_des = phi_anchor
            phi1_des = phi_anchor + self.phase_diff

            # Phase correction (PLL)
            e0 = float(wrap_to_pi(phi0_des - phi0))
            e1 = float(wrap_to_pi(phi1_des - phi1))
            phi0_corr = phi0 + a * e0
            phi1_corr = phi1 + a * e1

            # Re-project back to oscillator state (unit circle)
            z0, v0 = math.sin(phi0_corr), math.cos(phi0_corr)
            z1, v1 = math.sin(phi1_corr), math.cos(phi1_corr)

        return ch0, ch1


# =============================================================================
# Data Generation
# =============================================================================

def generate_target_signals(cfg: EX5Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate target signals and a noisy anchor *phase* observation.

    We treat the anchor as an environmental clock:
      phi_anchor_true(t) = t
      phi_anchor_obs(t)  = wrap(phi_anchor_true + N(0, sigma))

    The anchor signal itself is sin(phi_anchor_obs), but the *open* controller
    consumes the phase observation (clock), not a non-invertible arcsin.
    """
    t = np.linspace(0.0, cfg.t_max, cfg.seq_len)

    # Clean target signals
    ch0_target = np.sin(t)
    ch1_target = np.sin(t + cfg.phase_diff)

    # Anchor as noisy phase (clock)
    phase_noise = np.random.normal(0.0, cfg.anchor_noise_std, cfg.seq_len)
    anchor_phase_obs = wrap_to_pi(t + phase_noise)

    # Anchor signal (for plotting intuition)
    anchor_signal = np.sin(anchor_phase_obs)

    return t, ch0_target, ch1_target, anchor_signal, anchor_phase_obs


# =============================================================================
# Analysis
# =============================================================================

def compute_instantaneous_phase_diff_hilbert(ch0: np.ndarray, ch1: np.ndarray) -> np.ndarray:
    """Instantaneous phase difference via analytic signal + unwrap."""
    a0 = hilbert_analytic(ch0)
    a1 = hilbert_analytic(ch1)

    p0 = np.unwrap(np.angle(a0))
    p1 = np.unwrap(np.angle(a1))

    d = wrap_to_pi(p1 - p0)
    return d


def compute_binding_metrics(
    ch0: np.ndarray,
    ch1: np.ndarray,
    ch0_target: np.ndarray,
    ch1_target: np.ndarray,
    phase_diff: float,
    cfg: EX5Config,
) -> Dict:
    """Compute binding-related metrics."""
    # Individual channel MSE
    mse_ch0 = float(np.mean((ch0 - ch0_target) ** 2))
    mse_ch1 = float(np.mean((ch1 - ch1_target) ** 2))

    # Phase difference over time (binding information)
    phase_diffs = compute_instantaneous_phase_diff_hilbert(ch0, ch1)

    # Phase error (wrapped)
    phase_errors = np.abs(wrap_to_pi(phase_diffs - phase_diff))

    # Early vs Late
    early_phase_err = float(np.mean(phase_errors[: cfg.short_term_end]))
    late_phase_err = float(np.mean(phase_errors[cfg.long_term_start :]))

    early_mse = float(
        np.mean((ch0[: cfg.short_term_end] - ch0_target[: cfg.short_term_end]) ** 2)
        + np.mean((ch1[: cfg.short_term_end] - ch1_target[: cfg.short_term_end]) ** 2)
    )
    late_mse = float(
        np.mean((ch0[cfg.long_term_start :] - ch0_target[cfg.long_term_start :]) ** 2)
        + np.mean((ch1[cfg.long_term_start :] - ch1_target[cfg.long_term_start :]) ** 2)
    )

    # === NEW METRICS ===
    
    # stability_duration: fraction of time where phase error < threshold (0.3 rad ≈ 17°)
    threshold = 0.3
    stable_mask = phase_errors < threshold
    # Find longest consecutive stable run
    if np.any(stable_mask):
        # Count consecutive True values
        changes = np.diff(np.concatenate([[0], stable_mask.astype(int), [0]]))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        run_lengths = run_ends - run_starts
        longest_run = np.max(run_lengths) if len(run_lengths) > 0 else 0
        stability_duration = float(longest_run / len(phase_errors))
    else:
        stability_duration = 0.0
    
    # plv_late: Phase Locking Value in late window
    # PLV = |mean(exp(i * phase_diff_error))| - measures consistency of phase relationship
    late_phase_diff = phase_diffs[cfg.long_term_start:]
    late_phase_err_raw = late_phase_diff - phase_diff  # not wrapped, for PLV calculation
    plv_late = float(np.abs(np.mean(np.exp(1j * late_phase_err_raw))))

    return {
        "mse_ch0": mse_ch0,
        "mse_ch1": mse_ch1,
        "mse_total": mse_ch0 + mse_ch1,
        "early_mse": early_mse,
        "late_mse": late_mse,
        "early_phase_err": early_phase_err,
        "late_phase_err": late_phase_err,
        "stability_duration": stability_duration,
        "plv_late": plv_late,
        "phase_diffs": phase_diffs,
        "phase_errors": phase_errors,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(cfg: EX5Config) -> Dict:
    """Run binding experiment."""
    np.random.seed(cfg.seed)

    # Generate data
    t, ch0_target, ch1_target, anchor_signal, anchor_phase_obs = generate_target_signals(cfg)

    # dt consistent with linspace
    dt = cfg.t_max / (cfg.seq_len - 1)

    # Create integrator
    integrator = DualHarmonicIntegrator(
        dt=dt,
        freq_error_0=cfg.freq_error_ch0,
        freq_error_1=cfg.freq_error_ch1,
        phase_diff=cfg.phase_diff,
        pll_alpha=cfg.pll_alpha,
    )

    # Run both conditions
    print("Running Closed (no anchor)...")
    ch0_closed, ch1_closed = integrator.run_closed(cfg.seq_len)

    print(f"Running Open (with anchor PLL alpha={cfg.pll_alpha})...")
    ch0_open, ch1_open = integrator.run_open(cfg.seq_len, anchor_phase_obs)

    # Compute metrics
    metrics_closed = compute_binding_metrics(ch0_closed, ch1_closed, ch0_target, ch1_target, cfg.phase_diff, cfg)
    metrics_open = compute_binding_metrics(ch0_open, ch1_open, ch0_target, ch1_target, cfg.phase_diff, cfg)

    return {
        "t": t,
        "ch0_target": ch0_target,
        "ch1_target": ch1_target,
        "anchor": anchor_signal,
        "anchor_phase_obs": anchor_phase_obs,
        "ch0_closed": ch0_closed,
        "ch1_closed": ch1_closed,
        "ch0_open": ch0_open,
        "ch1_open": ch1_open,
        "metrics_closed": metrics_closed,
        "metrics_open": metrics_open,
    }


def plot_results(results: Dict, cfg: EX5Config):
    """Create visualization."""
    t = results["t"]
    mc = results["metrics_closed"]
    mo = results["metrics_open"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Row 1: Channel trajectories (zoomed to show drift)
    late_slice = slice(cfg.long_term_start, None)

    ax = axes[0, 0]
    ax.plot(t[late_slice], results["ch0_target"][late_slice], "k-", alpha=0.3, linewidth=2, label="Target ch0")
    ax.plot(t[late_slice], results["ch0_closed"][late_slice], "b-", alpha=0.8, label="Closed ch0")
    ax.plot(t[late_slice], results["ch0_open"][late_slice], "r--", alpha=0.8, label="Open ch0")
    ax.set_ylabel("Amplitude")
    ax.set_title("Channel 0 (Late Phase - Showing Drift)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t[late_slice], results["ch1_target"][late_slice], "k-", alpha=0.3, linewidth=2, label="Target ch1")
    ax.plot(t[late_slice], results["ch1_closed"][late_slice], "b-", alpha=0.8, label="Closed ch1")
    ax.plot(t[late_slice], results["ch1_open"][late_slice], "r--", alpha=0.8, label="Open ch1")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Channel 1 (Late Phase - φ = {cfg.phase_diff:.2f} rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Phase difference (binding information)
    ax = axes[1, 0]
    ax.axhline(cfg.phase_diff, color="k", linestyle=":", linewidth=2, label=f"Target φ = {cfg.phase_diff:.2f}")
    ax.plot(t, mc["phase_diffs"], "b-", alpha=0.8, label="Closed")
    ax.plot(t, mo["phase_diffs"], "r-", alpha=0.8, label="Open")
    ax.axvline(x=t[cfg.short_term_end], color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=t[cfg.long_term_start], color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Phase Difference (rad)")
    ax.set_title("BINDING: Instantaneous Phase Relationship (Hilbert)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-np.pi, np.pi)

    # Row 2 Right: Phase error over time
    ax = axes[1, 1]
    ax.semilogy(t, mc["phase_errors"] + 1e-8, "b-", alpha=0.8, label="Closed")
    ax.semilogy(t, mo["phase_errors"] + 1e-8, "r-", alpha=0.8, label="Open")
    ax.axvline(x=t[cfg.short_term_end], color="gray", linestyle=":", alpha=0.5, label="Short/Long")
    ax.axvline(x=t[cfg.long_term_start], color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("|Phase Error| (rad, log scale)")
    ax.set_xlabel("Time")
    ax.set_title("Binding Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3: Summary comparisons
    labels = ["Short-term\nBinding Error", "Long-term\nBinding Error"]
    x_pos = np.arange(len(labels))
    width = 0.35

    ax = axes[2, 0]
    vals_closed = [mc["early_phase_err"], mc["late_phase_err"]]
    vals_open = [mo["early_phase_err"], mo["late_phase_err"]]
    ax.bar(x_pos - width / 2, vals_closed, width, label="Closed", alpha=0.7)
    ax.bar(x_pos + width / 2, vals_open, width, label="Open", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Phase Error (rad)")
    ax.set_title("Binding Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (vc, vo) in enumerate(zip(vals_closed, vals_open)):
        if vo > 1e-12:
            ratio = vc / vo
            ax.annotate(f"{ratio:.1f}x", xy=(i, max(vc, vo) * 1.12), ha="center", fontsize=11, fontweight="bold")

    ax = axes[2, 1]
    labels2 = ["Short-term\nTotal MSE", "Long-term\nTotal MSE"]
    vals_closed2 = [mc["early_mse"], mc["late_mse"]]
    vals_open2 = [mo["early_mse"], mo["late_mse"]]
    ax.bar(x_pos - width / 2, vals_closed2, width, label="Closed", alpha=0.7)
    ax.bar(x_pos + width / 2, vals_open2, width, label="Open", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels2)
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (vc, vo) in enumerate(zip(vals_closed2, vals_open2)):
        if vo > 1e-12:
            ratio = vc / vo
            ax.annotate(f"{ratio:.1f}x", xy=(i, max(vc, vo) * 1.12), ha="center", fontsize=11, fontweight="bold")

    plt.suptitle(f"EX5: Binding via Common Anchor Clock (PLL α={cfg.pll_alpha})", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(cfg.output_dir, "ex5_binding.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to: {save_path}")


def save_metrics_csv(results: Dict, cfg: EX5Config):
    """Save metrics to CSV for reproducibility."""
    import csv
    
    mc = results["metrics_closed"]
    mo = results["metrics_open"]
    
    csv_path = os.path.join(cfg.output_dir, "ex5_metrics.csv")
    
    fieldnames = [
        "condition", "early_phase_err", "late_phase_err", 
        "stability_duration", "plv_late",
        "early_mse", "late_mse", "mse_ch0", "mse_ch1", "mse_total",
        "phase_diff", "freq_error_ch0", "freq_error_ch1", 
        "anchor_noise_std", "pll_alpha", "seed"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for cond, m in [("closed", mc), ("open", mo)]:
            row = {
                "condition": cond,
                "early_phase_err": m["early_phase_err"],
                "late_phase_err": m["late_phase_err"],
                "stability_duration": m["stability_duration"],
                "plv_late": m["plv_late"],
                "early_mse": m["early_mse"],
                "late_mse": m["late_mse"],
                "mse_ch0": m["mse_ch0"],
                "mse_ch1": m["mse_ch1"],
                "mse_total": m["mse_total"],
                "phase_diff": cfg.phase_diff,
                "freq_error_ch0": cfg.freq_error_ch0,
                "freq_error_ch1": cfg.freq_error_ch1,
                "anchor_noise_std": cfg.anchor_noise_std,
                "pll_alpha": cfg.pll_alpha,
                "seed": cfg.seed,
            }
            writer.writerow(row)
    
    print(f"Saved metrics to: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="CSCT EX5: Binding Problem — Module Synchronization via Common Anchor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Critical Parameters:
  --phase-diff: Target binding relation (default: 1.5 rad ≈ 86°)
      The relation that must be preserved over time.
      
  --freq-error-ch0 / --freq-error-ch1: OPPOSITE drift directions
      Default: +1.5% / -1.5%. Modules drift APART, destroying the relation.
      At t=200: relative drift ≈ 6 rad ≈ 2π (complete phase wrap).
      
  --anchor-noise-std: Phase noise on common clock (default: 0.1 rad ≈ 6°)
      Realistic noise level — tests binding under noisy synchronization.
      
  --pll-alpha: Synchronization strength (default: 0.1 = soft lock)
      α=0.1 balances noise rejection with drift prevention.
      α=1.0 would import all noise; α=0 would be equivalent to Closed.

Example:
  python csct_ex5_binding.py --seq-len 2000 --t-max 200 \\
      --phase-diff 1.5 --freq-error-ch0 0.015 --freq-error-ch1 -0.015 \\
      --anchor-noise-std 0.10 --pll-alpha 0.10
"""
    )

    p.add_argument("--seq-len", type=int, default=2000,
                   help="Sequence length (default: 2000, consistent with EX4)")
    p.add_argument("--t-max", type=float, default=200.0,
                   help="Maximum time (default: 200.0, consistent with EX4)")
    p.add_argument("--phase-diff", type=float, default=1.5,
                   help="Target phase difference φ to preserve, 1.5 rad ≈ 86° (default: 1.5)")
    p.add_argument("--freq-error-ch0", type=float, default=0.015,
                   help="Module 0 frequency error, +1.5%% (default: 0.015)")
    p.add_argument("--freq-error-ch1", type=float, default=-0.015,
                   help="Module 1 frequency error, -1.5%% opposite direction (default: -0.015)")
    p.add_argument("--anchor-noise-std", type=float, default=0.10,
                   help="Anchor phase noise std, 0.1 rad ≈ 6° (default: 0.10)")
    p.add_argument("--pll-alpha", type=float, default=0.10,
                   help="PLL correction strength, 0.1 = soft lock (default: 0.10)")
    p.add_argument("--short-term-end", type=int, default=200,
                   help="End of short-term analysis window (default: 200)")
    p.add_argument("--long-term-start", type=int, default=1600,
                   help="Start of long-term analysis window (default: 1600)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducibility (default: 0)")
    p.add_argument("--output-dir", type=str, default="./results_ex5",
                   help="Output directory (default: ./results_ex5)")

    args = p.parse_args()

    cfg = EX5Config(
        seq_len=args.seq_len,
        t_max=args.t_max,
        phase_diff=args.phase_diff,
        freq_error_ch0=args.freq_error_ch0,
        freq_error_ch1=args.freq_error_ch1,
        anchor_noise_std=args.anchor_noise_std,
        pll_alpha=args.pll_alpha,
        short_term_end=args.short_term_end,
        long_term_start=args.long_term_start,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print("# CSCT EX5: Binding Problem — Module Synchronization")
    print(f"{'#'*70}")
    print("#")
    print("# BINDING TARGET:")
    print(f"#   Phase difference φ = {cfg.phase_diff:.2f} rad ≈ {cfg.phase_diff/3.14159*180:.0f}°")
    print("#")
    print("# CRITICAL PARAMETERS:")
    print(f"#   freq_error_ch0 = {cfg.freq_error_ch0*100:+.1f}%")
    print(f"#   freq_error_ch1 = {cfg.freq_error_ch1*100:+.1f}%")
    print("#     → OPPOSITE directions: relative drift = 3%/cycle")
    print(f"#     → At t={cfg.t_max}: drift ≈ 6 rad ≈ 2π (complete wrap)")
    print(f"#   anchor_noise_std = {cfg.anchor_noise_std:.2f} rad ≈ {cfg.anchor_noise_std/3.14159*180:.0f}°")
    print(f"#   pll_alpha = {cfg.pll_alpha:.2f} (soft lock, 10% correction)")
    print("#")
    print("# CONDITIONS:")
    print("#   Closed: Independent drift (no anchor) → binding degrades")
    print("#   Open:   PLL sync to common clock → binding preserved")
    print(f"{'#'*70}")

    results = run_experiment(cfg)

    mc = results["metrics_closed"]
    mo = results["metrics_open"]

    print(f"\n{'='*70}")
    print("EX5 SUMMARY: Binding Problem (patched)")
    print(f"{'='*70}")
    print(f"\n{'Metric':28s} | {'Closed':>12s} | {'Open':>12s} | {'Ratio':>10s}")
    print("-" * 70)

    metrics_to_show = [
        ("Short-term Binding Error", "early_phase_err"),
        ("Long-term Binding Error", "late_phase_err"),
        ("Short-term Total MSE", "early_mse"),
        ("Long-term Total MSE", "late_mse"),
    ]

    for label, key in metrics_to_show:
        vc = mc[key]
        vo = mo[key]
        ratio = (vc / vo) if vo > 1e-12 else float("inf")
        winner = "◀" if vc > vo else ""
        print(f"{label:28s} | {vc:12.6f} | {vo:12.6f}{winner:1s} | {ratio:10.2f}x")

    print(f"\n{'='*70}")
    print("INTERPRETATION (Binding):")
    print("-" * 70)

    early_ratio = (mc["early_phase_err"] / mo["early_phase_err"]) if mo["early_phase_err"] > 1e-12 else float("inf")
    late_ratio = (mc["late_phase_err"] / mo["late_phase_err"]) if mo["late_phase_err"] > 1e-12 else float("inf")

    print(f"  Short-term binding error ratio: {early_ratio:.2f}x")
    print(f"  Long-term  binding error ratio: {late_ratio:.2f}x")
    print("")

    if late_ratio > max(2.0, 1.5 * early_ratio):
        print("  ✓ BINDING PROBLEM DEMONSTRATED:")
    else:
        print("  ? Weak separation. Try increasing freq errors or reducing pll_alpha.")

    plot_results(results, cfg)
    
    # Save CSV
    save_metrics_csv(results, cfg)
    
    print("\n[EX5] Done.")


if __name__ == "__main__":
    main()
