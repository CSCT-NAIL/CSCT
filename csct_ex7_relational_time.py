#!/usr/bin/env python3
"""
CSCT EX7: Relational Internal Time
==================================

Tests how internal time (operationally defined as code transition rate)
dilates under different anchor tempo conditions.

Purpose:
  - Demonstrate that internal time "stretches/compresses" with anchor tempo
  - Validate that no input leads to no update (zero transitions)
  - Show event-rate dilation as a measurable phenomenon

Hypothesis:
  - H1: Faster anchor tempo → higher code transition rate (time acceleration)
  - H2: Slower anchor tempo → lower code transition rate (time dilation)
  - H3: Null input (signal=0, anchor=0) → zero transitions (no update)

Task Design:
  - Train model in Standard World (constant 0.5Hz anchor)
  - Deploy to four worlds with different tempo profiles:
    - Standard: Constant 0.5Hz (baseline)
    - Fast: Accelerating 0.5Hz → 1.5Hz
    - Slow: Decelerating 0.5Hz → 0.15Hz
    - Null: No input (signal=0, anchor=0)

Critical Parameters:
  - n_clocks = 4: Number of discrete codes (K)
      Small K makes transitions clearly observable.
  
  - base_freq = 0.5 Hz: Baseline anchor frequency
      Standard world uses this constant frequency.
  
  - Fast profile: 0.5Hz → 1.5Hz (3x acceleration)
      Tests whether internal events speed up with faster anchor.
  
  - Slow profile: 0.5Hz → 0.15Hz (3.3x deceleration)
      Tests whether internal events slow down with slower anchor.
  
  - Null profile: signal=0, anchor=0
      Sanity check: system should not update without input.

Theoretical Significance:
  Internal time is operationally defined as the progression of discrete
  code transitions over physical time. This provides a measurable proxy
  for the "flow of time" within the CSCT framework.
  
  Key insight: Time is relational, not absolute.
  - Faster external events → faster internal time
  - No external events → no internal time progression

Key Metrics:
  - n_transitions: Total code transitions (internal events)
  - trans_per_sec: Transition rate over physical time
  - unique_codes: Number of distinct codes used

Outputs:
  - ex7_relational_time.png: Visualization of tempo dilation
  - ex7_metrics.csv: Quantitative results per world

"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Import CSCT Engine
from csct_engine import CSCT_Engine, CSCTConfig


# =============================================================================
# Config
# =============================================================================

@dataclass
class EX7Config:
    # Model parameters
    n_clocks: int = 4        # K = number of discrete codes (small for clear transitions)
    hidden_dim: int = 32     # Hidden dimension
    input_dim: int = 1       # Input dimension

    # Training parameters
    seq_len: int = 800       # Sequence length
    n_epochs: int = 800      # Training epochs
    lr: float = 0.01         # Learning rate

    # Anchor parameters
    base_freq: float = 0.5   # Baseline frequency (Hz) for Standard world

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "./results_ex7"


# =============================================================================
# World Generation
# =============================================================================

def generate_world(cfg: EX7Config, speed_profile: str = "standard",
                   device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Generate a world with specific time flow (anchor speed).

    Args:
        speed_profile:
          - 'standard': Constant 0.5Hz
          - 'fast': Accelerates 0.5Hz → 1.5Hz
          - 'slow': Decelerates 0.5Hz → 0.15Hz
          - 'null': No input world (signal=0, anchor=0) for "no update" check

    Returns:
        Dictionary with signal, anchor, time, frequency, phase
    """
    t = torch.linspace(0, 20, cfg.seq_len, device=device)
    dt = t[1] - t[0]

    # Define frequency schedule (flow of time)
    if speed_profile == "standard":
        freq = torch.ones_like(t) * cfg.base_freq
    elif speed_profile == "fast":
        # Accelerate: 0.5Hz → 1.5Hz
        freq = cfg.base_freq * (1.0 + 2.0 * (t / t.max()) ** 1.5)
    elif speed_profile == "slow":
        # Decelerate: 0.5Hz → 0.15Hz
        freq = cfg.base_freq * torch.exp(-1.2 * t / t.max())
        freq = torch.clamp(freq, min=0.15)
    elif speed_profile == "null":
        # No input world
        freq = torch.zeros_like(t)
    else:
        raise ValueError(f"Unknown profile: {speed_profile}")

    # Phase integration: φ = ∫ω dt
    if speed_profile == "null":
        phase = torch.zeros_like(t)
    else:
        phase = torch.cumsum(freq * dt, dim=0) * 2 * np.pi

    # Anchor
    if speed_profile == "null":
        anchor = torch.zeros((1, cfg.seq_len, 1), device=device)
    else:
        anchor = torch.sin(phase).view(1, -1, 1)

    # Signal (locked to phase in non-null worlds)
    if speed_profile == "null":
        signal = torch.zeros((1, cfg.seq_len, 1), device=device)
    else:
        signal = (torch.sin(phase) + 0.5 * torch.sin(2.0 * phase + 1.0)).view(1, -1, 1)
        signal = signal / (signal.abs().max() + 1e-12)

    return {
        "signal": signal,
        "anchor": anchor,
        "t": t,
        "freq": freq,
        "phase": phase,
        "profile": speed_profile,
    }


# =============================================================================
# Training
# =============================================================================

def train_model(cfg: EX7Config, device: str = "cpu") -> Tuple[CSCT_Engine, List[float]]:
    """Train CSCT model in Standard World."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Generate standard world
    world = generate_world(cfg, "standard", device)
    x = world["signal"]
    anchor = world["anchor"]

    # Initialize CSCT Engine
    csct_cfg = CSCTConfig(
        n_clocks=cfg.n_clocks,
        hidden_dim=cfg.hidden_dim,
        input_dim=cfg.input_dim,
        gate_tau=0.5,
    )
    model = CSCT_Engine(csct_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    losses: List[float] = []
    for epoch in range(cfg.n_epochs):
        model.train()
        optimizer.zero_grad()

        # Anneal beta for sharper transitions
        beta = 0.1 if epoch < 200 else (1.0 if epoch < 400 else 10.0)

        out = model(x, anchor, beta=beta)
        loss = out["loss"]

        loss.backward()
        optimizer.step()

        # Temperature annealing (engine-defined)
        if epoch % 50 == 0:
            model.anneal_temperature()

        losses.append(float(loss.item()))

        if epoch % 200 == 0:
            recon = float(out["losses"]["recon"]) if "losses" in out and "recon" in out["losses"] else float("nan")
            print(f"    Epoch {epoch}: Loss={loss.item():.4f}, Recon={recon:.4f}")

    return model, losses


# =============================================================================
# Analysis
# =============================================================================

def analyze_worlds(model: CSCT_Engine, cfg: EX7Config,
                   device: str = "cpu") -> Dict[str, Dict]:
    """Deploy model to all worlds and collect results."""
    profiles = ["standard", "fast", "slow", "null"]
    results: Dict[str, Dict] = {}

    model.eval()

    for profile in profiles:
        world = generate_world(cfg, profile, device)

        with torch.no_grad():
            out = model(world["signal"], world["anchor"], beta=10.0)

        indices = out["indices"].squeeze().detach().cpu().numpy()
        # force integer indices for plotting/metrics
        indices_i = indices.astype(np.int64) if indices.size > 0 else indices

        # "Update" = discrete code transition events
        n_transitions = int(np.sum(np.diff(indices_i) != 0)) if indices_i.size > 1 else 0
        duration = float(world["t"][-1].item() - world["t"][0].item())
        trans_per_sec = float(n_transitions / duration) if duration > 0 else 0.0

        results[profile] = {
            "t": world["t"].cpu().numpy(),
            "signal": world["signal"].squeeze().cpu().numpy(),
            "recon": out["recon"].squeeze().cpu().numpy(),
            "anchor": world["anchor"].squeeze().cpu().numpy(),
            "indices": indices_i,
            "freq": world["freq"].cpu().numpy(),
            "phase": world["phase"].cpu().numpy(),
            "n_transitions": n_transitions,
            "trans_per_sec": trans_per_sec,
        }

    return results


def compute_relational_metrics(results: Dict[str, Dict], cfg: EX7Config) -> Dict[str, Dict[str, float]]:
    """
    Compute relational time metrics:
      - n_transitions: number of code transitions (internal events)
      - trans_per_sec: internal event rate over physical time
      - unique_codes: number of unique codes used
      - freq_min/max: tempo range
    """
    metrics: Dict[str, Dict[str, float]] = {}
    for profile, res in results.items():
        indices = res.get("indices", np.array([], dtype=np.int64))
        uniq = int(len(np.unique(indices))) if indices.size > 0 else 0
        freq = res.get("freq", np.array([0.0], dtype=np.float64))
        metrics[profile] = {
            "n_transitions": int(res.get("n_transitions", 0)),
            "trans_per_sec": float(res.get("trans_per_sec", 0.0)),
            "unique_codes": float(uniq),
            "freq_min": float(np.min(freq)) if freq.size > 0 else 0.0,
            "freq_max": float(np.max(freq)) if freq.size > 0 else 0.0,
        }
    return metrics


# =============================================================================
# Visualization (minimal)
# =============================================================================

def plot_results(results: Dict[str, Dict], metrics: Dict[str, Dict[str, float]], cfg: EX7Config) -> None:
    """
    Visualization (2 rows x 4 columns):
      Row 1: Reconstruction colored by code + anchor overlay + transition markers
      Row 2: Cumulative transitions over physical time (internal time proxy)
    """
    profiles = ["standard", "fast", "slow", "null"]
    profile_names = {
        "standard": "Standard (0.5Hz const)",
        "fast": "Fast (accel 0.5→1.5Hz)",
        "slow": "Slow (decel 0.5→0.15Hz)",
        "null": "Null (signal=0, anchor=0)",
    }

    colors = [hsv_to_rgb((k / cfg.n_clocks, 0.8, 0.9)) for k in range(cfg.n_clocks)]

    fig = plt.figure(figsize=(5.2 * len(profiles), 7.0))

    for col, profile in enumerate(profiles):
        res = results[profile]
        t = res["t"]
        recon = res["recon"]
        anchor = res["anchor"]
        indices = res["indices"]

        # Transition times
        changes = np.where(np.diff(indices) != 0)[0] if indices.size > 1 else np.array([], dtype=np.int64)
        trans_t = t[changes] if changes.size > 0 else np.array([])

        # --- Row 1: recon colored by code ---
        ax1 = plt.subplot(2, len(profiles), col + 1)
        for i in range(len(t) - 1):
            code = int(indices[i]) if indices.size > 0 else 0
            code = max(0, min(cfg.n_clocks - 1, code))
            ax1.plot([t[i], t[i + 1]], [recon[i], recon[i + 1]],
                     color=colors[code], linewidth=1.5, alpha=0.9)

        ax1.plot(t, anchor, "k:", alpha=0.25, linewidth=1.0, label="anchor")
        for tt in trans_t:
            ax1.axvline(tt, color="k", alpha=0.12, linewidth=1.0)

        ax1.set_title(profile_names[profile], fontsize=11, fontweight="bold")
        ax1.set_xlim(float(t[0]), float(t[-1]))
        ax1.set_ylabel("recon")
        ax1.grid(True, alpha=0.25)
        if col == 0:
            ax1.legend(loc="upper right", fontsize=8)

        # --- Row 2: cumulative transitions (internal time proxy) ---
        ax2 = plt.subplot(2, len(profiles), len(profiles) + col + 1)
        cum = np.zeros_like(t, dtype=np.float64)
        if changes.size > 0:
            for idx in changes:
                cum[idx + 1:] += 1.0

        ax2.plot(t, cum, linewidth=2.0)
        ax2.set_xlim(float(t[0]), float(t[-1]))
        ax2.set_xlabel("physical time (s)")
        ax2.set_ylabel("cum transitions")
        ax2.grid(True, alpha=0.25)

        m = metrics[profile]
        ax2.set_title(
            f"unique={int(m['unique_codes'])}, n_trans={int(m['n_transitions'])}, rate={m['trans_per_sec']:.3f}/s",
            fontsize=10
        )

    plt.suptitle("EX7: Relational Internal Time — Event-Rate Dilation", fontsize=14, y=0.98)
    plt.tight_layout()

    save_path = os.path.join(cfg.output_dir, "ex7_relational_time.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to: {save_path}")


def print_summary(results: Dict[str, Dict], metrics: Dict[str, Dict[str, float]], cfg: EX7Config) -> None:
    """Print summary table."""
    print(f"\n{'=' * 80}")
    print("EX7 SUMMARY: Relational Internal Time")
    print(f"{'=' * 80}")

    print(f"\nModel: SingleGate CSCT with K={cfg.n_clocks} codes")
    print("Trained in: Standard World (constant 0.5Hz)")
    print("Deployed to: Standard, Fast, Slow, and Null (signal=0, anchor=0)")

    print(f"\n{'World':>10s} | {'Tempo(Hz)':>14s} | {'Unique':>6s} | {'Transitions':>11s} | {'Rate(/s)':>8s}")
    print("-" * 62)

    for profile in ["standard", "fast", "slow", "null"]:
        m = metrics[profile]
        freq_range = f"{m['freq_min']:.2f}-{m['freq_max']:.2f}"
        print(f"{profile:>10s} | {freq_range:>14s} | {int(m['unique_codes']):>6d} | "
              f"{int(m['n_transitions']):>11d} | {m['trans_per_sec']:>8.3f}")

    print("\nNotes:")
    print("  - Internal time proxy = cumulative code transitions over physical time.")
    print("  - Null world should show ~0 transitions (no update without input).")
    print(f"{'=' * 80}")


def save_metrics_csv(metrics: Dict[str, Dict[str, float]], cfg: EX7Config) -> None:
    """Save metrics to CSV for reproducibility."""
    import csv
    
    csv_path = os.path.join(cfg.output_dir, "ex7_metrics.csv")
    
    fieldnames = [
        "world", "freq_min", "freq_max", "unique_codes",
        "n_transitions", "trans_per_sec", "n_clocks", "base_freq"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for profile in ["standard", "fast", "slow", "null"]:
            m = metrics[profile]
            row = {
                "world": profile,
                "freq_min": m["freq_min"],
                "freq_max": m["freq_max"],
                "unique_codes": int(m["unique_codes"]),
                "n_transitions": int(m["n_transitions"]),
                "trans_per_sec": m["trans_per_sec"],
                "n_clocks": cfg.n_clocks,
                "base_freq": cfg.base_freq,
            }
            writer.writerow(row)
    
    print(f"Saved metrics to: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSCT EX7: Relational Internal Time — Event-Rate Dilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Critical Parameters:
  --n-clocks: Number of discrete codes K (default: 4)
      Small K makes transitions clearly observable.
      
  --base-freq: Baseline anchor frequency (default: 0.5 Hz)
      Standard world uses this constant frequency.
      
  Tempo profiles (hardcoded):
    - Standard: constant 0.5 Hz
    - Fast: accelerating 0.5 → 1.5 Hz (3x)
    - Slow: decelerating 0.5 → 0.15 Hz (3.3x)
    - Null: signal=0, anchor=0 (sanity check)

Example:
  python csct_ex7_relational_time.py --n-clocks 4 --n-epochs 800
"""
    )

    parser.add_argument("--n-clocks", type=int, default=4,
                        help="Number of discrete codes K (default: 4)")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Hidden dimension (default: 32)")
    parser.add_argument("--seq-len", type=int, default=800,
                        help="Sequence length (default: 800)")
    parser.add_argument("--n-epochs", type=int, default=800,
                        help="Training epochs (default: 800)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="./results_ex7",
                        help="Output directory (default: ./results_ex7)")

    args = parser.parse_args()

    cfg = EX7Config(
        n_clocks=args.n_clocks,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        n_epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'#' * 70}")
    print("# CSCT EX7: Relational Internal Time — Event-Rate Dilation")
    print(f"{'#' * 70}")
    print("#")
    print("# CRITICAL PARAMETERS:")
    print(f"#   n_clocks = {cfg.n_clocks} (small K for clear transitions)")
    print(f"#   base_freq = {cfg.base_freq} Hz (Standard world)")
    print("#")
    print("# TEMPO PROFILES:")
    print("#   Standard: constant 0.5 Hz (baseline)")
    print("#   Fast: accelerating 0.5 → 1.5 Hz (3x)")
    print("#   Slow: decelerating 0.5 → 0.15 Hz (3.3x)")
    print("#   Null: signal=0, anchor=0 (sanity check)")
    print("#")

    print(f"{'#' * 70}")

    # Train
    print("\n--- Phase 1: Training in Standard World ---")
    model, _losses = train_model(cfg, device)

    # Analyze
    print("\n--- Phase 2: Deploying to Multiverse ---")
    results = analyze_worlds(model, cfg, device)

    # Metrics
    print("\n--- Phase 3: Computing Relational Metrics ---")
    metrics = compute_relational_metrics(results, cfg)

    # Summary and visualization
    print_summary(results, metrics, cfg)
    plot_results(results, metrics, cfg)
    
    # Save CSV
    save_metrics_csv(metrics, cfg)

    print("\n[EX7] Done.")


if __name__ == "__main__":
    main()
