#!/usr/bin/env python3
"""
CSCT Experiment Suite Runner
============================

Unified runner for CSCT experiments:
  - EX1: Single-channel waveform discretization (SingleGate vs MultiGate)
  - EX2: Multi-channel relational extraction (MultiGate)
  - EX3: K-dependency analysis (Lebesgue-like integration)
  - EX4: Anchor role — noise floor vs drift (A4 validation)
  - EX5: Binding problem — module synchronization via common anchor
  - EX6: Category recognition — frozen codebook categorical abstraction
  - EX7: Relational internal time — event-rate dilation
  - EX8: Semantic grounding — anchor alone provides meaning (A5)
  - EX9: Syntax emergence — composition rule learning from codes (A5)
  - EX10: Logic gate routing — command-as-anchor with MultiGate router

Experiment Design:
  EX1: Single-channel waveforms - compare SingleGate vs MultiGate (capacity control)
  EX2: Multi-channel Lissajous - MultiGate required for relational info
  EX3: K-dependency - Shows discretization geometry varies with K
  EX4: Anchor role - Shows long-term stability vs short-term accuracy tradeoff
  EX5: Binding - Shows common anchor enables inter-module relation preservation
  EX6: Category recognition - Shows frozen codebook performs categorical abstraction
  EX7: Relational time - Shows internal time dilates with anchor tempo
  EX8: Semantic grounding - Shows anchor alone can provide meaning to discrete codes
  EX9: Syntax - Shows composition rules require discrete codes

Usage:
  python csct_suite.py --run ex1              # Run EX1 only
  python csct_suite.py --run ex2              # Run EX2 only
  python csct_suite.py --run ex3              # Run EX3 only
  python csct_suite.py --run ex4              # Run EX4 only
  python csct_suite.py --run ex5              # Run EX5 only
  python csct_suite.py --run ex6              # Run EX6 only
  python csct_suite.py --run ex7              # Run EX7 only
  python csct_suite.py --run ex8              # Run EX8 only (sweep)
  python csct_suite.py --run ex9              # Run EX9 only (sweep)
  python csct_suite.py --run ex10             # Run EX10 only
  python csct_suite.py --run all              # Run all
  python csct_suite.py --run ex1              # EX1 runs SingleGate vs MultiGate (always)

Author: NAOKI (CSCT Research)
"""

import argparse
import os
import sys
import subprocess
import math
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[Warning] pandas not available, summary will be limited")

import numpy as np


@dataclass
class RunConfig:
    name: str
    script: str
    extra_args: List[str]


def _infer_group_cols(df, exp: str):
    """Infer grouping columns for seed-mean aggregation.

    Strategy:
      - Numeric columns are treated as metrics.
      - We group by non-numeric columns except 'seed'/'experiment'.
      - For some experiments we pin a stable ordering / grouping keys.
      - For experiments without condition variations (EX8, EX9), returns empty list.
    """
    import numpy as np

    # preferred group keys per experiment (if present)
    # EX8/EX9 have no condition variations - single task per seed
    preferred = {
        'EX1': ['wave', 'gate_type'],
        'EX2': ['gate_type', 'k_mode', 'k_nsegs', 'K'],
        'EX3': ['K'],
        'EX4': ['noise_std'],
        'EX5': ['condition'],
        'EX6': [],  # single condition - category recognition
        'EX7': ['world'],
        'EX8': [],  # single condition experiment
        'EX9': [],  # single condition experiment
        'EX10': ['anchor_mode', 'task'],
    }.get(exp, None)

    cols = list(df.columns)
    if preferred is not None:
        keys = [c for c in preferred if c in cols]
        return keys  # may be empty for EX8/EX9

    # fallback: group by non-numeric, excluding seed/experiment
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    group_cols = [c for c in cols if c not in numeric_cols and c not in ('seed', 'experiment')]
    return group_cols


def build_seed_summary_table(exp_df, exp: str):
    """Return a table containing per-seed rows + per-group MEAN/STD rows."""
    import numpy as np
    import pandas as pd

    if exp_df is None or len(exp_df) == 0:
        return None

    df = exp_df.copy()

    # ensure seed exists
    if 'seed' not in df.columns:
        df['seed'] = np.nan

    group_cols = _infer_group_cols(df, exp)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    # group_colsに含まれる列をnumeric_colsから除外（重複エラー防止）
    numeric_cols = [c for c in numeric_cols if c not in group_cols]

    if len(numeric_cols) == 0:
        return df

    # Handle empty group_cols (single-condition experiments like EX8/EX9)
    if len(group_cols) == 0:
        # No grouping - compute overall mean/std across all seeds
        mean_row = df[numeric_cols].mean().to_frame().T
        std_row = df[numeric_cols].std(ddof=0).to_frame().T
        
        # Rename std columns
        for c in numeric_cols:
            std_row.rename(columns={c: f'{c}_std'}, inplace=True)
        
        merged = pd.concat([mean_row, std_row], axis=1)
        merged['seed'] = 'MEAN'
        merged['experiment'] = exp
        
        df['experiment'] = exp
        out = pd.concat([df, merged], ignore_index=True, sort=False)
        return out

    # group-by seed mean rows
    g = df.groupby(group_cols, dropna=False)
    mean_df = g[numeric_cols].mean().reset_index()
    std_df = g[numeric_cols].std(ddof=0).reset_index()

    # attach suffix std
    for c in numeric_cols:
        std_df.rename(columns={c: f'{c}_std'}, inplace=True)

    merged = mean_df.merge(std_df, on=group_cols, how='left')
    merged['seed'] = 'MEAN'
    merged['experiment'] = exp

    df['experiment'] = exp

    # order: group cols, seed, then rest
    out = pd.concat([df, merged], ignore_index=True, sort=False)

    # Try to sort nicely
    sort_cols = [c for c in (group_cols + ['seed']) if c in out.columns]
    try:
        out = out.sort_values(sort_cols)
    except Exception:
        pass

    return out


def save_per_experiment_summaries(all_results: List[Dict], base_dir: Path) -> None:
    """Write per-experiment CSVs with per-seed rows + SEED-mean rows.

    Output:
      {output_root}/{exN}/summary.csv
    """
    if not all_results or not HAS_PANDAS:
        return

    import pandas as pd

    df = pd.DataFrame(all_results)
    if 'experiment' not in df.columns:
        return

    for exp in sorted(df['experiment'].dropna().unique()):
        exp_df = df[df['experiment'] == exp].copy()
        out_dir = base_dir / exp.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        table = build_seed_summary_table(exp_df, exp)
        if table is None:
            continue

        out_path = out_dir / 'summary.csv'
        table.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")


def print_seed_mean_summaries(all_results: List[Dict]) -> None:
    """CLI: collect each seed summary and print SEED-mean summaries."""
    if not all_results or not HAS_PANDAS:
        return

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(all_results)
    if 'experiment' not in df.columns:
        return

    metric_priority = {
        'EX1': ['recon_loss', 'code_entropy_norm', 'unique_codes'],
        'EX2': ['recon_loss', 'k_mae', 'stability', 'unique_codes'],
        'EX3': ['recon_loss', 'zero_cross_ratio', 'extrema_ratio'],
        'EX4': ['short_mse_closed', 'short_mse_open', 'long_mse_closed', 'long_mse_open', 'crossover_idx'],
        'EX5': ['early_phase_err', 'late_phase_err', 'stability_duration', 'plv_late'],
        'EX6': ['within_trained_jsd', 'to_withheld_jsd', 'diff_jsd', 'ari_trans', 'nmi_trans', 'mse_ratio', 'mean_trained_mse', 'mean_withheld_mse'],
        'EX7': ['n_transitions', 'trans_per_sec'],
        'EX8': ['final_loss', 'mean_trained_sim', 'withheld_sim', 'trans_rate', 'n_unique_codes', 'maxp_A', 'maxp_C', 'ent_C'],
        'EX9': ['final_loss', 'withheld_sim', 'mean_trained_sim', 'trans_rate', 'maxp_a+b', 'maxp_b+c', 'ent_a+b', 'ent_b+c'],
        'EX10': ['accuracy', 'gate_acc'],
    }

    print("\n" + "=" * 70)
    print("[CLI Summary] Per-seed rows + SEED-mean (by condition)")
    print("=" * 70)

    for exp in sorted(df['experiment'].dropna().unique()):
        sub = df[df['experiment'] == exp].copy()
        group_cols = _infer_group_cols(sub, exp)

        metrics = [m for m in metric_priority.get(exp, []) if m in sub.columns]
        if not metrics:
            num_cols = list(sub.select_dtypes(include=[np.number]).columns)
            metrics = [c for c in num_cols if c not in ('seed',)]
            metrics = metrics[:4]

        print(f"\n{exp}:")

        if 'seed' not in sub.columns or not group_cols:
            mean_vals = sub[metrics].mean(numeric_only=True)
            std_vals = sub[metrics].std(numeric_only=True, ddof=0)
            mean_str = " ".join([
                f"{m}={mean_vals[m]:.4g}±{std_vals[m]:.3g}" for m in metrics if m in mean_vals
            ])
            print(f"  MEAN {mean_str}")
            continue

        for keys, gdf in sub.groupby(group_cols, dropna=False):
            keys = (keys,) if not isinstance(keys, tuple) else keys
            label = ", ".join([f"{k}={v}" for k, v in zip(group_cols, keys)])

            gdf2 = gdf.sort_values('seed') if 'seed' in gdf.columns else gdf
            for _, r in gdf2.iterrows():
                seed = r.get('seed', '?')
                vals = " ".join([
                    f"{m}={float(r[m]):.4g}" if pd.notna(r.get(m)) else f"{m}=nan"
                    for m in metrics
                ])
                print(f"    seed{seed}: {vals} | {label}")

            mean_vals = gdf2[metrics].mean(numeric_only=True)
            std_vals = gdf2[metrics].std(numeric_only=True, ddof=0)
            mean_str = " ".join([
                f"{m}={mean_vals[m]:.4g}±{std_vals[m]:.3g}" for m in metrics if m in mean_vals
            ])
            print(f"    MEAN : {mean_str} | {label}")

    print("\n" + "=" * 70)



def aggregate_convergence_data(base_dir: Path) -> None:
    """Aggregate convergence curves from all experiments into summary CSVs and plots."""
    import matplotlib.pyplot as plt
    import glob
    
    # Find all convergence PNG files and create summary
    convergence_files = list(base_dir.rglob("convergence_*.png"))
    if not convergence_files:
        print("[info] No convergence files found to aggregate")
        return
    
    print(f"\n[Convergence] Found {len(convergence_files)} convergence plots")
    
    # Group by experiment
    exp_files = {}
    for f in convergence_files:
        # Determine experiment from path
        path_str = str(f)
        if "ex1" in path_str.lower():
            exp = "EX1"
        elif "ex2" in path_str.lower():
            exp = "EX2"
        elif "ex3" in path_str.lower():
            exp = "EX3"
        elif "ex6" in path_str.lower():
            exp = "EX6"
        elif "ex7" in path_str.lower():
            exp = "EX7"
        elif "ex8" in path_str.lower():
            exp = "EX8"
        elif "ex9" in path_str.lower():
            exp = "EX9"
        else:
            exp = "OTHER"
        
        if exp not in exp_files:
            exp_files[exp] = []
        exp_files[exp].append(f)
    
    # Report
    for exp, files in sorted(exp_files.items()):
        print(f"  {exp}: {len(files)} convergence plots")
    
    # Create convergence summary figure (montage of all experiments)
    if exp_files:
        try:
            n_exps = len(exp_files)
            fig, axes = plt.subplots(1, n_exps, figsize=(4*n_exps, 4))
            if n_exps == 1:
                axes = [axes]
            
            for ax, (exp, files) in zip(axes, sorted(exp_files.items())):
                ax.text(0.5, 0.5, f"{exp}\n{len(files)} runs", 
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(exp)
            
            plt.suptitle("Convergence Summary by Experiment", fontsize=14)
            plt.tight_layout()
            summary_path = base_dir / "convergence_summary.png"
            fig.savefig(summary_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {summary_path}")
        except Exception as e:
            print(f"[warn] Could not create convergence summary: {e}")


def run_experiment(python_exe: str, script: str, args: List[str], output_dir: Path = None, 
                   use_output_dir_arg: bool = True) -> int:
    """Run a single experiment and return exit code.
    
    Args:
        python_exe: Python executable path
        script: Script filename
        args: Command line arguments
        output_dir: Output directory (for display only if use_output_dir_arg=False)
        use_output_dir_arg: If True, add --output-dir argument. If False, skip it.
    """
    if use_output_dir_arg and output_dir is not None:
        cmd = [python_exe, "-u", script] + args + ["--output-dir", str(output_dir)]
    else:
        cmd = [python_exe, "-u", script] + args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_ex1_suite(python_exe: str, base_dir: Path, waves: List[str],
                  common_args: List[str], seeds: int = 1) -> List[Dict]:
    """Run EX1 waveform experiments."""
    results = []
    # EX1 is a comparison experiment (SingleGate vs MultiGate)
    gate_configs = [
        ("SingleGate", []),
        ("MultiGate", ["--use-multigate"]),
    ]
    
    for wave in waves:
        for gate_name, gate_args in gate_configs:
            for seed in range(seeds):
                out_dir = base_dir / 'ex1' / wave / gate_name / f'seed{seed}'
                out_dir.mkdir(parents=True, exist_ok=True)
                args = common_args + gate_args + ["--wave", wave, "--seed", str(seed)]
                
                rc = run_experiment(python_exe, "csct_ex1_waveforms.py", args, out_dir)
                
                # Collect metrics if successful
                metrics_path = out_dir / "metrics_history.csv"
                if rc == 0 and metrics_path.exists():
                    if HAS_PANDAS:
                        df = pd.read_csv(metrics_path)
                        if len(df) > 0:
                            last = df.iloc[-1].to_dict()
                            last["experiment"] = "EX1"
                            last["wave"] = wave
                            last["seed"] = seed
                            last["gate_type"] = gate_name
                            results.append(last)
                    else:
                        import csv
                        with open(metrics_path, 'r') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if rows:
                                last = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v 
                                       for k, v in rows[-1].items()}
                                last["experiment"] = "EX1"
                                last["wave"] = wave
                                last["seed"] = seed
                                last["gate_type"] = gate_name
                                results.append(last)
    
    return results


def run_ex2_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  seeds: int = 5,
                  k_mode: str = "piecewise",
                  k_nsegs: int = 4,
                  steps: int = 1000,
                  log_every: int = 50) -> List[Dict]:
    """Run EX2 Lissajous/k(t) experiments.

    Tests relational information extraction: frequency ratio k(t) between channels.

    NOTE: This experiment is a comparison (SingleGate vs MultiGate).
    """
    results = []

    # gate comparison (restore original intent)
    gate_configs = [
        ("SingleGate", ["--single-gate"]),
        ("MultiGate", []),
    ]

    device = common_args[common_args.index("--device") + 1] if "--device" in common_args else "cuda"
    K = common_args[common_args.index("--n-clocks") + 1] if "--n-clocks" in common_args else "8"

    for gate_name, gate_args in gate_configs:
        for seed in range(seeds):
            out_dir = base_dir / 'ex2' / gate_name / f'seed{seed}'
            out_dir.mkdir(parents=True, exist_ok=True)

            args = [
                "--device", device,
                "--steps", str(steps),
                "--K", str(K),
                "--seed", str(seed),
                "--k-mode", k_mode,
                "--k-nsegs", str(k_nsegs),
                "--log-every", str(log_every),
                "--output-dir", str(out_dir),
            ] + gate_args

            rc = run_experiment(python_exe, "csct_ex2_relational.py", args, out_dir, use_output_dir_arg=False)

            metrics_path = out_dir / "metrics_history.csv"
            if rc == 0 and metrics_path.exists():
                if HAS_PANDAS:
                    df = pd.read_csv(metrics_path)
                    if len(df) > 0:
                        last = df.iloc[-1].to_dict()
                        last["experiment"] = "EX2"
                        last["seed"] = seed
                        last["k_mode"] = k_mode
                        last["k_nsegs"] = k_nsegs
                        last["gate_type"] = gate_name
                        last["K"] = int(K)
                        results.append(last)
                else:
                    import csv
                    with open(metrics_path, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v
                                   for k, v in rows[-1].items()}
                            last["experiment"] = "EX2"
                            last["seed"] = seed
                            last["k_mode"] = k_mode
                            last["k_nsegs"] = k_nsegs
                            last["gate_type"] = gate_name
                            last["K"] = int(K)
                            results.append(last)
            elif rc == 0:
                results.append({
                    "experiment": "EX2",
                    "seed": seed,
                    "k_mode": k_mode,
                    "k_nsegs": k_nsegs,
                    "gate_type": gate_name,
                    "K": int(K),
                    "status": "completed",
                })

    return results


def run_ex3_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  k_values: List[int], seeds: int = 1) -> List[Dict]:
    """Run EX3 K-dependency analysis.

    Tests how codebook size K determines discretization geometry.
    Uses SingleGate with sine wave for clean analysis.
    """
    results = []

    for seed in range(seeds):
        out_dir = base_dir / 'ex3' / f'seed{seed}'
        
        # Build args for csct_ex3_kdependency.py
        k_str = ",".join(str(k) for k in k_values)
        args = [
            "--device", common_args[common_args.index("--device") + 1] if "--device" in common_args else "cuda",
            "--steps", "2000",  # EX3 uses fewer steps
            "--k-values", k_str,
            "--seed", str(seed),
            "--tau", common_args[common_args.index("--beta") + 1] if "--beta" in common_args else "50.0",
        ]

        rc = run_experiment(python_exe, "csct_ex3_kdependency.py", args, out_dir)

        # Collect metrics
        metrics_path = out_dir / "k_dependency_metrics.csv"
        if rc == 0 and metrics_path.exists():
            if HAS_PANDAS:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    r = row.to_dict()
                    r["experiment"] = "EX3"
                    r["seed"] = seed
                    results.append(r)
            else:
                import csv
                with open(metrics_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v
                             for k, v in row.items()}
                        r["experiment"] = "EX3"
                        r["seed"] = seed
                        results.append(r)

    return results


def run_ex4_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  noise_levels: List[float], seeds: int = 1) -> List[Dict]:
    """Run EX4 anchor role / noise floor experiments.

    Tests the tradeoff between noise import (anchored) and drift (free-running).
    """
    results = []

    for seed in range(seeds):
        out_dir = base_dir / 'ex4' / f'seed{seed}'
        
        args = [
            "--noise-levels", *[str(n) for n in noise_levels],
            "--seed", str(seed),
        ]

        rc = run_experiment(python_exe, "csct_ex4_noise_floor.py", args, out_dir)

        # Collect metrics
        metrics_path = out_dir / "ex4_metrics.csv"
        if rc == 0 and metrics_path.exists():
            if HAS_PANDAS:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    r = row.to_dict()
                    r["experiment"] = "EX4"
                    r["seed"] = seed
                    results.append(r)
            else:
                import csv
                with open(metrics_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v
                             for k, v in row.items()}
                        r["experiment"] = "EX4"
                        r["seed"] = seed
                        results.append(r)

    return results


def run_ex5_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  phase_diff: float, pll_alpha: float, seeds: int = 1) -> List[Dict]:
    """Run EX5 binding problem experiments.

    Tests whether common anchor enables inter-module relation preservation.
    """
    results = []

    for seed in range(seeds):
        out_dir = base_dir / 'ex5' / f'seed{seed}'
        
        args = [
            "--phase-diff", str(phase_diff),
            "--pll-alpha", str(pll_alpha),
            "--seed", str(seed),
            "--output-dir", str(out_dir),
        ]

        rc = run_experiment(python_exe, "csct_ex5_binding.py", args, out_dir)

        # Collect metrics
        metrics_path = out_dir / "ex5_metrics.csv"
        if rc == 0 and metrics_path.exists():
            if HAS_PANDAS:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    r = row.to_dict()
                    r["experiment"] = "EX5"
                    r["seed"] = seed
                    results.append(r)
            else:
                import csv
                with open(metrics_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v
                             for k, v in row.items()}
                        r["experiment"] = "EX5"
                        r["seed"] = seed
                        results.append(r)

    return results


def run_ex6_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  n_seeds: int = 20) -> List[Dict]:
    """Run EX6 category recognition experiments.

    Tests frozen codebook's ability to exclude unseen frequency-ratio shapes.
    Design: Train on ShapeA (1:1 circle) and ShapeB (2:1 figure-8)
            Test on ShapeC (3:1 trefoil) - WITHHELD during training
    
    Key metrics: JSD (Jensen-Shannon Divergence), ARI, MSE ratio
    """
    results = []

    for seed in range(n_seeds):
        out_dir = base_dir / 'ex6' / f'seed{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        args = [
            "--seed", str(seed),
            "--output-dir", str(out_dir),
        ]

        rc = run_experiment(python_exe, "csct_ex6_category_recognition.py", args, out_dir, use_output_dir_arg=False)

        # Collect metrics
        metrics_path = out_dir / "ex6_metrics.csv"
        if rc == 0 and metrics_path.exists():
            if HAS_PANDAS:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    r = row.to_dict()
                    r["experiment"] = "EX6"
                    r["seed"] = seed
                    results.append(r)
            else:
                import csv
                with open(metrics_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v
                             for k, v in row.items()}
                        r["experiment"] = "EX6"
                        r["seed"] = seed
                        results.append(r)

    return results


def run_ex7_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  n_clocks: int, n_epochs: int, seeds: int = 1) -> List[Dict]:
    """Run EX7 relational internal time experiments.

    Tests event-rate dilation under tempo warp.
    """
    results = []

    for seed in range(seeds):
        out_dir = base_dir / 'ex7' / f'seed{seed}'
        
        args = [
            "--n-clocks", str(n_clocks),
            "--n-epochs", str(n_epochs),
            "--seed", str(seed),
            "--output-dir", str(out_dir),
        ]

        rc = run_experiment(python_exe, "csct_ex7_relational_time.py", args, out_dir)

        # Collect metrics
        metrics_path = out_dir / "ex7_metrics.csv"
        if rc == 0 and metrics_path.exists():
            if HAS_PANDAS:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    r = row.to_dict()
                    r["experiment"] = "EX7"
                    r["seed"] = seed
                    results.append(r)
            else:
                import csv
                with open(metrics_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v
                             for k, v in row.items()}
                        r["experiment"] = "EX7"
                        r["seed"] = seed
                        results.append(r)

    return results


def run_ex8_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  train_steps: int = 4000, test_steps: int = 6000, seeds: int = 30,
                  hull_conditions: bool = True) -> List[Dict]:
    """Run EX8 meaning emergence experiments.

    Tests if anchor alone can provide semantic grounding via fixed codebook.
    Training: A, B (single) + A+B, A+C, B+C (composite)
    Test: C (withheld as single) - can model extract C's meaning?
    
    If hull_conditions=True:
        - Run seeds/3 with hull_mode='in_hull' (C inside convex hull)
        - Run seeds/3 with hull_mode='out_hull' (C outside convex hull)
        - Run seeds/3 with hull_mode='random' (baseline)
        - Perform statistical test comparing conditions
    """
    results = []
    
    if hull_conditions:
        # Split seeds between 3 conditions
        seeds_per_condition = seeds // 3
        conditions = [
            ('in_hull', seeds_per_condition),
            ('out_hull', seeds_per_condition),
            ('random', seeds_per_condition)
        ]
    else:
        conditions = [('random', seeds)]
    
    for hull_mode, n_seeds in conditions:
        for seed in range(n_seeds):
            out_dir = base_dir / 'ex8' / f'{hull_mode}_seed{seed}'
            out_dir.mkdir(parents=True, exist_ok=True)
            
            args = [
                "--train-steps", str(train_steps),
                "--test-steps", str(test_steps),
                "--seed", str(seed),
                "--hull-mode", hull_mode,
                "--output-dir", str(out_dir),
            ]
            
            rc = run_experiment(python_exe, "csct_ex8_meaning.py", args, out_dir, use_output_dir_arg=False)
            
            # Collect metrics from CSV
            metrics_path = out_dir / "ex8_metrics.csv"
            if rc == 0 and metrics_path.exists():
                if HAS_PANDAS:
                    df = pd.read_csv(metrics_path)
                    for _, row in df.iterrows():
                        r = row.to_dict()
                        r["experiment"] = "EX8"
                        r["seed"] = seed
                        r["hull_mode"] = hull_mode
                        results.append(r)
                else:
                    import csv
                    with open(metrics_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v 
                                 for k, v in row.items()}
                            r["experiment"] = "EX8"
                            r["seed"] = seed
                            r["hull_mode"] = hull_mode
                            results.append(r)
    
    # Statistical test if hull_conditions
    if hull_conditions and len(results) > 0:
        _perform_ex8_hull_test(results)
    
    return results


def _perform_ex8_hull_test(results: List[Dict]) -> None:
    """Perform statistical test comparing in_hull vs out_hull vs random conditions."""
    import numpy as np
    
    in_hull_sims = [r['withheld_sim'] for r in results if r.get('hull_mode') == 'in_hull']
    out_hull_sims = [r['withheld_sim'] for r in results if r.get('hull_mode') == 'out_hull']
    random_sims = [r['withheld_sim'] for r in results if r.get('hull_mode') == 'random']
    
    print(f"\n{'='*60}")
    print("EX8 CONVEX HULL HYPOTHESIS TEST (3 conditions)")
    print(f"{'='*60}")
    
    conditions_data = [
        ('IN_HULL', in_hull_sims, 'C inside convex hull of A, B'),
        ('OUT_HULL', out_hull_sims, 'C orthogonal to A and B'),
        ('RANDOM', random_sims, 'C random (baseline)'),
    ]
    
    for name, sims, desc in conditions_data:
        if len(sims) == 0:
            print(f"\nCondition: {name} - No data")
            continue
        sims_arr = np.array(sims)
        success = (sims_arr > 0.90).sum()
        print(f"\nCondition: {name} ({desc})")
        print(f"  N = {len(sims_arr)}")
        print(f"  Withheld sim: {sims_arr.mean():.3f} ± {sims_arr.std():.3f}")
        print(f"  Success rate: {success}/{len(sims_arr)} ({100*success/len(sims_arr):.0f}%)")
    
    # Statistical tests
    try:
        from scipy.stats import kruskal, mannwhitneyu
        
        # Kruskal-Wallis H-test (non-parametric ANOVA)
        groups = [g for g in [in_hull_sims, out_hull_sims, random_sims] if len(g) > 0]
        if len(groups) >= 2:
            stat_h, p_h = kruskal(*groups)
            print(f"\n--- Statistical Tests ---")
            print(f"  Kruskal-Wallis H-test: H={stat_h:.2f}, p={p_h:.6f}")
            
            if p_h < 0.05:
                print(f"  → SIGNIFICANT overall difference (p < 0.05)")
                
                # Post-hoc pairwise comparisons
                print(f"\n  Post-hoc pairwise (Mann-Whitney U, one-sided):")
                
                if len(in_hull_sims) > 0 and len(out_hull_sims) > 0:
                    _, p_io = mannwhitneyu(in_hull_sims, out_hull_sims, alternative='greater')
                    print(f"    IN_HULL > OUT_HULL: p = {p_io:.6f} {'*' if p_io < 0.05 else ''}")
                
                if len(in_hull_sims) > 0 and len(random_sims) > 0:
                    _, p_ir = mannwhitneyu(in_hull_sims, random_sims, alternative='greater')
                    print(f"    IN_HULL > RANDOM:   p = {p_ir:.6f} {'*' if p_ir < 0.05 else ''}")
                
                if len(random_sims) > 0 and len(out_hull_sims) > 0:
                    _, p_ro = mannwhitneyu(random_sims, out_hull_sims, alternative='greater')
                    print(f"    RANDOM > OUT_HULL:  p = {p_ro:.6f} {'*' if p_ro < 0.05 else ''}")
            else:
                print(f"  → No significant difference between conditions")
                
    except ImportError:
        # Fallback: simple comparison
        print(f"\n--- Simple Comparison ---")
        if len(in_hull_sims) > 0 and len(out_hull_sims) > 0:
            diff = np.mean(in_hull_sims) - np.mean(out_hull_sims)
            print(f"  IN_HULL - OUT_HULL: {diff:.3f}")
    
    # Conclusion
    if len(in_hull_sims) > 0 and len(out_hull_sims) > 0:
        in_mean = np.mean(in_hull_sims)
        out_mean = np.mean(out_hull_sims)
        if in_mean > 0.90 and out_mean < 0.5:
            print(f"\n→ CONVEX HULL HYPOTHESIS STRONGLY SUPPORTED")
            print(f"  Meaning extraction requires C INSIDE the hull")
        elif in_mean > out_mean + 0.2:
            print(f"\n→ Convex hull hypothesis supported (moderate)")
    
    print(f"{'='*60}\n")


def run_ex9_suite(python_exe: str, base_dir: Path, common_args: List[str],
                  train_steps: int = 4000, seeds: int = 30,
                  hull_conditions: bool = True) -> List[Dict]:
    """Run EX9 syntax emergence experiments.

    Tests composition rule inference via discrete codes.
    Training: A, B, C + A+B, A+C (B+C withheld)
    Test: b+c → B+C? (inference of unseen composition)
    
    If hull_conditions=True:
        - Run seeds/3 with hull_mode='in_hull' (C inside convex hull)
        - Run seeds/3 with hull_mode='out_hull' (C outside convex hull)
        - Run seeds/3 with hull_mode='random' (baseline)
        - Perform statistical test comparing conditions
    """
    results = []
    
    if hull_conditions:
        seeds_per_condition = seeds // 3
        conditions = [
            ('in_hull', seeds_per_condition),
            ('out_hull', seeds_per_condition),
            ('random', seeds_per_condition)
        ]
    else:
        conditions = [('random', seeds)]
    
    for hull_mode, n_seeds in conditions:
        for seed in range(n_seeds):
            out_dir = base_dir / 'ex9' / f'{hull_mode}_seed{seed}'
            out_dir.mkdir(parents=True, exist_ok=True)
            
            args = [
                "--train-steps", str(train_steps),
                "--seed", str(seed),
                "--hull-mode", hull_mode,
                "--output-dir", str(out_dir),
            ]
            
            rc = run_experiment(python_exe, "csct_ex9_syntax.py", args, out_dir, use_output_dir_arg=False)
            
            # Collect metrics from CSV
            metrics_path = out_dir / "ex9_metrics.csv"
            if rc == 0 and metrics_path.exists():
                if HAS_PANDAS:
                    df = pd.read_csv(metrics_path)
                    for _, row in df.iterrows():
                        r = row.to_dict()
                        r["experiment"] = "EX9"
                        r["seed"] = seed
                        r["hull_mode"] = hull_mode
                        results.append(r)
                else:
                    import csv
                    with open(metrics_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v 
                                 for k, v in row.items()}
                            r["experiment"] = "EX9"
                            r["seed"] = seed
                            r["hull_mode"] = hull_mode
                            results.append(r)
            elif rc == 0:
                results.append({
                    "experiment": "EX9",
                    "seed": seed,
                    "hull_mode": hull_mode,
                    "status": "completed",
                })
    
    # Statistical test if hull_conditions
    if hull_conditions and len(results) > 0:
        _perform_ex9_hull_test(results)
    
    return results


def _perform_ex9_hull_test(results: List[Dict]) -> None:
    """Perform statistical test comparing in_hull vs out_hull vs random conditions for EX9."""
    import numpy as np
    
    in_hull_sims = [r['withheld_sim'] for r in results if r.get('hull_mode') == 'in_hull' and 'withheld_sim' in r]
    out_hull_sims = [r['withheld_sim'] for r in results if r.get('hull_mode') == 'out_hull' and 'withheld_sim' in r]
    random_sims = [r['withheld_sim'] for r in results if r.get('hull_mode') == 'random' and 'withheld_sim' in r]
    
    print(f"\n{'='*60}")
    print("EX9 CONVEX HULL HYPOTHESIS TEST (3 conditions)")
    print("  Syntax Inference: Can model infer B+C from B, C, A+B, A+C?")
    print(f"{'='*60}")
    
    conditions_data = [
        ('IN_HULL', in_hull_sims, 'C inside convex hull of A, B'),
        ('OUT_HULL', out_hull_sims, 'C orthogonal to A and B'),
        ('RANDOM', random_sims, 'C random (baseline)'),
    ]
    
    for name, sims, desc in conditions_data:
        if len(sims) == 0:
            print(f"\nCondition: {name} - No data")
            continue
        sims_arr = np.array(sims)
        success = (sims_arr > 0.90).sum()
        print(f"\nCondition: {name} ({desc})")
        print(f"  N = {len(sims_arr)}")
        print(f"  Withheld sim (B+C): {sims_arr.mean():.3f} ± {sims_arr.std():.3f}")
        print(f"  Success rate: {success}/{len(sims_arr)} ({100*success/len(sims_arr):.0f}%)")
    
    # Statistical tests
    try:
        from scipy.stats import kruskal, mannwhitneyu
        
        groups = [g for g in [in_hull_sims, out_hull_sims, random_sims] if len(g) > 0]
        if len(groups) >= 2:
            stat_h, p_h = kruskal(*groups)
            print(f"\n--- Statistical Tests ---")
            print(f"  Kruskal-Wallis H-test: H={stat_h:.2f}, p={p_h:.6f}")
            
            if p_h < 0.05:
                print(f"  → SIGNIFICANT overall difference (p < 0.05)")
                
                print(f"\n  Post-hoc pairwise (Mann-Whitney U, one-sided):")
                
                if len(in_hull_sims) > 0 and len(out_hull_sims) > 0:
                    _, p_io = mannwhitneyu(in_hull_sims, out_hull_sims, alternative='greater')
                    print(f"    IN_HULL > OUT_HULL: p = {p_io:.6f} {'*' if p_io < 0.05 else ''}")
                
                if len(in_hull_sims) > 0 and len(random_sims) > 0:
                    _, p_ir = mannwhitneyu(in_hull_sims, random_sims, alternative='greater')
                    print(f"    IN_HULL > RANDOM:   p = {p_ir:.6f} {'*' if p_ir < 0.05 else ''}")
                
                if len(random_sims) > 0 and len(out_hull_sims) > 0:
                    _, p_ro = mannwhitneyu(random_sims, out_hull_sims, alternative='greater')
                    print(f"    RANDOM > OUT_HULL:  p = {p_ro:.6f} {'*' if p_ro < 0.05 else ''}")
            else:
                print(f"  → No significant difference between conditions")
                
    except ImportError:
        print(f"\n--- Simple Comparison ---")
        if len(in_hull_sims) > 0 and len(out_hull_sims) > 0:
            diff = np.mean(in_hull_sims) - np.mean(out_hull_sims)
            print(f"  IN_HULL - OUT_HULL: {diff:.3f}")
    
    # Conclusion
    if len(in_hull_sims) > 0 and len(out_hull_sims) > 0:
        in_mean = np.mean(in_hull_sims)
        out_mean = np.mean(out_hull_sims)
        if in_mean > 0.90 and out_mean < 0.5:
            print(f"\n→ SYNTAX INFERENCE depends on vector geometry")
            print(f"  Compositional rules require C INSIDE the hull")
        elif in_mean > out_mean + 0.2:
            print(f"\n→ Geometry effect detected (moderate)")
    
    print(f"{'='*60}\n")


def run_ex10_suite(python_exe: str, base_dir: Path, common_args: List[str],
                   steps: int = 6000, seeds: int = 5) -> List[Dict]:
    """Run EX10 logic gate experiments.

    Tests command-as-anchor with MultiGate router (AND/OR/NOR).
    """
    results = []
    
    for seed in range(seeds):
        out_dir = base_dir / 'ex10' / f'seed{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        args = [
            "--device", common_args[common_args.index("--device") + 1] if "--device" in common_args else "cuda",
            "--steps", str(steps),
            "--seed", str(seed),
            "--outdir", str(out_dir),
        ]
        
        rc = run_experiment(python_exe, "csct_ex10_logic.py", args, out_dir, use_output_dir_arg=False)
        
        # Collect metrics from ablation_results.csv
        ablation_path = out_dir / "ablation_results.csv"
        if rc == 0 and ablation_path.exists():
            if HAS_PANDAS:
                df = pd.read_csv(ablation_path)
                for _, row in df.iterrows():
                    r = row.to_dict()
                    r["experiment"] = "EX10"
                    r["seed"] = seed
                    results.append(r)
            else:
                import csv
                with open(ablation_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = {k: float(v) if v.replace('.','',1).replace('-','',1).isdigit() else v
                             for k, v in row.items()}
                        r["experiment"] = "EX10"
                        r["seed"] = seed
                        results.append(r)
    
    return results


def create_ex1_aggregate_figures(base_dir: Path) -> None:
    """Create EX1 aggregate figures from all seeds.
    
    Reads metrics from summary.csv and creates:
    1. SingleGate vs MultiGate comparison with error bars per waveform
    2. Overall comparison across all waveforms
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex1_dir = base_dir / 'ex1'
    if not ex1_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Read summary.csv
    summary_path = ex1_dir / 'summary.csv'
    if not summary_path.exists():
        print("[info] No EX1 summary.csv found")
        return
    
    df = pd.read_csv(summary_path)
    
    # Filter out MEAN rows for per-seed analysis
    df_seeds = df[df['seed'].astype(str) != 'MEAN'].copy()
    
    if len(df_seeds) == 0:
        print("[info] No per-seed data in EX1 summary.csv")
        return
    
    n_seeds = df_seeds['seed'].nunique()
    print(f"[EX1] Creating aggregate figures ({n_seeds} seeds)")
    
    # Get gate types and waves
    gate_types = sorted(df_seeds['gate_type'].unique()) if 'gate_type' in df_seeds.columns else ['SingleGate']
    waves = sorted(df_seeds['wave'].unique()) if 'wave' in df_seeds.columns else []
    
    if not waves:
        print("[info] No wave data in EX1 summary.csv")
        return
    
    # Figure 1: Per-waveform comparison with error bars
    n_waves = len(waves)
    fig, ax = plt.subplots(figsize=(max(12, n_waves * 1.5), 6))
    
    x = np.arange(n_waves)
    width = 0.35
    n_gates = len(gate_types)
    
    colors = ['C0', 'C1', 'C2', 'C3']
    
    for gi, gate in enumerate(gate_types):
        df_gate = df_seeds[df_seeds['gate_type'] == gate]
        means = []
        stds = []
        for wave in waves:
            df_wave = df_gate[df_gate['wave'] == wave]
            if len(df_wave) > 0 and 'recon_loss' in df_wave.columns:
                means.append(df_wave['recon_loss'].mean())
                stds.append(df_wave['recon_loss'].std())
            else:
                means.append(0)
                stds.append(0)
        
        offset = (gi - (n_gates - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=gate, color=colors[gi % len(colors)], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Waveform Type', fontsize=11)
    ax.set_ylabel('Reconstruction Loss', fontsize=11)
    ax.set_title(f'EX1: Waveform Discretization - Per-waveform Comparison ({n_seeds} seeds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(waves, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    p = out_dir / 'ex1_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')
    
    # Figure 2: Overall gate comparison (aggregated across waves)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Mean recon_loss per gate type
    ax = axes[0]
    gate_means = []
    gate_stds = []
    for gate in gate_types:
        df_gate = df_seeds[df_seeds['gate_type'] == gate]
        if 'recon_loss' in df_gate.columns:
            gate_means.append(df_gate['recon_loss'].mean())
            gate_stds.append(df_gate['recon_loss'].std())
        else:
            gate_means.append(0)
            gate_stds.append(0)
    
    x_pos = np.arange(len(gate_types))
    bars = ax.bar(x_pos, gate_means, yerr=gate_stds, capsize=5,
                  color=['C0', 'C1'][:len(gate_types)], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gate_types, fontsize=11)
    ax.set_ylabel('Reconstruction Loss (mean over all waves)', fontsize=11)
    ax.set_title('Overall Performance', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for bar, mean, std in zip(bars, gate_means, gate_stds):
        ax.annotate(f'{mean:.4f}±{std:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', fontsize=10)
    
    # Right: Win rate per wave
    ax = axes[1]
    if len(gate_types) == 2:
        wins = {gate: 0 for gate in gate_types}
        for wave in waves:
            wave_means = {}
            for gate in gate_types:
                df_gw = df_seeds[(df_seeds['gate_type'] == gate) & (df_seeds['wave'] == wave)]
                if len(df_gw) > 0 and 'recon_loss' in df_gw.columns:
                    wave_means[gate] = df_gw['recon_loss'].mean()
            if len(wave_means) == 2:
                winner = min(wave_means, key=wave_means.get)
                wins[winner] += 1
        
        bars = ax.bar(list(wins.keys()), list(wins.values()),
                      color=['C0', 'C1'], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Number of Waveforms Won', fontsize=11)
        ax.set_title(f'Win Count (lower loss = win)\nTotal: {len(waves)} waveforms', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, wins.values()):
            ax.annotate(f'{val}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Win rate requires\nexactly 2 gate types',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    fig.suptitle(f'EX1: SingleGate vs MultiGate Summary ({n_seeds} seeds)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex1_gate_comparison.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')


def create_ex2_aggregate_figures(base_dir: Path) -> None:
    """Create EX2 aggregate figures from all seeds.
    
    Reads k_code_mapping.csv from each seed directory and creates:
    1. k(t) vs code scatter (all seeds overlaid)
    2. Code distribution histogram (all seeds)
    3. k estimation accuracy per gate type
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex2_dir = base_dir / 'ex2'
    if not ex2_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all k_code_mapping.csv files
    all_data = {}  # gate_type -> list of (seed, df)
    
    for gate_dir in ex2_dir.iterdir():
        if not gate_dir.is_dir():
            continue
        gate_type = gate_dir.name  # SingleGate or MultiGate
        if gate_type not in all_data:
            all_data[gate_type] = []
        
        for seed_dir in gate_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            csv_path = seed_dir / 'k_code_mapping.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    seed_num = seed_dir.name.replace('seed', '')
                    all_data[gate_type].append((seed_num, df))
                except Exception as e:
                    print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX2 k_code_mapping.csv files found")
        return
    
    # Define consistent colors for codes
    code_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for gate_type, seed_data in all_data.items():
        if not seed_data:
            continue
        
        n_seeds = len(seed_data)
        print(f"[EX2] Creating aggregate figures for {gate_type} ({n_seeds} seeds)")
        
        # Figure 1: k(t) vs code scatter (all seeds overlaid)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: k_true vs code
        ax = axes[0]
        for seed_num, df in seed_data:
            ax.scatter(df['k_true'], df['code'], 
                      alpha=0.3, s=10, label=f'seed{seed_num}')
        ax.set_xlabel('k(t) true')
        ax.set_ylabel('Code index')
        ax.set_title(f'{gate_type}: k(t) vs Code (all seeds)')
        ax.grid(True, alpha=0.3)
        if n_seeds <= 5:
            ax.legend(fontsize=8)
        
        # Right: Code distribution (histogram)
        ax = axes[1]
        all_codes = []
        for seed_num, df in seed_data:
            all_codes.extend(df['code'].tolist())
        
        if all_codes:
            max_code = max(all_codes)
            bins = np.arange(-0.5, max_code + 1.5, 1)
            ax.hist(all_codes, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Code index')
            ax.set_ylabel('Count (all seeds)')
            ax.set_title(f'{gate_type}: Code Distribution ({n_seeds} seeds)')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'EX2 {gate_type} Aggregate Analysis', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = out_dir / f'ex2_{gate_type}_aggregate.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')
        
        # Figure 2: Per-seed k vs code heatmap
        fig, axes = plt.subplots(1, min(n_seeds, 5), figsize=(4 * min(n_seeds, 5), 4))
        if n_seeds == 1:
            axes = [axes]
        
        for i, (seed_num, df) in enumerate(seed_data[:5]):
            ax = axes[i]
            # Create 2D histogram (k_true bins vs code)
            k_bins = np.linspace(df['k_true'].min() - 0.1, df['k_true'].max() + 0.1, 20)
            code_bins = np.arange(-0.5, df['code'].max() + 1.5, 1)
            
            h, xedges, yedges = np.histogram2d(df['k_true'], df['code'], 
                                               bins=[k_bins, code_bins])
            im = ax.imshow(h.T, aspect='auto', origin='lower',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          cmap='Blues')
            ax.set_xlabel('k(t)')
            ax.set_ylabel('Code')
            ax.set_title(f'seed{seed_num}')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        fig.suptitle(f'EX2 {gate_type}: k(t) vs Code Heatmap (per seed)', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = out_dir / f'ex2_{gate_type}_heatmaps.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')
        
        # Figure 3: Lissajous with code coloring (sample from each seed)
        fig, axes = plt.subplots(1, min(n_seeds, 5), figsize=(4 * min(n_seeds, 5), 4))
        if n_seeds == 1:
            axes = [axes]
        
        for i, (seed_num, df) in enumerate(seed_data[:5]):
            ax = axes[i]
            # Plot true trajectory in gray first
            ax.plot(df['ch0'], df['ch1'], color='gray', alpha=0.3, linewidth=1, label='true')
            # Then scatter reconstructed with code colors
            sc = ax.scatter(df['recon_ch0'], df['recon_ch1'], 
                           c=df['code'], cmap='tab10', s=5, alpha=0.7)
            ax.set_xlabel('ch0')
            ax.set_ylabel('ch1')
            ax.set_title(f'seed{seed_num}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, shrink=0.8)
        
        fig.suptitle(f'EX2 {gate_type}: Lissajous colored by code (gray=true)', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = out_dir / f'ex2_{gate_type}_lissajous.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')
    
    # Figure 4: Comparison between gate types (if both exist)
    if 'SingleGate' in all_data and 'MultiGate' in all_data:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for gi, (gate_type, seed_data) in enumerate(all_data.items()):
            # k estimation error
            ax = axes[0, gi]
            for seed_num, df in seed_data:
                k_err = np.abs(df['k_hat'] - df['k_true'])
                ax.plot(df['t'], k_err, alpha=0.5, label=f'seed{seed_num}')
            ax.set_xlabel('Time')
            ax.set_ylabel('|k_hat - k_true|')
            ax.set_title(f'{gate_type}: k estimation error')
            ax.grid(True, alpha=0.3)
            
            # Code usage
            ax = axes[1, gi]
            all_codes = []
            for seed_num, df in seed_data:
                all_codes.extend(df['code'].tolist())
            if all_codes:
                max_code = max(all_codes)
                bins = np.arange(-0.5, max_code + 1.5, 1)
                ax.hist(all_codes, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Code index')
            ax.set_ylabel('Count')
            ax.set_title(f'{gate_type}: Code distribution')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('EX2: SingleGate vs MultiGate Comparison', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = out_dir / 'ex2_gate_comparison.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')
    
    # Figure 5: Metrics comparison with error bars (from metrics_history.csv)
    # Collect metrics from all seeds
    all_metrics = {}  # gate_type -> list of final metrics dicts
    
    for gate_dir in ex2_dir.iterdir():
        if not gate_dir.is_dir():
            continue
        gate_type = gate_dir.name
        if gate_type not in all_metrics:
            all_metrics[gate_type] = []
        
        for seed_dir in gate_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            metrics_path = seed_dir / 'metrics_history.csv'
            if metrics_path.exists():
                try:
                    df = pd.read_csv(metrics_path)
                    if len(df) > 0:
                        last = df.iloc[-1].to_dict()
                        seed_num = seed_dir.name.replace('seed', '')
                        last['seed'] = int(seed_num)
                        all_metrics[gate_type].append(last)
                except Exception as e:
                    print(f"[warn] Could not read {metrics_path}: {e}")
    
    if all_metrics and len(all_metrics) >= 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        gate_types = list(all_metrics.keys())
        n_gates = len(gate_types)
        x_pos = np.arange(n_gates)
        
        # Count total seeds
        total_seeds = sum(len(v) for v in all_metrics.values())
        n_seeds_per_gate = {g: len(v) for g, v in all_metrics.items()}
        
        # Left: recon_loss comparison
        ax = axes[0]
        means = []
        stds = []
        for gate in gate_types:
            if all_metrics[gate]:
                vals = [m.get('recon_loss', float('nan')) for m in all_metrics[gate]]
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            else:
                means.append(0)
                stds.append(0)
        
        colors = ['C0', 'C1'][:n_gates]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(gate_types, fontsize=11)
        ax.set_ylabel('Reconstruction Loss', fontsize=11)
        ax.set_title('Reconstruction Quality', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for bar, mean, std in zip(bars, means, stds):
            ax.annotate(f'{mean:.3f}±{std:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=9)
        
        # Right: k_mae comparison (if available)
        ax = axes[1]
        means = []
        stds = []
        for gate in gate_types:
            if all_metrics[gate]:
                vals = [m.get('k_mae', float('nan')) for m in all_metrics[gate]]
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(gate_types, fontsize=11)
        ax.set_ylabel('k(t) MAE', fontsize=11)
        ax.set_title('Relational Extraction Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for bar, mean, std in zip(bars, means, stds):
            if not np.isnan(mean):
                ax.annotate(f'{mean:.3f}±{std:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', fontsize=9)
        
        seeds_str = ', '.join([f'{g}:{n}' for g, n in n_seeds_per_gate.items()])
        fig.suptitle(f'EX2: Metrics Comparison with Error Bars (seeds: {seeds_str})', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = out_dir / 'ex2_metrics_errorbars.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')


def create_ex3_aggregate_figures(base_dir: Path) -> None:
    """Create EX3 aggregate figures from all seeds.
    
    Reads k_dependency_metrics.csv from each seed directory and creates:
    1. K vs boundary ratios (zero_cross, extrema) with error bars
    2. K vs recon_loss with error bars
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex3_dir = base_dir / 'ex3'
    if not ex3_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all k_dependency_metrics.csv files
    all_data = []
    
    for seed_dir in ex3_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'k_dependency_metrics.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                seed_num = seed_dir.name.replace('seed', '')
                df['seed'] = int(seed_num)
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX3 k_dependency_metrics.csv files found")
        return
    
    # Combine all seeds
    df_all = pd.concat(all_data, ignore_index=True)
    n_seeds = df_all['seed'].nunique()
    print(f"[EX3] Creating aggregate figures ({n_seeds} seeds)")
    
    # Group by K and compute mean/std
    grouped = df_all.groupby('K')
    K_values = sorted(df_all['K'].unique())
    
    # Figure 1: K vs boundary ratios (zero_cross_ratio, extrema_ratio)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: zero_cross_ratio and extrema_ratio
    ax = axes[0]
    for metric, label, marker, color in [
        ('zero_cross_ratio', 'Zero-crossing', 'o', 'C0'),
        ('extrema_ratio', 'Extrema', '^', 'C1'),
    ]:
        means = grouped[metric].mean()
        stds = grouped[metric].std()
        ax.errorbar(K_values, means.values * 100, yerr=stds.values * 100,
                   marker=marker, color=color, capsize=4, linewidth=2,
                   label=f'{label} ({n_seeds} seeds)', alpha=0.8)
    
    ax.set_xlabel('K (codebook size)', fontsize=11)
    ax.set_ylabel('Boundary hit ratio (%)', fontsize=11)
    ax.set_title('EX3: Transition boundaries vs K', fontsize=12)
    ax.set_xticks(K_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Theoretical expectation annotations
    ax.annotate('K=2: zero-cross\ndominant', xy=(2, 80), fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.annotate('K=4: extrema\nemerge', xy=(4, 60), fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: recon_loss
    ax = axes[1]
    means = grouped['recon_loss'].mean()
    stds = grouped['recon_loss'].std()
    ax.errorbar(K_values, means.values, yerr=stds.values,
               marker='s', color='C2', capsize=4, linewidth=2,
               label=f'Recon loss ({n_seeds} seeds)', alpha=0.8)
    ax.set_xlabel('K (codebook size)', fontsize=11)
    ax.set_ylabel('Reconstruction loss', fontsize=11)
    ax.set_title('EX3: Reconstruction quality vs K', fontsize=12)
    ax.set_xticks(K_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('EX3: K-Dependency Analysis (Lebesgue-like discretization)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex3_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')
    
    # Figure 2: Per-seed detail (small multiples showing consistency)
    if n_seeds >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot each seed as a faint line, mean as bold
        for seed in df_all['seed'].unique():
            df_seed = df_all[df_all['seed'] == seed].sort_values('K')
            ax.plot(df_seed['K'], df_seed['zero_cross_ratio'] * 100,
                   'o-', alpha=0.3, color='C0', linewidth=1, markersize=4)
            ax.plot(df_seed['K'], df_seed['extrema_ratio'] * 100,
                   '^-', alpha=0.3, color='C1', linewidth=1, markersize=4)
        
        # Bold mean lines
        means_zc = grouped['zero_cross_ratio'].mean() * 100
        means_ex = grouped['extrema_ratio'].mean() * 100
        ax.plot(K_values, means_zc.values, 'o-', color='C0', linewidth=3,
               markersize=8, label='Zero-crossing (mean)')
        ax.plot(K_values, means_ex.values, '^-', color='C1', linewidth=3,
               markersize=8, label='Extrema (mean)')
        
        ax.set_xlabel('K (codebook size)', fontsize=11)
        ax.set_ylabel('Boundary hit ratio (%)', fontsize=11)
        ax.set_title(f'EX3: Seed consistency ({n_seeds} seeds, faint=individual)', fontsize=12)
        ax.set_xticks(K_values)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        fig.tight_layout()
        p = out_dir / 'ex3_seed_consistency.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')


def create_ex4_aggregate_figures(base_dir: Path) -> None:
    """Create EX4 aggregate figures from all seeds.
    
    Reads ex4_metrics.csv from each seed directory and creates:
    1. noise_std vs MSE (short/long × closed/open) with error bars
    2. Crossover point vs noise_std
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex4_dir = base_dir / 'ex4'
    if not ex4_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all ex4_metrics.csv files
    all_data = []
    
    for seed_dir in ex4_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'ex4_metrics.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                seed_num = seed_dir.name.replace('seed', '')
                df['seed'] = int(seed_num)
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX4 ex4_metrics.csv files found")
        return
    
    # Combine all seeds
    df_all = pd.concat(all_data, ignore_index=True)
    n_seeds = df_all['seed'].nunique()
    print(f"[EX4] Creating aggregate figures ({n_seeds} seeds)")
    
    # Group by noise_std and compute mean/std
    grouped = df_all.groupby('noise_std')
    noise_values = sorted(df_all['noise_std'].unique())
    
    # Figure 1: MSE comparison (Short-term vs Long-term, Closed vs Open)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(noise_values))
    width = 0.35
    
    # Left: Short-term MSE
    ax = axes[0]
    short_closed_mean = grouped['short_mse_closed'].mean().values
    short_closed_std = grouped['short_mse_closed'].std().values
    short_open_mean = grouped['short_mse_open'].mean().values
    short_open_std = grouped['short_mse_open'].std().values
    
    bars1 = ax.bar(x - width/2, short_closed_mean, width, yerr=short_closed_std,
                   label='Closed (free-run)', color='C0', alpha=0.7, capsize=3)
    bars2 = ax.bar(x + width/2, short_open_mean, width, yerr=short_open_std,
                   label='Open (anchored)', color='C1', alpha=0.7, capsize=3)
    
    ax.set_xlabel('Noise σ', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Short-term: Closed WINS\n(anchor imports noise)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n:.2f}' for n in noise_values])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Long-term MSE
    ax = axes[1]
    long_closed_mean = grouped['long_mse_closed'].mean().values
    long_closed_std = grouped['long_mse_closed'].std().values
    long_open_mean = grouped['long_mse_open'].mean().values
    long_open_std = grouped['long_mse_open'].std().values
    
    bars1 = ax.bar(x - width/2, long_closed_mean, width, yerr=long_closed_std,
                   label='Closed (free-run)', color='C0', alpha=0.7, capsize=3)
    bars2 = ax.bar(x + width/2, long_open_mean, width, yerr=long_open_std,
                   label='Open (anchored)', color='C1', alpha=0.7, capsize=3)
    
    ax.set_xlabel('Noise σ', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Long-term: Open WINS\n(anchor prevents drift)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n:.2f}' for n in noise_values])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'EX4: Anchor Role — Noise Floor vs Drift ({n_seeds} seeds)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex4_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')
    
    # Figure 2: Crossover point analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    crossover_mean = grouped['crossover_idx'].mean().values
    crossover_std = grouped['crossover_idx'].std().values
    
    # Plot each seed as faint points
    for seed in df_all['seed'].unique():
        df_seed = df_all[df_all['seed'] == seed].sort_values('noise_std')
        ax.scatter(df_seed['noise_std'], df_seed['crossover_idx'],
                  alpha=0.3, color='C2', s=30)
    
    # Bold mean line with error bars
    ax.errorbar(noise_values, crossover_mean, yerr=crossover_std,
               marker='o', color='C2', linewidth=2, markersize=8,
               capsize=4, label=f'Mean ± std ({n_seeds} seeds)')
    
    ax.set_xlabel('Noise σ', fontsize=11)
    ax.set_ylabel('Crossover step', fontsize=11)
    ax.set_title('EX4: Crossover Point (when Open beats Closed)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotation
    ax.annotate('Earlier crossover = anchor\nmore valuable despite noise',
               xy=(noise_values[-1], crossover_mean[-1]),
               xytext=(noise_values[-1] - 0.05, crossover_mean[-1] + 100),
               fontsize=9, ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    fig.tight_layout()
    p = out_dir / 'ex4_crossover.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')


def create_ex5_aggregate_figures(base_dir: Path) -> None:
    """Create EX5 aggregate figures from all seeds.
    
    Reads ex5_metrics.csv from each seed directory and creates:
    1. Binding error comparison (Closed vs Open, early/late)
    2. Stability duration and PLV comparison
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex5_dir = base_dir / 'ex5'
    if not ex5_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all ex5_metrics.csv files
    all_data = []
    
    for seed_dir in ex5_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'ex5_metrics.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                seed_num = seed_dir.name.replace('seed', '')
                df['seed'] = int(seed_num)
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX5 ex5_metrics.csv files found")
        return
    
    # Combine all seeds
    df_all = pd.concat(all_data, ignore_index=True)
    n_seeds = df_all['seed'].nunique()
    print(f"[EX5] Creating aggregate figures ({n_seeds} seeds)")
    
    # Separate closed and open
    df_closed = df_all[df_all['condition'] == 'closed']
    df_open = df_all[df_all['condition'] == 'open']
    
    if len(df_closed) == 0 or len(df_open) == 0:
        print("[warn] Missing closed or open condition data")
        return
    
    # Figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Phase error bar chart (early vs late, closed vs open)
    ax = axes[0]
    
    metrics = ['early_phase_err', 'late_phase_err']
    labels = ['Short-term\n(t < 200)', 'Long-term\n(t > 1600)']
    x = np.arange(len(metrics))
    width = 0.35
    
    closed_means = [df_closed[m].mean() for m in metrics]
    closed_stds = [df_closed[m].std() for m in metrics]
    open_means = [df_open[m].mean() for m in metrics]
    open_stds = [df_open[m].std() for m in metrics]
    
    bars1 = ax.bar(x - width/2, closed_means, width, yerr=closed_stds,
                   label='Closed (no anchor)', color='C0', alpha=0.7, capsize=4)
    bars2 = ax.bar(x + width/2, open_means, width, yerr=open_stds,
                   label='Open (anchored)', color='C1', alpha=0.7, capsize=4)
    
    ax.set_xlabel('Time Window', fontsize=11)
    ax.set_ylabel('Binding Error (rad)', fontsize=11)
    ax.set_title('Binding Error: Closed vs Open', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, (vc, vo) in enumerate(zip(closed_means, open_means)):
        if vo > 1e-9:
            ratio = vc / vo
            ax.annotate(f'{ratio:.1f}×', xy=(i, max(vc, vo) * 1.15),
                       ha='center', fontsize=11, fontweight='bold')
    
    # Middle: Stability duration comparison
    ax = axes[1]
    
    stab_metrics = ['stability_duration']
    stab_labels = ['Stability\nDuration']
    x = np.arange(len(stab_metrics))
    
    # Check if stability_duration exists
    if 'stability_duration' in df_closed.columns:
        closed_stab = [df_closed['stability_duration'].mean()]
        closed_stab_std = [df_closed['stability_duration'].std()]
        open_stab = [df_open['stability_duration'].mean()]
        open_stab_std = [df_open['stability_duration'].std()]
        
        bars1 = ax.bar(x - width/2, closed_stab, width, yerr=closed_stab_std,
                       label='Closed', color='C0', alpha=0.7, capsize=4)
        bars2 = ax.bar(x + width/2, open_stab, width, yerr=open_stab_std,
                       label='Open', color='C1', alpha=0.7, capsize=4)
        
        ax.set_ylabel('Stability Duration (fraction)', fontsize=11)
        ax.set_title('Stability Duration\n(phase err < 0.3 rad)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(stab_labels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        # Add value annotations
        for bar, val in zip(bars1, closed_stab):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
        for bar, val in zip(bars2, open_stab):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'stability_duration\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Stability Duration', fontsize=12)
    
    # Right: PLV late comparison
    ax = axes[2]
    
    if 'plv_late' in df_closed.columns:
        plv_metrics = ['plv_late']
        x = np.arange(len(plv_metrics))
        
        closed_plv = [df_closed['plv_late'].mean()]
        closed_plv_std = [df_closed['plv_late'].std()]
        open_plv = [df_open['plv_late'].mean()]
        open_plv_std = [df_open['plv_late'].std()]
        
        bars1 = ax.bar(x - width/2, closed_plv, width, yerr=closed_plv_std,
                       label='Closed', color='C0', alpha=0.7, capsize=4)
        bars2 = ax.bar(x + width/2, open_plv, width, yerr=open_plv_std,
                       label='Open', color='C1', alpha=0.7, capsize=4)
        
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect sync')
        ax.set_ylabel('Phase Locking Value', fontsize=11)
        ax.set_title('PLV (Late Window)\nHigher = Better Sync', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(['PLV Late'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        # Add value annotations
        for bar, val in zip(bars1, closed_plv):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
        for bar, val in zip(bars2, open_plv):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'plv_late\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title('PLV Late', fontsize=12)
    
    fig.suptitle(f'EX5: Binding Problem — Common Anchor Preserves Relations ({n_seeds} seeds)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex5_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')


def create_ex6_aggregate_figures(base_dir: Path) -> None:
    """Create EX6 aggregate figures from all seeds.
    
    Reads ex6_metrics.csv from each seed directory and creates:
    1. JSD comparison (trained vs withheld) - with error bars
    2. Clustering metrics (ARI, NMI)
    3. MSE comparison (trained vs withheld)
    4. JSD difference distribution
    
    Design: ShapeA (1:1), ShapeB (2:1) trained; ShapeC (3:1) WITHHELD
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex6_dir = base_dir / 'ex6'
    if not ex6_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all ex6_metrics.csv files
    all_data = []
    
    for seed_dir in ex6_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'ex6_metrics.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                seed_num = seed_dir.name.replace('seed', '')
                df['seed'] = int(seed_num)
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX6 ex6_metrics.csv files found")
        return
    
    # Combine all seeds
    df_all = pd.concat(all_data, ignore_index=True)
    n_seeds = df_all['seed'].nunique()
    print(f"[EX6] Creating aggregate figures ({n_seeds} seeds)")
    
    # Figure: Category Recognition Results (Lissajous Frequency-Ratio Design)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: JSD comparison (trained vs withheld)
    # JSD is distance: 0=identical, higher=different
    ax = axes[0, 0]
    if 'within_trained_jsd' in df_all.columns and 'to_withheld_jsd' in df_all.columns:
        within_jsd = df_all['within_trained_jsd']
        to_withheld_jsd = df_all['to_withheld_jsd']
        
        bars = ax.bar([0, 1], [within_jsd.mean(), to_withheld_jsd.mean()], 
                      yerr=[within_jsd.std(), to_withheld_jsd.std()], capsize=5,
                      color=['blue', 'red'], alpha=0.7, edgecolor='black')
        bars[1].set_linewidth(2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Trained-Trained\n(A-B: 1:1, 2:1)', 'Trained-Withheld*\n(A-C, B-C: 3:1)'], fontsize=10)
        ax.set_ylabel('Jensen-Shannon Divergence', fontsize=11)
        ax.set_title('Category Separation (JSD)\n(0=identical, higher=different)', fontsize=12)
        
        # Add annotation
        diff_jsd_mean = df_all['diff_jsd'].mean() if 'diff_jsd' in df_all.columns else 0
        diff_jsd_std = df_all['diff_jsd'].std() if 'diff_jsd' in df_all.columns else 0
        ax.annotate(f'Diff: {diff_jsd_mean:.3f} +/- {diff_jsd_std:.3f}', 
                   xy=(0.5, 0.95), xycoords='axes fraction',
                   ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(['* ShapeC (3:1 trefoil) WITHHELD'], fontsize=9, loc='upper right')
    
    # Top-right: Clustering metrics (ARI, NMI for both histogram and transition)
    ax = axes[0, 1]
    metrics = []
    labels = []
    stds = []
    colors_m = []
    
    if 'ari_hist' in df_all.columns:
        metrics.append(df_all['ari_hist'].mean())
        stds.append(df_all['ari_hist'].std())
        labels.append('ARI\n(Histogram)')
        colors_m.append('purple')
    if 'ari_trans' in df_all.columns:
        metrics.append(df_all['ari_trans'].mean())
        stds.append(df_all['ari_trans'].std())
        labels.append('ARI\n(Transition)')
        colors_m.append('darkviolet')
    if 'nmi_hist' in df_all.columns:
        metrics.append(df_all['nmi_hist'].mean())
        stds.append(df_all['nmi_hist'].std())
        labels.append('NMI\n(Histogram)')
        colors_m.append('orange')
    if 'nmi_trans' in df_all.columns:
        metrics.append(df_all['nmi_trans'].mean())
        stds.append(df_all['nmi_trans'].std())
        labels.append('NMI\n(Transition)')
        colors_m.append('darkorange')
    
    if metrics:
        ax.bar(range(len(metrics)), metrics, yerr=stds, capsize=5,
               color=colors_m, alpha=0.7, edgecolor='black')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.axhline(0.8, color='green', linestyle=':', alpha=0.5, label='Strong (0.8)')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Clustering Metrics\n(trained vs withheld separation)', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom-left: MSE comparison
    ax = axes[1, 0]
    if 'mean_trained_mse' in df_all.columns and 'mean_withheld_mse' in df_all.columns:
        trained_mse = df_all['mean_trained_mse']
        withheld_mse = df_all['mean_withheld_mse']
        
        bars = ax.bar([0, 1], [trained_mse.mean(), withheld_mse.mean()],
                      yerr=[trained_mse.std(), withheld_mse.std()], capsize=5,
                      color=['blue', 'red'], alpha=0.7, edgecolor='black')
        bars[1].set_linewidth(2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Trained\n(A: 1:1, B: 2:1)', 'Withheld*\n(C: 3:1)'], fontsize=10)
        ax.set_ylabel('Reconstruction MSE', fontsize=11)
        ax.set_title('Out-of-Distribution Detection\n(Frozen Codebook)', fontsize=12)
        
        # Add ratio annotation
        if 'mse_ratio' in df_all.columns:
            ratio_mean = df_all['mse_ratio'].mean()
            ratio_std = df_all['mse_ratio'].std()
            ax.annotate(f'Ratio: {ratio_mean:.2f}x +/- {ratio_std:.2f}', 
                       xy=(0.5, 0.95), xycoords='axes fraction',
                       ha='center', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom-right: JSD difference distribution
    ax = axes[1, 1]
    if 'diff_jsd' in df_all.columns:
        diffs = df_all['diff_jsd'].values
        ax.hist(diffs, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(diffs.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean={diffs.mean():.3f}')
        ax.axvline(0.05, color='green', linestyle=':', linewidth=2, label='Threshold=0.05')
        ax.axvline(0, color='red', linestyle='-', linewidth=1, label='No separation')
        ax.set_xlabel('JSD Difference (to_withheld - within_trained)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Category Separation Distribution (n={len(diffs)})', fontsize=12)
        ax.legend(fontsize=9)
        
        # Count success rate (diff_jsd > 0)
        success_rate = (diffs > 0).sum() / len(diffs) * 100
        ax.annotate(f'Success rate: {success_rate:.1f}%', 
                   xy=(0.5, 0.02), xycoords='axes fraction',
                   ha='center', fontsize=10, style='italic')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'EX6: Category Recognition via Withheld Frequency Shape ({n_seeds} seeds)\n'
                 f'Trained: ShapeA (1:1 circle), ShapeB (2:1 figure-8) | '
                 f'WITHHELD: ShapeC (3:1 trefoil)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = out_dir / 'ex6_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')


def create_ex7_aggregate_figures(base_dir: Path) -> None:
    """Create EX7 aggregate figures from all seeds.
    
    Reads ex7_metrics.csv from each seed directory and creates:
    1. World vs trans_per_sec bar chart with error bars
    
    Note: Null condition (trans_per_sec=0) is a valid result, not an error.
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex7_dir = base_dir / 'ex7'
    if not ex7_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all ex7_metrics.csv files
    all_data = []
    
    for seed_dir in ex7_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'ex7_metrics.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                seed_num = seed_dir.name.replace('seed', '')
                df['seed'] = int(seed_num)
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX7 ex7_metrics.csv files found")
        return
    
    # Combine all seeds
    df_all = pd.concat(all_data, ignore_index=True)
    n_seeds = df_all['seed'].nunique()
    print(f"[EX7] Creating aggregate figures ({n_seeds} seeds)")
    
    # Group by world and compute mean/std
    worlds = ['standard', 'fast', 'slow', 'null']
    
    # Filter to only these worlds
    df_filtered = df_all[df_all['world'].isin(worlds)]
    grouped = df_filtered.groupby('world')
    
    # Figure: World vs trans_per_sec
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Transition rate comparison
    ax = axes[0]
    trans_means = []
    trans_stds = []
    for w in worlds:
        if w in grouped.groups:
            vals = grouped.get_group(w)['trans_per_sec'].dropna()
            if len(vals) > 0:
                trans_means.append(vals.mean())
                trans_stds.append(vals.std() if len(vals) > 1 else 0.0)
            else:
                trans_means.append(0.0)
                trans_stds.append(0.0)
        else:
            trans_means.append(0.0)
            trans_stds.append(0.0)
    
    x_pos = np.arange(len(worlds))
    colors = ['C0', 'C1', 'C2', 'C3']
    bars = ax.bar(x_pos, trans_means, yerr=trans_stds, capsize=4,
                  color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Standard\n(0.5Hz)', 'Fast\n(0.5→1.5Hz)', 
                       'Slow\n(0.5→0.15Hz)', 'Null\n(no input)'], fontsize=10)
    ax.set_ylabel('Transition Rate (/sec)', fontsize=11)
    ax.set_title('Internal Time Dilation', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotation (only if fast mean > 0)
    if len(trans_means) > 1 and trans_means[1] > 0:
        ax.annotate('Faster anchor\n→ more events',
                   xy=(1, trans_means[1] * 0.8), fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: Ratio to standard (dilation factor)
    ax = axes[1]
    
    # Safe ratio calculation: avoid division by zero
    standard_mean = trans_means[0] if len(trans_means) > 0 else 0.0
    
    if standard_mean > 1e-9:  # Standard has meaningful transitions
        ratios = [m / standard_mean for m in trans_means]
        ratio_stds = [s / standard_mean if not np.isnan(s) else 0.0 for s in trans_stds]
    else:
        # Standard has no transitions - ratios are undefined
        # Use absolute values instead
        print("[warn] Standard world has ~0 transitions; showing absolute rates instead of ratios")
        ratios = trans_means
        ratio_stds = trans_stds
    
    # Handle NaN in ratios
    ratios = [r if not np.isnan(r) else 0.0 for r in ratios]
    ratio_stds = [s if not np.isnan(s) else 0.0 for s in ratio_stds]
    
    bars = ax.bar(x_pos, ratios, yerr=ratio_stds, capsize=4,
                  color=colors, alpha=0.8, edgecolor='black')
    
    if standard_mean > 1e-9:
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Standard baseline')
        ax.set_ylabel('Dilation Factor (relative to Standard)', fontsize=11)
        ax.set_title('Event-Rate Dilation Factor', fontsize=12)
    else:
        ax.set_ylabel('Transition Rate (/sec)', fontsize=11)
        ax.set_title('Transition Rate (absolute)', fontsize=12)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Standard', 'Fast', 'Slow', 'Null'], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotation for Slow (only if valid)
    if standard_mean > 1e-9 and len(ratios) > 2 and ratios[2] > 0:
        ax.annotate(f'Slow: {ratios[2]:.2f}× dilation',
                   xy=(2, ratios[2] * 1.1), fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    fig.suptitle(f'EX7: Relational Internal Time — Event-Rate Dilation ({n_seeds} seeds)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex7_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')


def create_ex8_aggregate_figures(base_dir: Path) -> None:
    """Create EX8 aggregate figures from all seeds.
    
    Reads ex8_metrics.csv from each seed directory and creates:
    1. Condition comparison (IN_HULL vs RANDOM vs OUT_HULL)
    2. Withheld similarity by condition
    3. Success rate by condition
    4. Statistical test visualization
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex8_dir = base_dir / 'ex8'
    if not ex8_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics from all seeds
    all_data = []
    
    for seed_dir in ex8_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'ex8_metrics.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Extract hull_mode from directory name
                dir_name = seed_dir.name
                if 'in_hull' in dir_name:
                    df['hull_mode'] = 'in_hull'
                elif 'out_hull' in dir_name:
                    df['hull_mode'] = 'out_hull'
                else:
                    df['hull_mode'] = 'random'
                seed_num = dir_name.replace('in_hull_seed', '').replace('out_hull_seed', '').replace('random_seed', '').replace('seed', '')
                df['seed'] = int(seed_num) if seed_num.isdigit() else 0
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX8 ex8_metrics.csv files found")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    n_total = len(df_all)
    print(f"[EX8] Creating aggregate figures ({n_total} runs)")
    
    # Check if we have hull_mode conditions
    has_conditions = 'hull_mode' in df_all.columns and df_all['hull_mode'].nunique() > 1
    
    # Figure: 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if has_conditions:
        conditions = ['in_hull', 'random', 'out_hull']
        colors = ['green', 'blue', 'red']
        labels = ['IN_HULL\n(inside)', 'RANDOM\n(baseline)', 'OUT_HULL\n(orthogonal)']
        
        # Get data for each condition
        cond_data = {}
        for cond in conditions:
            mask = df_all['hull_mode'] == cond
            if mask.sum() > 0:
                cond_data[cond] = df_all[mask]['withheld_sim'].astype(float)
        
        # Top-left: Condition comparison (bar chart)
        ax = axes[0, 0]
        x_pos = []
        means = []
        stds = []
        bar_colors = []
        bar_labels = []
        
        for i, cond in enumerate(conditions):
            if cond in cond_data:
                x_pos.append(i)
                means.append(cond_data[cond].mean())
                stds.append(cond_data[cond].std())
                bar_colors.append(colors[i])
                bar_labels.append(labels[i])
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel('Withheld (C) Similarity')
        ax.set_title('Convex Hull Hypothesis Test')
        ax.set_ylim(0, 1.1)
        ax.axhline(0.90, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Top-right: Success rate comparison
        ax = axes[0, 1]
        success_rates = []
        success_labels = []
        success_colors = []
        counts_text = []
        
        for i, cond in enumerate(conditions):
            if cond in cond_data:
                data = cond_data[cond]
                n_success = (data > 0.90).sum()
                n_total_cond = len(data)
                success_rates.append(100 * n_success / n_total_cond if n_total_cond > 0 else 0)
                success_labels.append(labels[i].split('\n')[0])
                success_colors.append(colors[i])
                counts_text.append(f'{n_success}/{n_total_cond}')
        
        bars = ax.bar(range(len(success_rates)), success_rates, color=success_colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(success_rates)))
        ax.set_xticklabels(success_labels)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Extraction Success (sim > 0.90)')
        ax.set_ylim(0, 110)
        
        # Add count labels
        for i, (rate, txt) in enumerate(zip(success_rates, counts_text)):
            ax.text(i, rate + 3, txt, ha='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom-left: Distribution of withheld_sim by condition
        ax = axes[1, 0]
        bins = np.linspace(-0.5, 1, 20)
        
        for i, cond in enumerate(conditions):
            if cond in cond_data:
                ax.hist(cond_data[cond], bins=bins, alpha=0.5, color=colors[i], 
                       label=f'{labels[i].split(chr(10))[0]} (n={len(cond_data[cond])})', edgecolor='black')
        
        ax.axvline(0.90, color='gray', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Withheld (C) Similarity')
        ax.set_ylabel('Count')
        ax.set_title('Distribution by Condition')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom-right: Statistical test result
        ax = axes[1, 1]
        ax.axis('off')
        
        # Build summary text
        lines = ["CONVEX HULL HYPOTHESIS TEST", ""]
        
        for cond in conditions:
            if cond in cond_data:
                data = cond_data[cond]
                n_success = (data > 0.90).sum()
                label = {'in_hull': 'IN_HULL', 'out_hull': 'OUT_HULL', 'random': 'RANDOM'}[cond]
                lines.append(f"{label}:")
                lines.append(f"  Mean: {data.mean():.3f} ± {data.std():.3f}")
                lines.append(f"  Success: {n_success}/{len(data)} ({100*n_success/len(data):.0f}%)")
                lines.append("")
        
        # Perform test
        try:
            from scipy.stats import kruskal
            groups = [cond_data[c].values for c in conditions if c in cond_data]
            if len(groups) >= 2:
                stat_h, p_h = kruskal(*groups)
                lines.append(f"Kruskal-Wallis: H={stat_h:.2f}, p={p_h:.6f}")
                if p_h < 0.001:
                    verdict = "HIGHLY SIGNIFICANT (p < 0.001)\n→ Convex Hull Hypothesis SUPPORTED"
                elif p_h < 0.05:
                    verdict = "SIGNIFICANT (p < 0.05)\n→ Hypothesis supported"
                else:
                    verdict = "NOT SIGNIFICANT"
                lines.append("")
                lines.append(verdict)
        except:
            lines.append("(scipy not available)")
        
        summary_text = '\n'.join(lines)
        ax.text(0.5, 0.5, summary_text, fontsize=10, ha='center', va='center',
                fontfamily='monospace', transform=ax.transAxes)
    else:
        # Fallback: single condition
        ax = axes[0, 0]
        if 'withheld_sim' in df_all.columns:
            w_sims = df_all['withheld_sim'].astype(float)
            ax.hist(w_sims, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(w_sims.mean(), color='red', linestyle='--', label=f'Mean={w_sims.mean():.3f}')
            ax.set_xlabel('Withheld (C) Similarity')
            ax.set_ylabel('Count')
            ax.set_title('Extraction Quality')
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        for ax in axes.flat[1:]:
            ax.axis('off')
    
    fig.suptitle(f'EX8: Convex Hull Hypothesis Test ({n_total} runs)\n'
                 f'Meaning extraction requires C INSIDE the training hull', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex8_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')


def create_ex9_aggregate_figures(base_dir: Path) -> None:
    """Create EX9 aggregate figures from all seeds.
    
    Reads ex9_metrics.csv from each seed directory and creates:
    1. Condition comparison (IN_HULL vs RANDOM vs OUT_HULL)
    2. Withheld similarity (B+C) by condition
    3. Success rate by condition
    4. Statistical test visualization
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex9_dir = base_dir / 'ex9'
    if not ex9_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics from all seeds
    all_data = []
    
    for seed_dir in ex9_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'ex9_metrics.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Extract hull_mode from directory name
                dir_name = seed_dir.name
                if 'in_hull' in dir_name:
                    df['hull_mode'] = 'in_hull'
                elif 'out_hull' in dir_name:
                    df['hull_mode'] = 'out_hull'
                else:
                    df['hull_mode'] = 'random'
                seed_num = dir_name.replace('in_hull_seed', '').replace('out_hull_seed', '').replace('random_seed', '').replace('seed', '')
                df['seed'] = int(seed_num) if seed_num.isdigit() else 0
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX9 ex9_metrics.csv files found")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    n_total = len(df_all)
    print(f"[EX9] Creating aggregate figures ({n_total} runs)")
    
    # Check if we have hull_mode conditions
    has_conditions = 'hull_mode' in df_all.columns and df_all['hull_mode'].nunique() > 1
    
    # Figure: 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if has_conditions:
        conditions = ['in_hull', 'random', 'out_hull']
        colors = ['green', 'blue', 'red']
        labels = ['IN_HULL\n(inside)', 'RANDOM\n(baseline)', 'OUT_HULL\n(orthogonal)']
        
        # Get data for each condition
        cond_data = {}
        for cond in conditions:
            mask = df_all['hull_mode'] == cond
            if mask.sum() > 0:
                cond_data[cond] = df_all[mask]['withheld_sim'].astype(float)
        
        # Top-left: Condition comparison (bar chart)
        ax = axes[0, 0]
        x_pos = []
        means = []
        stds = []
        bar_colors = []
        bar_labels = []
        
        for i, cond in enumerate(conditions):
            if cond in cond_data:
                x_pos.append(i)
                means.append(cond_data[cond].mean())
                stds.append(cond_data[cond].std())
                bar_colors.append(colors[i])
                bar_labels.append(labels[i])
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel('Withheld (B+C) Similarity')
        ax.set_title('Syntax Inference: Convex Hull Test')
        ax.set_ylim(0, 1.1)
        ax.axhline(0.90, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Top-right: Success rate comparison
        ax = axes[0, 1]
        success_rates = []
        success_labels = []
        success_colors = []
        counts_text = []
        
        for i, cond in enumerate(conditions):
            if cond in cond_data:
                data = cond_data[cond]
                n_success = (data > 0.90).sum()
                n_total_cond = len(data)
                success_rates.append(100 * n_success / n_total_cond if n_total_cond > 0 else 0)
                success_labels.append(labels[i].split('\n')[0])
                success_colors.append(colors[i])
                counts_text.append(f'{n_success}/{n_total_cond}')
        
        bars = ax.bar(range(len(success_rates)), success_rates, color=success_colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(success_rates)))
        ax.set_xticklabels(success_labels)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Composition Inference Success (sim > 0.90)')
        ax.set_ylim(0, 110)
        
        for i, (rate, txt) in enumerate(zip(success_rates, counts_text)):
            ax.text(i, rate + 3, txt, ha='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom-left: Distribution of withheld_sim by condition
        ax = axes[1, 0]
        bins = np.linspace(-0.5, 1, 20)
        
        for i, cond in enumerate(conditions):
            if cond in cond_data:
                ax.hist(cond_data[cond], bins=bins, alpha=0.5, color=colors[i], 
                       label=f'{labels[i].split(chr(10))[0]} (n={len(cond_data[cond])})', edgecolor='black')
        
        ax.axvline(0.90, color='gray', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Withheld (B+C) Similarity')
        ax.set_ylabel('Count')
        ax.set_title('Distribution by Condition')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom-right: Statistical test result
        ax = axes[1, 1]
        ax.axis('off')
        
        lines = ["SYNTAX INFERENCE: CONVEX HULL TEST", ""]
        lines.append("Training: A, B, C + A+B, A+C")
        lines.append("Withheld: B+C (can model infer?)")
        lines.append("")
        
        for cond in conditions:
            if cond in cond_data:
                data = cond_data[cond]
                n_success = (data > 0.90).sum()
                label = {'in_hull': 'IN_HULL', 'out_hull': 'OUT_HULL', 'random': 'RANDOM'}[cond]
                lines.append(f"{label}:")
                lines.append(f"  Mean: {data.mean():.3f} ± {data.std():.3f}")
                lines.append(f"  Success: {n_success}/{len(data)} ({100*n_success/len(data):.0f}%)")
                lines.append("")
        
        try:
            from scipy.stats import kruskal
            groups = [cond_data[c].values for c in conditions if c in cond_data]
            if len(groups) >= 2:
                stat_h, p_h = kruskal(*groups)
                lines.append(f"Kruskal-Wallis: H={stat_h:.2f}, p={p_h:.6f}")
                if p_h < 0.001:
                    verdict = "HIGHLY SIGNIFICANT (p < 0.001)"
                elif p_h < 0.05:
                    verdict = "SIGNIFICANT (p < 0.05)"
                else:
                    verdict = "NOT SIGNIFICANT"
                lines.append(verdict)
        except:
            lines.append("(scipy not available)")
        
        summary_text = '\n'.join(lines)
        ax.text(0.5, 0.5, summary_text, fontsize=10, ha='center', va='center',
                fontfamily='monospace', transform=ax.transAxes)
    else:
        # Fallback: single condition
        ax = axes[0, 0]
        if 'withheld_sim' in df_all.columns:
            w_sims = df_all['withheld_sim'].astype(float)
            ax.hist(w_sims, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(w_sims.mean(), color='red', linestyle='--', label=f'Mean={w_sims.mean():.3f}')
            ax.set_xlabel('Withheld (B+C) Similarity')
            ax.set_ylabel('Count')
            ax.set_title('Composition Inference Quality')
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        for ax in axes.flat[1:]:
            ax.axis('off')
    
    fig.suptitle(f'EX9: Syntax Emergence - Convex Hull Test ({n_total} runs)\n'
                 f'Can model infer B+C from B, C, A+B, A+C?', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex9_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')


def create_ex10_aggregate_figures(base_dir: Path) -> None:
    """Create EX10 aggregate figures from all seeds.
    
    Reads ablation_results.csv from each seed directory and creates:
    1. Anchor mode vs accuracy/gate_acc comparison
    2. Expert specialization accuracy
    """
    if not HAS_PANDAS:
        return
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ex10_dir = base_dir / 'ex10'
    if not ex10_dir.exists():
        return
    
    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect ablation results from all seeds
    all_data = []
    
    for seed_dir in ex10_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / 'ablation_results.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                seed_num = seed_dir.name.replace('seed', '')
                df['seed'] = int(seed_num)
                all_data.append(df)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
    
    if not all_data:
        print("[info] No EX10 ablation_results.csv files found")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    n_seeds = df_all['seed'].nunique()
    print(f"[EX10] Creating aggregate figures ({n_seeds} seeds)")
    
    # Figure: 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Anchor mode comparison (accuracy and gate_acc)
    ax = axes[0]
    modes = ['normal', 'zero', 'shuffle']
    mode_labels = ['Normal\nAnchor', 'Zero\nAnchor', 'Shuffled\nAnchor']
    
    grouped = df_all.groupby('anchor_mode')
    
    acc_means = [grouped.get_group(m)['accuracy'].mean() if m in grouped.groups else 0 for m in modes]
    acc_stds = [grouped.get_group(m)['accuracy'].std() if m in grouped.groups else 0 for m in modes]
    gate_means = [grouped.get_group(m)['gate_acc'].mean() if m in grouped.groups else 0 for m in modes]
    gate_stds = [grouped.get_group(m)['gate_acc'].std() if m in grouped.groups else 0 for m in modes]
    
    x_pos = np.arange(len(modes))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, acc_means, width, yerr=acc_stds, capsize=4,
                   label='Task Accuracy', color='C0', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, gate_means, width, yerr=gate_stds, capsize=4,
                   label='Gate Accuracy', color='C1', alpha=0.8, edgecolor='black')
    
    # Baselines
    ax.axhline(0.556, color='C0', linestyle='--', alpha=0.5, label='Random task (5/9)')
    ax.axhline(0.333, color='C1', linestyle='--', alpha=0.5, label='Random gate (1/3)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_labels, fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Anchor Ablation Study', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Expert specialization (normal anchor only)
    ax = axes[1]
    normal_df = df_all[df_all['anchor_mode'] == 'normal']
    
    expert_metrics = ['exp0_acc', 'exp1_acc', 'exp2_acc']
    expert_labels = ['Expert 0\n(AND)', 'Expert 1\n(OR)', 'Expert 2\n(NOR)']
    
    means = [normal_df[m].mean() for m in expert_metrics]
    stds = [normal_df[m].std() for m in expert_metrics]
    
    x_pos = np.arange(len(expert_metrics))
    colors = ['C2', 'C3', 'C4']
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect specialization')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(expert_labels, fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Expert Specialization (Normal Anchor)', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', fontsize=10)
    
    fig.suptitle(f'EX10: Logic Gate with MultiGate Router ({n_seeds} seeds)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / 'ex10_aggregate.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {p}')




def create_paper_figures(base_dir: Path) -> None:
    """Create paper-ready figures from SEED-mean summary tables.

    Reads {base_dir}/{ex}/summary.csv and uses rows with seed=='MEAN'.
    """
    if not HAS_PANDAS:
        print('[warn] pandas not available; skip paper figures')
        return

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    out_dir = base_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    def _read_mean(exp: str):
        p = base_dir / exp.lower() / 'summary.csv'
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if 'seed' in df.columns:
            df = df[df['seed'].astype(str) == 'MEAN']
        return df

    # EX1
    df1 = _read_mean('EX1')
    if df1 is not None and 'wave' in df1.columns and 'recon_loss' in df1.columns:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,1,1)
        # pivot by gate_type if present
        if 'gate_type' in df1.columns:
            piv = df1.pivot_table(index='wave', columns='gate_type', values='recon_loss', aggfunc='mean')
            piv.plot(kind='bar', ax=ax)
        else:
            ax.bar(df1['wave'], df1['recon_loss'])
        ax.set_ylabel('Recon loss')
        ax.set_title('EX1: Waveform Discretization (SEED-mean)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p = out_dir / 'paper_ex1.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX2
    df2 = _read_mean('EX2')
    if df2 is not None and 'gate_type' in df2.columns and 'recon_loss' in df2.columns:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        x = df2['gate_type'].astype(str)
        ax.bar(x, df2['recon_loss'].astype(float))
        ax.set_ylabel('Recon loss')
        ax.set_title('EX2: Relational Extraction (SEED-mean)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p = out_dir / 'paper_ex2_recon.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

        if 'k_mae' in df2.columns:
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(1,1,1)
            ax.bar(x, df2['k_mae'].astype(float))
            ax.set_ylabel('k MAE')
            ax.set_title('EX2: k(t) estimation (SEED-mean)')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            p = out_dir / 'paper_ex2_kmae.png'
            fig.savefig(p, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'[saved] {p}')

    # EX3
    df3 = _read_mean('EX3')
    if df3 is not None and 'K' in df3.columns:
        for y in ['zero_cross_ratio', 'extrema_ratio']:
            if y in df3.columns:
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(1,1,1)
                ax.plot(df3['K'], df3[y], marker='o')
                ax.set_xlabel('K')
                ax.set_ylabel(y)
                ax.set_title(f'EX3: {y} vs K (SEED-mean)')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                p = out_dir / f'paper_ex3_{y}.png'
                fig.savefig(p, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f'[saved] {p}')

    # EX4
    df4 = _read_mean('EX4')
    if df4 is not None and 'noise_std' in df4.columns:
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1,1,1)
        for col in ['short_mse_closed', 'short_mse_open', 'long_mse_closed', 'long_mse_open']:
            if col in df4.columns:
                ax.plot(df4['noise_std'], df4[col], marker='o', label=col)
        ax.set_xlabel('noise_std')
        ax.set_ylabel('MSE')
        ax.set_title('EX4: Noise floor vs drift (SEED-mean)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        p = out_dir / 'paper_ex4.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX5
    df5 = _read_mean('EX5')
    if df5 is not None and 'condition' in df5.columns and 'late_phase_err' in df5.columns:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        ax.bar(df5['condition'].astype(str), df5['late_phase_err'].astype(float))
        ax.set_ylabel('Late phase error')
        ax.set_title('EX5: Binding error (SEED-mean)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p = out_dir / 'paper_ex5.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX6 - Category Recognition (Lissajous Frequency-Ratio Design)
    df6 = _read_mean('EX6')
    if df6 is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        # Get all seed data for std
        p6 = base_dir / 'ex6' / 'summary.csv'
        if p6.exists():
            df6_all = pd.read_csv(p6)
            df6_seeds = df6_all[df6_all['seed'].astype(str) != 'MEAN']
            
            # Plot 1: JSD comparison (trained vs withheld)
            ax = axes[0]
            within_jsd = df6_seeds['within_trained_jsd'].astype(float) if 'within_trained_jsd' in df6_seeds.columns else pd.Series([0])
            to_w_jsd = df6_seeds['to_withheld_jsd'].astype(float) if 'to_withheld_jsd' in df6_seeds.columns else pd.Series([0])
            
            ax.bar(0, within_jsd.mean(), yerr=within_jsd.std(), capsize=5, color='blue', alpha=0.7, label='Within trained')
            ax.bar(1, to_w_jsd.mean(), yerr=to_w_jsd.std(), capsize=5, color='red', alpha=0.7, label='To withheld', edgecolor='black', linewidth=2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Trained-Trained\n(A-B)', 'Trained-Withheld*\n(A-C, B-C)'])
            ax.set_ylabel('Jensen-Shannon Divergence')
            ax.set_title('Category Separation (JSD)\n(0=same, higher=different)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Clustering metrics (ARI, NMI)
            ax = axes[1]
            metrics_plot = ['ari_hist', 'ari_trans', 'nmi_hist', 'nmi_trans']
            labels_m = ['ARI\n(Hist)', 'ARI\n(Trans)', 'NMI\n(Hist)', 'NMI\n(Trans)']
            colors_m = ['purple', 'darkviolet', 'orange', 'darkorange']
            vals = []
            stds = []
            for m in metrics_plot:
                if m in df6_seeds.columns:
                    vals.append(df6_seeds[m].astype(float).mean())
                    stds.append(df6_seeds[m].astype(float).std())
                else:
                    vals.append(0)
                    stds.append(0)
            ax.bar(range(len(metrics_plot)), vals, yerr=stds, capsize=5, color=colors_m, alpha=0.7)
            ax.axhline(0.8, color='green', linestyle=':', label='Strong (0.8)', alpha=0.5)
            ax.axhline(0.5, color='gray', linestyle='--', label='Threshold (0.5)', alpha=0.5)
            ax.set_xticks(range(len(metrics_plot)))
            ax.set_xticklabels(labels_m, fontsize=9)
            ax.set_ylabel('Score')
            ax.set_title('Clustering Metrics')
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: MSE comparison
            ax = axes[2]
            trained_mse = df6_seeds['mean_trained_mse'].astype(float) if 'mean_trained_mse' in df6_seeds.columns else pd.Series([0])
            withheld_mse = df6_seeds['mean_withheld_mse'].astype(float) if 'mean_withheld_mse' in df6_seeds.columns else pd.Series([0])
            
            ax.bar(0, trained_mse.mean(), yerr=trained_mse.std(), capsize=5, color='blue', alpha=0.7, label='Trained')
            ax.bar(1, withheld_mse.mean(), yerr=withheld_mse.std(), capsize=5, color='red', alpha=0.7, label='Withheld*', edgecolor='black', linewidth=2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Trained\n(A:1:1, B:2:1)', 'Withheld*\n(C:3:1)'])
            ax.set_ylabel('MSE')
            ax.set_title('Reconstruction Error')
            
            # Add ratio annotation
            if 'mse_ratio' in df6_seeds.columns:
                ratio_mean = df6_seeds['mse_ratio'].astype(float).mean()
                ratio_std = df6_seeds['mse_ratio'].astype(float).std()
                ax.annotate(f'Ratio: {ratio_mean:.2f}x +/- {ratio_std:.2f}', 
                           xy=(0.5, 0.95), xycoords='axes fraction',
                           ha='center', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: JSD difference distribution
            ax = axes[3]
            if 'diff_jsd' in df6_seeds.columns:
                diffs = df6_seeds['diff_jsd'].astype(float).values
                ax.hist(diffs, bins=15, alpha=0.7, color='purple', edgecolor='black')
                ax.axvline(diffs.mean(), color='blue', linestyle='--', label=f'Mean={diffs.mean():.3f}')
                ax.axvline(0.05, color='green', linestyle=':', label='Threshold=0.05')
                ax.axvline(0, color='red', linestyle='-', label='No diff')
                ax.set_xlabel('JSD Difference (to_withheld - within)')
                ax.set_ylabel('Count')
                ax.set_title(f'Category Separation (n={len(diffs)})')
                ax.legend(fontsize=8)
                
                # Success rate annotation
                success_rate = (diffs > 0).sum() / len(diffs) * 100
                ax.annotate(f'Success: {success_rate:.1f}%', 
                           xy=(0.5, 0.02), xycoords='axes fraction',
                           ha='center', fontsize=10, style='italic')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.grid(True, alpha=0.3, axis='y')
        
        n_seeds = len(df6_seeds) if p6.exists() else 20
        plt.suptitle(f'EX6: Category Recognition via Withheld Frequency Shape ({n_seeds} seeds, +/-std)\n'
                    f'Trained: ShapeA (1:1), ShapeB (2:1) | WITHHELD: ShapeC (3:1 trefoil)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        p = out_dir / 'paper_ex6.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')
        plt.tight_layout()
        p = out_dir / 'paper_ex6.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX7
    df7 = _read_mean('EX7')
    if df7 is not None and 'world' in df7.columns and 'trans_per_sec' in df7.columns:
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1,1,1)
        ax.bar(df7['world'].astype(str), df7['trans_per_sec'].astype(float))
        ax.set_ylabel('transitions / sec')
        ax.set_title('EX7: Internal time dilation (SEED-mean)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p = out_dir / 'paper_ex7.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX8 - Meaning Extraction (Withheld Design)
    df8 = _read_mean('EX8')
    if df8 is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        p8 = base_dir / 'ex8' / 'summary.csv'
        if p8.exists():
            df8_all = pd.read_csv(p8)
            df8_seeds = df8_all[df8_all['seed'].astype(str) != 'MEAN']
            
            # Plot 1: Trained vs Withheld similarity
            ax = axes[0]
            trained = df8_seeds['mean_trained_sim'].astype(float) if 'mean_trained_sim' in df8_seeds.columns else pd.Series([0])
            withheld = df8_seeds['withheld_sim'].astype(float) if 'withheld_sim' in df8_seeds.columns else pd.Series([0])
            
            bars = ax.bar([0, 1], [trained.mean(), withheld.mean()],
                         yerr=[trained.std(), withheld.std()], capsize=5,
                         color=['blue', 'red'], alpha=0.7)
            bars[1].set_edgecolor('black')
            bars[1].set_linewidth(2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Trained\n(A, B)', 'Withheld*\n(C)'])
            ax.set_ylabel('Similarity')
            ax.set_title('Meaning Extraction\n(*C never seen as single)')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Per-token similarity
            ax = axes[1]
            tokens = ['A', 'B', 'C']
            vals = []
            stds = []
            for t in tokens:
                col = f'sim_{t}'
                if col in df8_seeds.columns:
                    vals.append(df8_seeds[col].astype(float).mean())
                    stds.append(df8_seeds[col].astype(float).std())
                else:
                    vals.append(0)
                    stds.append(0)
            colors_t = ['blue', 'orange', 'red']
            bars = ax.bar(range(3), vals, yerr=stds, capsize=5, color=colors_t, alpha=0.7)
            bars[2].set_edgecolor('black')
            bars[2].set_linewidth(2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(['A (trained)', 'B (trained)', 'C* (withheld)'])
            ax.set_ylabel('Similarity')
            ax.set_title('Per-Token Similarity')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Vector angle vs withheld_sim (KEY)
            ax = axes[2]
            if 'min_angle_to_C' in df8_seeds.columns and 'withheld_sim' in df8_seeds.columns:
                angles = df8_seeds['min_angle_to_C'].astype(float).values
                w_sims = df8_seeds['withheld_sim'].astype(float).values
                
                # Color by success/failure
                success_mask = w_sims > 0.90
                ax.scatter(angles[success_mask], w_sims[success_mask], 
                           c='green', s=60, alpha=0.7, label=f'Success', edgecolor='black')
                ax.scatter(angles[~success_mask], w_sims[~success_mask], 
                           c='red', s=60, alpha=0.7, label=f'Failure', edgecolor='black')
                
                # Correlation
                corr = np.corrcoef(angles, w_sims)[0, 1] if len(angles) > 1 else 0
                ax.set_xlabel('Min Angle to C (°)')
                ax.set_ylabel('Withheld Similarity')
                ax.set_title(f'Geometry vs Success\n(r={corr:.2f})')
                ax.axhline(0.90, color='gray', linestyle='--', alpha=0.5)
                ax.legend(fontsize=8)
                
                # Trend line
                if len(angles) > 2:
                    z = np.polyfit(angles, w_sims, 1)
                    p_fit = np.poly1d(z)
                    x_line = np.linspace(angles.min(), angles.max(), 100)
                    ax.plot(x_line, p_fit(x_line), 'b--', alpha=0.5, linewidth=2)
            else:
                ax.text(0.5, 0.5, 'No angle data', ha='center', va='center')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Success rate summary
            ax = axes[3]
            if 'withheld_sim' in df8_seeds.columns:
                w_vals = df8_seeds['withheld_sim'].astype(float).values
                n_success = (w_vals > 0.90).sum()
                n_total = len(w_vals)
                
                ax.bar([0, 1], [n_success, n_total - n_success],
                       color=['green', 'red'], alpha=0.7, edgecolor='black')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Success\n(sim>0.90)', 'Failure\n(sim≤0.90)'])
                ax.set_ylabel('Count')
                ax.set_title(f'Extraction Outcome\n({n_success}/{n_total} = {100*n_success/n_total:.0f}%)')
                
                # Add percentage annotation
                for i, v in enumerate([n_success, n_total - n_success]):
                    ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('EX8: Meaning Extraction (Withheld Design, 20 seeds)\nC NEVER seen as single during training', fontsize=13, fontweight='bold')
        plt.tight_layout()
        p = out_dir / 'paper_ex8.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX9 - Syntax Emergence
    df9 = _read_mean('EX9')
    if df9 is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        p9 = base_dir / 'ex9' / 'summary.csv'
        if p9.exists():
            df9_all = pd.read_csv(p9)
            df9_seeds = df9_all[df9_all['seed'].astype(str) != 'MEAN']
            
            # Plot 1: Trained vs Withheld similarity
            ax = axes[0]
            metrics = ['mean_single_sim', 'mean_trained_sim', 'withheld_sim']
            labels_s = ['Single\n(baseline)', 'Trained\n(A+B, A+C)', 'Withheld\n(B+C)']
            colors_s = ['blue', 'purple', 'red']
            vals = []
            stds = []
            for m in metrics:
                if m in df9_seeds.columns:
                    vals.append(df9_seeds[m].astype(float).mean())
                    stds.append(df9_seeds[m].astype(float).std())
                else:
                    vals.append(0)
                    stds.append(0)
            bars = ax.bar(range(3), vals, yerr=stds, capsize=5, color=colors_s, alpha=0.7)
            bars[-1].set_edgecolor('black')
            bars[-1].set_linewidth(2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(labels_s, fontsize=9)
            ax.set_ylabel('Similarity')
            ax.set_title('Composition Similarity (±std)')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: maxp and ent per composition
            ax = axes[1]
            comps = ['a+b', 'a+c', 'b+c']
            x = np.arange(3)
            width = 0.35
            maxp_vals = []
            ent_vals = []
            for comp in comps:
                maxp_col = f'maxp_{comp}'
                ent_col = f'ent_{comp}'
                maxp_vals.append(df9_seeds[maxp_col].astype(float).mean() if maxp_col in df9_seeds.columns else 0)
                ent_vals.append(df9_seeds[ent_col].astype(float).mean() if ent_col in df9_seeds.columns else 0)
            ax.bar(x - width/2, maxp_vals, width, label='maxp', color='purple', alpha=0.7)
            ax.bar(x + width/2, ent_vals, width, label='entropy', color='orange', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(comps)
            ax.set_ylabel('Value')
            ax.set_title('Discretization (maxp/ent)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Trans rate histogram
            ax = axes[2]
            if 'trans_rate' in df9_seeds.columns:
                rates = df9_seeds['trans_rate'].astype(float).values
                ax.hist(rates, bins=15, alpha=0.7, color='cyan', edgecolor='black')
                ax.axvline(rates.mean(), color='blue', linestyle='--', label=f'Mean={rates.mean():.3f}')
                ax.set_xlabel('Transition Rate')
                ax.set_ylabel('Count')
                ax.set_title(f'Inference Activity (n={len(rates)})')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No trans_rate data', ha='center', va='center')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Withheld distribution
            ax = axes[3]
            if 'withheld_sim' in df9_seeds.columns:
                w_vals = df9_seeds['withheld_sim'].astype(float).values
                ax.hist(w_vals, bins=20, alpha=0.7, color='red', edgecolor='black')
                ax.axvline(w_vals.mean(), color='blue', linestyle='--', 
                           label=f'Mean={w_vals.mean():.3f}')
                ax.axvline(0.90, color='green', linestyle=':', label='Threshold=0.90')
                ax.set_xlabel('Withheld (B+C) Similarity')
                ax.set_ylabel('Count')
                ax.set_title(f'Inference Quality (n={len(w_vals)})')
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('EX9: Syntax Emergence (20 seeds, ±std)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        p = out_dir / 'paper_ex9.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX10
    df10 = _read_mean('EX10')
    if df10 is not None and 'anchor_mode' in df10.columns and 'accuracy' in df10.columns:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        ax.bar(df10['anchor_mode'].astype(str), df10['accuracy'].astype(float))
        ax.set_ylabel('accuracy')
        ax.set_title('EX10: Logic under anchor ablations (SEED-mean)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p = out_dir / 'paper_ex10.png'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {p}')

    # EX1 detailed aggregate figures (waveform comparison)
    create_ex1_aggregate_figures(base_dir)
    
    # EX2 detailed aggregate figures (k vs code scatter, etc.)
    create_ex2_aggregate_figures(base_dir)
    
    # EX3 detailed aggregate figures (K vs boundary ratios, etc.)
    create_ex3_aggregate_figures(base_dir)
    
    # EX4 detailed aggregate figures (noise vs MSE, crossover)
    create_ex4_aggregate_figures(base_dir)
    
    # EX5 detailed aggregate figures (binding error comparison)
    create_ex5_aggregate_figures(base_dir)
    
    # EX6 detailed aggregate figures (phase space alignment)
    create_ex6_aggregate_figures(base_dir)
    
    # EX7 detailed aggregate figures (event-rate dilation)
    create_ex7_aggregate_figures(base_dir)
    
    # EX8 detailed aggregate figures (signal separation)
    create_ex8_aggregate_figures(base_dir)
    
    # EX9 detailed aggregate figures (composition extraction)
    create_ex9_aggregate_figures(base_dir)
    
    # EX10 detailed aggregate figures (logic gate routing)
    create_ex10_aggregate_figures(base_dir)


def main():
    ap = argparse.ArgumentParser(description="CSCT Experiment Suite Runner")
    
    ap.add_argument("--run", type=str, default="all",
                    choices=["ex1", "ex2", "ex3", "ex4", "ex5", "ex6", "ex7", "ex8", "ex9", "ex10", "all"],
                    help="Which experiments to run")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--output-root", type=str, default="./results_suite")
    
    # Common training args
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("-K", "--n-clocks", type=int, default=8, help="Codebook size K")
    ap.add_argument("--tau", "--beta", type=float, default=50.0, dest="beta", help="Transition penalty tau")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=30, help="Number of seeds (default: 30, for EX8: 3 conditions × 10)")
    
    # EX1 specific
    ap.add_argument("--ex1-waves", type=str, 
                    default="sine,chirp,am,fm,ecg,saw_bl,composite,noisy,burst")
    
    # EX2 specific (Lissajous / k(t) task)
    ap.add_argument("--ex2-k-mode", type=str, default="piecewise",
                    choices=["linear", "piecewise"],
                    help="k(t) schedule mode: linear or piecewise")
    ap.add_argument("--ex2-steps", type=int, default=1500,
                    help="Training steps for EX2 (default: 1000)")
    ap.add_argument("--ex2-log-every", type=int, default=50,
                    help="Logging interval for EX2 (default: 50)")
    ap.add_argument("--ex2-k-nsegs", type=int, default=4,
                    help="Number of segments for piecewise k(t)")
    
    # EX3 specific (K-dependency)
    ap.add_argument("--ex3-k-values", type=str, default="2,4,8,16",
                    help="Comma-separated K values to test in EX3")
    
    # EX4 specific (Anchor role / noise floor)
    ap.add_argument("--ex4-noise-levels", type=str, default="0.05,0.1,0.2,0.3",
                    help="Comma-separated noise levels for EX4")
    
    # EX5 specific (Binding problem)
    ap.add_argument("--ex5-phase-diff", type=float, default=1.5,
                    help="Target phase difference for EX5 (default: 1.5 rad)")
    ap.add_argument("--ex5-pll-alpha", type=float, default=0.1,
                    help="PLL sync strength for EX5 (default: 0.1)")
    
    # EX6 specific (Category Recognition)
    ap.add_argument("--ex6-n-seeds", type=int, default=20,
                    help="Seeds for EX6 category recognition (default: 20)")
    
    # EX7 specific (Relational internal time)
    ap.add_argument("--ex7-n-clocks", type=int, default=4,
                    help="Number of codes K for EX7 (default: 4)")
    ap.add_argument("--ex7-n-epochs", type=int, default=500,
                    help="Training epochs for EX7 (default: 800)")
    
    # EX8 specific (Meaning Emergence)
    ap.add_argument("--ex8-train-steps", type=int, default=4000,
                    help="Training steps for EX8 (default: 4000)")
    ap.add_argument("--ex8-test-steps", type=int, default=6000,
                    help="Test steps for EX8 (default: 6000)")
    
    # EX9 specific (Syntax Emergence)
    ap.add_argument("--ex9-train-steps", type=int, default=4000,
                    help="Training steps for EX9 (default: 4000)")
    
    # EX10 specific (Logic Gate with MultiGate)
    ap.add_argument("--ex10-steps", type=int, default=6000,
                    help="Training steps for EX10 (default: 6000)")
    
    # Gate comparison
    ap.add_argument("--compare-gates", action="store_true", default=False,
                    help="Deprecated: EX1 always compares SingleGate vs MultiGate")
    
    args = ap.parse_args()
    
    base_dir = Path(args.output_root)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    common_args = [
        "--device", args.device,
        "--steps", str(args.steps),
        "--n-clocks", str(args.n_clocks),
        "--beta", str(args.beta),
        "--lr", str(args.lr),
    ]
    
    all_results = []
    
    # EX1: Waveform experiments
    if args.run in ["ex1", "all"]:
        waves = [w.strip() for w in args.ex1_waves.split(",") if w.strip()]
        print(f"\n{'#'*70}")
        print(f"# EX1: Waveform Discretization ({len(waves)} waves)")
        print(f"{'#'*70}\n")
        
        ex1_results = run_ex1_suite(
            args.python, base_dir, waves, common_args, args.seeds
        )
        all_results.extend(ex1_results)
    
    # EX2: Lissajous / k(t) experiments
    if args.run in ["ex2", "all"]:
        print(f"\n{'#'*70}")
        print(f"# EX2: Lissajous Relational Extraction")
        print(f"{'#'*70}\n")
        
        ex2_results = run_ex2_suite(
            args.python, base_dir, common_args,
            args.seeds,
            k_mode=args.ex2_k_mode,
            k_nsegs=args.ex2_k_nsegs,
            steps=args.ex2_steps,
            log_every=args.ex2_log_every
        )
        all_results.extend(ex2_results)
        
        if ex2_results:
            print(f"\n{'-'*50}")
            print(f"EX2 completed: {len(ex2_results)} runs")
            print(f"{'-'*50}\n")
    
    # EX3: K-dependency analysis
    if args.run in ["ex3", "all"]:
        k_values = [int(k.strip()) for k in args.ex3_k_values.split(",")]
        print(f"\n{'#'*70}")
        print(f"# EX3: K-Dependency Analysis (K={k_values})")
        print(f"{'#'*70}\n")
        
        ex3_results = run_ex3_suite(
            args.python, base_dir, common_args,
            k_values, args.seeds
        )
        all_results.extend(ex3_results)
    
    # EX4: Anchor role / noise floor
    if args.run in ["ex4", "all"]:
        noise_levels = [float(n.strip()) for n in args.ex4_noise_levels.split(",")]
        print(f"\n{'#'*70}")
        print(f"# EX4: Anchor Role / Noise Floor (σ={noise_levels})")
        print(f"{'#'*70}\n")
        
        ex4_results = run_ex4_suite(
            args.python, base_dir, common_args,
            noise_levels, args.seeds
        )
        all_results.extend(ex4_results)
    
    # EX5: Binding problem
    if args.run in ["ex5", "all"]:
        print(f"\n{'#'*70}")
        print(f"# EX5: Binding Problem (φ={args.ex5_phase_diff}, α={args.ex5_pll_alpha})")
        print(f"{'#'*70}\n")
        
        ex5_results = run_ex5_suite(
            args.python, base_dir, common_args,
            args.ex5_phase_diff, args.ex5_pll_alpha, args.seeds
        )
        all_results.extend(ex5_results)
    
    # EX6: Category Recognition
    if args.run in ["ex6", "all"]:
        print(f"\n{'#'*70}")
        print(f"# EX6: Category Recognition via Frozen Codebook")
        print(f"{'#'*70}\n")
        
        ex6_results = run_ex6_suite(
            args.python, base_dir, common_args,
            args.seeds
        )
        all_results.extend(ex6_results)
        
        if ex6_results:
            print(f"\n{'-'*50}")
            print("EX6 Summary:")
            print(f"{'-'*50}")
            
            # JSD metrics (0=identical, higher=different)
            within_jsd = np.mean([r.get("within_trained_jsd", float('nan')) for r in ex6_results])
            to_withheld_jsd = np.mean([r.get("to_withheld_jsd", float('nan')) for r in ex6_results])
            diff_jsd = np.mean([r.get("diff_jsd", float('nan')) for r in ex6_results])
            diff_jsd_std = np.std([r.get("diff_jsd", float('nan')) for r in ex6_results])
            
            # Clustering metrics
            ari_trans = np.mean([r.get("ari_trans", float('nan')) for r in ex6_results])
            ari_trans_std = np.std([r.get("ari_trans", float('nan')) for r in ex6_results])
            nmi_trans = np.mean([r.get("nmi_trans", float('nan')) for r in ex6_results])
            
            # MSE
            mse_ratio = np.mean([r.get("mse_ratio", float('nan')) for r in ex6_results])
            mse_ratio_std = np.std([r.get("mse_ratio", float('nan')) for r in ex6_results])
            
            print(f"  [JSD] (0=same, higher=different)")
            print(f"    Within trained: {within_jsd:.3f}")
            print(f"    To withheld: {to_withheld_jsd:.3f}")
            print(f"    Difference: {diff_jsd:.3f} +/- {diff_jsd_std:.3f}")
            print(f"  [Clustering]")
            print(f"    ARI (trans): {ari_trans:.3f} +/- {ari_trans_std:.3f}")
            print(f"    NMI (trans): {nmi_trans:.3f}")
            print(f"  [MSE]")
            print(f"    Ratio (withheld/trained): {mse_ratio:.2f}x +/- {mse_ratio_std:.2f}")
            
            # Determine success
            success_count = sum(1 for r in ex6_results 
                               if r.get("ari_trans", 0) >= 0.8 or r.get("mse_ratio", 0) >= 3.0)
            success_rate = success_count / len(ex6_results) * 100
            
            print(f"\n  Success rate: {success_rate:.1f}% ({success_count}/{len(ex6_results)})")
            
            if success_rate >= 50:
                print("\n  -> Category recognition CONFIRMED across seeds!")
            print(f"{'-'*50}\n")
    
    # EX7: Relational internal time
    if args.run in ["ex7", "all"]:
        print(f"\n{'#'*70}")
        print(f"# EX7: Relational Internal Time (K={args.ex7_n_clocks})")
        print(f"{'#'*70}\n")
        
        ex7_results = run_ex7_suite(
            args.python, base_dir, common_args,
            args.ex7_n_clocks, args.ex7_n_epochs, args.seeds
        )
        all_results.extend(ex7_results)
    
    # EX8: Meaning Emergence
    if args.run in ["ex8", "all"]:
        print(f"\n{'#'*70}")
        print(f"# EX8: Meaning Emergence via Convex Hull Test (3 conditions)")
        print(f"#   IN_HULL:  C inside convex hull → expect HIGH success")
        print(f"#   RANDOM:   C random (baseline)")
        print(f"#   OUT_HULL: C orthogonal to A,B → expect LOW success")
        print(f"{'#'*70}\n")
        
        ex8_results = run_ex8_suite(
            args.python, base_dir, common_args,
            args.ex8_train_steps, args.ex8_test_steps, args.seeds,
            hull_conditions=True
        )
        all_results.extend(ex8_results)
        
        if ex8_results:
            print(f"\n{'-'*50}")
            print("EX8 Summary (By Condition):")
            print(f"{'-'*50}")
            
            # Separate by hull_mode
            for mode in ['in_hull', 'random', 'out_hull']:
                data = [r for r in ex8_results if r.get('hull_mode') == mode]
                if data:
                    withheld = np.mean([r.get("withheld_sim", float('nan')) for r in data])
                    withheld_std = np.std([r.get("withheld_sim", float('nan')) for r in data])
                    success = sum(1 for r in data if r.get("withheld_sim", 0) > 0.90)
                    label = {'in_hull': 'IN_HULL', 'out_hull': 'OUT_HULL', 'random': 'RANDOM'}[mode]
                    print(f"  {label} (n={len(data)}):")
                    print(f"    Withheld sim: {withheld:.3f} ± {withheld_std:.3f}")
                    print(f"    Success rate: {success}/{len(data)} ({100*success/len(data):.0f}%)")
            
            print(f"{'-'*50}\n")
    
    # EX9: Syntax Emergence
    if args.run in ["ex9", "all"]:
        print(f"\n{'#'*70}")
        print(f"# EX9: Syntax Emergence via Convex Hull Test (3 conditions)")
        print(f"#   IN_HULL:  C inside convex hull → expect HIGH success")
        print(f"#   RANDOM:   C random (baseline)")
        print(f"#   OUT_HULL: C orthogonal to A,B → expect LOW success")
        print(f"{'#'*70}\n")
        
        ex9_results = run_ex9_suite(
            args.python, base_dir, common_args,
            args.ex9_train_steps, args.seeds,
            hull_conditions=True
        )
        all_results.extend(ex9_results)
        
        if ex9_results:
            print(f"\n{'-'*50}")
            print("EX9 Summary (By Condition):")
            print(f"{'-'*50}")
            
            # Separate by hull_mode
            for mode in ['in_hull', 'random', 'out_hull']:
                data = [r for r in ex9_results if r.get('hull_mode') == mode]
                if data:
                    withheld = np.mean([r.get("withheld_sim", float('nan')) for r in data])
                    withheld_std = np.std([r.get("withheld_sim", float('nan')) for r in data])
                    success = sum(1 for r in data if r.get("withheld_sim", 0) > 0.90)
                    label = {'in_hull': 'IN_HULL', 'out_hull': 'OUT_HULL', 'random': 'RANDOM'}[mode]
                    print(f"  {label} (n={len(data)}):")
                    print(f"    Withheld sim (B+C): {withheld:.3f} ± {withheld_std:.3f}")
                    print(f"    Success rate: {success}/{len(data)} ({100*success/len(data):.0f}%)")
            
            print(f"{'-'*50}\n")
    
    # EX10: Logic Gate with MultiGate
    if args.run in ["ex10", "all"]:
        print(f"\n{'#'*70}")
        print(f"# EX10: Logic Gate with MultiGate (AND/OR/NOR)")
        print(f"{'#'*70}\n")
        
        ex10_results = run_ex10_suite(
            args.python, base_dir, common_args,
            args.ex10_steps, args.seeds
        )
        all_results.extend(ex10_results)
        
        if ex10_results:
            print(f"\n{'-'*50}")
            print("EX10 Summary:")
            print(f"{'-'*50}")
            for mode in ["normal", "zero", "shuffle"]:
                subset = [r for r in ex10_results if r.get("anchor_mode") == mode]
                if subset:
                    acc = np.mean([r.get("accuracy", float('nan')) for r in subset])
                    gate = np.mean([r.get("gate_acc", float('nan')) for r in subset])
                    print(f"  {mode:8s}: acc={acc:.3f}, gate_acc={gate:.3f}")
            print(f"{'-'*50}\n")
    
    # CLI summary: per-seed + SEED-mean
    if all_results:
        print_seed_mean_summaries(all_results)
    # Save per-experiment CSVs (per-seed + MEAN rows)
    if all_results:
        print("\n" + "=" * 70)
        print("Saving per-experiment summaries...")
        print(f"{'='*70}")
        save_per_experiment_summaries(all_results, base_dir)

    # Create paper figures from SEED-mean
    if all_results:
        print("\n" + "=" * 70)
        print("Creating paper figures (SEED-mean)...")
        print(f"{'='*70}")
        create_paper_figures(base_dir)

    # Aggregate convergence curves

    print(f"\n{'='*70}")
    print("Aggregating convergence data...")
    print(f"{'='*70}")
    aggregate_convergence_data(base_dir)
    
    print("\n[Suite] Done.")


if __name__ == "__main__":
    main()
