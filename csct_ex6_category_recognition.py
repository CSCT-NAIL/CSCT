#!/usr/bin/env python3
"""
CSCT EX6: Category Recognition via Withheld Frequency Shape

Design:
  3 shape categories defined by integer frequency ratios (Lissajous curves):
    - ShapeA: Circle (1:1) - x=sin(t), y=cos(t)
    - ShapeB: Figure-8 (2:1) - x=sin(t), y=cos(2t)  
    - ShapeC: Trefoil (3:1) - x=sin(t), y=cos(3t) [WITHHELD]

  All shapes have small amplitude/phase perturbations (within convex hull).

  Phase 1 (Training): ShapeA and ShapeB ONLY
  Phase 2 (Test): All 3 shapes including ShapeC (NEVER SEEN)

Key Question: Does the frozen codebook exclude unseen ShapeC from trained categories?

Hypothesis: If codebook performs categorical abstraction:
  1. ShapeC should use different code distribution than A, B
  2. ShapeC should have higher reconstruction error
  3. ShapeC should cluster separately from trained shapes

Metrics:
  1. Code histogram similarity (cosine)
  2. Reconstruction MSE
  3. KMeans clustering (k=2: trained vs withheld)
  4. Adjusted Rand Index / Normalized Mutual Info
  5. Linear separability (Logistic Regression CV)
  6. Gate pattern differentiation (Na⁺, θ, NMDA)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
from scipy.spatial.distance import cosine, jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import json
import csv
import argparse

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from csct_engine import CSCT_Engine, CSCTConfig
except ImportError:
    print("Warning: csct_engine not found")


@dataclass 
class EX6Config:
    """Configuration for EX6 experiment."""
    seed: int = 42
    train_steps: int = 2000
    test_steps: int = 6000  # 3 shapes × 2 cycles × 1000 steps
    block_size: int = 1000
    
    # Model
    n_clocks: int = 16
    hidden_dim: int = 32
    z_dim: int = 2
    
    # Signal
    seq_len: int = 128
    base_freq: float = 2.0
    
    # Training
    train_lr: float = 1e-3
    
    # Perturbation ranges (small, within convex hull)
    amp_perturb_range: Tuple[float, float] = (0.95, 1.05)  # ±5%
    phase_perturb_range: Tuple[float, float] = (-0.1, 0.1)  # small phase shift
    
    output_dir: str = "results_ex6"


# =============================================================================
# Signal Generation: 3 Frequency Ratio Shapes
# =============================================================================

def generate_shape_A(cfg: EX6Config, device: torch.device,
                     seed: int = 0, trial: int = 0) -> Tuple[torch.Tensor, Dict]:
    """
    ShapeA: Circle (frequency ratio 1:1)
    x = sin(t), y = cos(t)
    With small perturbations within convex hull.
    """
    seed32 = (int(seed) + 1000003 * int(trial)) % (2**32)
    rng = np.random.RandomState(seed32)
    
    # Small perturbations
    amp_x = rng.uniform(*cfg.amp_perturb_range)
    amp_y = rng.uniform(*cfg.amp_perturb_range)
    phase_x = rng.uniform(*cfg.phase_perturb_range)
    phase_y = rng.uniform(*cfg.phase_perturb_range)
    
    t = torch.linspace(0, 2 * np.pi * cfg.base_freq, cfg.seq_len, device=device)
    x = amp_x * torch.sin(t + phase_x)
    y = amp_y * torch.cos(t + phase_y)
    signal = torch.stack([x, y], dim=-1)
    
    params = {
        'shape': 'shapeA', 'freq_ratio': '1:1', 'category': 'trained',
        'amp_x': float(amp_x), 'amp_y': float(amp_y), 'trial': trial
    }
    return signal.unsqueeze(0), params


def generate_shape_B(cfg: EX6Config, device: torch.device,
                     seed: int = 0, trial: int = 0) -> Tuple[torch.Tensor, Dict]:
    """
    ShapeB: Figure-8 / Lemniscate (frequency ratio 2:1)
    x = sin(t), y = cos(2t)
    With small perturbations within convex hull.
    """
    seed32 = (int(seed) + 1000003 * int(trial)) % (2**32)
    rng = np.random.RandomState(seed32)
    
    amp_x = rng.uniform(*cfg.amp_perturb_range)
    amp_y = rng.uniform(*cfg.amp_perturb_range)
    phase_x = rng.uniform(*cfg.phase_perturb_range)
    phase_y = rng.uniform(*cfg.phase_perturb_range)
    
    t = torch.linspace(0, 2 * np.pi * cfg.base_freq, cfg.seq_len, device=device)
    x = amp_x * torch.sin(t + phase_x)
    y = amp_y * torch.cos(2 * t + phase_y)  # 2:1 ratio
    signal = torch.stack([x, y], dim=-1)
    
    params = {
        'shape': 'shapeB', 'freq_ratio': '2:1', 'category': 'trained',
        'amp_x': float(amp_x), 'amp_y': float(amp_y), 'trial': trial
    }
    return signal.unsqueeze(0), params


def generate_shape_C(cfg: EX6Config, device: torch.device,
                     seed: int = 0, trial: int = 0) -> Tuple[torch.Tensor, Dict]:
    """
    ShapeC: Trefoil (frequency ratio 3:1) [WITHHELD]
    x = sin(t), y = cos(3t)
    With small perturbations within convex hull.
    """
    seed32 = (int(seed) + 1000003 * int(trial)) % (2**32)
    rng = np.random.RandomState(seed32)
    
    amp_x = rng.uniform(*cfg.amp_perturb_range)
    amp_y = rng.uniform(*cfg.amp_perturb_range)
    phase_x = rng.uniform(*cfg.phase_perturb_range)
    phase_y = rng.uniform(*cfg.phase_perturb_range)
    
    t = torch.linspace(0, 2 * np.pi * cfg.base_freq, cfg.seq_len, device=device)
    x = amp_x * torch.sin(t + phase_x)
    y = amp_y * torch.cos(3 * t + phase_y)  # 3:1 ratio
    signal = torch.stack([x, y], dim=-1)
    
    params = {
        'shape': 'shapeC', 'freq_ratio': '3:1', 'category': 'withheld',
        'amp_x': float(amp_x), 'amp_y': float(amp_y), 'trial': trial
    }
    return signal.unsqueeze(0), params


def generate_training_signal(cfg: EX6Config, device: torch.device,
                             seed: int, trial: int) -> Tuple[torch.Tensor, Dict]:
    """Training: ShapeA (1:1) and ShapeB (2:1) only. ShapeC (3:1) is WITHHELD."""
    seed32 = (int(seed) + 1000003 * int(trial)) % (2**32)
    rng = np.random.RandomState(seed32)
    choice = rng.random()
    
    # 50% ShapeA, 50% ShapeB (NO ShapeC)
    if choice < 0.5:
        return generate_shape_A(cfg, device, seed, trial)
    else:
        return generate_shape_B(cfg, device, seed, trial)


# =============================================================================
# Feature Extraction
# =============================================================================

def compute_code_signature(indices: torch.Tensor, n_clocks: int) -> Dict:
    """Extract code histogram, transition matrix, and dominant codes."""
    indices_np = indices.cpu().numpy().flatten()
    counter = Counter(indices_np)
    
    # 0th-order statistics: histogram
    histogram = np.zeros(n_clocks)
    for code, count in counter.items():
        if 0 <= code < n_clocks:
            histogram[code] = count
    histogram = histogram / (histogram.sum() + 1e-9)
    
    # 1st-order statistics: transition matrix (from_code -> to_code)
    transition_matrix = np.zeros((n_clocks, n_clocks))
    for i in range(len(indices_np) - 1):
        from_code = int(indices_np[i])
        to_code = int(indices_np[i + 1])
        if 0 <= from_code < n_clocks and 0 <= to_code < n_clocks:
            transition_matrix[from_code, to_code] += 1
    
    # Row-wise normalization (transition probability from each code)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_prob = transition_matrix / (row_sums + 1e-9)
    
    # Flatten transition matrix (for similarity computation)
    transition_flat = transition_matrix.flatten()
    transition_flat = transition_flat / (transition_flat.sum() + 1e-9)
    
    dominant = sorted(counter.items(), key=lambda x: -x[1])[:3]
    return {
        'histogram': histogram,
        'transition_matrix': transition_matrix,
        'transition_prob': transition_prob,
        'transition_flat': transition_flat,
        'dominant_codes': dominant,
        'n_unique_codes': len(counter),
    }


def compute_gate_pattern(gate_info: Dict) -> Dict:
    """Extract gate activation patterns."""
    na_gate = gate_info.get('na_mean', 0.0)
    nmda_gate = gate_info.get('nmda_mean', 0.0)
    theta_phase = gate_info.get('theta_mean', 0.0)
    
    if isinstance(na_gate, torch.Tensor):
        na_gate = na_gate.item()
    if isinstance(nmda_gate, torch.Tensor):
        nmda_gate = nmda_gate.item()
    if isinstance(theta_phase, torch.Tensor):
        theta_phase = theta_phase.item()
    
    return {
        'na_gate': na_gate,
        'nmda_gate': nmda_gate,
        'theta_phase': theta_phase,
        'gate_vector': [na_gate, nmda_gate, theta_phase],
    }


def compute_confidence(probs: torch.Tensor) -> Dict:
    """Compute entropy and max probability."""
    p = probs.mean(dim=(0, 1))
    entropy = -(p * (p + 1e-9).log()).sum().item()
    max_prob = p.max().item()
    return {'entropy': entropy, 'max_prob': max_prob}


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute reconstruction MSE."""
    return ((original - reconstructed) ** 2).mean().item()


def compute_n_transitions(indices: torch.Tensor) -> int:
    """Count code transitions."""
    indices_np = indices.cpu().numpy().flatten()
    return int(np.sum(indices_np[:-1] != indices_np[1:]))


# =============================================================================
# Training
# =============================================================================

def train_model(cfg: EX6Config, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Train on ShapeA (1:1) and ShapeB (2:1) only. ShapeC (3:1) is withheld."""
    
    print("\n" + "="*60)
    print("PHASE 1: Training (ShapeC WITHHELD)")
    print("  Trained shapes: ShapeA (1:1 circle), ShapeB (2:1 figure-8)")
    print("  Withheld shape: ShapeC (3:1 trefoil) - NEVER shown")
    print("="*60)
    
    engine_cfg = CSCTConfig(
        n_clocks=cfg.n_clocks,
        hidden_dim=cfg.hidden_dim,
        z_dim=cfg.z_dim,
        input_dim=2,
    )
    model = CSCT_Engine(engine_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
    
    history = {'step': [], 'loss': [], 'maxp': []}
    
    for step in range(1, cfg.train_steps + 1):
        model.train()
        opt.zero_grad()
        
        signal, params = generate_training_signal(cfg, device, cfg.seed, step)
        result = model(x_target=signal, y_anchor=signal, beta=20.0)
        
        loss = result['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        
        if step == 1 or step % 200 == 0 or step == cfg.train_steps:
            maxp = result['probs'].max(dim=-1)[0].mean().item()
            history['step'].append(step)
            history['loss'].append(loss.item())
            history['maxp'].append(maxp)
            print(f"  Step {step:5d}: loss={loss.item():.4f} maxp={maxp:.3f} shape={params['shape']} ({params['freq_ratio']})")
    
    print(f"\n[Phase 1 Complete] Final loss={history['loss'][-1]:.4f}")
    return model, history


# =============================================================================
# Testing with Withheld Shape
# =============================================================================

def test_with_withheld(model: nn.Module, cfg: EX6Config, device: torch.device) -> Tuple[Dict, List]:
    """Test all 3 shapes including withheld ShapeC."""
    
    print("\n" + "="*60)
    print("PHASE 2: Testing (Including WITHHELD ShapeC)")
    print("  Schedule: ShapeA(1:1) -> ShapeB(2:1) -> ShapeC(3:1,WITHHELD) (repeat)")
    print("  Codebook is FROZEN after training")
    print("="*60)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Schedule: 3 shapes (A, B, C withheld)
    shape_sequence = ['shapeA', 'shapeB', 'shapeC'] * 2
    generators = {
        'shapeA': generate_shape_A,
        'shapeB': generate_shape_B,
        'shapeC': generate_shape_C,
    }
    
    observations = {
        'step': [], 'shape': [], 'category': [], 'freq_ratio': [],
        'code_histogram': [], 'transition_flat': [], 'dominant_codes': [], 'n_unique_codes': [],
        'na_gate': [], 'nmda_gate': [], 'theta_phase': [], 'gate_vector': [],
        'entropy': [], 'max_prob': [], 'mse': [], 'n_transitions': [],
    }
    
    csv_data = []
    
    with torch.no_grad():
        for step in range(cfg.test_steps):
            block_idx = step // cfg.block_size
            shape_name = shape_sequence[block_idx % len(shape_sequence)]
            
            signal, params = generators[shape_name](cfg, device, seed=cfg.seed, trial=step)
            result = model(x_target=signal, y_anchor=signal, beta=20.0)
            
            code_sig = compute_code_signature(result['indices'], cfg.n_clocks)
            gate_pat = compute_gate_pattern(result.get('gate_info', [{}])[0] if result.get('gate_info') else {})
            conf = compute_confidence(result['probs'])
            mse = compute_mse(signal, result['recon'])
            n_trans = compute_n_transitions(result['indices'])
            
            category = 'withheld' if shape_name == 'shapeC' else 'trained'
            freq_ratio = params['freq_ratio']
            
            csv_row = {
                'step': step, 'shape': shape_name, 'category': category,
                'freq_ratio': freq_ratio,
                'mse': mse, 'entropy': conf['entropy'], 'max_prob': conf['max_prob'],
                'na_gate': gate_pat['na_gate'], 'nmda_gate': gate_pat['nmda_gate'],
                'theta_phase': gate_pat['theta_phase'], 'n_transitions': n_trans,
                'n_unique_codes': code_sig['n_unique_codes'],
            }
            csv_data.append(csv_row)
            
            # Sample every 50 steps
            if step % 50 == 0:
                observations['step'].append(step)
                observations['shape'].append(shape_name)
                observations['category'].append(category)
                observations['freq_ratio'].append(freq_ratio)
                observations['code_histogram'].append(code_sig['histogram'])
                observations['transition_flat'].append(code_sig['transition_flat'])
                observations['dominant_codes'].append(code_sig['dominant_codes'])
                observations['n_unique_codes'].append(code_sig['n_unique_codes'])
                observations['na_gate'].append(gate_pat['na_gate'])
                observations['nmda_gate'].append(gate_pat['nmda_gate'])
                observations['theta_phase'].append(gate_pat['theta_phase'])
                observations['gate_vector'].append(gate_pat['gate_vector'])
                observations['entropy'].append(conf['entropy'])
                observations['max_prob'].append(conf['max_prob'])
                observations['mse'].append(mse)
                observations['n_transitions'].append(n_trans)
                
                is_withheld = "*** WITHHELD ***" if shape_name == 'shapeC' else "(trained)"
                if step % 500 == 0:
                    print(f"  Step {step:5d}: {shape_name} ({freq_ratio}) mse={mse:.4f} {is_withheld}")
    
    # Final block summary
    print("\n[Block Summary]")
    for shape in ['shapeA', 'shapeB', 'shapeC']:
        idxs = [i for i, s in enumerate(observations['shape']) if s == shape]
        if idxs:
            mses = [observations['mse'][i] for i in idxs]
            is_withheld = "*** WITHHELD ***" if shape == 'shapeC' else "(trained)"
            freq_ratio = observations['freq_ratio'][idxs[0]] if idxs else '?'
            print(f"  {shape} ({freq_ratio}): MSE={np.mean(mses):.4f}±{np.std(mses):.4f} {is_withheld}")
    
    return observations, csv_data


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(observations: Dict, cfg: EX6Config) -> Dict:
    """Analyze category separation between trained and withheld shapes."""
    
    print("\n" + "="*60)
    print("ANALYSIS: Category Recognition")
    print("="*60)
    
    shapes = observations['shape']
    histograms = observations['code_histogram']
    transitions = observations['transition_flat']
    gate_vectors = observations['gate_vector']
    mses = observations['mse']
    categories = observations['category']
    
    # ===== 0th-order: Code histogram similarity =====
    print("\n[0th-order: Code Histogram (Cosine)]")
    shape_histograms = {}
    for shape in ['shapeA', 'shapeB', 'shapeC']:
        idxs = [i for i, s in enumerate(shapes) if s == shape]
        if idxs:
            shape_histograms[shape] = np.mean([histograms[i] for i in idxs], axis=0)
    
    shape_similarities_hist = {}
    pairs = [('shapeA', 'shapeB'), ('shapeA', 'shapeC'), ('shapeB', 'shapeC')]
    for s1, s2 in pairs:
        if s1 in shape_histograms and s2 in shape_histograms:
            sim = 1 - cosine(shape_histograms[s1], shape_histograms[s2])
            shape_similarities_hist[f'{s1}-{s2}'] = sim
            marker = "<- WITHHELD" if 'shapeC' in (s1, s2) else ""
            print(f"  {s1}-{s2}: {sim:.3f} {marker}")
    
    within_trained_hist = shape_similarities_hist.get('shapeA-shapeB', 0)
    to_withheld_hist = np.mean([shape_similarities_hist.get('shapeA-shapeC', 0), 
                                shape_similarities_hist.get('shapeB-shapeC', 0)])
    diff_hist = within_trained_hist - to_withheld_hist
    print(f"  -> Within trained: {within_trained_hist:.3f}, To withheld: {to_withheld_hist:.3f}, Diff: {diff_hist:.3f}")
    
    # ===== 1st-order: Transition matrix (Cosine) =====
    print("\n[1st-order: Transition Matrix (Cosine)]")
    shape_transitions = {}
    for shape in ['shapeA', 'shapeB', 'shapeC']:
        idxs = [i for i, s in enumerate(shapes) if s == shape]
        if idxs:
            shape_transitions[shape] = np.mean([transitions[i] for i in idxs], axis=0)
    
    shape_similarities_trans = {}
    for s1, s2 in pairs:
        if s1 in shape_transitions and s2 in shape_transitions:
            sim = 1 - cosine(shape_transitions[s1], shape_transitions[s2])
            shape_similarities_trans[f'{s1}-{s2}'] = sim
            marker = "<- WITHHELD" if 'shapeC' in (s1, s2) else ""
            print(f"  {s1}-{s2}: {sim:.3f} {marker}")
    
    within_trained_trans = shape_similarities_trans.get('shapeA-shapeB', 0)
    to_withheld_trans = np.mean([shape_similarities_trans.get('shapeA-shapeC', 0), 
                                  shape_similarities_trans.get('shapeB-shapeC', 0)])
    diff_trans = within_trained_trans - to_withheld_trans
    print(f"  -> Within trained: {within_trained_trans:.3f}, To withheld: {to_withheld_trans:.3f}, Diff: {diff_trans:.3f}")
    
    # ===== Jensen-Shannon Divergence (JSD) - RECOMMENDED =====
    # JSD is a distance: 0 = identical, 1 = completely different
    # Trained shapes should have LOW JSD, withheld should have HIGH JSD
    print("\n[Jensen-Shannon Divergence (JSD)] *RECOMMENDED*")
    print("  (Distance: 0=identical, higher=different)")
    
    shape_jsd = {}
    for s1, s2 in pairs:
        if s1 in shape_transitions and s2 in shape_transitions:
            # Add small epsilon to avoid zero probabilities
            p1 = shape_transitions[s1] + 1e-10
            p2 = shape_transitions[s2] + 1e-10
            # Normalize to probability distributions
            p1 = p1 / p1.sum()
            p2 = p2 / p2.sum()
            jsd = jensenshannon(p1, p2)
            shape_jsd[f'{s1}-{s2}'] = jsd
            marker = "<- WITHHELD" if 'shapeC' in (s1, s2) else ""
            print(f"  {s1}-{s2}: {jsd:.3f} {marker}")
    
    within_trained_jsd = shape_jsd.get('shapeA-shapeB', 0)
    to_withheld_jsd = np.mean([shape_jsd.get('shapeA-shapeC', 0), 
                               shape_jsd.get('shapeB-shapeC', 0)])
    diff_jsd = to_withheld_jsd - within_trained_jsd  # Note: reversed (higher = more different)
    print(f"  -> Within trained: {within_trained_jsd:.3f}, To withheld: {to_withheld_jsd:.3f}, Diff: {diff_jsd:.3f}")
    
    # ===== Clustering comparison =====
    true_labels = [0 if cat == 'trained' else 1 for cat in categories]
    
    print("\n[Clustering (k=2): Histogram vs Transition]")
    X_hist = np.array(histograms)
    X_trans = np.array(transitions)
    
    if len(X_hist) > 10:
        # Histogram-based
        kmeans_hist = KMeans(n_clusters=2, random_state=cfg.seed, n_init=10)
        pred_hist = kmeans_hist.fit_predict(X_hist)
        ari_hist = adjusted_rand_score(true_labels, pred_hist)
        nmi_hist = normalized_mutual_info_score(true_labels, pred_hist)
        
        # Transition-based
        kmeans_trans = KMeans(n_clusters=2, random_state=cfg.seed, n_init=10)
        pred_trans = kmeans_trans.fit_predict(X_trans)
        ari_trans = adjusted_rand_score(true_labels, pred_trans)
        nmi_trans = normalized_mutual_info_score(true_labels, pred_trans)
        
        print(f"  Histogram:   ARI={ari_hist:.3f}, NMI={nmi_hist:.3f}")
        print(f"  Transition:  ARI={ari_trans:.3f}, NMI={nmi_trans:.3f} *")
    else:
        ari_hist, nmi_hist = 0, 0
        ari_trans, nmi_trans = 0, 0
    
    # ===== Linear separability =====
    print("\n[Linear Separability (5-fold CV)]")
    if len(X_hist) > 20:
        clf = LogisticRegression(random_state=cfg.seed, max_iter=500)
        
        # Histogram
        scores_hist = cross_val_score(clf, X_hist, true_labels, cv=5)
        acc_hist = scores_hist.mean()
        acc_hist_std = scores_hist.std()
        
        # Transition
        scores_trans = cross_val_score(clf, X_trans, true_labels, cv=5)
        acc_trans = scores_trans.mean()
        acc_trans_std = scores_trans.std()
        
        # Gate patterns
        X_gate = np.array(gate_vectors)
        scores_gate = cross_val_score(clf, X_gate, true_labels, cv=5)
        acc_gate = scores_gate.mean()
        acc_gate_std = scores_gate.std()
        
        print(f"  Histogram:   {acc_hist:.3f} ± {acc_hist_std:.3f}")
        print(f"  Transition:  {acc_trans:.3f} ± {acc_trans_std:.3f} *")
        print(f"  Gate (Na⁺,θ,NMDA): {acc_gate:.3f} ± {acc_gate_std:.3f}")
    else:
        acc_hist, acc_hist_std = 0.5, 0
        acc_trans, acc_trans_std = 0.5, 0
        acc_gate, acc_gate_std = 0.5, 0
    
    # MSE comparison
    trained_mses = [mses[i] for i, c in enumerate(categories) if c == 'trained']
    withheld_mses = [mses[i] for i, c in enumerate(categories) if c == 'withheld']
    
    mean_trained_mse = np.mean(trained_mses) if trained_mses else 0
    mean_withheld_mse = np.mean(withheld_mses) if withheld_mses else 0
    std_trained_mse = np.std(trained_mses) if trained_mses else 0
    std_withheld_mse = np.std(withheld_mses) if withheld_mses else 0
    mse_ratio = mean_withheld_mse / (mean_trained_mse + 1e-9)
    
    print(f"\n[Reconstruction MSE]")
    print(f"  Trained (A, B): {mean_trained_mse:.4f} ± {std_trained_mse:.4f}")
    print(f"  WITHHELD (C): {mean_withheld_mse:.4f} ± {std_withheld_mse:.4f}")
    print(f"  Ratio (withheld/trained): {mse_ratio:.2f}x")
    
    # Verdict: using multiple criteria
    # - ARI >= 0.8: near-perfect clustering separation
    # - MSE ratio >= 3.0: withheld has much higher reconstruction error
    # - JSD diff >= 0.05: clear divergence in transition patterns
    if ari_trans >= 0.8 or mse_ratio >= 3.0 or diff_jsd >= 0.1:
        verdict = "CONFIRMED: Withheld shape excluded from trained categories"
    elif ari_trans >= 0.5 or mse_ratio >= 2.0 or diff_jsd >= 0.05:
        verdict = "PARTIAL: Some category separation observed"
    else:
        verdict = "NOT CONFIRMED: No clear category separation"
    
    print(f"\n  -> {verdict}")
    
    return {
        # Histogram-based (0th-order)
        'shape_similarities': shape_similarities_hist,
        'shape_histograms': shape_histograms,
        'within_trained_sim': within_trained_hist,
        'to_withheld_sim': to_withheld_hist,
        'sim_difference': diff_hist,
        'ari': ari_hist,
        'nmi': nmi_hist,
        'linear_sep_hist': acc_hist,
        'linear_sep_hist_std': acc_hist_std,
        # Transition-based (1st-order)
        'shape_transitions': shape_transitions,
        'shape_similarities_trans': shape_similarities_trans,
        'within_trained_trans': within_trained_trans,
        'to_withheld_trans': to_withheld_trans,
        'trans_difference': diff_trans,
        'ari_trans': ari_trans,
        'nmi_trans': nmi_trans,
        'linear_sep_trans': acc_trans,
        'linear_sep_trans_std': acc_trans_std,
        # JSD (recommended)
        'shape_jsd': shape_jsd,
        'within_trained_jsd': within_trained_jsd,
        'to_withheld_jsd': to_withheld_jsd,
        'jsd_difference': diff_jsd,
        # Gate
        'linear_sep_gate': acc_gate,
        'linear_sep_gate_std': acc_gate_std,
        # MSE
        'mean_trained_mse': mean_trained_mse,
        'mean_withheld_mse': mean_withheld_mse,
        'std_trained_mse': std_trained_mse,
        'std_withheld_mse': std_withheld_mse,
        'mse_ratio': mse_ratio,
        'verdict': verdict,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_results(observations: Dict, analysis: Dict, train_history: Dict,
                 cfg: EX6Config, output_dir: Path):
    """Plot comprehensive results."""
    
    fig = plt.figure(figsize=(20, 16))
    
    steps = np.array(observations['step'])
    shapes = observations['shape']
    
    colors = {'shapeA': 'blue', 'shapeB': 'green', 'shapeC': 'red'}
    color_list = [colors[s] for s in shapes]
    
    def add_bg(ax):
        """Add background bands for shape blocks."""
        block_colors = ['blue', 'green', 'red'] * 2
        for i in range(6):
            ax.axvspan(i*1000, (i+1)*1000, alpha=0.1, color=block_colors[i])
    
    # Row 1: Training
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.plot(train_history['step'], train_history['loss'], 'b-', lw=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Phase 1: Training Loss\n(ShapeC withheld)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(4, 4, 2)
    ax2.plot(train_history['step'], train_history['maxp'], 'g-', lw=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Max Prob')
    ax2.set_title('Training Code Confidence')
    ax2.grid(True, alpha=0.3)
    
    # MSE by shape
    ax3 = fig.add_subplot(4, 4, 3)
    mse_by_shape = {}
    for shape in ['shapeA', 'shapeB', 'shapeC']:
        idxs = [i for i, s in enumerate(shapes) if s == shape]
        if idxs:
            mse_by_shape[shape] = [observations['mse'][i] for i in idxs]
    
    positions = range(3)
    for i, shape in enumerate(['shapeA', 'shapeB', 'shapeC']):
        mses = mse_by_shape.get(shape, [0])
        color = 'red' if shape == 'shapeC' else colors[shape]
        edgecolor = 'black' if shape == 'shapeC' else None
        lw = 2 if shape == 'shapeC' else 0
        ax3.bar(i, np.mean(mses), yerr=np.std(mses), color=color, alpha=0.7, 
                capsize=5, edgecolor=edgecolor, linewidth=lw)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['A (1:1)', 'B (2:1)', 'C* (3:1)'])
    ax3.set_ylabel('MSE')
    ax3.set_title('Reconstruction MSE\n(*=WITHHELD)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # JSD comparison (distance: 0=identical, higher=different)
    ax4 = fig.add_subplot(4, 4, 4)
    jsd_data = analysis.get('shape_jsd', {})
    within_trained_jsd = jsd_data.get('shapeA-shapeB', 0)
    to_withheld_jsd_vals = [jsd_data.get('shapeA-shapeC', 0), jsd_data.get('shapeB-shapeC', 0)]
    ax4.bar(0, within_trained_jsd, color='blue', alpha=0.7, label='Within trained (A-B)')
    ax4.bar(1, np.mean(to_withheld_jsd_vals), yerr=np.std(to_withheld_jsd_vals), color='red', alpha=0.7,
            label='To withheld', capsize=5, edgecolor='black', linewidth=2)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Trained-Trained\n(A-B)', 'Trained-Withheld\n(A-C, B-C)'])
    ax4.set_ylabel('JSD (distance)')
    ax4.set_title('Category Separation\n(JSD: 0=same, higher=different)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Time series
    ax5 = fig.add_subplot(4, 4, 5)
    ax5.scatter(steps, observations['mse'], c=color_list, alpha=0.7, s=20)
    add_bg(ax5)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('MSE')
    ax5.set_title('MSE Over Time')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(4, 4, 6)
    ax6.scatter(steps, observations['max_prob'], c=color_list, alpha=0.7, s=20)
    add_bg(ax6)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Max Prob')
    ax6.set_title('Code Confidence Over Time')
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(4, 4, 7)
    ax7.scatter(steps, observations['entropy'], c=color_list, alpha=0.7, s=20)
    add_bg(ax7)
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Entropy')
    ax7.set_title('Code Entropy Over Time')
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(4, 4, 8)
    ax8.scatter(steps, observations['n_unique_codes'], c=color_list, alpha=0.7, s=20)
    add_bg(ax8)
    ax8.set_xlabel('Step')
    ax8.set_ylabel('N Unique Codes')
    ax8.set_title('Code Diversity Over Time')
    ax8.grid(True, alpha=0.3)
    
    # Row 3: Gate patterns (Na⁺, NMDA focus)
    ax9 = fig.add_subplot(4, 4, 9)
    ax9.scatter(steps, observations['na_gate'], c=color_list, alpha=0.7, s=20)
    add_bg(ax9)
    ax9.set_xlabel('Step')
    ax9.set_ylabel('Na⁺ Gate')
    ax9.set_title('Na⁺ Gate Over Time')
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(4, 4, 10)
    ax10.scatter(steps, observations['nmda_gate'], c=color_list, alpha=0.7, s=20)
    add_bg(ax10)
    ax10.set_xlabel('Step')
    ax10.set_ylabel('NMDA Gate')
    ax10.set_title('NMDA Gate Over Time')
    ax10.grid(True, alpha=0.3)
    
    # Gate by shape (Na⁺)
    ax11 = fig.add_subplot(4, 4, 11)
    na_by_shape = {}
    for shape in ['shapeA', 'shapeB', 'shapeC']:
        idxs = [i for i, s in enumerate(shapes) if s == shape]
        if idxs:
            na_by_shape[shape] = [observations['na_gate'][i] for i in idxs]
    
    for i, shape in enumerate(['shapeA', 'shapeB', 'shapeC']):
        vals = na_by_shape.get(shape, [0])
        color = 'red' if shape == 'shapeC' else colors[shape]
        edgecolor = 'black' if shape == 'shapeC' else None
        lw = 2 if shape == 'shapeC' else 0
        ax11.bar(i, np.mean(vals), yerr=np.std(vals), color=color, alpha=0.7, 
                 capsize=5, edgecolor=edgecolor, linewidth=lw)
    ax11.set_xticks(range(3))
    ax11.set_xticklabels(['A (1:1)', 'B (2:1)', 'C* (3:1)'])
    ax11.set_ylabel('Na⁺ Gate')
    ax11.set_title('Na⁺ Gate by Shape\n(*=WITHHELD)')
    ax11.grid(True, alpha=0.3, axis='y')
    
    # Gate by shape (NMDA)
    ax12 = fig.add_subplot(4, 4, 12)
    nmda_by_shape = {}
    for shape in ['shapeA', 'shapeB', 'shapeC']:
        idxs = [i for i, s in enumerate(shapes) if s == shape]
        if idxs:
            nmda_by_shape[shape] = [observations['nmda_gate'][i] for i in idxs]
    
    for i, shape in enumerate(['shapeA', 'shapeB', 'shapeC']):
        vals = nmda_by_shape.get(shape, [0])
        color = 'red' if shape == 'shapeC' else colors[shape]
        edgecolor = 'black' if shape == 'shapeC' else None
        lw = 2 if shape == 'shapeC' else 0
        ax12.bar(i, np.mean(vals), yerr=np.std(vals), color=color, alpha=0.7, 
                 capsize=5, edgecolor=edgecolor, linewidth=lw)
    ax12.set_xticks(range(3))
    ax12.set_xticklabels(['A (1:1)', 'B (2:1)', 'C* (3:1)'])
    ax12.set_ylabel('NMDA Gate')
    ax12.set_title('NMDA Gate by Shape\n(*=WITHHELD)')
    ax12.grid(True, alpha=0.3, axis='y')
    
    # Row 4: Code histograms and similarity matrix
    ax13 = fig.add_subplot(4, 4, 13)
    shape_hists = analysis['shape_histograms']
    x_pos = np.arange(cfg.n_clocks)
    width = 0.25
    for j, shape in enumerate(['shapeA', 'shapeB', 'shapeC']):
        if shape in shape_hists:
            color = 'red' if shape == 'shapeC' else colors[shape]
            label = f'{shape}*' if shape == 'shapeC' else shape
            ax13.bar(x_pos + j*width, shape_hists[shape], width, alpha=0.7, 
                     color=color, label=label)
    ax13.set_xlabel('Code Index')
    ax13.set_ylabel('Frequency')
    ax13.set_title('Code Histograms\n(*=WITHHELD)')
    ax13.legend(fontsize=8)
    ax13.grid(True, alpha=0.3, axis='y')
    
    # Code similarity matrix (Transition-based *)
    ax14 = fig.add_subplot(4, 4, 14)
    shape_list = ['shapeA', 'shapeB', 'shapeC']
    sims_trans = analysis.get('shape_similarities_trans', analysis['shape_similarities'])
    sim_matrix = np.zeros((3, 3))
    for i, s1 in enumerate(shape_list):
        for j, s2 in enumerate(shape_list):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                key = f'{s1}-{s2}' if f'{s1}-{s2}' in sims_trans else f'{s2}-{s1}'
                sim_matrix[i, j] = sims_trans.get(key, 0)
    
    im = ax14.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax14.set_xticks(range(3))
    ax14.set_yticks(range(3))
    ax14.set_xticklabels(['A (1:1)', 'B (2:1)', 'C* (3:1)'])
    ax14.set_yticklabels(['A (1:1)', 'B (2:1)', 'C* (3:1)'])
    ax14.set_title('Transition Similarity *\n(*=WITHHELD)')
    for i in range(3):
        for j in range(3):
            ax14.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', 
                     fontsize=10, color='black' if sim_matrix[i,j] > 0.5 else 'white')
    fig.colorbar(im, ax=ax14, shrink=0.8)
    
    # Summary text
    ax15 = fig.add_subplot(4, 4, 15)
    ax15.axis('off')
    summary_text = f"""
    EX6 Summary: Category Recognition
    
    Design:
      Trained: ShapeA (1:1), ShapeB (2:1)
      Withheld: ShapeC (3:1)
    
    [JSD] (0=same, higher=different)
      Within trained: {analysis.get('within_trained_jsd', 0):.3f}
      To withheld: {analysis.get('to_withheld_jsd', 0):.3f}
      Diff: {analysis.get('jsd_difference', 0):.3f}
    
    [Clustering] ARI: {analysis.get('ari_trans', 0):.3f}
    
    [MSE]
      Trained: {analysis['mean_trained_mse']:.4f}
      Withheld: {analysis['mean_withheld_mse']:.4f}
      Ratio: {analysis['mse_ratio']:.2f}x
    
    {analysis['verdict']}
    """
    ax15.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
              verticalalignment='center', transform=ax15.transAxes)
    
    # Shape examples
    ax16 = fig.add_subplot(4, 4, 16)
    t = np.linspace(0, 2 * np.pi * cfg.base_freq, cfg.seq_len)
    
    # ShapeA (1:1)
    x_a, y_a = np.sin(t), np.cos(t)
    ax16.plot(x_a, y_a, 'b-', lw=2, label='A (1:1)')
    
    # ShapeB (2:1)
    x_b, y_b = np.sin(t), np.cos(2*t)
    ax16.plot(x_b + 2.5, y_b, 'g-', lw=2, label='B (2:1)')
    
    # ShapeC (3:1) - withheld
    x_c, y_c = np.sin(t), np.cos(3*t)
    ax16.plot(x_c + 5, y_c, 'r-', lw=2, label='C* (3:1)')
    
    ax16.set_xlim(-1.5, 6.5)
    ax16.set_ylim(-1.5, 1.5)
    ax16.set_aspect('equal')
    ax16.set_title('Shape Examples\n(*=WITHHELD)')
    ax16.legend(fontsize=8)
    ax16.grid(True, alpha=0.3)
    
    fig.suptitle('EX6: Category Recognition via Withheld Frequency Shape\n'
                 'Blue=ShapeA(1:1), Green=ShapeB(2:1), Red=ShapeC(3:1,WITHHELD)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    p = output_dir / 'ex6_results.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[saved] {p}")


def plot_shape_signals(cfg: EX6Config, device: torch.device, output_dir: Path):
    """Plot example signals from each shape category."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    generators = {
        'shapeA': (generate_shape_A, '1:1 Circle'),
        'shapeB': (generate_shape_B, '2:1 Figure-8'),
        'shapeC': (generate_shape_C, '3:1 Trefoil (WITHHELD)'),
    }
    
    for col, (shape_name, (gen_func, title)) in enumerate(generators.items()):
        signal, params = gen_func(cfg, device, seed=cfg.seed, trial=0)
        signal_np = signal.cpu().numpy()[0]
        
        # Top: 2D trajectory
        ax = axes[0, col]
        color = 'red' if shape_name == 'shapeC' else ('blue' if shape_name == 'shapeA' else 'green')
        ax.plot(signal_np[:, 0], signal_np[:, 1], color=color, lw=2)
        ax.scatter(signal_np[0, 0], signal_np[0, 1], c='black', s=50, marker='o', zorder=5, label='Start')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        title_suffix = ' (WITHHELD)' if shape_name == 'shapeC' else ''
        ax.set_title(f'{title}{title_suffix}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Bottom: Time series
        ax = axes[1, col]
        t = np.arange(cfg.seq_len)
        ax.plot(t, signal_np[:, 0], 'b-', label='X', alpha=0.7)
        ax.plot(t, signal_np[:, 1], 'r-', label='Y', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{shape_name} X/Y Components')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('EX6: Shape Categories (Frequency Ratios)\n'
                 'ShapeA(1:1) and ShapeB(2:1) trained, ShapeC(3:1) WITHHELD',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    p = output_dir / 'ex6_shapes.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {p}")


def plot_reconstruction(model: nn.Module, cfg: EX6Config, device: torch.device, output_dir: Path):
    """
    Plot reconstruction comparison for frozen model.
    Shows input vs reconstruction for each shape category.
    ShapeC (WITHHELD) was never seen during training.
    """
    
    model.eval()
    
    generators = {
        'shapeA': (generate_shape_A, '1:1 Circle'),
        'shapeB': (generate_shape_B, '2:1 Figure-8'),
        'shapeC': (generate_shape_C, '3:1 Trefoil'),
    }
    
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    
    with torch.no_grad():
        for row in range(n_rows):
            for col, (shape_name, (gen_func, title)) in enumerate(generators.items()):
                ax = axes[row, col]
                
                # Generate signal with different trial for each row
                trial = row * 100 + col
                signal, params = gen_func(cfg, device, seed=cfg.seed, trial=trial)
                
                # Forward pass through frozen model
                result = model(x_target=signal, y_anchor=signal, beta=20.0)
                recon = result['recon']
                indices = result['indices']
                
                # Compute MSE
                mse = ((signal - recon) ** 2).mean().item()
                
                # Get dominant codes
                indices_np = indices.cpu().numpy().flatten()
                code_counts = Counter(indices_np)
                top_codes = [c for c, _ in code_counts.most_common(3)]
                
                # Convert to numpy
                signal_np = signal.cpu().numpy()[0]
                recon_np = recon.cpu().numpy()[0]
                
                # Plot input (blue solid) and reconstruction (red dashed)
                ax.plot(signal_np[:, 0], signal_np[:, 1], 'b-', lw=2, label='Input')
                ax.plot(recon_np[:, 0], recon_np[:, 1], 'r--', lw=2, label='Recon')
                
                # Title with shape name, MSE, and codes
                is_withheld = shape_name == 'shapeC'
                title_text = f"{title}"
                if is_withheld:
                    title_text += " (WITHHELD)"
                ax.set_title(f"{title_text}\nmse={mse:.4f}, codes={top_codes}", fontsize=10,
                            color='red' if is_withheld else 'black',
                            fontweight='bold' if is_withheld else 'normal')
                
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                if row == 0 and col == 0:
                    ax.legend(fontsize=8, loc='upper right')
    
    fig.suptitle(f'EX6: Reconstruction (seed={cfg.seed})\n'
                 f'ShapeC was NEVER seen during training (codebook FROZEN)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    p = output_dir / 'ex6_reconstruction.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {p}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CSCT EX6: Category Recognition")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-steps', type=int, default=2000)
    parser.add_argument('--test-steps', type=int, default=6000)
    parser.add_argument('--output-dir', type=str, default='results_ex6')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    cfg = EX6Config(
        seed=args.seed,
        train_steps=args.train_steps,
        test_steps=args.test_steps,
        output_dir=args.output_dir,
    )
    
    device = torch.device(args.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CSCT EX6: Category Recognition via Withheld Frequency Shape")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {cfg.seed}")
    print(f"\nDesign:")
    print(f"  Trained: ShapeA (1:1 circle), ShapeB (2:1 figure-8)")
    print(f"  Withheld: ShapeC (3:1 trefoil) - NEVER seen during training")
    print("="*70)
    
    # Plot shape examples
    plot_shape_signals(cfg, device, output_dir)
    
    # Phase 1: Train
    model, train_history = train_model(cfg, device)
    
    # Phase 2: Test with withheld
    observations, csv_data = test_with_withheld(model, cfg, device)
    
    # Plot reconstruction (frozen model)
    plot_reconstruction(model, cfg, device, output_dir)
    
    # Analysis
    analysis = analyze_results(observations, cfg)
    
    # Visualization
    plot_results(observations, analysis, train_history, cfg, output_dir)
    
    # Save CSV
    csv_path = output_dir / 'ex6_test_data.csv'
    if csv_data:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"[saved] {csv_path}")
    
    # Save metrics
    metrics = {
        'seed': cfg.seed,
        # JSD (recommended)
        'within_trained_jsd': analysis.get('within_trained_jsd', 0),
        'to_withheld_jsd': analysis.get('to_withheld_jsd', 0),
        'diff_jsd': analysis.get('jsd_difference', 0),
        # 0th-order (Histogram)
        'ari_hist': analysis['ari'],
        'nmi_hist': analysis['nmi'],
        'linear_sep_hist': analysis['linear_sep_hist'],
        # 1st-order (Transition)
        'ari_trans': analysis.get('ari_trans', 0),
        'nmi_trans': analysis.get('nmi_trans', 0),
        'linear_sep_trans': analysis.get('linear_sep_trans', 0),
        # Gate
        'linear_sep_gate': analysis['linear_sep_gate'],
        # MSE
        'mean_trained_mse': analysis['mean_trained_mse'],
        'mean_withheld_mse': analysis['mean_withheld_mse'],
        'mse_ratio': analysis['mse_ratio'],
    }
    
    metrics_path = output_dir / 'ex6_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[saved] {metrics_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("EX6 FINAL SUMMARY")
    print("="*60)
    print(f"\n  Trained: ShapeA (1:1), ShapeB (2:1)")
    print(f"  Withheld: ShapeC (3:1)")
    
    print(f"\n  [JSD] (0=same, higher=different)")
    print(f"    Within trained: {analysis.get('within_trained_jsd', 0):.3f}")
    print(f"    To withheld: {analysis.get('to_withheld_jsd', 0):.3f}")
    print(f"    Diff: {analysis.get('jsd_difference', 0):.3f}")
    
    print(f"\n  [Clustering]")
    print(f"    ARI: {analysis.get('ari_trans', 0):.3f}, NMI: {analysis.get('nmi_trans', 0):.3f}")
    
    print(f"\n  [MSE]")
    print(f"    Trained: {analysis['mean_trained_mse']:.4f}")
    print(f"    Withheld: {analysis['mean_withheld_mse']:.4f}")
    print(f"    Ratio: {analysis['mse_ratio']:.2f}x")
    
    print(f"\n  -> {analysis['verdict']}")
    print("="*60)


if __name__ == "__main__":
    main()
