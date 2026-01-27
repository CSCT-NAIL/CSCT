#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSCT EX8: Meaning Emergence via Withheld Token Extraction
==========================================================

Design (mirror of EX9):
  Phase 1 (Training): A, B (single) + A+B, A+C, B+C (composite)
  Phase 2 (Test): C (withheld single) - NEVER seen as single

Key Question: Can the model EXTRACT meaning of C from composites A+C and B+C?

Symmetry with EX9:
  - EX9: All singles (A,B,C) seen, composite B+C withheld → Syntax (composition inference)
  - EX8: All composites seen, single C withheld → Meaning (extraction inference)

Hypothesis: If the model learns semantic grounding, it should:
  1. Infer C's meaning from A+C and B+C (knowing A and B)
  2. Show high similarity for withheld C
  3. Use consistent code for C

Metrics:
  - trained_sim: Similarity for A, B (seen as singles)
  - withheld_sim: Similarity for C (never seen as single)
  - composite_sim: Similarity for A+B, A+C, B+C
  - trans_rate: Code transition rate (inference activity)
"""

from __future__ import annotations

import os
import math
import argparse
import csv
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return -torch.sum(p * torch.log(p + eps), dim=-1)


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
@dataclass
class DataCfg:
    n_tokens: int = 3
    D: int = 3
    T: int = 128
    dt: float = 0.05
    omega: float = 2.0 * math.pi
    phi0: float = 0.0
    harm2: float = 0.3
    noise: float = 0.01


@dataclass
class ModelCfg:
    hidden_dim: int = 64
    n_heads: int = 4


@dataclass
class TaskCfg:
    train_steps: int = 3000
    test_steps: int = 6000
    switch_interval: int = 1000
    lr: float = 1e-3


# -----------------------------------------------------------------------------
# Convex Hull Utilities
# -----------------------------------------------------------------------------
def point_in_hull(point: np.ndarray, hull_points: np.ndarray) -> bool:
    """Check if point is inside convex hull of hull_points.
    
    Uses linear programming: point is in hull iff it can be expressed as
    convex combination of hull_points (coefficients >= 0, sum = 1).
    """
    from scipy.optimize import linprog
    
    n_points = hull_points.shape[0]
    
    # We want: hull_points.T @ coeffs = point, coeffs >= 0, sum(coeffs) = 1
    # Reformulate for linprog
    A_eq = np.vstack([hull_points.T, np.ones(n_points)])
    b_eq = np.append(point, 1.0)
    
    # Minimize 0 (feasibility check)
    c = np.zeros(n_points)
    bounds = [(0, None) for _ in range(n_points)]
    
    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        return result.success
    except:
        return False


def compute_hull_distance(C: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Compute approximate distance from C to convex hull of training points.
    
    Training points: A, B, (A+B)/norm, (A+C)/norm, (B+C)/norm
    But we don't have C during training, so hull is: A, B, (A+B)/norm
    
    Returns positive if outside (scaled), negative if inside.
    """
    # Normalize
    A = A / (np.linalg.norm(A) + 1e-9)
    B = B / (np.linalg.norm(B) + 1e-9)
    C = C / (np.linalg.norm(C) + 1e-9)
    
    AB = (A + B)
    AB = AB / (np.linalg.norm(AB) + 1e-9)
    
    # Training sees A+C and B+C, so hull includes those directions
    AC = (A + C)
    AC = AC / (np.linalg.norm(AC) + 1e-9)
    BC = (B + C)
    BC = BC / (np.linalg.norm(BC) + 1e-9)
    
    hull_points = np.array([A, B, AB, AC, BC])
    
    in_hull = point_in_hull(C, hull_points)
    
    # Approximate distance: project C onto span of hull, measure residual
    # Simple heuristic: can C be written as positive combination?
    if in_hull:
        return -1.0  # Inside
    else:
        # Distance heuristic: min distance to any hull point
        dists = [np.linalg.norm(C - hp) for hp in hull_points]
        return min(dists)


# -----------------------------------------------------------------------------
# Dual Codebook
# -----------------------------------------------------------------------------
class DualCodebook:
    """Fixed codebooks for X (output) and Y (anchor).
    
    hull_mode controls C's position relative to A, B:
      - 'random': Standard random initialization
      - 'in_hull': C is convex combination of A and B (inside hull)
      - 'out_hull': C is orthogonal to BOTH A and B (outside hull, true extrapolation)
    """
    
    def __init__(self, cfg: DataCfg, device: torch.device, seed: int = 42,
                 hull_mode: str = 'random'):
        rng = np.random.RandomState(seed)
        
        # X-codebook (output space)
        Vx = np.zeros((cfg.n_tokens, cfg.D), dtype=np.float32)
        
        # A and B are always random
        Vx[0] = rng.randn(cfg.D).astype(np.float32)
        Vx[0] /= np.linalg.norm(Vx[0]) + 1e-9
        Vx[1] = rng.randn(cfg.D).astype(np.float32)
        Vx[1] /= np.linalg.norm(Vx[1]) + 1e-9
        
        if hull_mode == 'random':
            Vx[2] = rng.randn(cfg.D).astype(np.float32)
            Vx[2] /= np.linalg.norm(Vx[2]) + 1e-9
        
        elif hull_mode == 'in_hull':
            # C = convex combination of A and B (inside hull)
            alpha = rng.uniform(0.3, 0.7)
            Vx[2] = alpha * Vx[0] + (1 - alpha) * Vx[1]
            # Add small noise to avoid exact degeneracy
            Vx[2] += 0.1 * rng.randn(cfg.D).astype(np.float32)
            Vx[2] /= np.linalg.norm(Vx[2]) + 1e-9
        
        elif hull_mode == 'out_hull':
            # C = orthogonal to BOTH A and B
            # In 3D, this is the cross product direction (unique up to sign)
            C_cross = np.cross(Vx[0], Vx[1])
            if np.linalg.norm(C_cross) < 0.01:
                # A and B are nearly parallel, add perturbation
                Vx[1] += 0.1 * rng.randn(cfg.D).astype(np.float32)
                Vx[1] /= np.linalg.norm(Vx[1]) + 1e-9
                C_cross = np.cross(Vx[0], Vx[1])
            
            # Random sign
            sign = 1 if rng.rand() > 0.5 else -1
            Vx[2] = sign * C_cross / (np.linalg.norm(C_cross) + 1e-9)
        
        else:
            raise ValueError(f"Unknown hull_mode: {hull_mode}")
        
        Wx = rng.randn(cfg.n_tokens, cfg.D).astype(np.float32)
        proj = np.sum(Wx * Vx, axis=1, keepdims=True) * Vx
        Wx = Wx - proj
        Wx /= (np.linalg.norm(Wx, axis=1, keepdims=True) + 1e-9)
        
        # Y-codebook (anchor space) - same structure
        Vy = np.zeros((cfg.n_tokens, cfg.D), dtype=np.float32)
        Vy[0] = rng.randn(cfg.D).astype(np.float32)
        Vy[0] /= np.linalg.norm(Vy[0]) + 1e-9
        Vy[1] = rng.randn(cfg.D).astype(np.float32)
        Vy[1] /= np.linalg.norm(Vy[1]) + 1e-9
        
        if hull_mode == 'random':
            Vy[2] = rng.randn(cfg.D).astype(np.float32)
            Vy[2] /= np.linalg.norm(Vy[2]) + 1e-9
        elif hull_mode == 'in_hull':
            alpha = rng.uniform(0.3, 0.7)
            Vy[2] = alpha * Vy[0] + (1 - alpha) * Vy[1]
            Vy[2] += 0.1 * rng.randn(cfg.D).astype(np.float32)
            Vy[2] /= np.linalg.norm(Vy[2]) + 1e-9
        elif hull_mode == 'out_hull':
            C_cross = np.cross(Vy[0], Vy[1])
            if np.linalg.norm(C_cross) < 0.01:
                Vy[1] += 0.1 * rng.randn(cfg.D).astype(np.float32)
                Vy[1] /= np.linalg.norm(Vy[1]) + 1e-9
                C_cross = np.cross(Vy[0], Vy[1])
            sign = 1 if rng.rand() > 0.5 else -1
            Vy[2] = sign * C_cross / (np.linalg.norm(C_cross) + 1e-9)
        
        Wy = rng.randn(cfg.n_tokens, cfg.D).astype(np.float32)
        proj = np.sum(Wy * Vy, axis=1, keepdims=True) * Vy
        Wy = Wy - proj
        Wy /= (np.linalg.norm(Wy, axis=1, keepdims=True) + 1e-9)
        
        self.Vx = torch.tensor(Vx, device=device)
        self.Wx = torch.tensor(Wx, device=device)
        self.Vy = torch.tensor(Vy, device=device)
        self.Wy = torch.tensor(Wy, device=device)
        
        self.names = ['A', 'B', 'C'][:cfg.n_tokens]
        self.hull_mode = hull_mode
        
        # Compute and store hull membership
        self.c_in_hull = point_in_hull(
            Vx[2] / (np.linalg.norm(Vx[2]) + 1e-9),
            np.array([
                Vx[0] / (np.linalg.norm(Vx[0]) + 1e-9),
                Vx[1] / (np.linalg.norm(Vx[1]) + 1e-9),
                (Vx[0] + Vx[1]) / (np.linalg.norm(Vx[0] + Vx[1]) + 1e-9),
                (Vx[0] + Vx[2]) / (np.linalg.norm(Vx[0] + Vx[2]) + 1e-9),
                (Vx[1] + Vx[2]) / (np.linalg.norm(Vx[1] + Vx[2]) + 1e-9),
            ])
        )
    
    def get_x_codebook(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Vx, self.Wx
    
    def get_y_codebook(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Vy, self.Wy


# -----------------------------------------------------------------------------
# Training Patterns (Withheld Design)
# -----------------------------------------------------------------------------
# C is NEVER seen as single during training
TRAINING_PATTERNS = [
    ('single', [0]),        # A (single) - TRAINED
    ('single', [1]),        # B (single) - TRAINED
    ('composite', [0, 1]),  # A+B - TRAINED
    ('composite', [0, 2]),  # A+C - TRAINED (C only in composite)
    ('composite', [1, 2]),  # B+C - TRAINED (C only in composite)
]

# C (index 2) is withheld as single
WITHHELD_TOKEN = 2  # C


# -----------------------------------------------------------------------------
# Data Generator
# -----------------------------------------------------------------------------
class DataGenerator:
    """Generate training and test data."""
    
    def __init__(self, cfg: DataCfg, codebook: DualCodebook, device: torch.device):
        self.cfg = cfg
        self.codebook = codebook
        self.device = device
    
    def _generate_x_stream(self, tok: int) -> torch.Tensor:
        """Generate X signal for token. Returns [1, T, D]"""
        cfg = self.cfg
        cb = self.codebook
        t = torch.arange(cfg.T, device=self.device, dtype=torch.float32) * cfg.dt
        th1 = cfg.omega * t + cfg.phi0
        th2 = 2.0 * cfg.omega * t + cfg.phi0
        v = cb.Vx[tok]
        w = cb.Wx[tok]
        s = (v[None, :] * torch.sin(th1)[:, None] +
             cfg.harm2 * w[None, :] * torch.cos(th2)[:, None])
        return s.unsqueeze(0)
    
    def _generate_y_stream(self, tok: int) -> torch.Tensor:
        """Generate Y signal for token. Returns [1, T, D]"""
        cfg = self.cfg
        cb = self.codebook
        t = torch.arange(cfg.T, device=self.device, dtype=torch.float32) * cfg.dt
        th1 = cfg.omega * t + cfg.phi0
        th2 = 2.0 * cfg.omega * t + cfg.phi0
        v = cb.Vy[tok]
        w = cb.Wy[tok]
        s = (v[None, :] * torch.sin(th1)[:, None] +
             cfg.harm2 * w[None, :] * torch.cos(th2)[:, None])
        return s.unsqueeze(0)
    
    def sample_training_batch(self, step: int) -> Dict[str, torch.Tensor]:
        """Generate training batch based on step."""
        pattern_idx = step % len(TRAINING_PATTERNS)
        p_type, tokens = TRAINING_PATTERNS[pattern_idx]
        
        if p_type == 'single':
            x_target = self._generate_x_stream(tokens[0])
            y_anchor = self._generate_y_stream(tokens[0])
        else:  # composite
            x1 = self._generate_x_stream(tokens[0])
            x2 = self._generate_x_stream(tokens[1])
            y1 = self._generate_y_stream(tokens[0])
            y2 = self._generate_y_stream(tokens[1])
            x_target = (x1 + x2) / 2.0
            y_anchor = (y1 + y2) / 2.0
        
        # Add noise
        if self.cfg.noise > 0:
            x_target = x_target + self.cfg.noise * torch.randn_like(x_target)
            y_anchor = y_anchor + self.cfg.noise * torch.randn_like(y_anchor) * 0.5
        
        return {
            'x_target': x_target,
            'y_anchor': y_anchor,
            'pattern_idx': pattern_idx,
            'tokens': tokens,
            'pattern_type': p_type,
        }
    
    def sample_test_single(self, tok: int) -> Dict[str, torch.Tensor]:
        """Generate test with single anchor."""
        y_anchor = self._generate_y_stream(tok)
        x_expected = self._generate_x_stream(tok)
        x_others = [self._generate_x_stream(i) for i in range(self.cfg.n_tokens) if i != tok]
        
        is_withheld = (tok == WITHHELD_TOKEN)
        
        return {
            'y_anchor': y_anchor,
            'x_expected': x_expected,
            'x_others': x_others,
            'tok': tok,
            'is_withheld': is_withheld,
        }
    
    def sample_test_composite(self, tok1: int, tok2: int) -> Dict[str, torch.Tensor]:
        """Generate test with composite anchor."""
        y1 = self._generate_y_stream(tok1)
        y2 = self._generate_y_stream(tok2)
        x1 = self._generate_x_stream(tok1)
        x2 = self._generate_x_stream(tok2)
        
        y_anchor = (y1 + y2) / 2.0
        x_expected = (x1 + x2) / 2.0
        
        return {
            'y_anchor': y_anchor,
            'x_expected': x_expected,
            'tok1': tok1,
            'tok2': tok2,
        }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MeaningExtractor(nn.Module):
    """Anchor-driven extractor that outputs X signal from Y anchor."""
    
    def __init__(self, model_cfg: ModelCfg, task_cfg: TaskCfg,
                 Vx: torch.Tensor, Wx: torch.Tensor):
        super().__init__()
        self.model_cfg = model_cfg
        self.task_cfg = task_cfg
        
        # FIXED X-codebook for output decoding (not learned)
        self.register_buffer("Vx", Vx)
        self.register_buffer("Wx", Wx)
        
        n_codes = Vx.shape[0]
        D = Vx.shape[1]
        H = model_cfg.hidden_dim
        
        # Learnable components
        self.anchor_proj = nn.Linear(D, H)
        self.rnn = nn.GRU(H, H, batch_first=True)
        self.code_head = nn.Linear(H, n_codes)
        
        self.n_codes = n_codes
        self.D = D
    
    def forward(self, y_anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_anchor: [B, T, D] anchor signal
        Returns:
            x_hat: [B, T, D] reconstructed X signal
            code_probs: [B, T, K] code probabilities
            indices: [B, T] selected code indices
        """
        B, T, D = y_anchor.shape
        
        # Process anchor
        h = self.anchor_proj(y_anchor)  # [B, T, H]
        h, _ = self.rnn(h)  # [B, T, H]
        
        # Code probabilities
        logits = self.code_head(h)  # [B, T, K]
        code_probs = F.softmax(logits, dim=-1)
        
        # Hard selection (for analysis)
        indices = torch.argmax(code_probs, dim=-1)  # [B, T]
        
        # Soft reconstruction
        t = torch.arange(T, device=y_anchor.device, dtype=torch.float32) * 0.05
        omega = 2.0 * math.pi
        th1 = omega * t
        th2 = 2.0 * omega * t
        
        x_hat = torch.zeros(B, T, D, device=y_anchor.device)
        for k in range(self.n_codes):
            v = self.Vx[k]
            w = self.Wx[k]
            s_k = (v[None, :] * torch.sin(th1)[:, None] +
                   0.3 * w[None, :] * torch.cos(th2)[:, None])
            x_hat = x_hat + code_probs[:, :, k:k+1] * s_k[None, :, :]
        
        # Compute metrics
        maxp = code_probs.max(dim=-1).values
        ent = _entropy_from_probs(code_probs)
        
        return {
            'x_hat': x_hat,
            'code_probs': code_probs,
            'indices': indices,
            'maxp': maxp,
            'ent': ent,
        }


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_similarity(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """Compute cosine similarity."""
    x1_flat = x1.flatten()
    x2_flat = x2.flatten()
    sim = F.cosine_similarity(x1_flat.unsqueeze(0), x2_flat.unsqueeze(0)).item()
    return sim


def compute_dominant_code(indices: torch.Tensor) -> int:
    """Get most frequent code."""
    codes = indices.flatten().cpu().numpy()
    counts = Counter(codes)
    return counts.most_common(1)[0][0]


def compute_losses(x_target: torch.Tensor, x_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute reconstruction loss."""
    recon_loss = F.mse_loss(x_hat, x_target)
    return {"loss": recon_loss, "recon_loss": recon_loss}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(model: MeaningExtractor, gen: DataGenerator,
          cfg: Dict, device: torch.device) -> Dict:
    """Train the model."""
    
    print("\n" + "="*60)
    print("PHASE 1: Training (C withheld as single)")
    print("  Patterns: A, B (single) + A+B, A+C, B+C (composite)")
    print("  C is NEVER seen as single during training")
    print("="*60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    
    history = {
        'step': [], 'loss': [], 'recon_loss': [],
        'maxp': [], 'ent': [],
    }
    
    for step in range(1, cfg['train_steps'] + 1):
        model.train()
        optimizer.zero_grad()
        
        batch = gen.sample_training_batch(step)
        out = model(batch['y_anchor'])
        
        losses = compute_losses(batch['x_target'], out['x_hat'])
        
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if step == 1 or step % 200 == 0 or step == cfg['train_steps']:
            maxp = out['maxp'].mean().item()
            ent = out['ent'].mean().item()
            
            history['step'].append(step)
            history['loss'].append(losses['loss'].item())
            history['recon_loss'].append(losses['recon_loss'].item())
            history['maxp'].append(maxp)
            history['ent'].append(ent)
            
            p_type = batch['pattern_type']
            tokens = batch['tokens']
            names = gen.codebook.names
            if p_type == 'single':
                pattern_str = f"{names[tokens[0]]}"
            else:
                pattern_str = f"{names[tokens[0]]}+{names[tokens[1]]}"
            
            print(f"  Step {step:5d}: loss={losses['loss'].item():.4f} "
                  f"maxp={maxp:.3f} ent={ent:.3f} pattern={pattern_str}")
    
    print(f"\n[Phase 1 Complete] Final loss: {history['loss'][-1]:.4f}")
    return history


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
def test_extraction(model: MeaningExtractor, gen: DataGenerator,
                    cfg: Dict, device: torch.device) -> Dict:
    """Test meaning extraction including withheld C."""
    
    print("\n" + "="*60)
    print("PHASE 2: Frozen Test (Including WITHHELD C)")
    print("  Schedule: A → B → C(WITHHELD) → A+B → A+C → B+C")
    print("  Codebook is FROZEN after training")
    print("="*60)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Test schedule: singles then composites
    # A(0), B(1) are trained; C(2) is withheld
    test_schedule = [
        ('single', 0),       # A - trained
        ('single', 1),       # B - trained
        ('single', 2),       # C - WITHHELD
        ('composite', 0, 1), # A+B - trained
        ('composite', 0, 2), # A+C - trained
        ('composite', 1, 2), # B+C - trained
    ]
    
    results = {
        'step': [], 'test_type': [], 'tokens': [],
        'similarity': [], 'maxp': [], 'ent': [], 'dominant_code': [],
    }
    
    per_single = {0: {'sim': [], 'maxp': [], 'ent': [], 'codes': []},
                  1: {'sim': [], 'maxp': [], 'ent': [], 'codes': []},
                  2: {'sim': [], 'maxp': [], 'ent': [], 'codes': []}}  # 2 = C (withheld)
    
    per_composite = {'A+B': {'sim': [], 'maxp': [], 'ent': []},
                     'A+C': {'sim': [], 'maxp': [], 'ent': []},
                     'B+C': {'sim': [], 'maxp': [], 'ent': []}}
    
    all_codes = []
    
    with torch.no_grad():
        for step in range(cfg['test_steps']):
            phase = step // cfg['switch_interval']
            test_item = test_schedule[phase % len(test_schedule)]
            
            if test_item[0] == 'single':
                tok = test_item[1]
                batch = gen.sample_test_single(tok)
                out = model(batch['y_anchor'])
                
                sim = compute_similarity(out['x_hat'], batch['x_expected'])
                maxp = out['maxp'].mean().item()
                ent = out['ent'].mean().item()
                dominant = compute_dominant_code(out['indices'])
                
                per_single[tok]['sim'].append(sim)
                per_single[tok]['maxp'].append(maxp)
                per_single[tok]['ent'].append(ent)
                per_single[tok]['codes'].append(dominant)
                
                test_type = 'single'
                tokens = [tok]
                
            else:  # composite
                tok1, tok2 = test_item[1], test_item[2]
                batch = gen.sample_test_composite(tok1, tok2)
                out = model(batch['y_anchor'])
                
                sim = compute_similarity(out['x_hat'], batch['x_expected'])
                maxp = out['maxp'].mean().item()
                ent = out['ent'].mean().item()
                dominant = compute_dominant_code(out['indices'])
                
                names = gen.codebook.names
                comp_name = f"{names[tok1]}+{names[tok2]}"
                per_composite[comp_name]['sim'].append(sim)
                per_composite[comp_name]['maxp'].append(maxp)
                per_composite[comp_name]['ent'].append(ent)
                
                test_type = 'composite'
                tokens = [tok1, tok2]
            
            results['step'].append(step)
            results['test_type'].append(test_type)
            results['tokens'].append(tokens)
            results['similarity'].append(sim)
            results['maxp'].append(maxp)
            results['ent'].append(ent)
            results['dominant_code'].append(dominant)
            
            all_codes.append(dominant)
            
            if step % 500 == 0:
                names = gen.codebook.names
                if test_type == 'single':
                    name = names[tokens[0]]
                    marker = "*** WITHHELD ***" if tokens[0] == WITHHELD_TOKEN else "(trained)"
                else:
                    name = f"{names[tokens[0]]}+{names[tokens[1]]}"
                    marker = "(composite)"
                print(f"  Step {step:4d}: {name} {marker}, sim={sim:.3f}, maxp={maxp:.3f}")
    
    # Calculate transition rate
    n_transitions = sum(1 for i in range(1, len(all_codes)) if all_codes[i] != all_codes[i-1])
    trans_rate = n_transitions / (len(all_codes) - 1) if len(all_codes) > 1 else 0
    
    # Summary
    print(f"\n--- Single Token Results ---")
    single_results = {}
    names = gen.codebook.names
    for tok in [0, 1, 2]:
        sims = per_single[tok]['sim']
        maxps = per_single[tok]['maxp']
        ents = per_single[tok]['ent']
        codes = per_single[tok]['codes']
        
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        mean_maxp = np.mean(maxps)
        mean_ent = np.mean(ents)
        code_counts = Counter(codes)
        modal_code = code_counts.most_common(1)[0][0]
        consistency = code_counts[modal_code] / len(codes)
        
        single_results[tok] = {
            'mean_sim': mean_sim, 'std_sim': std_sim,
            'mean_maxp': mean_maxp, 'mean_ent': mean_ent,
            'modal_code': modal_code, 'consistency': consistency,
        }
        
        marker = "*** WITHHELD ***" if tok == WITHHELD_TOKEN else "(trained)"
        print(f"  {names[tok]} {marker}: sim={mean_sim:.3f}±{std_sim:.3f}, "
              f"maxp={mean_maxp:.3f}, ent={mean_ent:.3f}, code={modal_code}, cons={consistency:.2%}")
    
    print(f"\n--- Composite Results ---")
    composite_results = {}
    for comp_name in ['A+B', 'A+C', 'B+C']:
        sims = per_composite[comp_name]['sim']
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        mean_maxp = np.mean(per_composite[comp_name]['maxp'])
        mean_ent = np.mean(per_composite[comp_name]['ent'])
        
        composite_results[comp_name] = {
            'mean_sim': mean_sim, 'std_sim': std_sim,
            'mean_maxp': mean_maxp, 'mean_ent': mean_ent,
        }
        print(f"  {comp_name}: sim={mean_sim:.3f}±{std_sim:.3f}, maxp={mean_maxp:.3f}, ent={mean_ent:.3f}")
    
    # Overall summary
    trained_sims = per_single[0]['sim'] + per_single[1]['sim']  # A, B
    withheld_sims = per_single[2]['sim']  # C
    
    mean_trained = np.mean(trained_sims)
    mean_withheld = np.mean(withheld_sims)
    
    print(f"\n[Phase 2 Complete]")
    print(f"  Trained singles (A, B): {mean_trained:.3f}")
    print(f"  *** WITHHELD (C): {mean_withheld:.3f} ***")
    print(f"  Transition Rate: {trans_rate:.3f} ({n_transitions} transitions)")
    
    results['single_results'] = single_results
    results['composite_results'] = composite_results
    results['summary'] = {
        'mean_trained_sim': mean_trained,
        'withheld_sim': mean_withheld,
        'trans_rate': trans_rate,
        'n_transitions': n_transitions,
    }
    
    return results


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def save_results_plot(train_history: Dict, test_results: Dict,
                      model: MeaningExtractor, gen: DataGenerator,
                      cfg: Dict, output_dir: str):
    """Save comprehensive results plot."""
    
    fig = plt.figure(figsize=(16, 12))
    names = gen.codebook.names
    
    # Row 1: Training
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.plot(train_history['step'], train_history['loss'], 'b-', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss\n(C withheld as single)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.plot(train_history['step'], train_history['maxp'], 'g-', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Max Prob')
    ax2.set_title('Code Confidence')
    ax2.grid(True, alpha=0.3)
    
    # Single similarity comparison
    ax3 = fig.add_subplot(3, 4, 3)
    single_results = test_results['single_results']
    sims = [single_results[0]['mean_sim'], single_results[1]['mean_sim'], single_results[2]['mean_sim']]
    stds = [single_results[0]['std_sim'], single_results[1]['std_sim'], single_results[2]['std_sim']]
    colors = ['blue', 'orange', 'red']
    bars = ax3.bar(range(3), sims, yerr=stds, capsize=5, color=colors, alpha=0.7)
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['A\n(trained)', 'B\n(trained)', 'C\n(WITHHELD)'])
    ax3.set_ylabel('Similarity')
    ax3.set_title('Single Token Similarity')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Composite similarity
    ax4 = fig.add_subplot(3, 4, 4)
    comp_results = test_results['composite_results']
    comp_sims = [comp_results['A+B']['mean_sim'], comp_results['A+C']['mean_sim'], comp_results['B+C']['mean_sim']]
    comp_stds = [comp_results['A+B']['std_sim'], comp_results['A+C']['std_sim'], comp_results['B+C']['std_sim']]
    ax4.bar(range(3), comp_sims, yerr=comp_stds, capsize=5, color=['purple', 'brown', 'cyan'], alpha=0.7)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(['A+B', 'A+C', 'B+C'])
    ax4.set_ylabel('Similarity')
    ax4.set_title('Composite Similarity')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Time series
    ax5 = fig.add_subplot(3, 4, 5)
    steps = test_results['step']
    sims_ts = test_results['similarity']
    ax5.plot(steps, sims_ts, 'b-', alpha=0.5, lw=0.5)
    window = 50
    if len(sims_ts) > window:
        ma = np.convolve(sims_ts, np.ones(window)/window, mode='valid')
        ax5.plot(range(window-1, len(sims_ts)), ma, 'b-', lw=2)
    for i in range(1, 6):
        ax5.axvline(i * cfg['switch_interval'], color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Similarity')
    ax5.set_title('Similarity Over Time')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.plot(steps, test_results['maxp'], 'g-', alpha=0.5, lw=0.5)
    for i in range(1, 6):
        ax6.axvline(i * cfg['switch_interval'], color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Max Prob')
    ax6.set_title('Confidence Over Time')
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.plot(steps, test_results['ent'], 'purple', alpha=0.5, lw=0.5)
    for i in range(1, 6):
        ax7.axvline(i * cfg['switch_interval'], color='red', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Entropy')
    ax7.set_title('Code Entropy Over Time')
    ax7.grid(True, alpha=0.3)
    
    # maxp/ent per token
    ax8 = fig.add_subplot(3, 4, 8)
    x = np.arange(3)
    width = 0.35
    maxps = [single_results[i]['mean_maxp'] for i in range(3)]
    ents = [single_results[i]['mean_ent'] for i in range(3)]
    ax8.bar(x - width/2, maxps, width, label='maxp', color='purple', alpha=0.7)
    ax8.bar(x + width/2, ents, width, label='entropy', color='orange', alpha=0.7)
    ax8.set_xticks(x)
    ax8.set_xticklabels(['A', 'B', 'C*'])
    ax8.set_ylabel('Value')
    ax8.set_title('Discretization (*=WITHHELD)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Code analysis
    ax9 = fig.add_subplot(3, 4, 9)
    codes = [single_results[i]['modal_code'] for i in range(3)]
    cons = [single_results[i]['consistency'] for i in range(3)]
    bars = ax9.bar(range(3), codes, color=colors, alpha=0.7)
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)
    ax9.set_xticks(range(3))
    ax9.set_xticklabels(['A', 'B', 'C*'])
    ax9.set_ylabel('Modal Code')
    ax9.set_title('Code Assignment (*=WITHHELD)')
    ax9.grid(True, alpha=0.3, axis='y')
    
    ax10 = fig.add_subplot(3, 4, 10)
    bars = ax10.bar(range(3), cons, color=colors, alpha=0.7)
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)
    ax10.set_xticks(range(3))
    ax10.set_xticklabels(['A', 'B', 'C*'])
    ax10.set_ylabel('Consistency')
    ax10.set_title('Code Consistency (*=WITHHELD)')
    ax10.set_ylim(0, 1.1)
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Summary text
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('off')
    summary = test_results['summary']
    summary_text = f"""
    EX8 SUMMARY (Withheld Design)
    
    Training: A, B (single) + A+B, A+C, B+C
    Withheld: C (never seen as single)
    
    Results:
    • Trained (A, B): {summary['mean_trained_sim']:.3f}
    • WITHHELD (C): {summary['withheld_sim']:.3f}
    • Trans rate: {summary['trans_rate']:.3f}
    
    Per-token:
    • A: sim={single_results[0]['mean_sim']:.3f}, code={single_results[0]['modal_code']}
    • B: sim={single_results[1]['mean_sim']:.3f}, code={single_results[1]['modal_code']}
    • C*: sim={single_results[2]['mean_sim']:.3f}, code={single_results[2]['modal_code']}
    
    Unique codes: {len(set(codes))}
    """
    ax11.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
              fontfamily='monospace')
    
    # Verdict
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    withheld_sim = summary['withheld_sim']
    trained_sim = summary['mean_trained_sim']
    
    if withheld_sim > 0.95:
        verdict = "STRONG"
        verdict_text = "→ MEANING EMERGES!\nC extracted from composites"
    elif withheld_sim > 0.85:
        verdict = "CONFIRMED"
        verdict_text = "→ Meaning extraction confirmed"
    elif withheld_sim > 0.7:
        verdict = "PARTIAL"
        verdict_text = "→ Partial extraction"
    else:
        verdict = "NOT_CONFIRMED"
        verdict_text = "→ Extraction failed"
    
    ax12.text(0.5, 0.5, f"VERDICT: {verdict}\n\n{verdict_text}",
              fontsize=14, ha='center', va='center', fontweight='bold')
    
    plt.suptitle(f'EX8: Meaning Extraction via Withheld Token\n'
                 f'C NEVER seen as single | withheld_sim={withheld_sim:.3f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    p = os.path.join(output_dir, 'ex8_results.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {p}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CSCT EX8: Meaning Extraction")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-steps', type=int, default=3000)
    parser.add_argument('--test-steps', type=int, default=6000)
    parser.add_argument('--switch-interval', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-tokens', type=int, default=3)
    parser.add_argument('--output-dir', type=str, default='results_ex8')
    parser.add_argument('--hull-mode', type=str, default='random',
                        choices=['random', 'in_hull', 'out_hull'],
                        help='Codebook geometry: random, in_hull (C inside), out_hull (C outside)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CSCT EX8: Meaning Extraction via Withheld Token")
    print("="*70)
    print(f"\nDesign (mirror of EX9):")
    print(f"  Training: A, B (single) + A+B, A+C, B+C (composite)")
    print(f"  Withheld: C (never seen as single)")
    print(f"\nHypothesis: Model extracts C's meaning from A+C and B+C")
    print(f"  Given: A (known), B (known), A+C, B+C")
    print(f"  Infer: C = (A+C) - A ≈ (B+C) - B")
    print(f"\nHull Mode: {args.hull_mode}")
    print("="*70)
    
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Seed: {args.seed}")
    
    safe_makedirs(args.output_dir)
    
    # Setup
    data_cfg = DataCfg(n_tokens=args.n_tokens)
    model_cfg = ModelCfg()
    task_cfg = TaskCfg(
        train_steps=args.train_steps,
        test_steps=args.test_steps,
        switch_interval=args.switch_interval,
        lr=args.lr
    )
    
    codebook = DualCodebook(data_cfg, device, seed=args.seed, hull_mode=args.hull_mode)
    gen = DataGenerator(data_cfg, codebook, device)
    
    Vx, Wx = codebook.get_x_codebook()
    model = MeaningExtractor(model_cfg, task_cfg, Vx, Wx).to(device)
    
    cfg = {
        'train_steps': args.train_steps,
        'test_steps': args.test_steps,
        'switch_interval': args.switch_interval,
        'lr': args.lr,
    }
    
    # Print codebook
    print("\nCodebooks (FIXED):")
    for i in range(args.n_tokens):
        v = codebook.Vx[i].cpu().numpy()
        print(f"  V_x[{codebook.names[i]}]: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]")
    
    # Train
    train_history = train(model, gen, cfg, device)
    
    # Test
    test_results = test_extraction(model, gen, cfg, device)
    
    # Save results
    save_results_plot(train_history, test_results, model, gen, cfg, args.output_dir)
    
    # Save metrics
    summary = test_results['summary']
    single_results = test_results['single_results']
    composite_results = test_results['composite_results']
    
    # Calculate vector angles (cosine similarity between codebook vectors)
    Vx = codebook.Vx.cpu().numpy()
    cos_AC = np.dot(Vx[0], Vx[2]) / (np.linalg.norm(Vx[0]) * np.linalg.norm(Vx[2]) + 1e-9)
    cos_BC = np.dot(Vx[1], Vx[2]) / (np.linalg.norm(Vx[1]) * np.linalg.norm(Vx[2]) + 1e-9)
    cos_AB = np.dot(Vx[0], Vx[1]) / (np.linalg.norm(Vx[0]) * np.linalg.norm(Vx[1]) + 1e-9)
    
    # Minimum angle to C (how "unique" C is)
    min_cos_to_C = max(abs(cos_AC), abs(cos_BC))  # Higher = C is closer to A or B
    
    # Angular separation (lower = harder to extract)
    angle_AC_deg = np.degrees(np.arccos(np.clip(cos_AC, -1, 1)))
    angle_BC_deg = np.degrees(np.arccos(np.clip(cos_BC, -1, 1)))
    min_angle_to_C = min(angle_AC_deg, angle_BC_deg)
    
    metrics = {
        'seed': args.seed,
        'hull_mode': args.hull_mode,
        'c_in_hull': int(codebook.c_in_hull),  # 1 if C is inside hull, 0 otherwise
        'final_loss': train_history['loss'][-1] if train_history['loss'] else float('nan'),
        'mean_trained_sim': summary['mean_trained_sim'],
        'withheld_sim': summary['withheld_sim'],
        'trans_rate': summary['trans_rate'],
        'n_transitions': summary['n_transitions'],
        # Vector geometry
        'cos_AC': cos_AC,
        'cos_BC': cos_BC,
        'cos_AB': cos_AB,
        'min_cos_to_C': min_cos_to_C,
        'angle_AC_deg': angle_AC_deg,
        'angle_BC_deg': angle_BC_deg,
        'min_angle_to_C': min_angle_to_C,
    }
    
    # Print hull status
    print(f"\n[Hull Analysis]")
    print(f"  Hull mode: {args.hull_mode}")
    print(f"  C in hull: {codebook.c_in_hull}")
    
    # Per-token metrics
    for tok in range(args.n_tokens):
        name = codebook.names[tok]
        metrics[f'sim_{name}'] = single_results[tok]['mean_sim']
        metrics[f'maxp_{name}'] = single_results[tok]['mean_maxp']
        metrics[f'ent_{name}'] = single_results[tok]['mean_ent']
        metrics[f'cons_{name}'] = single_results[tok]['consistency']
        metrics[f'code_{name}'] = single_results[tok]['modal_code']
    
    # Composite metrics
    for comp_name in ['A+B', 'A+C', 'B+C']:
        metrics[f'sim_{comp_name}'] = composite_results[comp_name]['mean_sim']
    
    # Unique codes
    codes = [single_results[i]['modal_code'] for i in range(args.n_tokens)]
    metrics['n_unique_codes'] = len(set(codes))
    
    csv_path = os.path.join(args.output_dir, "ex8_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[saved] {csv_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("EX8 FINAL SUMMARY")
    print("="*60)
    print(f"\n  Training: A, B (single) + A+B, A+C, B+C")
    print(f"  Withheld: C (never seen as single)")
    print(f"\n  Trained (A, B): {summary['mean_trained_sim']:.3f}")
    print(f"  *** WITHHELD (C): {summary['withheld_sim']:.3f} ***")
    print(f"  Transition Rate: {summary['trans_rate']:.3f}")
    print(f"  Unique Codes: {metrics['n_unique_codes']}/3")
    
    withheld_sim = summary['withheld_sim']
    if withheld_sim > 0.95:
        print(f"\n→ STRONG: Meaning EXTRACTED! (Perfect)")
        print("  Model inferred C from composites A+C and B+C")
    elif withheld_sim > 0.85:
        print(f"\n→ CONFIRMED: Meaning emerges! (Good extraction)")
    elif withheld_sim > 0.7:
        print(f"\n→ PARTIAL: Some meaning extraction observed.")
    else:
        print(f"\n→ NOT CONFIRMED: Extraction failed.")
    
    print("\n[EX8 Complete]")


if __name__ == "__main__":
    main()
