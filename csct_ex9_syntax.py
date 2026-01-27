#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSCT EX9: Syntax Emergence via Anchor-Driven Decoding
======================================================

Mirror of EX8: Fixed-codebook decoding to demonstrate SYNTAX emergence.
Learning via reconstruction only - no direct supervision.

Purpose:
  - Demonstrate that composition rules (syntax) emerge from separated signals
  - Validate A5: Syntax requires discrete symbols to operate on
  - Show that compositional understanding emerges without explicit training

Task Design (MIRROR of EX8):
  Training Phase:
    - ONLY separated signals: (X=A, Y=a), (X=B, Y=b), (X=C, Y=c)
    - Learn single correspondence Y[k] → X[k]
    
  Frozen Test Phase:
    - Anchor Y: COMPOSITE signal (y = a + b)
    - Expected: COMPOSITE output (X = A + B)
    - Question: Does the model compose without being taught composition?

Theoretical Significance:
  Symmetry with EX8 (Meaning):
  - EX8: Train on composite → Test on separated (meaning emerges)
  - EX9: Train on separated → Test on composite (syntax emerges)
  
  Together they show:
  - Meaning requires grounding via external reference (EX8)
  - Syntax requires discrete symbols to operate on (EX9)
  - Both can EMERGE from anchor-driven learning (A5 validation)

Key Metrics:
  - similarity: Cosine similarity between output and expected composite
  - composition_accuracy: How well the model combines learned components
  - code_distribution: Does the model activate multiple codes for composite?

Author: NAOKI (CSCT Research)
"""

from __future__ import annotations

import os
import math
import time
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
    """Entropy over last dim."""
    p = p.clamp(min=eps, max=1.0)
    return -(p * p.log()).sum(dim=-1)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class DataCfg:
    """Data generation parameters."""
    n_tokens: int = 3  # A, B, C
    input_dim: int = 3  # 3D vectors
    T: int = 24
    dt: float = 0.15
    omega: float = 1.0
    phi0: float = math.pi / 2
    harm2: float = 0.35
    noise: float = 0.01


@dataclass
class ModelCfg:
    """Model parameters."""
    hidden_dim: int = 128
    z_dim: int = 64
    input_dim: int = 3


@dataclass
class TaskCfg:
    """Task-specific parameters."""
    n_tokens: int = 3
    T: int = 24
    dt: float = 0.15
    omega: float = 1.0
    phi0: float = math.pi / 2
    harm2: float = 0.35


# -----------------------------------------------------------------------------
# Dual Codebook: X (output) and Y (anchor) - FIXED
# -----------------------------------------------------------------------------
class DualCodebook:
    """
    Two FIXED codebooks for X (output) and Y (anchor).
    
    The correspondence X[k] ↔ Y[k] is ARBITRARY but fixed.
    
    hull_mode controls C's position relative to A, B:
      - 'random': Standard random initialization
      - 'in_hull': C is convex combination of A and B (inside hull)
      - 'out_hull': C is orthogonal to BOTH A and B (outside hull)
    """
    
    def __init__(self, cfg: DataCfg, seed: int, device: torch.device,
                 hull_mode: str = 'random'):
        self.cfg = cfg
        self.device = device
        self.hull_mode = hull_mode
        
        rng = np.random.RandomState(seed)
        
        # X-codebook (output space)
        Vx = np.zeros((cfg.n_tokens, cfg.input_dim), dtype=np.float32)
        
        # A and B are always random
        Vx[0] = rng.randn(cfg.input_dim).astype(np.float32)
        Vx[0] /= np.linalg.norm(Vx[0]) + 1e-9
        Vx[1] = rng.randn(cfg.input_dim).astype(np.float32)
        Vx[1] /= np.linalg.norm(Vx[1]) + 1e-9
        
        if hull_mode == 'random':
            Vx[2] = rng.randn(cfg.input_dim).astype(np.float32)
            Vx[2] /= np.linalg.norm(Vx[2]) + 1e-9
        
        elif hull_mode == 'in_hull':
            # C = convex combination of A and B (inside hull)
            alpha = rng.uniform(0.3, 0.7)
            Vx[2] = alpha * Vx[0] + (1 - alpha) * Vx[1]
            Vx[2] += 0.1 * rng.randn(cfg.input_dim).astype(np.float32)
            Vx[2] /= np.linalg.norm(Vx[2]) + 1e-9
        
        elif hull_mode == 'out_hull':
            # C = orthogonal to BOTH A and B (cross product)
            C_cross = np.cross(Vx[0], Vx[1])
            if np.linalg.norm(C_cross) < 0.01:
                Vx[1] += 0.1 * rng.randn(cfg.input_dim).astype(np.float32)
                Vx[1] /= np.linalg.norm(Vx[1]) + 1e-9
                C_cross = np.cross(Vx[0], Vx[1])
            sign = 1 if rng.rand() > 0.5 else -1
            Vx[2] = sign * C_cross / (np.linalg.norm(C_cross) + 1e-9)
        
        else:
            raise ValueError(f"Unknown hull_mode: {hull_mode}")
        
        Wx = rng.randn(cfg.n_tokens, cfg.input_dim).astype(np.float32)
        proj = (Wx * Vx).sum(axis=1, keepdims=True) * Vx
        Wx = Wx - proj
        Wx /= (np.linalg.norm(Wx, axis=1, keepdims=True) + 1e-9)
        
        # Y-codebook (anchor space) - same structure
        Vy = np.zeros((cfg.n_tokens, cfg.input_dim), dtype=np.float32)
        Vy[0] = rng.randn(cfg.input_dim).astype(np.float32)
        Vy[0] /= np.linalg.norm(Vy[0]) + 1e-9
        Vy[1] = rng.randn(cfg.input_dim).astype(np.float32)
        Vy[1] /= np.linalg.norm(Vy[1]) + 1e-9
        
        if hull_mode == 'random':
            Vy[2] = rng.randn(cfg.input_dim).astype(np.float32)
            Vy[2] /= np.linalg.norm(Vy[2]) + 1e-9
        elif hull_mode == 'in_hull':
            alpha = rng.uniform(0.3, 0.7)
            Vy[2] = alpha * Vy[0] + (1 - alpha) * Vy[1]
            Vy[2] += 0.1 * rng.randn(cfg.input_dim).astype(np.float32)
            Vy[2] /= np.linalg.norm(Vy[2]) + 1e-9
        elif hull_mode == 'out_hull':
            C_cross = np.cross(Vy[0], Vy[1])
            if np.linalg.norm(C_cross) < 0.01:
                Vy[1] += 0.1 * rng.randn(cfg.input_dim).astype(np.float32)
                Vy[1] /= np.linalg.norm(Vy[1]) + 1e-9
                C_cross = np.cross(Vy[0], Vy[1])
            sign = 1 if rng.rand() > 0.5 else -1
            Vy[2] = sign * C_cross / (np.linalg.norm(C_cross) + 1e-9)
        
        Wy = rng.randn(cfg.n_tokens, cfg.input_dim).astype(np.float32)
        proj = (Wy * Vy).sum(axis=1, keepdims=True) * Vy
        Wy = Wy - proj
        Wy /= (np.linalg.norm(Wy, axis=1, keepdims=True) + 1e-9)
        
        self.Vx = torch.tensor(Vx, device=device)
        self.Wx = torch.tensor(Wx, device=device)
        self.Vy = torch.tensor(Vy, device=device)
        self.Wy = torch.tensor(Wy, device=device)
        
        self.names = ['A', 'B', 'C'][:cfg.n_tokens]
        
        # Store vector angles for analysis
        self.angle_AC = np.degrees(np.arccos(np.clip(np.dot(Vx[0], Vx[2]), -1, 1)))
        self.angle_BC = np.degrees(np.arccos(np.clip(np.dot(Vx[1], Vx[2]), -1, 1)))
        self.angle_AB = np.degrees(np.arccos(np.clip(np.dot(Vx[0], Vx[1]), -1, 1)))


# -----------------------------------------------------------------------------
# Training Patterns: Separated + PARTIAL composites (B+C withheld)
# -----------------------------------------------------------------------------
TRAINING_PATTERNS = [
    ('single', [0]),      # X = A, Y = a
    ('single', [1]),      # X = B, Y = b
    ('single', [2]),      # X = C, Y = c
    ('composite', [0, 1]), # X = A+B, Y = a+b
    ('composite', [0, 2]), # X = A+C, Y = a+c
    # B+C is WITHHELD - will test if model can infer it
]


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
        """Generate training batch: separated + partial composites."""
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
            'type': p_type,
        }
    
    def sample_test_composite(self, tok1: int, tok2: int) -> Dict[str, torch.Tensor]:
        """Generate test with COMPOSITE anchor (never seen during training)."""
        y1 = self._generate_y_stream(tok1)
        y2 = self._generate_y_stream(tok2)
        y_anchor = (y1 + y2) / 2.0  # Composite anchor
        
        x1 = self._generate_x_stream(tok1)
        x2 = self._generate_x_stream(tok2)
        x_expected = (x1 + x2) / 2.0  # Expected composite output
        
        return {
            'y_anchor': y_anchor,
            'x_expected': x_expected,
            'x_components': [x1, x2],
            'tokens': [tok1, tok2],
        }
    
    def sample_test_single(self, tok: int) -> Dict[str, torch.Tensor]:
        """Generate test with single anchor (baseline)."""
        y_anchor = self._generate_y_stream(tok)
        x_expected = self._generate_x_stream(tok)
        
        return {
            'y_anchor': y_anchor,
            'x_expected': x_expected,
            'tok': tok,
        }


# -----------------------------------------------------------------------------
# Model: Anchor-Driven Syntax Extractor with Fixed Codebook
# -----------------------------------------------------------------------------
class SyntaxExtractor(nn.Module):
    """
    Anchor-driven extractor that outputs X signal from Y anchor.
    Same architecture as EX8's MeaningExtractor.
    """
    
    def __init__(self, model_cfg: ModelCfg, task_cfg: TaskCfg,
                 Vx: torch.Tensor, Wx: torch.Tensor):
        super().__init__()
        self.model_cfg = model_cfg
        self.task_cfg = task_cfg
        
        # FIXED X-codebook for output decoding (not learned)
        self.register_buffer("Vx", Vx)
        self.register_buffer("Wx", Wx)
        
        n_tokens = Vx.shape[0]
        self.n_tokens = n_tokens
        
        # Anchor encoder
        self.anchor_in = nn.Linear(model_cfg.input_dim, model_cfg.hidden_dim, bias=True)
        
        # Internal dynamics
        self.dyn = nn.Sequential(
            nn.Linear(model_cfg.z_dim + model_cfg.hidden_dim, model_cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(model_cfg.hidden_dim, model_cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(model_cfg.hidden_dim, model_cfg.z_dim),
        )
        
        # Output head: code selection
        self.out_head = nn.Sequential(
            nn.Linear(model_cfg.z_dim + model_cfg.hidden_dim, model_cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(model_cfg.hidden_dim, n_tokens),
        )
    
    def _decode_stream(self, tok: int) -> torch.Tensor:
        """Decode token into X signal. Returns [1, T, D]"""
        cfg = self.task_cfg
        device = self.Vx.device
        t = torch.arange(cfg.T, device=device, dtype=torch.float32) * cfg.dt
        th1 = cfg.omega * t + cfg.phi0
        th2 = 2.0 * cfg.omega * t + cfg.phi0
        v = self.Vx[tok]
        w = self.Wx[tok]
        s = (v[None, :] * torch.sin(th1)[:, None] +
             cfg.harm2 * w[None, :] * torch.cos(th2)[:, None])
        return s.unsqueeze(0)
    
    def _decode_all_streams(self) -> torch.Tensor:
        """Decode all tokens. Returns [n_tokens, T, D]"""
        streams = [self._decode_stream(i).squeeze(0) for i in range(self.n_tokens)]
        return torch.stack(streams, dim=0)
    
    def forward(self, y_anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B, T, D = y_anchor.shape
        
        u = torch.zeros(B, self.model_cfg.z_dim, device=y_anchor.device, dtype=y_anchor.dtype)
        u_hist: List[torch.Tensor] = []
        code_hist: List[torch.Tensor] = []
        
        for t in range(T):
            anchor_t = y_anchor[:, t, :]
            h = self.anchor_in(anchor_t)
            
            dyn_input = torch.cat([u, h], dim=-1)
            du = self.dyn(dyn_input)
            u = u + self.task_cfg.dt * du
            
            out_in = torch.cat([u, h], dim=-1)
            logits = self.out_head(out_in)
            code_probs = F.softmax(logits, dim=-1)
            
            u_hist.append(u)
            code_hist.append(code_probs)
        
        code_probs = torch.stack(code_hist, dim=1)
        u_stack = torch.stack(u_hist, dim=1)
        
        # Decode to X signal
        all_x = self._decode_all_streams()
        
        x_hat_list = []
        for t in range(T):
            code_probs_t = code_probs[:, t, :]
            x_t = all_x[:, t, :]
            x_hat_t = torch.matmul(code_probs_t, x_t)
            x_hat_list.append(x_hat_t)
        
        x_hat = torch.stack(x_hat_list, dim=1)
        
        maxp = code_probs.max(dim=-1).values
        ent = _entropy_from_probs(code_probs)
        indices = code_probs.argmax(dim=-1)
        
        return {
            "code_probs": code_probs,
            "x_hat": x_hat,
            "u": u_stack,
            "maxp": maxp,
            "ent": ent,
            "indices": indices,
        }


# -----------------------------------------------------------------------------
# Loss and Evaluation
# -----------------------------------------------------------------------------
def compute_losses(x_target: torch.Tensor, x_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(x_hat, x_target)
    return {"loss": recon_loss, "recon_loss": recon_loss}


def compute_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().cpu().numpy().flatten()
    b_flat = b.detach().cpu().numpy().flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def compute_dominant_code(indices: torch.Tensor) -> int:
    indices_np = indices.cpu().numpy().flatten()
    counts = Counter(indices_np)
    return counts.most_common(1)[0][0] if counts else -1


def compute_code_distribution(code_probs: torch.Tensor) -> Dict[int, float]:
    """Compute average code probability distribution."""
    mean_probs = code_probs.mean(dim=(0, 1)).cpu().numpy()
    return {i: float(mean_probs[i]) for i in range(len(mean_probs))}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(model: SyntaxExtractor, gen: DataGenerator,
          cfg: Dict, device: torch.device) -> Dict:
    """Train the model on SEPARATED signals only."""
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    
    history = {
        'step': [], 'loss': [], 'recon_loss': [],
        'maxp': [], 'ent': [],
    }
    
    print("\n" + "="*60)
    print("PHASE 1: Training (Separated + Partial Composites)")
    print("  Patterns: A, B, C, A+B, A+C")
    print("  Withheld: B+C (will test inference)")
    print("="*60)
    
    for step in range(1, cfg['train_steps'] + 1):
        model.train()
        opt.zero_grad()
        
        batch = gen.sample_training_batch(step - 1)
        out = model(batch['y_anchor'])
        losses = compute_losses(batch['x_target'], out['x_hat'])
        
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        
        if step == 1 or step % cfg['log_every'] == 0 or step == cfg['train_steps']:
            maxp = out['maxp'].mean().item()
            ent = out['ent'].mean().item()
            
            history['step'].append(step)
            history['loss'].append(losses['loss'].item())
            history['recon_loss'].append(losses['recon_loss'].item())
            history['maxp'].append(maxp)
            history['ent'].append(ent)
            
            p_idx = batch['pattern_idx']
            p_type, tokens = TRAINING_PATTERNS[p_idx]
            if p_type == 'single':
                pattern_str = gen.codebook.names[tokens[0]]
            else:
                pattern_str = f"{gen.codebook.names[tokens[0]]}+{gen.codebook.names[tokens[1]]}"
            
            print(f"  Step {step:5d}: loss={losses['loss'].item():.4f} "
                  f"maxp={maxp:.3f} ent={ent:.3f} pattern={pattern_str}")
    
    print(f"\n[Phase 1 Complete] Final loss: {history['loss'][-1]:.4f}")
    
    return history


# -----------------------------------------------------------------------------
# Testing: Composite Anchor (never seen during training)
# -----------------------------------------------------------------------------
def test_composition(model: SyntaxExtractor, gen: DataGenerator,
                     cfg: Dict, device: torch.device) -> Dict:
    """Test composition inference with anchor switching."""
    
    print("\n" + "="*60)
    print("PHASE 2: Frozen Test (COMPOSITION INFERENCE)")
    print("  Schedule: a+b → a+c → b+c (repeat)")
    print("  Switch interval: 1000 steps, Total: 6000 steps")
    print("  KEY TEST: b+c → B+C? (NEVER SEEN)")
    print("="*60)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    n_tokens = gen.cfg.n_tokens
    switch_interval = 1000
    test_steps = 6000
    
    # Schedule: a+b, a+c, b+c, a+b, a+c, b+c
    anchor_schedule = [
        (0, 1),  # a+b (trained)
        (0, 2),  # a+c (trained)
        (1, 2),  # b+c (WITHHELD)
        (0, 1),  # a+b
        (0, 2),  # a+c
        (1, 2),  # b+c
    ]
    
    results = {
        'step': [],
        'anchor_type': [],
        'tokens': [],
        'similarity': [],
        'maxp': [],
        'ent': [],
        'code_dist': [],
        'dominant_code': [],
    }
    
    # Per-type tracking
    per_type = {
        'a+b': {'sim': [], 'maxp': [], 'ent': []},
        'a+c': {'sim': [], 'maxp': [], 'ent': []},
        'b+c': {'sim': [], 'maxp': [], 'ent': []},
    }
    
    # For transition rate
    all_codes = []
    
    with torch.no_grad():
        for step in range(test_steps):
            phase = step // switch_interval
            tok1, tok2 = anchor_schedule[phase % len(anchor_schedule)]
            
            batch = gen.sample_test_composite(tok1, tok2)
            out = model(batch['y_anchor'])
            
            sim = compute_similarity(out['x_hat'], batch['x_expected'])
            code_dist = compute_code_distribution(out['code_probs'])
            maxp = out['maxp'].mean().item()
            ent = out['ent'].mean().item()
            dominant = compute_dominant_code(out['indices'])
            
            name1 = gen.codebook.names[tok1]
            name2 = gen.codebook.names[tok2]
            anchor_type = f"{name1.lower()}+{name2.lower()}"
            
            results['step'].append(step)
            results['anchor_type'].append(anchor_type)
            results['tokens'].append([tok1, tok2])
            results['similarity'].append(sim)
            results['maxp'].append(maxp)
            results['ent'].append(ent)
            results['code_dist'].append(code_dist)
            results['dominant_code'].append(dominant)
            
            per_type[anchor_type]['sim'].append(sim)
            per_type[anchor_type]['maxp'].append(maxp)
            per_type[anchor_type]['ent'].append(ent)
            
            all_codes.append(dominant)
            
            if step % 500 == 0:
                is_withheld = "(WITHHELD)" if anchor_type == 'b+c' else "(trained)"
                print(f"  Step {step:4d}: {anchor_type} {is_withheld}, sim={sim:.3f}, maxp={maxp:.3f}")
    
    # Calculate transition rate
    n_transitions = sum(1 for i in range(1, len(all_codes)) if all_codes[i] != all_codes[i-1])
    trans_rate = n_transitions / (len(all_codes) - 1) if len(all_codes) > 1 else 0
    
    # Summary per type
    print(f"\n--- Per-Type Results ---")
    type_results = {}
    for atype in ['a+b', 'a+c', 'b+c']:
        sims = per_type[atype]['sim']
        maxps = per_type[atype]['maxp']
        ents = per_type[atype]['ent']
        
        type_results[atype] = {
            'mean_sim': np.mean(sims),
            'std_sim': np.std(sims),
            'mean_maxp': np.mean(maxps),
            'mean_ent': np.mean(ents),
        }
        
        is_withheld = "*** WITHHELD ***" if atype == 'b+c' else "(trained)"
        print(f"  {atype} {is_withheld}: sim={np.mean(sims):.3f}±{np.std(sims):.3f}, "
              f"maxp={np.mean(maxps):.3f}, ent={np.mean(ents):.3f}")
    
    # Also test single (baseline)
    print(f"\n--- Single Tests (Baseline) ---")
    single_results = {}
    with torch.no_grad():
        for tok in [0, 1, 2]:
            batch = gen.sample_test_single(tok)
            out = model(batch['y_anchor'])
            
            sim = compute_similarity(out['x_hat'], batch['x_expected'])
            maxp = out['maxp'].mean().item()
            ent = out['ent'].mean().item()
            dominant = compute_dominant_code(out['indices'])
            
            name = gen.codebook.names[tok]
            single_results[name.lower()] = {
                'similarity': sim,
                'maxp': maxp,
                'ent': ent,
                'dominant_code': dominant,
            }
            
            print(f"  {name.lower()} → {name}: sim={sim:.3f}, code={dominant}")
    
    # Overall summary
    trained_sims = per_type['a+b']['sim'] + per_type['a+c']['sim']
    withheld_sims = per_type['b+c']['sim']
    single_sims = [r['similarity'] for r in single_results.values()]
    
    mean_trained = np.mean(trained_sims)
    mean_withheld = np.mean(withheld_sims)
    mean_single = np.mean(single_sims)
    
    print(f"\n[Phase 2 Complete]")
    print(f"  Single (baseline): {mean_single:.3f}")
    print(f"  Trained (a+b, a+c): {mean_trained:.3f}")
    print(f"  *** WITHHELD (b+c): {mean_withheld:.3f} ***")
    print(f"  Transition Rate: {trans_rate:.3f} ({n_transitions} transitions)")
    
    results['type_results'] = type_results
    results['single_results'] = single_results
    results['summary'] = {
        'mean_single_sim': mean_single,
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
                      model: SyntaxExtractor, gen: DataGenerator,
                      cfg: Dict, output_dir: str):
    """Save comprehensive results plot with time series."""
    
    fig = plt.figure(figsize=(16, 14))
    n_tokens = gen.cfg.n_tokens
    colors = ['blue', 'orange', 'green'][:n_tokens]
    
    # Row 1: Training
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.plot(train_history['step'], train_history['loss'], 'b-', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(4, 4, 2)
    ax2.plot(train_history['step'], train_history['maxp'], 'g-', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Max Prob')
    ax2.set_title('Code Confidence (Training)')
    ax2.grid(True, alpha=0.3)
    
    # Per-type similarity
    ax3 = fig.add_subplot(4, 4, 3)
    type_results = test_results['type_results']
    types = ['a+b', 'a+c', 'b+c']
    sims = [type_results[t]['mean_sim'] for t in types]
    stds = [type_results[t]['std_sim'] for t in types]
    bar_colors = ['purple', 'brown', 'red']
    bars = ax3.bar(range(3), sims, yerr=stds, color=bar_colors, alpha=0.7, capsize=5)
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['a+b\n(trained)', 'a+c\n(trained)', 'b+c\n(WITHHELD)'])
    ax3.set_ylabel('Similarity')
    ax3.set_title('Composite Similarity')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Single baseline
    ax4 = fig.add_subplot(4, 4, 4)
    single_results = test_results['single_results']
    single_names = list(single_results.keys())
    single_sims = [single_results[k]['similarity'] for k in single_names]
    ax4.bar(range(len(single_names)), single_sims, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(single_names)))
    ax4.set_xticklabels([f'{n}→{n.upper()}' for n in single_names])
    ax4.set_ylabel('Similarity')
    ax4.set_title('Single (Baseline)')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Time series
    ax5 = fig.add_subplot(4, 4, 5)
    steps = test_results['step']
    sims_ts = test_results['similarity']
    ax5.plot(steps, sims_ts, 'b-', alpha=0.5, lw=0.5)
    window = 50
    if len(sims_ts) > window:
        ma = np.convolve(sims_ts, np.ones(window)/window, mode='valid')
        ax5.plot(range(window-1, len(sims_ts)), ma, 'b-', lw=2)
    for i in range(1, 6):
        ax5.axvline(i * 1000, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Similarity')
    ax5.set_title('Similarity Over Time')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(4, 4, 6)
    anchor_types = test_results['anchor_type']
    type_to_num = {'a+b': 0, 'a+c': 1, 'b+c': 2}
    type_nums = [type_to_num[t] for t in anchor_types]
    ax6.scatter(steps[::10], [type_nums[i] for i in range(0, len(steps), 10)],
                c='blue', alpha=0.5, s=10, label='Anchor')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Type')
    ax6.set_yticks([0, 1, 2])
    ax6.set_yticklabels(['a+b', 'a+c', 'b+c'])
    ax6.set_title('Anchor Schedule')
    for i in range(1, 6):
        ax6.axvline(i * 1000, color='red', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(4, 4, 7)
    maxps = test_results['maxp']
    ax7.plot(steps, maxps, 'g-', alpha=0.5, lw=0.5)
    if len(maxps) > window:
        ma = np.convolve(maxps, np.ones(window)/window, mode='valid')
        ax7.plot(range(window-1, len(maxps)), ma, 'g-', lw=2)
    for i in range(1, 6):
        ax7.axvline(i * 1000, color='red', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Max Prob')
    ax7.set_title('Code Confidence Over Time')
    ax7.grid(True, alpha=0.3)
    
    # Summary
    ax8 = fig.add_subplot(4, 4, 8)
    summary = test_results['summary']
    summary_names = ['Single', 'Trained', 'Withheld']
    summary_vals = [
        summary['mean_single_sim'],
        summary['mean_trained_sim'],
        summary['withheld_sim'],
    ]
    ax8.bar(range(3), summary_vals, color=['blue', 'purple', 'red'], alpha=0.7)
    ax8.set_xticks(range(3))
    ax8.set_xticklabels(summary_names)
    ax8.set_ylabel('Similarity')
    ax8.set_title('Summary')
    ax8.set_ylim(0, 1.1)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Code distributions
    ax9 = fig.add_subplot(4, 4, 9)
    # Get sample code distributions for each type
    for i, atype in enumerate(['a+b', 'a+c', 'b+c']):
        # Find first occurrence of this type
        for j, t in enumerate(test_results['anchor_type']):
            if t == atype:
                cd = test_results['code_dist'][j]
                x_pos = np.arange(n_tokens) + i * 0.25
                ax9.bar(x_pos, [cd[k] for k in range(n_tokens)], width=0.2, 
                        label=atype, alpha=0.7)
                break
    ax9.set_xticks(np.arange(n_tokens) + 0.25)
    ax9.set_xticklabels([f'Code {i}\n({gen.codebook.names[i]})' for i in range(n_tokens)])
    ax9.set_ylabel('Probability')
    ax9.set_title('Code Distribution by Type')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Entropy comparison
    ax10 = fig.add_subplot(4, 4, 10)
    ents = [type_results[t]['mean_ent'] for t in types]
    ax10.bar(range(3), ents, color=['purple', 'brown', 'red'], alpha=0.7)
    ax10.set_xticks(range(3))
    ax10.set_xticklabels(['a+b', 'a+c', 'b+c'])
    ax10.set_ylabel('Entropy')
    ax10.set_title('Code Entropy\n(Higher = Multiple Codes)')
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Entropy over time
    ax11 = fig.add_subplot(4, 4, 11)
    ents_ts = test_results['ent']
    ax11.plot(steps, ents_ts, 'r-', alpha=0.5, lw=0.5)
    if len(ents_ts) > window:
        ma = np.convolve(ents_ts, np.ones(window)/window, mode='valid')
        ax11.plot(range(window-1, len(ents_ts)), ma, 'r-', lw=2)
    for i in range(1, 6):
        ax11.axvline(i * 1000, color='red', linestyle='--', alpha=0.5)
    ax11.set_xlabel('Step')
    ax11.set_ylabel('Entropy')
    ax11.set_title('Entropy Over Time')
    ax11.grid(True, alpha=0.3)
    
    # Placeholder
    ax12 = fig.add_subplot(4, 4, 12)
    ax12.axis('off')
    ax12.text(0.5, 0.5, f"Withheld Test:\nb+c → B+C\nsim = {summary['withheld_sim']:.3f}",
              ha='center', va='center', fontsize=14, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Row 4: Signal examples
    with torch.no_grad():
        # a+b (trained)
        ax = fig.add_subplot(4, 4, 13)
        batch = gen.sample_test_composite(0, 1)
        out = model(batch['y_anchor'])
        expected_np = batch['x_expected'].cpu().numpy().squeeze()
        output_np = out['x_hat'].cpu().numpy().squeeze()
        ax.plot(expected_np[:, 0], 'b-', lw=2, label='Expected', alpha=0.7)
        ax.plot(output_np[:, 0], 'r--', lw=2, label='Output', alpha=0.8)
        sim = compute_similarity(out['x_hat'], batch['x_expected'])
        ax.set_title(f'a+b→A+B (trained): sim={sim:.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        
        # a+c (trained)
        ax = fig.add_subplot(4, 4, 14)
        batch = gen.sample_test_composite(0, 2)
        out = model(batch['y_anchor'])
        expected_np = batch['x_expected'].cpu().numpy().squeeze()
        output_np = out['x_hat'].cpu().numpy().squeeze()
        ax.plot(expected_np[:, 0], 'b-', lw=2, label='Expected', alpha=0.7)
        ax.plot(output_np[:, 0], 'r--', lw=2, label='Output', alpha=0.8)
        sim = compute_similarity(out['x_hat'], batch['x_expected'])
        ax.set_title(f'a+c→A+C (trained): sim={sim:.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # b+c (WITHHELD - KEY)
        ax = fig.add_subplot(4, 4, 15)
        batch = gen.sample_test_composite(1, 2)
        out = model(batch['y_anchor'])
        expected_np = batch['x_expected'].cpu().numpy().squeeze()
        output_np = out['x_hat'].cpu().numpy().squeeze()
        ax.plot(expected_np[:, 0], 'b-', lw=2, label='Expected', alpha=0.7)
        ax.plot(output_np[:, 0], 'r--', lw=2, label='Output', alpha=0.8)
        sim = compute_similarity(out['x_hat'], batch['x_expected'])
        ax.set_title(f'b+c→B+C (WITHHELD): sim={sim:.2f}', fontsize=10, color='red')
        ax.grid(True, alpha=0.3)
        
        # Single example
        ax = fig.add_subplot(4, 4, 16)
        batch = gen.sample_test_single(0)
        out = model(batch['y_anchor'])
        expected_np = batch['x_expected'].cpu().numpy().squeeze()
        output_np = out['x_hat'].cpu().numpy().squeeze()
        ax.plot(expected_np[:, 0], 'b-', lw=2, label='Expected', alpha=0.7)
        ax.plot(output_np[:, 0], 'r--', lw=2, label='Output', alpha=0.8)
        sim = compute_similarity(out['x_hat'], batch['x_expected'])
        ax.set_title(f'a→A (single): sim={sim:.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    # Title
    summary = test_results['summary']
    withheld_sim = summary['withheld_sim']
    trained_sim = summary['mean_trained_sim']
    
    if withheld_sim > 0.95:
        conclusion = "Syntax emerges! (Perfect)"
    elif withheld_sim > 0.85:
        conclusion = "Syntax emerges! (Good)"
    elif withheld_sim > 0.7:
        conclusion = "Partial syntax"
    else:
        conclusion = "No syntax"
    
    plt.suptitle(f'EX9: Syntax Emergence (seed={cfg["seed"]})\n'
                 f'Trained={trained_sim:.2f}, WITHHELD(B+C)={withheld_sim:.2f} | {conclusion}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, "ex9_results.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CSCT EX9: Syntax Emergence")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train-steps', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--output-dir', type=str, default='results_ex9')
    parser.add_argument('--n-tokens', type=int, default=3)
    parser.add_argument('--hull-mode', type=str, default='random',
                       choices=['random', 'in_hull', 'out_hull'],
                       help='Control C position: random, in_hull, out_hull')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    seed_all(args.seed)
    safe_makedirs(args.output_dir)
    
    cfg = {
        'seed': args.seed,
        'train_steps': args.train_steps,
        'lr': args.lr,
        'log_every': args.log_every,
        'hull_mode': args.hull_mode,
    }
    
    # Configurations
    data_cfg = DataCfg(n_tokens=args.n_tokens)
    model_cfg = ModelCfg()
    task_cfg = TaskCfg(n_tokens=args.n_tokens)
    
    # Create codebook and data generator
    codebook = DualCodebook(data_cfg, args.seed, device, hull_mode=args.hull_mode)
    gen = DataGenerator(data_cfg, codebook, device)
    
    # Create model
    model = SyntaxExtractor(model_cfg, task_cfg, codebook.Vx, codebook.Wx).to(device)
    
    print("="*70)
    print("CSCT EX9: Syntax Emergence via Composition Inference")
    print("="*70)
    print(f"Device: {device}, Seed: {args.seed}, Hull Mode: {args.hull_mode}")
    print(f"Design: Fixed codebook + Anchor-driven dynamics")
    print(f"Training: A, B, C (single) + A+B, A+C (composite)")
    print(f"Withheld: B+C (test if model can INFER this composition)")
    print("="*70)
    
    # Print codebook
    print("\nCodebooks (FIXED, 3D vectors):")
    print("  X-codebook (output):")
    for i in range(args.n_tokens):
        v = codebook.Vx[i].cpu().numpy()
        print(f"    V_x[{codebook.names[i]}]: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]")
    print("  Y-codebook (anchor):")
    for i in range(args.n_tokens):
        v = codebook.Vy[i].cpu().numpy()
        print(f"    V_y[{codebook.names[i].lower()}]: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]")
    print(f"\n  Vector Angles: A-C={codebook.angle_AC:.1f}°, B-C={codebook.angle_BC:.1f}°, A-B={codebook.angle_AB:.1f}°")
    
    # Train
    train_history = train(model, gen, cfg, device)
    
    # Test
    test_results = test_composition(model, gen, cfg, device)
    
    # Save results
    save_results_plot(train_history, test_results, model, gen, cfg, args.output_dir)
    
    # Save metrics
    summary = test_results['summary']
    type_results = test_results['type_results']
    metrics = {
        'seed': args.seed,
        'hull_mode': args.hull_mode,
        # Vector angles
        'angle_AC': codebook.angle_AC,
        'angle_BC': codebook.angle_BC,
        'angle_AB': codebook.angle_AB,
        # Final loss
        'final_loss': train_history['loss'][-1] if train_history['loss'] else float('nan'),
        # Similarity metrics
        'mean_single_sim': summary['mean_single_sim'],
        'mean_trained_sim': summary['mean_trained_sim'],
        'withheld_sim': summary['withheld_sim'],
        # Transition rate
        'trans_rate': summary.get('trans_rate', 0),
        'n_transitions': summary.get('n_transitions', 0),
    }
    
    # Add per-type metrics
    for atype in ['a+b', 'a+c', 'b+c']:
        metrics[f'sim_{atype}'] = type_results[atype]['mean_sim']
        metrics[f'maxp_{atype}'] = type_results[atype]['mean_maxp']
        metrics[f'ent_{atype}'] = type_results[atype]['mean_ent']
    
    csv_path = os.path.join(args.output_dir, "ex9_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[saved] {csv_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("EX9 FINAL SUMMARY")
    print("="*60)
    print(f"\n  Hull Mode: {args.hull_mode}")
    print(f"  Angles: A-C={codebook.angle_AC:.1f}°, B-C={codebook.angle_BC:.1f}°")
    print(f"\n  Training: A, B, C + A+B, A+C")
    print(f"  Withheld: B+C (never seen)")
    print(f"\n  Single (baseline): {summary['mean_single_sim']:.3f}")
    print(f"  Trained composites (A+B, A+C): {summary['mean_trained_sim']:.3f}")
    print(f"  *** WITHHELD (B+C): {summary['withheld_sim']:.3f} ***")
    
    if summary['withheld_sim'] > 0.95:
        print(f"\n→ CONFIRMED: Syntax emerges! (Perfect inference)")
        print("  Model INFERRED B+C composition without being trained on it")
    elif summary['withheld_sim'] > 0.85:
        print(f"\n→ CONFIRMED: Syntax emerges! (Good inference)")
        print("  Model learned compositional rules from A+B, A+C")
    elif summary['withheld_sim'] > 0.7:
        print(f"\n→ PARTIAL: Some syntactic ability observed.")
    else:
        print(f"\n→ NOT CONFIRMED: No clear syntax emergence.")
    
    print("\n[EX9 Complete]")


if __name__ == "__main__":
    main()
