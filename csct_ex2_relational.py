#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSCT EX2: Multi-Channel Relational Extraction (Lissajous)
==========================================================

Tests multi-channel relational information extraction with discrete codes.

Task:
  - ch0: Reference signal sin(2π f0 t)
  - ch1: Related signal sin(∫ 2π f0 k(t) dt) with time-varying frequency ratio k(t)
  - Goal: Learn discrete codes that capture the frequency ratio relationship

Theoretical Significance:
  Tests A3 (Multi-clock Selection):
  - Multiple discrete codes can capture inter-channel relationships
  - The frequency ratio k(t) represents a "relational" property
  - This demonstrates that discretization preserves structural relationships

Design Decision (Gate Input Separation):
  - Clock selection gate: sees INPUT (x_target) only → relation extracted from input
  - Anchor gate (A4): sees ANCHOR (y_anchor) only → regularization/synchronization
  - This prevents "label leakage" and clearly separates concerns
  - Implemented via CSCT_Engine_EX2 subclass (does NOT modify csct_engine.py)

Key Metrics:
  - recon_loss: Reconstruction error
  - k_mae: Mean absolute error of estimated k(t) via FFT Hilbert transform
  - stability: 1 - transition_rate (code persistence)
  - unique_codes: Number of distinct codes used

Critical Implementation Notes:
  1. dt is derived from linspace: dt = 1.0 / (T - 1), single source of truth
  2. Hilbert transform uses FFT (no scipy dependency)
  3. CSCT_Engine_EX2 inherits from CSCT_Engine, overrides forward() for gate input separation

"""

import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import CSCT Engine (unmodified)
from csct_engine import (
    CSCTConfig, 
    CSCT_Engine, 
    save_convergence_curve,
    extract_physical_features,
    straight_through_topk,
)


# =============================================================================
# CSCT_Engine_EX2: Subclass with Gate Input Separation
# =============================================================================

class CSCT_Engine_EX2(CSCT_Engine):
    """
    EX2-specific CSCT Engine with gate input separation.
    
    Key difference from parent:
      - Clock selection gate sees INPUT (feat_x) only
      - Anchor gate (A4) sees ANCHOR (feat_y_anchor) only
      - This clearly separates "relation extraction" from "regularization"
    
    Implementation:
      - Inherits from CSCT_Engine
      - Overrides __init__ to rebuild gate networks with correct input dimension
      - Overrides forward() to pass feat_x only to clock selection
    """
    
    def __init__(self, cfg: CSCTConfig = None, use_multigate: bool = False, **kwargs):
        # Call parent __init__ first
        super().__init__(cfg, use_multigate, **kwargs)
        
        # Now rebuild gate networks with input-only dimension (3*D instead of 6*D)
        D = int(getattr(self.cfg, "input_dim", 1))
        gate_in_dim = 3 * D  # feat_x only
        
        if self.use_multigate:
            # Rebuild MultiGate networks with new input dimension
            self.multigate.na_gate = nn.Sequential(
                nn.Linear(gate_in_dim, self.cfg.hidden_dim), nn.Tanh(),
                nn.Linear(self.cfg.hidden_dim, self.cfg.n_clocks),
            )
            self.multigate.theta_proj = nn.Linear(gate_in_dim, 1)
            self.multigate.nmda_gate = nn.Sequential(
                nn.Linear(gate_in_dim + 1, self.cfg.hidden_dim), nn.Tanh(),
                nn.Linear(self.cfg.hidden_dim, self.cfg.n_clocks),
            )
        else:
            # Rebuild SingleGate networks
            if getattr(self, "use_channel_top1", False):
                # Per-channel feature dim is now 3 (not 6)
                self.channel_gate_net = nn.Sequential(
                    nn.Linear(3, self.cfg.hidden_dim), nn.Tanh(),
                    nn.Linear(self.cfg.hidden_dim, 1),
                )
                single_gate_in_dim = 3
            else:
                single_gate_in_dim = gate_in_dim
            
            self.gate_net = nn.Sequential(
                nn.Linear(single_gate_in_dim, self.cfg.hidden_dim), nn.Tanh(),
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim), nn.Tanh(),
                nn.Linear(self.cfg.hidden_dim, self.cfg.n_clocks),
            )
    
    def forward(self, x_target: torch.Tensor, y_anchor: torch.Tensor, 
                beta: float = None) -> Dict:
        """
        Forward pass with gate input separation.
        
        Key difference: Clock selection gate sees feat_x only (not feat_y).
        Anchor gate (A4) still uses anchor for transition penalty.
        """
        device = x_target.device
        dtype = x_target.dtype
        cfg = self.cfg
        
        # β handling
        if beta is None:
            safe_log_beta = torch.clamp(self.log_beta, min=-5.0, max=5.0)
            beta_t = torch.exp(safe_log_beta)
        else:
            beta_t = torch.as_tensor(beta, device=device, dtype=dtype)
        
        B, T, D = x_target.shape
        
        # 1. Extract physical features
        feat_x = extract_physical_features(x_target)  # [B, T, 3*D]
        
        # Anchor features for anchor_gate_net (A4) - always from 1-channel
        y_anchor_ch0 = y_anchor[..., :1] if y_anchor.shape[-1] > 1 else y_anchor
        feat_y_anchor = extract_physical_features(y_anchor_ch0)  # [B, T, 3]
        
        # KEY DIFFERENCE: Clock gate sees feat_x ONLY (not feat_y)
        feat_for_clock_gate = feat_x  # [B, T, 3*D]
        
        # 2. Clock selection (A3) with temperature
        temp = self.temperature.item()
        effective_tau = cfg.gate_tau * temp
        
        gate_info_list = []
        
        if self.use_multigate:
            # Use Multi-Gate Clock Bank with feat_x only
            logits, gate_info = self.multigate(feat_for_clock_gate, t_rel=1.0)
            gate_info_list.append(gate_info)
        else:
            # SingleGate
            if getattr(self, "use_channel_top1", False):
                # Per-channel features: [B,T,3D] -> [B,T,D,3]
                feat_ch = feat_x.view(B, T, D, 3)
                
                # Channel scores
                ch_scores = self.channel_gate_net(feat_ch).squeeze(-1)  # [B,T,D]
                
                # Channel selection
                stick = float(getattr(cfg, "channel_stickiness", 0.0))
                if stick > 0.0 and T > 1:
                    if self.training:
                        sel_list = []
                        prev = None
                        for t_i in range(T):
                            s_t = ch_scores[:, t_i, :]
                            if prev is not None:
                                s_t = s_t + stick * prev
                            sel_t = straight_through_topk(
                                s_t, k=1, tau=effective_tau,
                                use_gumbel=cfg.use_gumbel,
                                gumbel_noise=cfg.gumbel_noise * temp,
                            )
                            sel_list.append(sel_t)
                            prev = sel_t.detach()
                        ch_sel = torch.stack(sel_list, dim=1)
                    else:
                        sel_list = []
                        prev = None
                        for t_i in range(T):
                            s_t = ch_scores[:, t_i, :]
                            if prev is not None:
                                s_t = s_t + stick * prev
                            idx = torch.argmax(s_t, dim=-1)
                            sel_t = F.one_hot(idx, num_classes=D).to(dtype=s_t.dtype)
                            sel_list.append(sel_t)
                            prev = sel_t
                        ch_sel = torch.stack(sel_list, dim=1)
                else:
                    if self.training:
                        ch_sel = straight_through_topk(
                            ch_scores, k=1, tau=effective_tau,
                            use_gumbel=cfg.use_gumbel,
                            gumbel_noise=cfg.gumbel_noise * temp,
                        )
                    else:
                        ch_sel = straight_through_topk(
                            ch_scores, k=1, tau=effective_tau, use_gumbel=False
                        )
                
                # Route selected channel features
                feat_sel = (ch_sel.unsqueeze(-1) * feat_ch).sum(dim=2)  # [B,T,3]
                logits = self.gate_net(feat_sel)
            else:
                # input_dim == 1
                logits = self.gate_net(feat_for_clock_gate)
        
        # Clock selection with straight-through
        if self.training:
            g = straight_through_topk(
                logits, k=cfg.gate_topk, tau=effective_tau,
                use_gumbel=cfg.use_gumbel, gumbel_noise=cfg.gumbel_noise * temp
            )
        else:
            g = straight_through_topk(
                logits, k=cfg.gate_topk, tau=effective_tau, use_gumbel=False
            )
        
        indices = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits / max(effective_tau, 0.01), dim=-1)
        
        # 3. Reconstruction
        recon = torch.einsum('btk,kd->btd', g, self.codebook) + self.bias.view(1, 1, -1)
        
        # 4. Detect transitions
        trans = torch.zeros(B, T, 1, device=device)
        trans[:, 1:] = (indices[:, 1:] != indices[:, :-1]).float().unsqueeze(-1)
        
        # 5. Anchor gate (A4) - ALWAYS uses anchor
        if bool(getattr(cfg, 'use_anchor_gate', True)):
            anchor_gate = self.anchor_gate_net(feat_y_anchor)
        else:
            anchor_gate = torch.ones(B, T, 1, device=device, dtype=dtype)
        
        # 6. Transition penalty
        penalty_weight = cfg.gate_floor + (1.0 - anchor_gate) * (1.0 - cfg.gate_floor)
        
        # 7. Losses
        trans_loss = (trans * penalty_weight).mean()
        recon_loss = F.mse_loss(recon, x_target)
        
        gate_sup_w = float(getattr(cfg, 'gate_sup_weight', 0.1))
        if bool(getattr(cfg, 'use_anchor_gate', True)) and gate_sup_w > 0.0:
            dy_a = torch.zeros(B, T, 1, device=device, dtype=dtype)
            dy_step = (y_anchor[:, 1:] - y_anchor[:, :-1]).abs()
            dy_a[:, 1:, 0] = dy_step.mean(dim=-1)
            dy_normalized = dy_a / (dy_a.max() + 1e-8)
            gate_supervision = F.mse_loss(anchor_gate, dy_normalized)
        else:
            gate_supervision = torch.zeros((), device=device, dtype=dtype)
        
        total_loss = recon_loss + beta_t * trans_loss + gate_sup_w * gate_supervision
        
        result = {
            "loss": total_loss,
            "beta": float(beta_t.detach().item()),
            "recon_loss": recon_loss,
            "trans_loss": trans_loss,
            "gate_supervision": gate_supervision,
            "losses": {
                "recon": recon_loss.item(),
                "trans": trans_loss.item(),
                "gate_sup": gate_supervision.item(),
            },
            "recon": recon,
            "indices": indices,
            "probs": probs,
            "gate": anchor_gate,
            "clock_selection": g,
            "trans": trans,
            "penalty_weight": penalty_weight,
            "temperature": temp,
        }
        
        if gate_info_list:
            result["gate_info"] = gate_info_list
        
        return result


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EX2Config:
    """Experiment-specific configuration for EX2.
    
    Note: dt is NOT stored here. It is always derived from seq_len as:
          dt = 1.0 / (seq_len - 1)
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    seq_len: int = 400
    f0: float = 5.0
    k_start: float = 1.0
    k_end: float = 2.0
    k_mode: str = "piecewise"  # linear | piecewise
    k_nsegs: int = 4
    k_jitter: float = 0.0
    
    # Model
    K: int = 8
    use_multigate: bool = True
    
    # Training
    steps: int = 3000
    lr: float = 0.01
    beta_start: float = 5.0
    beta_end: float = 50.0
    
    # Logging
    log_every: int = 50
    seed: int = 0
    output_dir: str = "results_ex2"
    
    @property
    def dt(self) -> float:
        """Derived dt from seq_len. Single source of truth."""
        return 1.0 / max(1, self.seq_len - 1)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# =============================================================================
# FFT-based Hilbert Transform (no scipy dependency)
# =============================================================================

def hilbert_analytic(x: np.ndarray) -> np.ndarray:
    """
    Compute analytic signal using FFT-based Hilbert transform.
    
    This replaces scipy.signal.hilbert for better reproducibility.
    
    Args:
        x: Real-valued signal [T]
        
    Returns:
        Analytic signal (complex) [T]
    """
    N = len(x)
    X = np.fft.fft(x)
    
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    
    return np.fft.ifft(X * h)


def estimate_k_from_signal(x: np.ndarray, f0: float, dt: float) -> np.ndarray:
    """
    Estimate instantaneous frequency ratio k(t) from signal using FFT Hilbert.
    
    Args:
        x: [T] waveform (ch1)
        f0: base frequency
        dt: sampling interval (derived from linspace)
        
    Returns:
        k_hat: [T] estimated ratio
    """
    z = hilbert_analytic(x)
    phase = np.unwrap(np.angle(z))
    dphase = np.diff(phase)
    inst_f = np.concatenate([[dphase[0] / (2 * np.pi * dt)], dphase / (2 * np.pi * dt)])
    k_hat = inst_f / max(1e-8, float(f0))
    return np.clip(k_hat, 0.0, 10.0)


# =============================================================================
# Data Generation
# =============================================================================

def make_k_schedule(cfg: EX2Config, T: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate k(t) schedule."""
    if cfg.k_mode == "linear":
        k_t = np.linspace(cfg.k_start, cfg.k_end, T)
    elif cfg.k_mode == "piecewise":
        nsegs = max(1, cfg.k_nsegs)
        seg_edges = np.linspace(0, T, nsegs + 1).astype(int)
        k_vals = rng.uniform(
            low=min(cfg.k_start, cfg.k_end),
            high=max(cfg.k_start, cfg.k_end),
            size=nsegs
        )
        k_t = np.empty(T, dtype=np.float32)
        for s in range(nsegs):
            a, b = seg_edges[s], seg_edges[s + 1]
            k_t[a:b] = k_vals[s]
    else:
        raise ValueError(f"Unknown k_mode: {cfg.k_mode}")
    
    if cfg.k_jitter > 0:
        k_t = k_t + rng.normal(scale=cfg.k_jitter, size=T)
        k_t = np.clip(k_t, 0.1, 10.0)
    
    return k_t.astype(np.float32)


def generate_lissajous_data(cfg: EX2Config) -> Dict[str, torch.Tensor]:
    """
    Generate Lissajous-like 2-channel signal with frequency ratio k(t).
    
    Critical: Uses linspace for time axis, dt is derived as 1/(T-1).
    """
    T = cfg.seq_len
    t = np.linspace(0, 1, T, dtype=np.float32)
    dt = cfg.dt  # Single source of truth
    
    rng = np.random.RandomState(cfg.seed)
    
    ch0 = np.sin(2 * np.pi * cfg.f0 * t).astype(np.float32)
    k_t = make_k_schedule(cfg, T, rng)
    phase_acc = np.cumsum(2 * np.pi * cfg.f0 * k_t * dt).astype(np.float32)
    ch1 = np.sin(phase_acc).astype(np.float32)
    
    x_target = np.stack([ch0, ch1], axis=-1)[None, :, :]
    y_anchor = ch0[None, :, None]
    k_t = k_t[None, :]
    
    device = torch.device(cfg.device)
    
    return {
        "x_target": torch.from_numpy(x_target).to(device),
        "y_anchor": torch.from_numpy(y_anchor).to(device),
        "k_t": torch.from_numpy(k_t).to(device),
    }


# =============================================================================
# Visualization
# =============================================================================

def save_plot(
    path: str,
    cfg: EX2Config,
    data: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    step: int,
    gate_type: str,
):
    """Save visualization."""
    x = data["x_target"][0].cpu().numpy()
    k_t = data["k_t"][0].cpu().numpy()
    
    recon = out["recon"][0].detach().cpu().numpy()
    indices = out["indices"][0].detach().cpu().numpy()
    
    T = len(k_t)
    t_axis = np.arange(T)
    dt = cfg.dt
    
    k_hat = estimate_k_from_signal(recon[:, 1], cfg.f0, dt)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    ax = axes[0, 0]
    ax.plot(t_axis, x[:, 0], "b-", label="ch0 (true)", alpha=0.7)
    ax.plot(t_axis, recon[:, 0], "b--", label="ch0 (recon)", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title("Channel 0 (Anchor)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t_axis, x[:, 1], "r-", label="ch1 (true)", alpha=0.7)
    ax.plot(t_axis, recon[:, 1], "r--", label="ch1 (recon)", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title("Channel 1 (Related)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(t_axis, k_t, "k-", linewidth=2, label="k(t) true")
    ax.plot(t_axis, k_hat, "r--", linewidth=2, label="k(t) estimated")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency Ratio")
    ax.set_title("k(t) Estimation (FFT Hilbert)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(x[:, 0], x[:, 1], "b-", alpha=0.3, label="true")
    sc = ax.scatter(recon[:, 0], recon[:, 1], c=indices, cmap="tab10", s=10, alpha=0.7)
    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.set_title("Lissajous (color=code)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax)
    
    ax = axes[1, 1]
    ax.step(t_axis, indices, where="post", color="k", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Code Index")
    ax.set_title("Discrete Codes")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    diffs = np.diff(indices) != 0
    ax.stem(range(len(diffs)), diffs, markerfmt=" ", basefmt="C0-")
    ax.set_xlabel("Time")
    ax.set_ylabel("Transition")
    ax.set_title("Code Transitions")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"EX2 Lissajous | {gate_type} | step={step}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Training
# =============================================================================

def run_ex2(cfg: EX2Config) -> Dict:
    """Main training loop using CSCT_Engine_EX2."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    
    data = generate_lissajous_data(cfg)
    x_target = data["x_target"]
    y_anchor = data["y_anchor"]
    k_t = data["k_t"]
    
    # Use EX2-specific engine (gate sees input only)
    model_cfg = CSCTConfig(
        input_dim=2,
        n_clocks=cfg.K,
        hidden_dim=64,
        z_dim=16,
        gate_topk=1,
        gate_tau=0.7,
        use_gumbel=True,
        gumbel_noise=0.5,
        beta=20.0,
        use_anchor_gate=True,
        gate_sup_weight=0.1,
    )
    model = CSCT_Engine_EX2(model_cfg, use_multigate=cfg.use_multigate).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    gate_type = "MultiGate" if cfg.use_multigate else "SingleGate"
    
    history = {
        "step": [],
        "loss": [],
        "loss_eval": [],
        "recon_loss": [],
        "trans_loss": [],
        "trans_rate": [],
        "unique_codes": [],
        "stability": [],
        "k_mae": [],
        "code_entropy": [],
        "code_entropy_norm": [],
        "beta": [],
    }
    
    print(f"EX2 Lissajous | {gate_type} | device={cfg.device} | steps={cfg.steps}")
    print(f"  K={cfg.K}, k_mode={cfg.k_mode}, k_nsegs={cfg.k_nsegs}")
    print(f"  Gate input: feat_x only (relation from input, anchor for A4 only)")
    print(f"  dt={cfg.dt:.6f} (derived from seq_len={cfg.seq_len})")
    print()
    
    t0 = time.time()
    dt = cfg.dt
    
    model.train()
    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad()
        
        progress = step / max(1, cfg.steps)
        beta = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * progress
        
        out = model(x_target, y_anchor, beta=beta)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        
        if hasattr(model, "anneal_temperature"):
            model.anneal_temperature()
        
        if step % cfg.log_every == 0 or step == cfg.steps:
            model.eval()
            with torch.no_grad():
                out_eval = model(x_target, y_anchor, beta=beta)
                recon = out_eval["recon"]
                indices = out_eval["indices"]
                
                loss_eval = out_eval["loss"].item()
                recon_loss = F.mse_loss(recon, x_target).item()
                trans_loss = out_eval["trans_loss"].item()
                trans_rate = out_eval["trans"].mean().item()
                unique_codes = len(torch.unique(indices))
                
                idx_np = indices[0].cpu().numpy()
                stability = 1.0 - (np.diff(idx_np) != 0).mean()
                
                # Code entropy (EX1 compatible)
                flat_idx = indices.reshape(-1).cpu()
                counts = torch.bincount(flat_idx, minlength=cfg.K).float()
                probs = counts / counts.sum().clamp(min=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                max_entropy = np.log(cfg.K) if cfg.K > 1 else 1.0
                entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
                
                # k(t) estimation
                recon_np = recon[0].cpu().numpy()
                k_true = k_t[0].cpu().numpy()
                k_hat = estimate_k_from_signal(recon_np[:, 1], cfg.f0, dt)
                k_mae = float(np.mean(np.abs(k_hat - k_true)))
                
                history["step"].append(step)
                history["loss"].append(float(loss.item()))
                history["loss_eval"].append(loss_eval)
                history["recon_loss"].append(recon_loss)
                history["trans_loss"].append(trans_loss)
                history["trans_rate"].append(trans_rate)
                history["unique_codes"].append(unique_codes)
                history["stability"].append(stability)
                history["k_mae"].append(k_mae)
                history["code_entropy"].append(entropy)
                history["code_entropy_norm"].append(entropy_norm)
                history["beta"].append(beta)
                
                print(
                    f"[step={step:4d}] loss={loss.item():.4f} "
                    f"recon={recon_loss:.4f} k_mae={k_mae:.3f} "
                    f"stability={stability:.3f} codes={unique_codes}"
                )
            model.train()
    
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    
    model.eval()
    with torch.no_grad():
        out = model(x_target, y_anchor, beta=cfg.beta_end)
    
    plot_path = os.path.join(cfg.output_dir, f"ex2_{gate_type}.png")
    save_plot(plot_path, cfg, data, out, cfg.steps, gate_type)
    print(f"[saved] {plot_path}")
    
    conv_path = os.path.join(cfg.output_dir, f"convergence_{gate_type}.png")
    save_convergence_curve(history, conv_path, f"EX2 {gate_type} Convergence")
    
    # Save k(t) vs code mapping for aggregate analysis
    import csv
    with torch.no_grad():
        recon = out["recon"][0].cpu().numpy()
        indices = out["indices"][0].cpu().numpy()
        k_true = data["k_t"][0].cpu().numpy()
        k_hat = estimate_k_from_signal(recon[:, 1], cfg.f0, cfg.dt)
        x_np = data["x_target"][0].cpu().numpy()
    
    # Save k_code_mapping.csv (for scatter plots)
    k_code_path = os.path.join(cfg.output_dir, "k_code_mapping.csv")
    with open(k_code_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "k_true", "k_hat", "code", "ch0", "ch1", "recon_ch0", "recon_ch1"])
        for t_i in range(len(k_true)):
            writer.writerow([
                t_i, k_true[t_i], k_hat[t_i], indices[t_i],
                x_np[t_i, 0], x_np[t_i, 1], recon[t_i, 0], recon[t_i, 1]
            ])
    print(f"[saved] {k_code_path}")
    
    # Save metrics history CSV (EX1 compatible format)
    csv_path = os.path.join(cfg.output_dir, "metrics_history.csv")
    if history["step"]:
        keys = list(history.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["gate_type"] + keys)
            writer.writeheader()
            for i in range(len(history["step"])):
                row = {"gate_type": gate_type}
                for k in keys:
                    row[k] = history[k][i]
                writer.writerow(row)
        print(f"[saved] {csv_path}")
    
    print("\n" + "=" * 60)
    print(f"EX2 LISSAJOUS SUMMARY ({gate_type})")
    print("=" * 60)
    print(f"Final recon_loss: {history['recon_loss'][-1]:.4f}")
    print(f"Final k_mae:      {history['k_mae'][-1]:.3f}")
    print(f"Final stability:  {history['stability'][-1]:.3f}")
    print(f"Unique codes:     {history['unique_codes'][-1]}")
    print(f"dt (derived):     {cfg.dt:.6f}")
    print("=" * 60)
    
    return {
        "recon_loss": history["recon_loss"][-1],
        "k_mae": history["k_mae"][-1],
        "stability": history["stability"][-1],
        "unique_codes": history["unique_codes"][-1],
        "history": history,
    }


# =============================================================================
# CLI
# =============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description="CSCT EX2: Lissajous Relational Extraction")
    p.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    p.add_argument("--output-dir", "--outdir", default="results_ex2", dest="output_dir")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seq-len", type=int, default=400)
    p.add_argument("--K", "--n_clocks", "--n-clocks", type=int, default=8, dest="K", help="codebook size / n_clocks")
    p.add_argument("--single-gate", action="store_true", help="Use SingleGate instead of MultiGate")
    p.add_argument("--f0", type=float, default=5.0)
    p.add_argument("--k-mode", "--k_mode", type=str, default="piecewise", choices=["linear", "piecewise"], dest="k_mode")
    p.add_argument("--k-start", type=float, default=1.0)
    p.add_argument("--k-end", type=float, default=2.0)
    p.add_argument("--k-nsegs", "--k_nsegs", type=int, default=4, dest="k_nsegs")
    p.add_argument("--k-jitter", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_argparser().parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = EX2Config(
        device=device,
        output_dir=args.output_dir,
        steps=args.steps,
        lr=args.lr,
        seq_len=args.seq_len,
        K=args.K,
        use_multigate=(not args.single_gate),
        f0=args.f0,
        k_mode=args.k_mode,
        k_start=args.k_start,
        k_end=args.k_end,
        k_nsegs=args.k_nsegs,
        k_jitter=args.k_jitter,
        log_every=args.log_every,
        seed=args.seed,
    )
    
    run_ex2(cfg)


if __name__ == "__main__":
    main()
