"""
Reasoning Core v5.0 — Wave Propagation + Resonance Memory

"Maybe Attention Is Not All You Need"

This is a fundamentally new sequence processing architecture that
replaces attention with TWO complementary mechanisms:

  1. CAUSAL WAVE PROPAGATION (local understanding)
     Multi-scale causal convolutions that capture syntax, grammar,
     and local phrase structure. Information flows causally through
     the sequence with each position integrating signals from its
     local neighborhood at multiple scales.

  2. RESONANCE MEMORY (global understanding)
     A shared neural whiteboard with K memory slots. Each position
     WRITES important information and READS relevant context via
     cumulative accumulation with exponential decay. This gives
     global context access in O(n·K·D) — linear in sequence length.

v5.0 additions:
  - Exponential decay in resonance memory (configurable via config)
  - Dropout after each sub-block
  - Per-layer learnable scaling (layer-scale initialization)
  - Gradient checkpointing support
  - Configurable resonance slots and memory dimension
"""

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "CausalConv1d",
    "WaveMixer",
    "ResonanceMemory",
    "WaveBlock",
    "ReasoningCore",
]

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import SageConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# =====================================================================
# MECHANISM 1: CAUSAL WAVE PROPAGATION (local understanding)
# =====================================================================

class CausalConv1d(nn.Module):
    """Causal 1D depthwise-separable convolution — position i sees only j <= i."""
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=0, groups=dim, bias=True)
        self.pw_conv = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.pad, 0))
        x_t = self.dw_conv(x_t)
        return self.pw_conv(x_t.transpose(1, 2))


class WaveMixer(nn.Module):
    """Multi-scale causal convolutions for local pattern extraction."""
    def __init__(self, dim: int, short_k: int = 3, mid_k: int = 11, long_k: int = 31):
        super().__init__()
        self.d_short = dim // 3
        self.d_mid = dim // 3
        self.d_long = dim - self.d_short - self.d_mid

        self.conv_short = CausalConv1d(self.d_short, short_k)
        self.conv_mid = CausalConv1d(self.d_mid, mid_k)
        self.conv_long = CausalConv1d(self.d_long, long_k)

        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_s = self.conv_short(x[..., :self.d_short])
        y_m = self.conv_mid(x[..., self.d_short:self.d_short + self.d_mid])
        y_l = self.conv_long(x[..., self.d_short + self.d_mid:])
        y = torch.cat([y_s, y_m, y_l], dim=-1)
        return torch.sigmoid(self.gate_proj(y)) * self.value_proj(y)


# =====================================================================
# MECHANISM 2: RESONANCE MEMORY (global understanding)
# =====================================================================

class ResonanceMemory(nn.Module):
    """
    A globally shared neural whiteboard with exponential decay.

    Mechanism:
      1. WRITE: Positions project state into K slots with importance weights.
      2. ACCUMULATE: Memory is built causally via cumsum with exponential decay.
         Older writes decay by factor decay^distance, preventing stale context.
      3. READ: Each position generates a read query and retrieves relevant
         information from its accumulated memory state.
      4. GATE: A learned gate controls how much retrieved context mixes
         with the current representation.
    """
    def __init__(self, dim: int, n_slots: int = 32, mem_dim: int = 64, decay: float = 0.999):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.mem_dim = mem_dim
        self.decay = decay

        self.write_key = nn.Linear(dim, n_slots, bias=False)
        self.write_value = nn.Linear(dim, mem_dim, bias=False)

        self.read_key = nn.Linear(dim, n_slots, bias=False)
        self.read_expand = nn.Linear(mem_dim, dim, bias=False)

        self.gate = nn.Linear(dim * 2, dim, bias=False)
        self.mem_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        wk = F.softmax(self.write_key(x), dim=-1)
        wv = self.write_value(x)

        mem_updates = wk.unsqueeze(-1) * wv.unsqueeze(-2)  # (B, L, K, mem_dim)

        if self.decay < 1.0:
            steps = torch.arange(L, device=x.device, dtype=x.dtype)
            decay_weights = (self.decay ** (L - 1 - steps)).view(1, L, 1, 1)
            mem_updates = mem_updates * decay_weights

        mem_state = torch.cumsum(mem_updates, dim=1)

        rk = F.softmax(self.read_key(x), dim=-1)
        retrieved_compressed = (rk.unsqueeze(-1) * mem_state).sum(dim=-2)

        retrieved = self.read_expand(retrieved_compressed)
        retrieved = self.mem_norm(retrieved)

        gate = torch.sigmoid(self.gate(torch.cat([x, retrieved], dim=-1)))
        return x + gate * retrieved


class WaveBlock(nn.Module):
    """
    Sage reasoning stack block.
    Flow: input -> [WaveMixer] -> [ResonanceMemory] -> [SwiGLU] -> output

    v5.0: dropout, layer-scale, gradient checkpointing support.
    """
    def __init__(self, config: SageConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        dim = config.core_dim

        depth_ratio = layer_idx / max(config.core_n_layers - 1, 1)
        short_k = 3
        mid_k = 5 + int(depth_ratio * 12)
        long_k = 11 + int(depth_ratio * 32)
        mid_k = mid_k | 1
        long_k = long_k | 1

        self.wave_norm = RMSNorm(dim)
        self.wave = WaveMixer(dim, short_k, mid_k, long_k)

        self.resonance_norm = RMSNorm(dim)
        self.resonance = ResonanceMemory(
            dim,
            n_slots=config.resonance_slots,
            mem_dim=config.resonance_mem_dim,
            decay=config.resonance_decay,
        )

        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, config.core_mlp_dim)

        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.wave_scale = nn.Parameter(torch.full((), config.layer_scale_init))
        self.resonance_scale = nn.Parameter(torch.full((), config.layer_scale_init))
        self.mlp_scale = nn.Parameter(torch.full((), config.layer_scale_init))

    def _wave_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.wave(self.wave_norm(x))) * self.wave_scale

    def _resonance_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.resonance(self.resonance_norm(x))) * self.resonance_scale

    def _mlp_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.mlp(self.mlp_norm(x))) * self.mlp_scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            x = x + grad_checkpoint(self._wave_fn, x, use_reentrant=False)
            x = x + grad_checkpoint(self._resonance_fn, x, use_reentrant=False)
            x = x + grad_checkpoint(self._mlp_fn, x, use_reentrant=False)
        else:
            x = x + self._wave_fn(x)
            x = x + self._resonance_fn(x)
            x = x + self._mlp_fn(x)
        return x


class ReasoningCore(nn.Module):
    """
    Sage 5.0 Reasoning Core: Wave Propagation + Resonance Memory

    Replaces the Transformer's attention mechanism with:
    1. Multi-scale causal convolutions (local patterns, O(n*k))
    2. Resonance memory with decay (global context, O(n*K*D))

    Combined: O(n * (k + K*D)) per layer — strictly linear in sequence length.
    """
    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            WaveBlock(config, layer_idx=i)
            for i in range(config.core_n_layers)
        ])

        self.output_norm = RMSNorm(config.core_dim)
        self.lm_head = nn.Linear(config.core_dim, config.text_vocab_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.output_norm(x)
