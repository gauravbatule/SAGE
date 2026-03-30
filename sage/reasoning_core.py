"""
Reasoning Core v4.0 — Wave Propagation + Resonance Memory

"Maybe Attention Is Not All You Need"

This is a fundamentally new sequence processing architecture that
replaces attention with TWO complementary mechanisms:

  1. CAUSAL WAVE PROPAGATION (local understanding)
     Multi-scale causal convolutions that capture syntax, grammar,
     and local phrase structure. Like how waves propagate through
     a medium — information flows causally, each position integrates
     signals from its local neighborhood.

  2. RESONANCE MEMORY (global understanding)
     A shared "neural whiteboard" with K memory slots. Each position
     WRITES important information to the whiteboard and READS context
     it needs. The whiteboard accumulates causally — position i's
     memory contains a compressed summary of ALL positions 0..i.
     This gives GLOBAL context access without O(n^2) attention.

WHY THIS IS NOT ATTENTION:
  - Attention: O(n^2) pairwise dot products between ALL positions,
    followed by softmax over sequence length, producing weighted
    value combinations. Every token explicitly compares to every other.
  - Resonance Memory: O(n*K*D) where K is memory slots (64-128).
    Tokens write to and read from a SHARED COMPRESSED MEMORY.
    No pairwise comparison. No softmax over sequence length.
    More like a neural RAM than a lookup table.

WHY THIS IS NOT MAMBA:
  - Mamba: Selective State Space Model with input-dependent transitions.
    Linear recurrence with selective scan. State evolves sequentially.
  - Resonance: No recurrence. Uses cumulative sum (parallelizable on GPU).
    No state space formulation. No selective scan operator.

WHY THIS IS NOT RWKV/LINEAR ATTENTION:
  - Linear attention approximates softmax(QK^T)V as Q(K^T V).
    It's a mathematical reformulation of attention.
  - Resonance uses explicit WRITE/READ operations with separate
    projections. It's a memory system, not a factored attention.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x_t = x.transpose(1, 2)                    # (B, D, L)
        x_t = F.pad(x_t, (self.pad, 0))            # causal pad left
        x_t = self.dw_conv(x_t)                     # (B, D, L)
        return self.pw_conv(x_t.transpose(1, 2))    # (B, L, D)


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
    A globally shared neural whiteboard.

    Mechanism:
      1. Write: Positions project state into K slots with importance weights.
      2. Accumulate: Memory is built causally via O(n) cumulative sum.
      3. Read: Queries retrieve relevant context from accumulated state.
      4. Gate: Controls mixture of retrieved context and current state.
      3. READ: Each position generates a read query and retrieves
         relevant information from its accumulated memory state.
      4. GATE: A learned gate controls how much retrieved context
         mixes with the current representation.

    """
    def __init__(self, dim: int, n_slots: int = 16, mem_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.mem_dim = mem_dim

        # Write: select slots + compress value to small dim
        self.write_key = nn.Linear(dim, n_slots, bias=False)
        self.write_value = nn.Linear(dim, mem_dim, bias=False)

        # Read: select slots + decompress back to full dim
        self.read_key = nn.Linear(dim, n_slots, bias=False)
        self.read_expand = nn.Linear(mem_dim, dim, bias=False)

        # Output gate
        self.gate = nn.Linear(dim * 2, dim, bias=False)
        self.mem_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: (B, L, D) — enriched with global context

        Memory: (B, L, K, mem_dim) — with K=16, mem_dim=32, this is
        only ~8MB per layer. Fully vectorized via cumsum.
        """
        B, L, D = x.shape

        wk = F.softmax(self.write_key(x), dim=-1)           # (B, L, K)
        wv = self.write_value(x)                            # (B, L, mem_dim)

        mem_updates = wk.unsqueeze(-1) * wv.unsqueeze(-2)   # (B, L, K, mem_dim)

        # Causal accumulation
        mem_state = torch.cumsum(mem_updates, dim=1)        # (B, L, K, mem_dim)

        rk = F.softmax(self.read_key(x), dim=-1)            # (B, L, K)
        retrieved_compressed = (rk.unsqueeze(-1) * mem_state).sum(dim=-2)

        retrieved = self.read_expand(retrieved_compressed)
        retrieved = self.mem_norm(retrieved)

        gate = torch.sigmoid(self.gate(torch.cat([x, retrieved], dim=-1)))
        return x + gate * retrieved


class WaveBlock(nn.Module):
    """
    Sage reasoning stack block.
    Flow: input -> [WaveMixer] -> [ResonanceMemory] -> [SwiGLU] -> output
    """
    def __init__(self, config: SageConfig, layer_idx: int = 0):
        super().__init__()
        dim = config.core_dim

        # Adaptive kernel sizes per layer (deeper = wider receptive field)
        depth_ratio = layer_idx / max(config.core_n_layers - 1, 1)
        short_k = 3
        mid_k = 5 + int(depth_ratio * 12)
        long_k = 11 + int(depth_ratio * 32)
        mid_k = mid_k | 1   # ensure odd
        long_k = long_k | 1

        # 1. Local causality
        self.wave_norm = RMSNorm(dim)
        self.wave = WaveMixer(dim, short_k, mid_k, long_k)

        # 2. Global context
        n_slots = 16
        mem_dim = 32
        self.resonance_norm = RMSNorm(dim)
        self.resonance = ResonanceMemory(dim, n_slots=n_slots, mem_dim=mem_dim)

        # 3. Position-wise mapping
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, config.core_mlp_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.wave(self.wave_norm(x))
        x = x + self.resonance(self.resonance_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ReasoningCore(nn.Module):
    """
    Sage 4.0 Reasoning Core: Wave Propagation + Resonance Memory

    "Maybe Attention Is Not All You Need"

    This replaces the Transformer's attention mechanism with:
    1. Multi-scale causal convolutions (local patterns, O(n*k))
    2. Resonance memory (global context, O(n*K*D))

    Combined: O(n * (k + K*D)) per layer — strictly linear in sequence length.
    No attention. No recurrence. No state spaces.
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
