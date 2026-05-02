"""
Sage 6.0 — Phase Encoding

Brain-inspired positional encoding based on hippocampal theta phase precession.

In the hippocampus, neurons encode sequential position through phase shifts
relative to ongoing theta oscillations — a neuron fires at progressively earlier
phases as an animal moves through its place field. This module applies the same
principle: position is encoded as a MULTIPLICATIVE amplitude modulation of each
channel, not as an additive embedding.

Key difference from RoPE/sinusoidal PE:
  - RoPE rotates pairs of dimensions (complex multiplication)
  - Sinusoidal PE adds a position-dependent vector
  - Phase encoding MODULATES the amplitude: x = x * (1 + alpha * phase(pos))

This preserves the magnitude structure of the embeddings while injecting
position information through amplitude variation, like how neural firing
rates are modulated by oscillatory phase.
"""

__all__ = ["PhaseEncoding"]

import math

import torch
import torch.nn as nn

from .config import SageConfig


class PhaseEncoding(nn.Module):
    def __init__(self, config: SageConfig):
        super().__init__()
        self.dim = config.core_dim
        self.alpha = nn.Parameter(torch.full((), 0.1))

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, config.core_dim, 2).float() / config.core_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device, dtype=x.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        phase = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        if phase.shape[-1] > D:
            phase = phase[:, :D]
        elif phase.shape[-1] < D:
            phase = torch.cat([phase, phase[:, :D - phase.shape[-1]]], dim=-1)
        return x * (1.0 + self.alpha * phase.unsqueeze(0))
