"""
Sage 4.0 — Temporal Binding

Pass-through module retained for API compatibility.

In Sage's architecture, position is implicit in the causal convolution
structure — position *i* can only receive signals from positions *j ≤ i*.
No explicit positional encoding is needed, unlike Transformers (learned/RoPE)
or Mamba (hidden state recurrence). This eliminates learned position tables
that cap context length and avoids RoPE computation overhead.

This module is preserved for future cross-modal binding experiments.
"""

__all__ = ["TemporalBinding"]

import torch
import torch.nn as nn

from .config import SageConfig


class TemporalBinding(nn.Module):
    """Position is implicit in causal convolutions. Clean pass-through."""

    def __init__(self, config: SageConfig):
        super().__init__()

    def bind(self, concept_embeddings: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return concept_embeddings
