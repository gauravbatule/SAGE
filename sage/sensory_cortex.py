"""
Sage 6.0 — Sensory Cortex

Maps raw text token IDs into node IDs for the graph substrate.
Handles out-of-vocabulary detection with runtime warnings.
"""

__all__ = ["SensoryCortex"]

import warnings

import torch
import torch.nn as nn
from typing import Tuple

from .config import SageConfig


class SensoryCortex(nn.Module):

    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config

    def ground_text(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = token_ids.shape
        oov_mask = (token_ids < 0) | (token_ids >= self.config.text_vocab_size)
        if oov_mask.any():
            n_oov = int(oov_mask.sum().item())
            warnings.warn(
                f"SensoryCortex.ground_text: {n_oov} token ID(s) out of range "
                f"[0, {self.config.text_vocab_size}). Clamping to valid range.",
                RuntimeWarning,
                stacklevel=2,
            )
        node_ids = token_ids.clamp(0, self.config.text_vocab_size - 1)
        energies = torch.ones(B, L, device=token_ids.device)
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        return node_ids, energies, positions
