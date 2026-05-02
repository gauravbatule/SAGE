"""
Sage 5.0 — Sensory Cortex

Multimodal input grounding. Maps raw inputs (text token IDs, vision patches,
audio frames) into the unified core_dim embedding space for downstream processing
by the reasoning core.
"""

__all__ = ["SensoryCortex"]

import warnings

import torch
import torch.nn as nn
from typing import List, Tuple

from .config import SageConfig


class SensoryCortex(nn.Module):

    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config

        self.vision_proj = nn.Sequential(
            nn.Linear(config.vision_patch_dim, config.core_dim),
            nn.SiLU(),
            nn.Linear(config.core_dim, config.core_dim),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_frame_dim, config.core_dim),
            nn.SiLU(),
            nn.Linear(config.core_dim, config.core_dim),
        )

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

    @staticmethod
    def merge_modalities(
        modality_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_ids = torch.cat([m[0] for m in modality_outputs], dim=1)
        all_energies = torch.cat([m[1] for m in modality_outputs], dim=1)
        all_positions = torch.cat([m[2] for m in modality_outputs], dim=1)
        return all_ids, all_energies, all_positions
