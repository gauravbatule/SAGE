"""
Sage 4.0 — Metacognitive Controller

Self-monitoring module that provides adaptive thinking depth. Easy tokens
get a single reasoning pass; complex tokens iterate up to ``max_think_iterations``
passes through the reasoning core. Includes confidence estimation, stagnation
detection, and retrieval query generation for dynamic graph re-retrieval.
"""

__all__ = ["MetacognitiveController"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import SageConfig


class MetacognitiveController(nn.Module):

    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config

        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(config.core_dim, config.metacog_dim),
            nn.SiLU(),
            nn.Linear(config.metacog_dim, config.metacog_dim),
            nn.SiLU(),
            nn.Linear(config.metacog_dim, 1),
            nn.Sigmoid(),
        )

        # Stagnation detector
        self.stagnation_detector = nn.Sequential(
            nn.Linear(config.core_dim * 2, config.metacog_dim),
            nn.SiLU(),
            nn.Linear(config.metacog_dim, 1),
            nn.Sigmoid(),
        )

        # Iteration embedding (small — metacog_dim, not core_dim)
        self.iteration_embed = nn.Embedding(config.max_think_iterations, config.metacog_dim)

        # Retrieval query generator (project to core_dim for similarity search)
        self.retrieval_query = nn.Linear(config.core_dim, config.core_dim, bias=False)

        self._state_history = []

    def assess(
        self,
        current_output: torch.Tensor,
        previous_output: torch.Tensor,
        iteration: int,
    ) -> Tuple[float, bool, torch.Tensor]:
        B = current_output.shape[0]

        curr_summary = current_output.mean(dim=1)  # (B, core_dim)
        prev_summary = previous_output.mean(dim=1)

        # Add iteration context (project metacog_dim to core_dim via padding)
        iter_idx = min(iteration, self.config.max_think_iterations - 1)
        iter_emb = self.iteration_embed(
            torch.tensor(iter_idx, device=current_output.device)
        )  # (metacog_dim,)

        # Confidence
        confidence = self.confidence_net(curr_summary).squeeze(-1)

        # Stagnation
        stag_input = torch.cat([curr_summary, prev_summary], dim=-1)
        stagnation = self.stagnation_detector(stag_input).squeeze(-1)

        needs_retrieval = bool(
            (stagnation.mean() > 0.7) and (confidence.mean() < self.config.confidence_threshold)
        )

        retrieval_vector = self.retrieval_query(curr_summary)

        return confidence.mean().item(), needs_retrieval, retrieval_vector

    def should_emit(self, confidence: float, iteration: int) -> bool:
        if iteration < self.config.min_think_iterations:
            return False
        if confidence >= self.config.confidence_threshold:
            return True
        if iteration >= self.config.max_think_iterations - 1:
            return True
        return False

    def reset(self):
        self._state_history.clear()
