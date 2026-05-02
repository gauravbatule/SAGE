"""
Sage 5.0 — Metacognitive Controller

Self-monitoring module that provides adaptive thinking depth. Easy tokens
get a single reasoning pass; complex tokens iterate up to ``max_think_iterations``
passes through the reasoning core. Includes confidence estimation, stagnation
detection, difficulty prediction, and retrieval query generation.

v5.0: Iteration embeddings are now actually used via learned projection.
Added difficulty estimator for predicting needed iterations.
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

        self.confidence_net = nn.Sequential(
            nn.Linear(config.core_dim, config.metacog_dim),
            nn.SiLU(),
            nn.Linear(config.metacog_dim, config.metacog_dim),
            nn.SiLU(),
            nn.Linear(config.metacog_dim, 1),
            nn.Sigmoid(),
        )

        self.stagnation_detector = nn.Sequential(
            nn.Linear(config.core_dim * 2, config.metacog_dim),
            nn.SiLU(),
            nn.Linear(config.metacog_dim, 1),
            nn.Sigmoid(),
        )

        self.iteration_embed = nn.Embedding(config.max_think_iterations, config.metacog_dim)
        self.iter_proj = nn.Linear(config.metacog_dim, config.core_dim, bias=False)

        self.difficulty_head = nn.Sequential(
            nn.Linear(config.core_dim, config.metacog_dim),
            nn.SiLU(),
            nn.Linear(config.metacog_dim, 1),
            nn.Sigmoid(),
        )

        self.retrieval_query = nn.Linear(config.core_dim, config.core_dim, bias=False)

        self._state_history = []

    def assess(
        self,
        current_output: torch.Tensor,
        previous_output: torch.Tensor,
        iteration: int,
    ) -> Tuple[float, bool, torch.Tensor, float]:
        """
        Assess the current reasoning state.

        Returns:
            confidence: Mean confidence score [0, 1].
            needs_retrieval: Whether to trigger graph re-retrieval.
            retrieval_vector: Query vector for similarity search.
            estimated_difficulty: Predicted difficulty [0, 1] (higher = harder).
        """
        curr_summary = current_output.mean(dim=1)
        prev_summary = previous_output.mean(dim=1)

        iter_idx = min(iteration, self.config.max_think_iterations - 1)
        iter_emb = self.iteration_embed(
            torch.tensor(iter_idx, device=current_output.device)
        )
        iter_context = self.iter_proj(iter_emb)
        curr_summary_ctx = curr_summary + iter_context.unsqueeze(0).expand_as(curr_summary)

        confidence = self.confidence_net(curr_summary_ctx).squeeze(-1)

        stag_input = torch.cat([curr_summary, prev_summary], dim=-1)
        stagnation = self.stagnation_detector(stag_input).squeeze(-1)

        needs_retrieval = bool(
            (stagnation.mean() > 0.7) and (confidence.mean() < self.config.confidence_threshold)
        )

        retrieval_vector = self.retrieval_query(curr_summary)

        estimated_difficulty = self.difficulty_head(curr_summary).squeeze(-1).mean().item()

        return confidence.mean().item(), needs_retrieval, retrieval_vector, estimated_difficulty

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
