"""
Sage 6.0 — Metacognitive Controller

Brain-inspired self-monitoring module implementing adaptive cognitive resource
allocation. Modeled after the prefrontal cortex's executive function:

  - Easy tokens get fast "reflexive" processing (1 pass, minimal compute)
  - Hard tokens get slow "deliberative" processing (multiple passes, full resonance)
  - The controller introspects on confidence, stagnation, and difficulty

v6.0 changes:
  - Per-token difficulty estimation (not just global mean)
  - Cognitive load routing signal for per-token layer skipping
  - Multi-iteration training support with refinement loss
  - Functional retrieval: when stagnating, generates a query for graph re-retrieval
"""

__all__ = ["MetacognitiveController"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

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

    def assess(
        self,
        current_output: torch.Tensor,
        previous_output: torch.Tensor,
        iteration: int,
    ) -> Dict[str, object]:
        """
        Assess the current reasoning state per-token and globally.

        Returns dict with:
            confidence: Mean confidence score [0, 1].
            per_token_difficulty: (B, L) difficulty scores for routing.
            needs_retrieval: Whether to trigger graph re-retrieval.
            retrieval_vector: Query vector for similarity search.
            estimated_difficulty: Global difficulty estimate [0, 1].
        """
        curr_summary = current_output.mean(dim=1)
        prev_summary = previous_output.mean(dim=1)

        iter_idx = min(iteration, self.config.max_think_iterations - 1)
        # Use torch.as_tensor to avoid boxing a Python int into a new tensor object
        # on every call — this reuses a scalar tensor when possible.
        iter_emb = self.iteration_embed(
            torch.as_tensor(iter_idx, device=current_output.device)
        )
        iter_context = self.iter_proj(iter_emb)
        curr_summary_ctx = curr_summary + iter_context.unsqueeze(0).expand_as(curr_summary)

        confidence = self.confidence_net(curr_summary_ctx).squeeze(-1)

        per_token_difficulty = self.difficulty_head(current_output).squeeze(-1)

        stag_input = torch.cat([curr_summary, prev_summary], dim=-1)
        stagnation = self.stagnation_detector(stag_input).squeeze(-1)

        needs_retrieval = bool(
            (stagnation.mean() > 0.7) and (confidence.mean() < self.config.confidence_threshold)
        )

        retrieval_vector = self.retrieval_query(curr_summary)

        estimated_difficulty = per_token_difficulty.mean().item()

        return {
            "confidence": confidence.mean().item(),
            "per_token_difficulty": per_token_difficulty,
            "needs_retrieval": needs_retrieval,
            "retrieval_vector": retrieval_vector,
            "estimated_difficulty": estimated_difficulty,
        }

    def should_emit(self, confidence: float, iteration: int) -> bool:
        if iteration < self.config.min_think_iterations:
            return False
        if confidence >= self.config.confidence_threshold:
            return True
        if iteration >= self.config.max_think_iterations - 1:
            return True
        return False

    def refinement_loss(
        self,
        logits_prev: torch.Tensor,
        logits_curr: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize iterations that produce WORSE predictions than the previous one.
        Encourages the model to refine rather than diverge across iterations.
        """
        loss_prev = F.cross_entropy(
            logits_prev.view(-1, logits_prev.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction='mean',
        )
        loss_curr = F.cross_entropy(
            logits_curr.view(-1, logits_curr.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction='mean',
        )
        regression = F.relu(loss_curr - loss_prev)
        return regression

    def reset(self):
        pass
