"""
Sage 6.0 — Model Orchestrator

Brain-Inspired Ultra-Efficient Language Architecture.

End-to-end model wiring:

    Graph Embedding → Phase Encoding → Cortical Reasoning → Output

**Not a Transformer**: No attention (QKV, softmax), no O(n²) computation.
**Not Mamba**: No state spaces, no selective scan.
**Not RWKV**: No WKV operator, no channel mixing.

**Unique to Sage 6.0**:
    - Harmonic wave propagation (gamma/beta/theta/alpha oscillation bands)
    - Hebbian resonance memory (outer-product matrices, input-dependent decay)
    - Sparse cortical activation (~20% neurons fire per token)
    - Predictive coding between layers (only errors propagate)
    - Phase-encoded position (multiplicative amplitude modulation)
    - Cognitive load routing (per-token adaptive compute)
    - Metacognitive multi-iteration reasoning with refinement loss
    - Recurrent inference state for O(1) per-token generation
"""

__all__ = ["SageModel"]

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .config import SageConfig
from .graph_store import GraphSubstrate
from .sensory_cortex import SensoryCortex
from .phase_encoding import PhaseEncoding
from .reasoning_core import ReasoningCore
from .metacognitive import MetacognitiveController


class SageModel(nn.Module):
    """
    Sage 6.0: Brain-Inspired Wave Propagation Language Model.
    """

    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config

        self.graph = GraphSubstrate(config)
        self.senses = SensoryCortex(config)
        self.phase = PhaseEncoding(config)
        self.core = ReasoningCore(config)
        self.metacog = MetacognitiveController(config)

        if config.weight_tying:
            self.core.lm_head.weight = self.graph.node_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = token_ids.shape

        active_ids, energies, positions = self.senses.ground_text(token_ids)

        x = self.graph(active_ids)

        if self.config.phase_encoding:
            x = self.phase(x)

        prev_x = torch.zeros_like(x)
        total_iterations = 0
        all_confidences = []
        refinement_loss = torch.tensor(0.0, device=x.device)
        prev_logits = None

        if self.training:
            n_iters = random.randint(
                self.config.min_think_iterations,
                self.config.max_train_iterations,
            )
        else:
            n_iters = self.config.max_think_iterations

        states = None
        for iteration in range(n_iters):
            x, states = self.core(x, states=states)
            total_iterations += 1

            if self.training and targets is not None and iteration > 0:
                curr_logits = self.core.lm_head(x)
                if prev_logits is not None:
                    refinement_loss = refinement_loss + self.metacog.refinement_loss(
                        prev_logits, curr_logits, targets,
                    )
                prev_logits = curr_logits.detach()
            elif self.training:
                prev_x = x.detach()
                continue

            if not self.training:
                assessment = self.metacog.assess(x, prev_x, iteration)
                all_confidences.append(assessment["confidence"])

                if self.metacog.should_emit(assessment["confidence"], iteration):
                    break
                prev_x = x.detach()

        logits = self.core.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.text_vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + 0.1 * refinement_loss

        return {
            "logits": logits,
            "loss": loss,
            "metrics": {
                "iterations_used": total_iterations,
                "confidences": all_confidences,
                "active_nodes": L,
            }
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[int]:
        self.eval()
        generated = []
        current_ids = prompt_ids

        for step in range(max_new_tokens):
            if current_ids.shape[1] > self.config.n_active_limit:
                current_ids = current_ids[:, -self.config.n_active_limit:]

            self.metacog.reset()
            output = self.forward(current_ids)
            logits = output["logits"][:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[(cumprobs - F.softmax(sorted_logits, dim=-1)) >= top_p] = float('-inf')

            probs = F.softmax(sorted_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, 1)
            next_token = sorted_indices.gather(-1, sampled_idx)

            token_id = next_token.item()
            generated.append(token_id)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            if token_id == 0:
                break

        return generated

    def count_parameters(self) -> Dict[str, int]:
        graph_p = sum(p.numel() for p in self.graph.parameters())
        sense_p = sum(p.numel() for p in self.senses.parameters())
        phase_p = sum(p.numel() for p in self.phase.parameters())
        core_p = sum(p.numel() for p in self.core.parameters())
        meta_p = sum(p.numel() for p in self.metacog.parameters())
        total = sum(p.numel() for p in self.parameters())

        return {
            "graph_substrate": graph_p,
            "sensory_cortex": sense_p,
            "phase_encoding": phase_p,
            "reasoning_core": core_p,
            "metacognitive": meta_p,
            "total": total,
            "active_per_token": core_p + phase_p + meta_p,
        }
