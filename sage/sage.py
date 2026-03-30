"""
Sage 4.0 — Model Orchestrator

End-to-end model that wires together the Sage architecture:

    Graph Embedding → Wave Propagation → Metacognitive Control → Output

**Not a Transformer**: No attention (QKV, softmax), no O(n²) computation.
Position is implicit in causal convolutions — no positional encoding needed.

**Not Mamba**: No state spaces, no recurrence, no selective scan.

**Unique to Sage**:
    - Graph-based knowledge substrate (scalable to billions on NVMe)
    - Multi-scale causal wave propagation (conv-based sequence mixing)
    - Resonance memory (compressed global context via cumulative write/read)
    - Metacognitive iterative reasoning (adaptive depth per token)
    - Weight tying (embedding == output head)
"""

__all__ = ["SageModel"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .config import SageConfig
from .graph_store import GraphSubstrate
from .sensory_cortex import SensoryCortex
from .temporal_binding import TemporalBinding
from .reasoning_core import ReasoningCore
from .metacognitive import MetacognitiveController


class SageModel(nn.Module):
    """
    Sage 4.0: Wave Propagation Language Model with Metacognitive Control.
    """

    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config

        # Components
        self.graph = GraphSubstrate(config)
        self.senses = SensoryCortex(config)
        self.binder = TemporalBinding(config)
        self.core = ReasoningCore(config)
        self.metacog = MetacognitiveController(config)

        # Weight tying: embedding and lm_head share weights
        if config.weight_tying:
            self.core.lm_head.weight = self.graph.node_embeddings.weight

        # Initialize
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
        vision_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = token_ids.shape
        device = token_ids.device

        # STEP 1: Ground inputs → node IDs
        active_ids, energies, positions = self.senses.ground_text(token_ids)

        # STEP 2: Graph embedding (direct to core_dim)
        x = self.graph.node_embeddings(active_ids)  # (B, L, core_dim)

        # STEP 3: Wave Propagation Reasoning
        # No causal mask needed — causality is built into the convolutions!
        # Training: 1 iteration (efficient). Inference: adaptive via metacog.
        prev_x = torch.zeros_like(x)
        total_iterations = 0
        all_confidences = []

        n_iters = self.config.min_think_iterations if self.training else self.config.max_think_iterations
        for iteration in range(n_iters):
            x = self.core(x)  # No mask needed — waves are inherently causal
            total_iterations += 1

            if self.training:
                prev_x = x.detach()
                continue

            # Inference: metacognitive early exit
            conf, needs_retrieval, _ = self.metacog.assess(x, prev_x, iteration)
            all_confidences.append(conf)

            if self.metacog.should_emit(conf, iteration):
                break
            prev_x = x.detach()

        # STEP 4: Output logits
        logits = self.core.lm_head(x)  # (B, L, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.text_vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

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
        bind_p = sum(p.numel() for p in self.binder.parameters())
        core_p = sum(p.numel() for p in self.core.parameters())
        meta_p = sum(p.numel() for p in self.metacog.parameters())
        total = sum(p.numel() for p in self.parameters())

        return {
            "graph_substrate": graph_p,
            "sensory_cortex": sense_p,
            "temporal_binding": bind_p,
            "reasoning_core": core_p,
            "metacognitive": meta_p,
            "total": total,
            "active_per_token": core_p + bind_p + meta_p,
        }
