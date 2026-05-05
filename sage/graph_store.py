"""
Sage 6.0 — Graph Substrate

Lean knowledge store that maps token/concept IDs directly to core_dim
vectors. At scale, this would be memory-mapped from NVMe with only
the active subgraph loaded into VRAM.
"""

__all__ = ["GraphSubstrate"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import SageConfig


class GraphSubstrate(nn.Module):
    """
    Lean knowledge store. Embedding table sized to vocab + concept buffer.
    At scale: memory-map from NVMe, only load active nodes.
    """

    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config
        actual_n_nodes = min(config.n_nodes, config.text_vocab_size + config.concept_buffer)
        self.node_embeddings = nn.Embedding(actual_n_nodes, config.core_dim)
        nn.init.normal_(self.node_embeddings.weight, std=config.init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.clamp(0, self.node_embeddings.num_embeddings - 1)
        embs = self.node_embeddings(ids)
        if self.config.normalize_embeddings:
            embs = F.normalize(embs, dim=-1)
        return embs

    def retrieve_by_similarity(
        self, query: torch.Tensor, top_k: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve most similar embeddings by cosine similarity."""
        vocab_weight = self.node_embeddings.weight[:self.config.text_vocab_size]
        all_embs = F.normalize(vocab_weight, dim=-1)
        query_norm = F.normalize(query, dim=-1)
        sims = torch.matmul(query_norm, all_embs.T)
        return torch.topk(sims, min(top_k, sims.shape[-1]), dim=-1)
