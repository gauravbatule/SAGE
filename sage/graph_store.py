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
    Lean knowledge store. Just embeddings, no dead weight.
    At scale: memory-map from NVMe, only load active nodes.
    """

    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config
        self.node_embeddings = nn.Embedding(config.n_nodes, config.core_dim)
        nn.init.normal_(self.node_embeddings.weight, std=config.init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        embs = self.node_embeddings(ids)
        if self.config.normalize_embeddings:
            embs = F.normalize(embs, dim=-1)
        return embs

    def retrieve_by_similarity(
        self, query: torch.Tensor, top_k: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embs = F.normalize(self.node_embeddings.weight, dim=-1)
        query_norm = F.normalize(query, dim=-1)
        sims = torch.matmul(query_norm, all_embs.T)
        return torch.topk(sims, top_k, dim=-1)
