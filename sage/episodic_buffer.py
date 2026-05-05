"""
Sage 6.0 — Episodic Buffer

Hippocampal short-term episodic memory for rapid, lossless context recall.

The hippocampus stores recent episodes with high fidelity before gradual
consolidation to neocortex. Unlike HebbianResonanceMemory, which compresses
context via outer products and applies exponential decay, the episodic buffer
retains the last N hidden states verbatim — no decay, no compression, no loss.
Retrieval is content-based (cosine similarity), not position-based.

Brain analogy: Hippocampal CA3/CA1 stores recent episodic memories as sparse
distributed patterns. Pattern completion retrieves a stored episode from a
partial cue (the current query). This complements the Hebbian "semantic" memory
with episodic "when/what happened" recall.

Design:
  - Ring buffer of capacity N storing D-dimensional states (no learned weights)
  - O(1) append, O(N) cosine-similarity retrieval
  - Learned injection gate: sigmoid(gate(cat[x, retrieved])) * out_proj(retrieved)
  - Gate initialized small so the buffer starts inert and learns to activate
  - Stored states are detached (write-once, like hippocampus — no gradient flows
    back through the stored episodes, only through the gate and projection)
  - buffer_state kept float32 regardless of AMP dtype for numerical stability
"""

__all__ = ["EpisodicBuffer"]

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicBuffer(nn.Module):
    """
    Hippocampal episodic buffer — stores recent hidden states verbatim.

    Brain basis: The hippocampus stores recent episodes with high fidelity
    before gradual consolidation to neocortex. Unlike Hebbian memory which
    compresses via outer products and decays exponentially, the episodic
    buffer provides exact recall for the last N states.

    During training: populated from left portion of sequence, retrieved for right portion.
    During inference: carried in recurrent_state, appended each step.
    """

    def __init__(self, dim: int, capacity: int = 512):
        super().__init__()
        self.dim = dim
        self.capacity = capacity

        # Learned injection gate: controls how much retrieved context is injected.
        # Initialized small so it starts with minimal effect.
        self.inject_gate = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, 1, bias=False),
        )
        nn.init.constant_(self.inject_gate[-1].weight, 0.01)

        self.out_proj = nn.Linear(dim, dim, bias=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, buffer_state: torch.Tensor, new_states: torch.Tensor) -> torch.Tensor:
        """Append new_states (B, T, D) into the ring buffer (B, N, D).

        Detaches new_states so gradients do not flow through stored episodes.
        When T >= capacity the buffer is replaced entirely with the last N frames.
        """
        new_states = new_states.detach().float()
        B, T, D = new_states.shape
        N = self.capacity

        if T >= N:
            return new_states[:, -N:, :]

        # Roll: drop the oldest T slots, append the new ones.
        return torch.cat([buffer_state[:, T:, :], new_states], dim=1)

    def _empty_buffer(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, self.capacity, self.dim, dtype=torch.float32, device=device)

    def retrieve(
        self,
        query: torch.Tensor,
        buffer_state: torch.Tensor,
        top_k: int = 4,
    ) -> torch.Tensor:
        """
        Retrieve top-k most similar states from buffer via cosine similarity.

        Args:
            query:        (B, D)
            buffer_state: (B, N, D)

        Returns:
            (B, D) weighted combination of top-k states.
        """
        # Normalize both sides for cosine similarity.
        q = F.normalize(query.float(), dim=-1).unsqueeze(1)      # (B, 1, D)
        buf = F.normalize(buffer_state.float(), dim=-1)           # (B, N, D)

        sims = torch.bmm(q, buf.transpose(1, 2)).squeeze(1)       # (B, N)

        k = min(top_k, buffer_state.shape[1])
        topk_sims, topk_idx = sims.topk(k, dim=-1)               # (B, k)

        weights = topk_sims.softmax(dim=-1).unsqueeze(-1)         # (B, k, 1)
        gathered = buffer_state.gather(
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, self.dim),
        )                                                          # (B, k, D)

        return (weights * gathered).sum(dim=1)                    # (B, D)

    def _inject(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        """Apply the gated injection: x + sigmoid(gate(cat[x, retrieved])) * out_proj(retrieved).

        x and retrieved must have the same shape (B, L, D) or (B, 1, D).
        """
        gate = torch.sigmoid(self.inject_gate(torch.cat([x, retrieved], dim=-1)))
        return x + gate * self.out_proj(retrieved)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        buffer_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full sequence during training.

        Args:
            x:            (B, L, D)
            buffer_state: optional (B, N, D) tensor of stored states.

        Strategy: use the first half of the sequence to populate the buffer,
        then retrieve from it for the second half. This teaches the model
        to use episodic recall during training.

        Returns:
            output:           (B, L, D)
            new_buffer_state: (B, N, D)
        """
        B, L, D = x.shape
        orig_dtype = x.dtype
        split = L // 2

        if buffer_state is None:
            buffer_state = self._empty_buffer(B, x.device)

        # Populate from the left half; detach inside _append.
        new_buffer_state = self._append(buffer_state, x[:, :split, :])

        # Determine which positions actually have content (non-zero rows).
        has_content = new_buffer_state.abs().sum(dim=-1).gt(0).any(dim=-1)  # (B,)

        # Right half retrieves from the populated buffer.
        right = x[:, split:, :]                                   # (B, R, D)
        R = L - split

        if has_content.any():
            queries = right.reshape(B * R, D)                     # (B*R, D)
            buf_expanded = new_buffer_state.unsqueeze(1).expand(-1, R, -1, -1).reshape(B * R, self.capacity, D)
            retrieved_flat = self.retrieve(queries, buf_expanded)  # (B*R, D)
            retrieved = retrieved_flat.reshape(B, R, D).to(orig_dtype)

            # Zero out retrieved context for batch items whose buffer is empty.
            if not has_content.all():
                mask = has_content.view(B, 1, 1).to(orig_dtype)
                retrieved = retrieved * mask

            right_out = self._inject(right, retrieved)
        else:
            right_out = right

        output = torch.cat([x[:, :split, :], right_out], dim=1)
        return output, new_buffer_state

    def forward_step(
        self,
        x_step: torch.Tensor,
        buffer_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-token inference step.

        Args:
            x_step:       (B, 1, D)
            buffer_state: (B, N, D) or None

        Returns:
            output:           (B, 1, D)
            new_buffer_state: (B, N, D)
        """
        B, _, D = x_step.shape
        orig_dtype = x_step.dtype

        if buffer_state is None:
            buffer_state = self._empty_buffer(B, x_step.device)

        has_content = buffer_state.abs().sum(dim=-1).gt(0).any(dim=-1)  # (B,)

        if has_content.any():
            query = x_step[:, 0, :]                               # (B, D)
            retrieved = self.retrieve(query, buffer_state).to(orig_dtype).unsqueeze(1)  # (B, 1, D)

            if not has_content.all():
                mask = has_content.view(B, 1, 1).to(orig_dtype)
                retrieved = retrieved * mask

            output = self._inject(x_step, retrieved)
        else:
            output = x_step

        # Append current step AFTER retrieval (hippocampus stores then replays).
        new_buffer_state = self._append(buffer_state, x_step)

        return output, new_buffer_state
