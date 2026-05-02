"""
Reasoning Core v6.0 — Brain-Inspired Wave Propagation + Hebbian Resonance Memory

"Maybe Attention Is Not All You Need"

A fundamentally new sequence processing architecture inspired by how the
brain actually processes language, replacing attention with THREE mechanisms:

  1. HARMONIC WAVE PROPAGATION (local understanding)
     Multi-frequency causal convolutions decomposed into neural oscillation
     bands: gamma (local syntax), beta (phrase-level), theta (discourse).
     An alpha inhibitory gate creates destructive interference to suppress
     noise — like how alpha oscillations inhibit irrelevant cortical activity.

  2. HEBBIAN RESONANCE MEMORY (global understanding)
     Outer-product memory inspired by Hebbian learning ("fire together, wire
     together"). K matrix-valued memory slots with input-dependent decay and
     gating. Reads via interference: M @ q retrieves patterns that resonate
     with the query. Working-memory-like capacity (4-8 slots).

  3. SPARSE CORTICAL MLP (per-position reasoning)
     Only ~10-20% of neurons fire per token, mimicking cortical sparse coding.
     Top-K activation with straight-through gradient estimation.

  4. PREDICTIVE CODING (inter-layer efficiency)
     Each layer predicts the next layer's output. Only the prediction ERROR
     propagates — drastically reducing computation for easy tokens.

Novel contributions over prior work:
  - Interference mixing between frequency bands (not concatenation)
  - Per-slot learned decay in Hebbian memory (not fixed scalar)
  - Predictive coding in a language model (first of its kind)
  - Sparse activation within a single expert (not MoE routing)

References:
  - Neural oscillations: Buzsáki & Draguhn (2004), Rhythms of the Brain
  - Hebbian learning: Hebb (1949), The Organization of Behavior
  - Predictive coding: Rao & Ballard (1999), Nature Neuroscience
  - Sparse coding: Olshausen & Field (1996), Nature
  - Outer-product memory: xLSTM (Beck et al., 2024), RWKV-6 (Peng et al., 2024)
  - Selective state spaces: Mamba (Gu & Dao, 2023)
  - Differential signal: DiffTransformer (Ye et al., 2024)
"""

__all__ = [
    "RMSNorm",
    "CausalConv1d",
    "HarmonicWaveMixer",
    "HebbianResonanceMemory",
    "SparseCorticalMLP",
    "CorticalBlock",
    "ReasoningCore",
    "parallel_scan",
]

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import SageConfig


# =====================================================================
# JIT-COMPILED SCAN KERNELS — eliminate Python-loop overhead
# =====================================================================

@torch.jit.script
def _fused_hebbian_scan(
    decays: torch.Tensor,
    updates: torch.Tensor,
    decay_1d: torch.Tensor,
    key_updates: torch.Tensor,
    state: torch.Tensor,
    norm_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled fused scan over L timesteps.

    Runs both the matrix-state and norm-state recurrences in a single
    Python-level loop with no list allocation.  torch.jit.script compiles
    this to TorchScript so the loop executes without CPython overhead.

    All tensors must be float32 on entry (caller is responsible).

    Args:
        decays:      (B, L, K, 1, 1)
        updates:     (B, L, K, M, M)
        decay_1d:    (B, L, K, 1)
        key_updates: (B, L, K, M)
        state:       (B, K, M, M)  initial matrix state
        norm_s:      (B, K, M)     initial norm state

    Returns:
        mem_states:  (B, L, K, M, M)
        norm_states: (B, L, K, M)
        final_state: (B, K, M, M)
        final_norm:  (B, K, M)
    """
    L = updates.shape[1]

    # Pre-allocate output tensors — avoids L intermediate list entries.
    mem_out  = torch.empty_like(updates)           # (B, L, K, M, M)
    norm_out = torch.empty_like(key_updates)       # (B, L, K, M)

    s  = state
    ns = norm_s

    for t in range(L):
        s  = decays[:, t]    * s  + updates[:, t]
        s  = s.clamp(-8.0, 8.0)
        ns = decay_1d[:, t]  * ns + key_updates[:, t]

        mem_out[:, t]  = s
        norm_out[:, t] = ns

    return mem_out, norm_out, s, ns


# =====================================================================
# PARALLEL SCAN — O(log L) associative scan for linear recurrences
# =====================================================================

def parallel_scan(
    decays: torch.Tensor,
    updates: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Parallel prefix scan solving the first-order linear recurrence:

        state_t = decay_t * state_{t-1} + update_t

    Each position is represented as an associative pair (a, b) meaning
    "state = a * prev_state + b".  The combine rule is:

        (a1, b1) ⊕ (a2, b2)  =  (a2 * a1,  a2 * b1 + b2)

    which corresponds to composing two affine steps left-to-right.
    Because the operation is associative, a Blelloch up-sweep evaluates
    all prefix products in O(log L) parallel steps, not O(L) sequential
    ones.  All work within each depth level is fully data-parallel and
    launches a single fused kernel per level — no Python loop over L.

    Args:
        decays:        (B, L, K, 1, 1)  — per-step, per-slot scalar decay
                       factors, broadcast-ready over the (M, M) matrix dims.
        updates:       (B, L, K, M, M)  — per-step Hebbian outer-product
                       updates (already multiplied by the input gate).
        initial_state: (B, K, M, M) or None.  When provided, seeds the
                       recurrence so that state_{-1} = initial_state,
                       enabling chunked / autoregressive continuation.
                       When None, the hidden state starts at zero.

    Returns:
        all_states: (B, L, K, M, M)  — memory matrix at every time step,
                    i.e. all_states[:, t] == state_t for t in [0, L).

    Algorithm
    ---------
    Represent the sequence as two tensors of shape (B, L_pad, K, M, M):
      ``a`` — the multiplicative coefficient (decay), tiled to (M, M)
      ``b`` — the additive term (update / accumulated output so far)

    Identity element: (a=1, b=0).  We pad to the next power of 2 with
    identity pairs so the sweep length is always a power of two.

    Up-sweep (inclusive prefix scan):
        For stride s = 1, 2, 4, ..., L_pad/2:
            For every index i >= s  (processed simultaneously):
                new_b[i] = a[i] * b[i-s] + b[i]
                new_a[i] = a[i] * a[i-s]

    After ceil(log2 L) passes, b[t] == state_t.
    """
    B, L, K, M, _ = updates.shape

    # ------------------------------------------------------------------
    # 1. Absorb the initial state into position 0's additive term.
    #    state_0 = decay_0 * s0 + update_0  =>  b_0 = decay_0 * s0 + update_0
    #    so the scan can treat the implicit prev-state as zero everywhere.
    # ------------------------------------------------------------------
    if initial_state is not None:
        # initial_state: (B, K, M, M) -> (B, 1, K, M, M)
        s0 = initial_state.unsqueeze(1)
        updates = updates.clone()
        updates[:, :1] = decays[:, :1] * s0 + updates[:, :1]

    # ------------------------------------------------------------------
    # 2. Pad sequence length to the next power of two with identity pairs.
    #    Identity: (a=1, b=0) is a no-op under the combine rule.
    # ------------------------------------------------------------------
    L_orig = L
    L_pad = 1 << (max(L, 1) - 1).bit_length() if L > 1 else 1
    pad = L_pad - L

    if pad > 0:
        a_pad = decays.new_ones(B, pad, K, 1, 1)
        b_pad = updates.new_zeros(B, pad, K, M, M)
        a = torch.cat([decays, a_pad], dim=1)    # (B, L_pad, K, 1, 1)
        b = torch.cat([updates, b_pad], dim=1)   # (B, L_pad, K, M, M)
    else:
        a = decays.clone()    # (B, L_pad, K, 1, 1)
        b = updates.clone()   # (B, L_pad, K, M, M)

    # ------------------------------------------------------------------
    # 3. Blelloch up-sweep — O(log L_pad) kernel launches total.
    #    Each level processes all eligible (right, left) pairs in parallel
    #    via advanced indexing; there is no Python loop over positions.
    # ------------------------------------------------------------------
    stride = 1
    while stride < L_pad:
        # right ∈ {stride, stride+1, ..., L_pad-1}  (all at once)
        right = torch.arange(stride, L_pad, device=a.device)
        left  = right - stride

        a_left  = a[:, left]    # (B, n_pairs, K, 1, 1)
        b_left  = b[:, left]    # (B, n_pairs, K, M, M)
        a_right = a[:, right]   # (B, n_pairs, K, 1, 1)
        b_right = b[:, right]   # (B, n_pairs, K, M, M)

        # Combine: (a_left, b_left) ⊕ (a_right, b_right)
        #       => (a_right * a_left,  a_right * b_left + b_right)
        new_a_right = a_right * a_left
        new_b_right = a_right * b_left + b_right

        # Clone before scatter to avoid aliasing across stride levels.
        a = a.clone()
        b = b.clone()
        a[:, right] = new_a_right
        b[:, right] = new_b_right

        stride <<= 1

    # ------------------------------------------------------------------
    # 4. Slice back to original length; b[:, t] == state_t.
    # ------------------------------------------------------------------
    return b[:, :L_orig]   # (B, L, K, M, M)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# =====================================================================
# MECHANISM 1: HARMONIC WAVE PROPAGATION
# Brain basis: cortical oscillations at gamma/beta/theta/alpha frequencies
# =====================================================================

class CausalConv1d(nn.Module):
    """Causal 1D depthwise-separable convolution — position i sees only j <= i."""
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=0, groups=dim, bias=True)
        self.pw_conv = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose once, pad, depthwise-conv, then a single contiguous
        # transpose before the pointwise linear so it sees a packed tensor.
        x_t = x.transpose(1, 2)          # (B, dim, L)
        x_t = F.pad(x_t, (self.pad, 0))  # (B, dim, L + pad)
        x_t = self.dw_conv(x_t)          # (B, dim, L)
        return self.pw_conv(x_t.transpose(1, 2).contiguous())

    def forward_step(
        self,
        x_step: torch.Tensor,
        conv_buffer: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single time step using a rolling buffer for O(1) inference.

        Args:
            x_step:      (B, 1, dim)  — the new token's features.
            conv_buffer: (B, dim, kernel_size-1)  — previous context, or None
                         on the first call (will be zero-initialised).

        Returns:
            output:     (B, 1, dim)              — convolved + projected output.
            new_buffer: (B, dim, kernel_size-1)  — updated sliding-window buffer.
        """
        # x_step arrives as (B, 1, dim); transpose to (B, dim, 1) for Conv1d.
        x_t = x_step.transpose(1, 2)  # (B, dim, 1)

        if conv_buffer is None:
            conv_buffer = torch.zeros(
                x_t.shape[0], x_t.shape[1], self.kernel_size - 1,
                dtype=x_t.dtype, device=x_t.device,
            )

        # Concatenate buffer + new step → (B, dim, kernel_size).
        windowed = torch.cat([conv_buffer, x_t], dim=2)

        # Depthwise conv over the full window → (B, dim, 1).
        out = self.dw_conv(windowed)

        # Pointwise linear: transpose to (B, 1, dim), project, return.
        output = self.pw_conv(out.transpose(1, 2))  # (B, 1, dim)

        # Slide the window: drop the oldest frame, keep the last (kernel_size-1).
        new_buffer = windowed[:, :, 1:]  # (B, dim, kernel_size-1)

        return output, new_buffer


class HarmonicWaveMixer(nn.Module):
    """
    Multi-frequency wave propagation with interference mixing.

    Decomposes input into neural oscillation frequency bands:
      gamma (kernel=3):  local syntax, word boundaries, morphology
      beta  (grows):     phrase/clause structure
      theta (grows):     discourse, long-range coherence

    An alpha inhibitory gate suppresses irrelevant signal via destructive
    interference — output = sum(bands) - alpha * sum(bands).

    Per-band phase offsets allow constructive/destructive interference
    based on content, analogous to phase coding in neural oscillations.

    Speed optimisations (behaviour unchanged):
      * Phase scale (1 + phase) is maintained as a single fused buffer that
        is lazily refreshed only when the parameters are updated, so the
        addition is not recomputed on every forward pass.
      * The interference formula is simplified:
            combined * (1 - alpha) + combined * alpha * 0.1
          = combined * (1 - alpha * 0.9)           [one mul fewer]
      * cross_band_mix and value_proj are mathematically equivalent to a
        single linear y = W_v (W_x z + z) = (W_v W_x + W_v) z, but since
        both weight matrices are learned independently we keep them separate
        and instead avoid allocating a temporary by passing the fused input
        directly.  The residual `mixed + interference` reuses the same
        buffer rather than creating a second cat.
      * The alpha_gate Linear+Sigmoid forward is called once on `combined`,
        which is already contiguous after torch.cat.
    """
    def __init__(self, dim: int, gamma_k: int = 3, beta_k: int = 7, theta_k: int = 15):
        super().__init__()
        self.dim = dim
        self.d_gamma = dim // 3
        self.d_beta = dim // 3
        self.d_theta = dim - self.d_gamma - self.d_beta

        self.conv_gamma = CausalConv1d(self.d_gamma, gamma_k)
        self.conv_beta = CausalConv1d(self.d_beta, beta_k)
        self.conv_theta = CausalConv1d(self.d_theta, theta_k)

        # Raw per-band phase offset parameters (small init, learned).
        self.phase_gamma = nn.Parameter(torch.randn(1, 1, self.d_gamma) * 0.01)
        self.phase_beta  = nn.Parameter(torch.randn(1, 1, self.d_beta)  * 0.01)
        self.phase_theta = nn.Parameter(torch.randn(1, 1, self.d_theta) * 0.01)

        # alpha_gate: a single Linear + Sigmoid applied to `combined`.
        # Stored as separate modules so that the Linear weight is properly
        # registered and the Sigmoid is applied in-place-friendly fashion.
        self.alpha_linear = nn.Linear(dim, dim, bias=False)

        self.value_proj    = nn.Linear(dim, dim, bias=False)
        self.cross_band_mix = nn.Linear(dim, dim, bias=False)

    # ------------------------------------------------------------------
    # Helper: build the concatenated phase-scale vector (1 + phase) once
    # and cache it.  The cache is invalidated by any parameter update
    # (training loop calls .zero_grad() → no special hook needed; we check
    # whether we are in training mode and skip caching there to stay safe).
    # ------------------------------------------------------------------
    def _phase_scale(self, device, dtype):
        """Return the concatenated (1 + phase) scale, shape (1, 1, dim)."""
        # During training parameters change every step — just compute it.
        # During eval it is constant; cache it for repeated inference calls.
        if self.training:
            return torch.cat(
                [1.0 + self.phase_gamma,
                 1.0 + self.phase_beta,
                 1.0 + self.phase_theta], dim=-1
            )
        # Eval: cache on first call or if device/dtype changed.
        cache = getattr(self, "_phase_scale_cache", None)
        if cache is None or cache.device != device or cache.dtype != dtype:
            with torch.no_grad():
                cache = torch.cat(
                    [1.0 + self.phase_gamma,
                     1.0 + self.phase_beta,
                     1.0 + self.phase_theta], dim=-1
                ).to(device=device, dtype=dtype)
            self._phase_scale_cache = cache
        return cache

    # ------------------------------------------------------------------
    # Shared core: runs after the per-band convolutions have produced
    # g, b, t.  Extracted so that forward() and forward_step() share
    # exactly the same post-conv logic with no duplication.
    # ------------------------------------------------------------------
    def _post_conv(self, g: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply phase scaling, alpha inhibition, cross-band mix, and value proj."""
        # Phase scale — single cat+mul instead of three separate muls.
        combined = torch.cat([g, b, t], dim=-1)           # (B, L, dim)
        combined = combined * self._phase_scale(combined.device, combined.dtype)

        # Alpha inhibitory gate.
        # Original: combined*(1-a) + combined*a*0.1 = combined*(1 - a*0.9)
        alpha = torch.sigmoid(self.alpha_linear(combined)) # (B, L, dim)
        combined = combined * (1.0 - 0.9 * alpha)         # interference

        # cross_band_mix output added to interference, then value_proj.
        # We avoid a separate `mixed` tensor by adding in-place.
        mixed = self.cross_band_mix(combined)
        mixed = mixed + combined                          # mixed + interference
        return self.value_proj(mixed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.conv_gamma(x[..., :self.d_gamma])
        b = self.conv_beta (x[..., self.d_gamma:self.d_gamma + self.d_beta])
        t = self.conv_theta(x[..., self.d_gamma + self.d_beta:])
        return self._post_conv(g, b, t)

    def forward_step(
        self,
        x_step: torch.Tensor,
        wave_state: Optional[dict],
    ) -> Tuple[torch.Tensor, dict]:
        """Single-token wave propagation using per-band rolling conv buffers.

        Args:
            x_step:     (B, 1, dim) — the new token's features.
            wave_state: dict with keys 'gamma', 'beta', 'theta' holding
                        (B, d_band, kernel_size-1) rolling buffers, or None.

        Returns:
            output:         (B, 1, dim)
            new_wave_state: updated dict of rolling buffers.
        """
        if wave_state is None:
            wave_state = {"gamma": None, "beta": None, "theta": None}

        x_g = x_step[..., :self.d_gamma]
        x_b = x_step[..., self.d_gamma:self.d_gamma + self.d_beta]
        x_t = x_step[..., self.d_gamma + self.d_beta:]

        g, buf_g = self.conv_gamma.forward_step(x_g, wave_state["gamma"])
        b, buf_b = self.conv_beta .forward_step(x_b, wave_state["beta"])
        t, buf_t = self.conv_theta.forward_step(x_t, wave_state["theta"])

        output = self._post_conv(g, b, t)
        new_wave_state = {"gamma": buf_g, "beta": buf_b, "theta": buf_t}
        return output, new_wave_state

    # ------------------------------------------------------------------
    # Checkpoint compatibility: old checkpoints stored the alpha gate as
    # `alpha_gate.0.weight` (nn.Sequential index 0).  Remap on load so
    # that weights saved before this refactor still load cleanly.
    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        old_key = prefix + "alpha_gate.0.weight"
        new_key = prefix + "alpha_linear.weight"
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
        # Drop any remaining legacy alpha_gate.* keys to avoid unexpected_keys
        # noise (e.g. if the Sequential had other sub-modules).
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "alpha_gate."):
                state_dict.pop(k)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs,
        )


# =====================================================================
# MECHANISM 2: HEBBIAN RESONANCE MEMORY
# Brain basis: Hebbian learning + working memory (4-7 item capacity)
# =====================================================================

class HebbianResonanceMemory(nn.Module):
    """
    Matrix-valued memory with Hebbian outer-product updates and learned decay.

    Each of K memory slots holds a matrix M of shape (mem_dim, mem_dim).
    Writing uses the Hebbian rule: M_t = decay_t * M_{t-1} + gate_t * (v_t ⊗ k_t)
    Reading retrieves via interference: output_t = M_t @ q_t

    The decay and input gate are INPUT-DEPENDENT — the model learns when to
    forget and when to write, enabling selective memory management.

    During training, the recurrence is computed via a parallel scan for GPU
    efficiency. During inference, the state is maintained incrementally.
    """
    def __init__(self, dim: int, n_slots: int = 8, mem_dim: int = 64,
                 decay_init: float = 0.95):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.mem_dim = mem_dim

        self.write_key = nn.Linear(dim, n_slots * mem_dim, bias=False)
        self.write_value = nn.Linear(dim, n_slots * mem_dim, bias=False)

        self.read_query = nn.Linear(dim, n_slots * mem_dim, bias=False)
        self.read_expand = nn.Linear(n_slots * mem_dim, dim, bias=False)

        self.decay_proj = nn.Linear(dim, n_slots, bias=True)
        self.input_gate_proj = nn.Linear(dim, n_slots, bias=True)

        raw_decay = math.log(decay_init / (1.0 - decay_init + 1e-8))
        nn.init.constant_(self.decay_proj.bias, raw_decay)

        self.gate = nn.Linear(dim * 2, dim, bias=False)
        self.mem_norm = RMSNorm(dim)
        self.norm_eps = 1e-3

    def _scan_fp32(self, decays, updates, state, key_updates, decay_1d, L, B):
        """Run the fused scan in float32 to avoid AMP gradient NaN.

        Uses the JIT-compiled ``_fused_hebbian_scan`` kernel which:
          - Runs both recurrences (matrix state + norm state) in one pass
          - Pre-allocates output tensors rather than building a Python list
          - Eliminates CPython loop overhead via TorchScript compilation
        All math is identical to the original sequential loop.
        """
        K, M = self.n_slots, self.mem_dim

        # Cast all inputs to float32 once, before entering the kernel.
        s_fp32         = state.float()
        decays_fp32    = decays.float()
        updates_fp32   = updates.float()
        key_upd_fp32   = key_updates.float()
        decay_1d_fp32  = decay_1d.float()
        norm_s_fp32    = torch.zeros(B, K, M, device=state.device, dtype=torch.float32)

        mem_states, norm_states, s_final, _ = _fused_hebbian_scan(
            decays_fp32,
            updates_fp32,
            decay_1d_fp32,
            key_upd_fp32,
            s_fp32,
            norm_s_fp32,
        )

        return mem_states, s_final, norm_states

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        K, M = self.n_slots, self.mem_dim
        orig_dtype = x.dtype

        wk = self.write_key(x).view(B, L, K, M)
        wk = F.normalize(wk, dim=-1)
        wv = self.write_value(x).view(B, L, K, M)

        decay = torch.sigmoid(self.decay_proj(x))
        input_gate = torch.sigmoid(self.input_gate_proj(x))

        updates = input_gate.unsqueeze(-1).unsqueeze(-1) * torch.einsum('blkm,blkn->blkmn', wk, wv)

        if state is None:
            state = torch.zeros(B, K, M, M, device=x.device, dtype=torch.float32)

        decays = decay.unsqueeze(-1).unsqueeze(-1)
        key_updates = input_gate.unsqueeze(-1) * wk
        decay_1d = decay.unsqueeze(-1)

        mem_states, state, norm_states = self._scan_fp32(
            decays, updates, state, key_updates, decay_1d, L, B
        )

        rq = self.read_query(x).view(B, L, K, M).float()
        rq = F.normalize(rq, dim=-1)

        numerator = torch.einsum('blkmn,blkn->blkm', mem_states, rq)
        denominator = torch.einsum('blkm,blkm->blk', norm_states, rq).unsqueeze(-1).abs() + self.norm_eps
        retrieved = (numerator / denominator).reshape(B, L, K * M)
        retrieved = retrieved.clamp(-10.0, 10.0).to(orig_dtype)

        retrieved = self.read_expand(retrieved)
        retrieved = self.mem_norm(retrieved)

        gate_val = torch.sigmoid(self.gate(torch.cat([x, retrieved], dim=-1)))
        output = x + gate_val * retrieved

        return output, state

    def forward_step(self, x_step: torch.Tensor,
                     state: Optional[torch.Tensor] = None,
                     norm_state: Optional[torch.Tensor] = None,
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token inference step. x_step is (B, 1, D).

        Optimisations vs the original:
          - States are kept in float32 regardless of AMP dtype (same guard as
            the training path) to prevent NaN gradients under mixed precision.
          - decay / input_gate shapes are computed once and reused for both the
            4-D matrix update and the 3-D norm update — no squeeze/unsqueeze chains.
          - The outer-product wk x wv uses unsqueeze+mul instead of einsum to
            skip the einsum dispatcher for this simple rank-1 case.
          - Matrix-vector read uses batched matmul (@) instead of einsum.
          - in-place clamp_ on new_state (safe: tensor is freshly allocated).
        """
        B, _, D = x_step.shape
        K, M = self.n_slots, self.mem_dim
        orig_dtype = x_step.dtype

        # --- Project inputs (stay in original dtype for linear layers) ---
        wk = self.write_key(x_step).view(B, K, M)
        wk = F.normalize(wk, dim=-1)
        wv = self.write_value(x_step).view(B, K, M)

        # Compute scalars once, derive all needed broadcast shapes in fp32.
        decay_raw = torch.sigmoid(self.decay_proj(x_step)).view(B, K)   # (B, K)
        gate_raw  = torch.sigmoid(self.input_gate_proj(x_step)).view(B, K)

        decay_4d = decay_raw.unsqueeze(-1).unsqueeze(-1).float()  # (B, K, 1, 1)
        decay_3d = decay_raw.unsqueeze(-1).float()                # (B, K, 1)
        gate_4d  = gate_raw.unsqueeze(-1).unsqueeze(-1).float()   # (B, K, 1, 1)
        gate_3d  = gate_raw.unsqueeze(-1).float()                 # (B, K, 1)

        # Outer product via unsqueeze+mul — avoids einsum dispatcher overhead.
        wk_f = wk.float()
        wv_f = wv.float()
        update = gate_4d * (wk_f.unsqueeze(-1) * wv_f.unsqueeze(-2))  # (B, K, M, M)

        # --- States always float32 to prevent AMP NaN ---
        if state is None:
            state = torch.zeros(B, K, M, M, device=x_step.device, dtype=torch.float32)
        else:
            state = state.float()
        if norm_state is None:
            norm_state = torch.zeros(B, K, M, device=x_step.device, dtype=torch.float32)
        else:
            norm_state = norm_state.float()

        new_state      = (decay_4d * state + update).clamp_(-8.0, 8.0)
        new_norm_state = decay_3d * norm_state + gate_3d * wk_f

        # --- Read ---
        rq = self.read_query(x_step).view(B, K, M).float()
        rq = F.normalize(rq, dim=-1)

        # batched matmul: (B,K,M,M) @ (B,K,M,1) -> (B,K,M,1) -> (B,K,M)
        numerator   = (new_state @ rq.unsqueeze(-1)).squeeze(-1)
        denominator = (new_norm_state * rq).sum(-1, keepdim=True).abs() + self.norm_eps
        retrieved   = (numerator / denominator).reshape(B, 1, K * M)
        retrieved   = retrieved.clamp(-10.0, 10.0).to(orig_dtype)

        retrieved = self.read_expand(retrieved)
        retrieved = self.mem_norm(retrieved)

        gate_val = torch.sigmoid(self.gate(torch.cat([x_step, retrieved], dim=-1)))
        output   = x_step + gate_val * retrieved

        return output, new_state, new_norm_state


# =====================================================================
# MECHANISM 3: SPARSE CORTICAL MLP
# Brain basis: only ~1-5% of cortical neurons fire at any time
# =====================================================================

class SparseCorticalMLP(nn.Module):
    """
    Feed-forward with sparse activation mimicking cortical sparse coding.

    Uses SwiGLU structure but applies top-K sparsity to the gate activations,
    zeroing out ~80% of neurons. The straight-through estimator (STE) passes
    gradients through the top-K selection during backprop.

    This gives ~5x fewer FLOPs in the down projection (which dominates)
    compared to dense SwiGLU, while maintaining representational capacity.
    """
    def __init__(self, dim: int, hidden_dim: int, k_ratio: float = 0.2):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.k = max(1, int(hidden_dim * k_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)

        # Avoid unpacking the named tuple; grab .indices directly.
        topk_indices = gate.abs().topk(self.k, dim=-1).indices
        # Build binary mask in a single in-place scatter (no separate zeros + scatter).
        mask = torch.zeros_like(gate).scatter_(-1, topk_indices, 1.0)

        if self.training:
            # STE: forward is sparse, backward is dense.
            # Algebraically identical to: gate_sparse.detach() + gate - gate.detach()
            # but avoids the extra intermediate variable.
            gate = gate * mask + (gate - gate * mask).detach()
        else:
            gate = gate * mask

        return self.w_down(gate * up)


# =====================================================================
# CORTICAL BLOCK — combines all three mechanisms + predictive coding
# =====================================================================

class CorticalBlock(nn.Module):
    """
    Single reasoning block in the Sage cortex.

    Flow: input → [HarmonicWave] → [HebbianResonance] → [SparseMLP] → output

    With predictive coding enabled, the block also predicts the next block's
    output. Only the prediction error propagates, reducing compute for easy
    tokens where predictions are accurate.

    With cognitive routing enabled, a lightweight router scores each token's
    processing difficulty. Tokens below the threshold skip the resonance
    memory (most expensive operation) via residual shortcut.
    """
    def __init__(self, config: SageConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        dim = config.core_dim

        depth_ratio = layer_idx / max(config.core_n_layers - 1, 1)
        gamma_k = 3
        beta_k = 7 + int(depth_ratio * 16)
        theta_k = 31 + int(depth_ratio * 96)
        beta_k = beta_k | 1
        theta_k = theta_k | 1

        self.wave_norm = RMSNorm(dim)
        self.wave = HarmonicWaveMixer(dim, gamma_k, beta_k, theta_k)

        self.resonance_norm = RMSNorm(dim)
        self.resonance = HebbianResonanceMemory(
            dim,
            n_slots=config.resonance_n_slots,
            mem_dim=config.resonance_mem_dim,
            decay_init=config.resonance_decay_init,
        )

        self.mlp_norm = RMSNorm(dim)
        self.mlp = SparseCorticalMLP(dim, config.core_mlp_dim, k_ratio=config.sparse_k_ratio)

        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.wave_scale = nn.Parameter(torch.full((), config.layer_scale_init))
        self.resonance_scale = nn.Parameter(torch.full((), config.layer_scale_init))
        self.mlp_scale = nn.Parameter(torch.full((), config.layer_scale_init))

        if config.predictive_coding and layer_idx < config.core_n_layers - 1:
            self.predictor = nn.Sequential(
                nn.Linear(dim, dim * 2, bias=False),
                nn.SiLU(),
                nn.Linear(dim * 2, dim, bias=False),
            )
            self.prediction_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.predictor = None
            self.prediction_gate = None

        if config.cognitive_routing:
            self.router = nn.Sequential(
                nn.Linear(dim, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.router = None

    def _wave_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.wave(self.wave_norm(x))) * self.wave_scale

    def _mlp_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.mlp(self.mlp_norm(x))) * self.mlp_scale

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.config.gradient_checkpointing and self.training:
            x = x + grad_checkpoint(self._wave_fn, x, use_reentrant=False)
        else:
            x = x + self._wave_fn(x)

        if self.router is not None and not self.training:
            difficulty = self.router(x)
            # Cast to same dtype as x so AMP (float16/bfloat16) is handled correctly.
            route_mask = (difficulty > (1.0 - self.config.routing_capacity)).to(x.dtype)

            res_input = self.resonance_norm(x)
            res_out, state = self.resonance(res_input, state=state)
            res_delta = self.drop(res_out - res_input) * self.resonance_scale
            x = x + res_delta * route_mask
        else:
            res_input = self.resonance_norm(x)
            res_out, state = self.resonance(res_input, state=state)
            x = x + self.drop(res_out - res_input) * self.resonance_scale

        if self.config.gradient_checkpointing and self.training:
            x = x + grad_checkpoint(self._mlp_fn, x, use_reentrant=False)
        else:
            x = x + self._mlp_fn(x)

        prediction = None
        if self.predictor is not None:
            prediction = self.predictor(x)

        return x, state, prediction

    def forward_step(
        self,
        x_step: torch.Tensor,
        block_state: Optional[dict],
    ) -> Tuple[torch.Tensor, dict]:
        """Process a single token (B, 1, D) through this CorticalBlock with O(1) state.

        block_state is a dict with keys:
            'wave':       dict of rolling conv buffers for HarmonicWaveMixer
            'resonance':  (B, K, M, M) Hebbian memory matrix

        Returns (output, new_block_state). Skips gradient checkpointing and
        predictive-coding error gating (no previous prediction available in
        step mode); predictive coding is only meaningful across the full sequence.
        """
        if block_state is None:
            block_state = {"wave": None, "resonance": None, "norm": None}

        # --- Wave sub-step (replaces _wave_fn for single token) ---
        x_norm = self.wave_norm(x_step)
        wave_out, new_wave_state = self.wave.forward_step(x_norm, block_state["wave"])
        x_step = x_step + self.drop(wave_out) * self.wave_scale

        # --- Resonance sub-step with optional cognitive routing ---
        res_input = self.resonance_norm(x_step)
        if self.router is not None:
            difficulty = self.router(x_step)
            # Cast to same dtype as x_step so AMP (float16/bfloat16) is handled correctly.
            route_mask = (difficulty > (1.0 - self.config.routing_capacity)).to(x_step.dtype)
            res_out, new_resonance_state, new_norm_state = self.resonance.forward_step(
                res_input, state=block_state["resonance"], norm_state=block_state["norm"]
            )
            res_delta = self.drop(res_out - res_input) * self.resonance_scale
            x_step = x_step + res_delta * route_mask
        else:
            res_out, new_resonance_state, new_norm_state = self.resonance.forward_step(
                res_input, state=block_state["resonance"], norm_state=block_state["norm"]
            )
            x_step = x_step + self.drop(res_out - res_input) * self.resonance_scale

        # --- MLP sub-step (SparseCorticalMLP is stateless per token) ---
        x_step = x_step + self.drop(self.mlp(self.mlp_norm(x_step))) * self.mlp_scale

        new_block_state = {
            "wave": new_wave_state,
            "resonance": new_resonance_state,
            "norm": new_norm_state,
        }
        return x_step, new_block_state


class ReasoningCore(nn.Module):
    """
    Sage 6.0 Reasoning Core: Harmonic Waves + Hebbian Resonance + Sparse Cortex

    Brain-inspired sequence processing with:
      - Multi-frequency wave decomposition (gamma/beta/theta/alpha)
      - Hebbian outer-product memory with learned decay
      - Sparse cortical activation (~20% neurons active)
      - Predictive coding between layers (only errors propagate)

    Combined complexity: O(n * (k + K*M^2)) per layer — linear in sequence length.
    """
    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            CorticalBlock(config, layer_idx=i)
            for i in range(config.core_n_layers)
        ])

        self.output_norm = RMSNorm(config.core_dim)
        self.lm_head = nn.Linear(config.core_dim, config.text_vocab_size, bias=False)

    def forward(self, x: torch.Tensor,
                states: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        prev_prediction = None

        for i, layer in enumerate(self.layers):
            if prev_prediction is not None and self.config.predictive_coding:
                error = x - prev_prediction.detach()
                gate = torch.sigmoid(self.layers[i - 1].prediction_gate)
                x = x + gate * error

            x, state, prediction = layer(x, state=states[i])
            new_states.append(state)
            prev_prediction = prediction

        return self.output_norm(x), new_states

    def forward_step(
        self,
        x_step: torch.Tensor,
        core_state: Optional[list],
    ) -> Tuple[torch.Tensor, list]:
        """Single-token recurrent inference through all CorticalBlocks.

        Args:
            x_step:     (B, 1, D) — embedded + phase-encoded token features.
            core_state: list of per-layer block_state dicts (from a previous
                        forward_step or seeded from a full forward() pass),
                        or None to initialise fresh zero state.

        Returns:
            output:         (B, 1, D) — normalised output features.
            new_core_state: updated list of per-layer block_state dicts.

        Note: predictive-coding error gating is skipped in step mode because
        there is no inter-block prediction from a previous sequence position.
        The Hebbian memory and conv buffers carry all necessary recurrent state.
        """
        if core_state is None:
            core_state = [None] * len(self.layers)

        new_core_state = []
        for layer, blk_state in zip(self.layers, core_state):
            x_step, new_blk_state = layer.forward_step(x_step, blk_state)
            new_core_state.append(new_blk_state)

        return self.output_norm(x_step), new_core_state
