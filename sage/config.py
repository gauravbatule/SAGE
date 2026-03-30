"""
Sage 4.0 — Configuration

Defines all hyperparameters and predefined model scales for the Sage architecture.
See :class:`SageConfig` for the full parameter list and :func:`get_config` for
predefined scale presets (alpha, beta, omega).
"""

__all__ = ["SageConfig", "get_config", "CONFIGS"]

from dataclasses import dataclass
import math


@dataclass
class SageConfig:
    name: str = "sage-alpha"

    # == Knowledge Graph ==
    n_nodes: int = 1_000_000        # Total concept nodes
    n_active_limit: int = 2048      # Max nodes in VRAM at once

    # == Tokenizer ==
    text_vocab_size: int = 32000    # Overridden by tiktoken (100277)

    # == Multimodal (kept for future) ==
    vision_patch_dim: int = 768
    audio_frame_dim: int = 80

    # == Dense Reasoning Core (where 80%+ of params should go) ==
    core_dim: int = 512             # Hidden dim
    core_n_heads: int = 8           # Reserved (wave arch doesn't use heads)
    core_n_layers: int = 6          # Depth
    core_mlp_ratio: float = 2.667   # SwiGLU expansion
    core_dropout: float = 0.0

    # == RoPE (Rotary Position Embedding) ==
    rope_theta: float = 10000.0     # Base frequency for RoPE
    max_seq_len: int = 131072       # Supports 128K context

    # == Metacognitive Controller ==
    max_think_iterations: int = 8
    min_think_iterations: int = 1
    confidence_threshold: float = 0.85
    metacog_dim: int = 128

    # == Training ==
    init_std: float = 0.02
    weight_tying: bool = True       # Share embedding <-> lm_head (saves params!)

    @property
    def core_mlp_dim(self) -> int:
        raw = int(self.core_dim * self.core_mlp_ratio)
        return ((raw + 7) // 8) * 8

    @property
    def head_dim(self) -> int:
        """Dimension per attention head (core_dim / core_n_heads)."""
        return self.core_dim // self.core_n_heads

    def validate(self) -> None:
        """Validate configuration for common errors."""
        assert self.core_dim % self.core_n_heads == 0, (
            f"core_dim ({self.core_dim}) must be divisible by core_n_heads ({self.core_n_heads})"
        )
        assert self.core_n_layers >= 1, "core_n_layers must be >= 1"
        assert self.text_vocab_size > 0, "text_vocab_size must be > 0"
        assert self.n_nodes >= self.text_vocab_size, (
            f"n_nodes ({self.n_nodes}) must be >= text_vocab_size ({self.text_vocab_size})"
        )



# Predefined Scales
CONFIGS = {
    "alpha": lambda: SageConfig(
        name="sage-alpha",
        n_nodes=1_000_000, core_dim=512, core_n_layers=6,
        n_active_limit=2048, text_vocab_size=32000,
    ),
    "beta": lambda: SageConfig(
        name="sage-beta",
        n_nodes=100_000_000, core_dim=1024, core_n_layers=8,
        core_n_heads=16, n_active_limit=4096, text_vocab_size=32000,
    ),
    "omega": lambda: SageConfig(
        name="sage-omega",
        n_nodes=10_000_000_000, core_dim=2048, core_n_layers=12,
        core_n_heads=32, n_active_limit=8192, text_vocab_size=64000,
        max_think_iterations=16,
    ),
}


def get_config(name: str = "alpha") -> SageConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown: '{name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()
