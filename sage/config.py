"""
Sage 6.0 — Configuration

Brain-Inspired Ultra-Efficient Language Architecture.

Defines all hyperparameters and predefined model scales. See :class:`SageConfig`
for the full parameter list and :func:`get_config` for predefined scale presets.
"""

__all__ = ["SageConfig", "get_config", "CONFIGS"]

from dataclasses import dataclass


@dataclass
class SageConfig:
    name: str = "sage-alpha"

    # == Knowledge Graph ==
    n_nodes: int = 1_000_000
    n_active_limit: int = 2048

    # == Tokenizer ==
    text_vocab_size: int = 32000

    # == Dense Reasoning Core ==
    core_dim: int = 512
    core_n_layers: int = 6
    core_mlp_ratio: float = 2.667

    # == Context ==
    context_length: int = 131072

    # == Resonance Memory (Hebbian outer-product) ==
    resonance_n_slots: int = 8
    resonance_mem_dim: int = 64
    resonance_decay_init: float = 0.95

    # == Sparse Cortical MLP ==
    sparse_k_ratio: float = 0.2

    # == Brain-Inspired Features ==
    phase_encoding: bool = True
    predictive_coding: bool = True
    cognitive_routing: bool = True
    routing_capacity: float = 0.5

    # == Regularization ==
    dropout: float = 0.1

    # == Gradient Checkpointing ==
    gradient_checkpointing: bool = False

    # == Layer Scale ==
    layer_scale_init: float = 1e-4

    # == Metacognitive Controller ==
    max_think_iterations: int = 8
    min_think_iterations: int = 1
    max_train_iterations: int = 3
    confidence_threshold: float = 0.85
    metacog_dim: int = 128

    # == Training ==
    init_std: float = 0.02
    weight_tying: bool = True

    # == Embedding ==
    normalize_embeddings: bool = False

    @property
    def core_mlp_dim(self) -> int:
        raw = int(self.core_dim * self.core_mlp_ratio)
        return ((raw + 7) // 8) * 8

    def validate(self) -> None:
        assert self.core_n_layers >= 1, "core_n_layers must be >= 1"
        assert self.text_vocab_size > 0, "text_vocab_size must be > 0"
        assert self.n_nodes >= self.text_vocab_size, (
            f"n_nodes ({self.n_nodes}) must be >= text_vocab_size ({self.text_vocab_size})"
        )
        assert 0.0 <= self.resonance_decay_init <= 1.0, "resonance_decay_init must be in [0, 1]"
        assert self.resonance_n_slots >= 1, "resonance_n_slots must be >= 1"
        assert self.resonance_mem_dim >= 1, "resonance_mem_dim must be >= 1"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert self.layer_scale_init > 0, "layer_scale_init must be > 0"
        assert 0.0 < self.sparse_k_ratio <= 1.0, "sparse_k_ratio must be in (0, 1]"
        assert 0.0 < self.routing_capacity <= 1.0, "routing_capacity must be in (0, 1]"


CONFIGS = {
    "alpha": lambda: SageConfig(
        name="sage-alpha",
        n_nodes=1_000_000, core_dim=512, core_n_layers=6,
        n_active_limit=2048, text_vocab_size=32000,
        resonance_n_slots=8, resonance_mem_dim=64, resonance_decay_init=0.95,
        sparse_k_ratio=0.4, routing_capacity=0.5,
        dropout=0.1, gradient_checkpointing=False, layer_scale_init=1e-4,
    ),
    "beta": lambda: SageConfig(
        name="sage-beta",
        n_nodes=100_000_000, core_dim=1024, core_n_layers=8,
        n_active_limit=4096, text_vocab_size=32000,
        resonance_n_slots=8, resonance_mem_dim=128, resonance_decay_init=0.97,
        sparse_k_ratio=0.15, routing_capacity=0.5,
        dropout=0.1, gradient_checkpointing=True, layer_scale_init=1e-5,
    ),
    "omega": lambda: SageConfig(
        name="sage-omega",
        n_nodes=10_000_000_000, core_dim=2048, core_n_layers=12,
        n_active_limit=8192, text_vocab_size=64000,
        max_think_iterations=16, max_train_iterations=4,
        resonance_n_slots=8, resonance_mem_dim=256, resonance_decay_init=0.98,
        sparse_k_ratio=0.1, routing_capacity=0.5,
        dropout=0.05, gradient_checkpointing=True, layer_scale_init=1e-6,
    ),
}


def get_config(name: str = "alpha") -> SageConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown: '{name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()
