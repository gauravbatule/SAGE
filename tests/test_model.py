"""
Test suite for Sage 4.0 architecture.

Covers: config validation, model instantiation, forward pass, weight tying,
generation utilities, and component-level checks.
"""

import pytest
import torch

from sage.config import SageConfig, get_config
from sage.generation import extract_assistant_response, top_p_sample
from sage.sage import SageModel


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def small_config() -> SageConfig:
    """Minimal config for fast testing."""
    return SageConfig(
        name="test",
        n_nodes=256,
        n_active_limit=32,
        text_vocab_size=256,
        core_dim=64,
        core_n_heads=4,
        core_n_layers=2,
        core_mlp_ratio=2.0,
        max_seq_len=64,
        max_think_iterations=2,
        min_think_iterations=1,
        metacog_dim=32,
        weight_tying=True,
        init_std=0.02,
    )


@pytest.fixture
def model(small_config: SageConfig) -> SageModel:
    return SageModel(small_config)


# ─── Config Tests ─────────────────────────────────────────────────────────────


class TestConfig:
    def test_predefined_configs(self):
        """All predefined configs should be instantiable."""
        for name in ("alpha", "beta", "omega"):
            config = get_config(name)
            assert config.name.startswith("sage-")
            config.validate()

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_config("nonexistent")

    def test_core_mlp_dim_aligned(self):
        """MLP dim should be 8-byte aligned."""
        config = get_config("alpha")
        assert config.core_mlp_dim % 8 == 0

    def test_head_dim(self):
        config = SageConfig(core_dim=512, core_n_heads=8)
        assert config.head_dim == 64

    def test_validation_catches_bad_heads(self):
        config = SageConfig(core_dim=100, core_n_heads=3)
        with pytest.raises(AssertionError, match="divisible"):
            config.validate()

    def test_validation_catches_bad_nodes(self):
        config = SageConfig(n_nodes=10, text_vocab_size=100)
        with pytest.raises(AssertionError, match="n_nodes"):
            config.validate()


# ─── Model Tests ──────────────────────────────────────────────────────────────


class TestModel:
    def test_instantiation(self, model: SageModel):
        """Model should instantiate without error."""
        assert model is not None

    def test_forward_pass(self, model: SageModel, small_config: SageConfig):
        """Forward pass should produce correct output shape."""
        B, L = 2, 16
        tokens = torch.randint(0, small_config.text_vocab_size, (B, L))
        output = model(tokens)

        assert "logits" in output
        assert "loss" in output
        assert "metrics" in output
        assert output["logits"].shape == (B, L, small_config.text_vocab_size)
        assert output["loss"] is None  # No targets provided

    def test_forward_with_targets(self, model: SageModel, small_config: SageConfig):
        """Forward pass with targets should produce loss."""
        B, L = 2, 16
        tokens = torch.randint(0, small_config.text_vocab_size, (B, L))
        targets = torch.randint(0, small_config.text_vocab_size, (B, L))
        output = model(tokens, targets=targets)

        assert output["loss"] is not None
        assert output["loss"].ndim == 0  # scalar
        assert output["loss"].item() > 0

    def test_weight_tying(self, model: SageModel):
        """Embedding and LM head should share the same weight tensor."""
        assert model.core.lm_head.weight is model.graph.node_embeddings.weight

    def test_no_weight_tying(self, small_config: SageConfig):
        """Without weight tying, weights should be independent."""
        small_config.weight_tying = False
        model = SageModel(small_config)
        assert model.core.lm_head.weight is not model.graph.node_embeddings.weight

    def test_parameter_count(self, model: SageModel):
        """count_parameters should return all expected keys."""
        counts = model.count_parameters()
        expected_keys = {
            "graph_substrate", "sensory_cortex", "temporal_binding",
            "reasoning_core", "metacognitive", "total", "active_per_token",
        }
        assert set(counts.keys()) == expected_keys
        assert counts["total"] > 0
        assert counts["temporal_binding"] == 0  # pass-through module

    def test_metrics(self, model: SageModel, small_config: SageConfig):
        """Metrics should report iteration count."""
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        output = model(tokens)
        assert output["metrics"]["iterations_used"] >= small_config.min_think_iterations


# ─── Generation Utility Tests ─────────────────────────────────────────────────


class TestGeneration:
    def test_top_p_sample_shape(self):
        """top_p_sample should return (B, 1) tensor."""
        logits = torch.randn(2, 100)
        result = top_p_sample(logits, temperature=0.8, top_p=0.9)
        assert result.shape == (2, 1)
        assert result.dtype == torch.int64

    def test_top_p_sample_in_range(self):
        """Sampled tokens should be valid indices."""
        vocab_size = 50
        logits = torch.randn(4, vocab_size)
        result = top_p_sample(logits, temperature=1.0, top_p=0.95)
        assert (result >= 0).all()
        assert (result < vocab_size).all()

    def test_low_temperature_deterministic(self):
        """Very low temperature should produce near-deterministic output."""
        logits = torch.zeros(1, 10)
        logits[0, 7] = 100.0  # Make token 7 dominant
        results = [top_p_sample(logits, temperature=0.01).item() for _ in range(10)]
        assert all(r == 7 for r in results)

    def test_extract_response_basic(self):
        text = "User: Hello\nAssistant: Hi there!"
        assert extract_assistant_response(text) == "Hi there!"

    def test_extract_response_multi_turn(self):
        text = "User: Hello\nAssistant: Hi!\nUser: How are you?"
        assert extract_assistant_response(text) == "Hi!"

    def test_extract_response_fallback(self):
        text = "Some raw output text"
        assert extract_assistant_response(text, prompt_length=5) == "raw output text"


# ─── Component Tests ──────────────────────────────────────────────────────────


class TestComponents:
    def test_graph_substrate(self, small_config: SageConfig):
        from sage.graph_store import GraphSubstrate
        graph = GraphSubstrate(small_config)
        assert graph.node_embeddings.weight.shape == (
            small_config.n_nodes, small_config.core_dim,
        )

    def test_sensory_cortex(self, small_config: SageConfig):
        from sage.sensory_cortex import SensoryCortex
        cortex = SensoryCortex(small_config)
        tokens = torch.randint(0, small_config.text_vocab_size, (2, 8))
        ids, energies, positions = cortex.ground_text(tokens)
        assert ids.shape == (2, 8)
        assert energies.shape == (2, 8)

    def test_temporal_binding_passthrough(self, small_config: SageConfig):
        from sage.temporal_binding import TemporalBinding
        binder = TemporalBinding(small_config)
        x = torch.randn(2, 8, small_config.core_dim)
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        out = binder.bind(x, positions)
        assert torch.equal(out, x)  # Should be pure pass-through

    def test_reasoning_core(self, small_config: SageConfig):
        from sage.reasoning_core import ReasoningCore
        core = ReasoningCore(small_config)
        x = torch.randn(2, 8, small_config.core_dim)
        out = core(x)
        assert out.shape == x.shape

    def test_metacognitive_assess(self, small_config: SageConfig):
        from sage.metacognitive import MetacognitiveController
        metacog = MetacognitiveController(small_config)
        current = torch.randn(2, 8, small_config.core_dim)
        previous = torch.randn(2, 8, small_config.core_dim)
        confidence, needs_retrieval, query = metacog.assess(current, previous, 0)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(needs_retrieval, bool)
        assert query.shape == (2, small_config.core_dim)
