"""
Test suite for Sage 6.0 architecture.

Covers: config validation, model instantiation, forward pass, weight tying,
generation utilities, component-level checks, and v6.0 features (harmonic waves,
Hebbian memory, sparse MLP, predictive coding, phase encoding, cognitive routing,
multi-iteration training, recurrent state).
"""

import pytest
import torch

from sage.config import SageConfig, get_config
from sage.generation import extract_assistant_response, top_p_sample
from sage.sage import SageModel


# --- Fixtures ---


@pytest.fixture
def small_config() -> SageConfig:
    return SageConfig(
        name="test",
        n_nodes=256,
        n_active_limit=32,
        text_vocab_size=256,
        core_dim=64,
        core_n_layers=2,
        core_mlp_ratio=2.0,
        context_length=64,
        max_think_iterations=2,
        min_think_iterations=1,
        max_train_iterations=2,
        metacog_dim=32,
        weight_tying=True,
        init_std=0.02,
        resonance_n_slots=4,
        resonance_mem_dim=16,
        resonance_decay_init=0.95,
        sparse_k_ratio=0.3,
        dropout=0.0,
        layer_scale_init=1e-4,
        phase_encoding=True,
        predictive_coding=True,
        cognitive_routing=True,
        routing_capacity=0.5,
    )


@pytest.fixture
def model(small_config: SageConfig) -> SageModel:
    return SageModel(small_config)


# --- Config Tests ---


class TestConfig:
    def test_predefined_configs(self):
        for name in ("alpha", "beta", "omega"):
            config = get_config(name)
            assert config.name.startswith("sage-")
            config.validate()

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_config("nonexistent")

    def test_core_mlp_dim_aligned(self):
        config = get_config("alpha")
        assert config.core_mlp_dim % 8 == 0

    def test_validation_catches_bad_nodes(self):
        config = SageConfig(n_nodes=10, text_vocab_size=100)
        with pytest.raises(AssertionError, match="n_nodes"):
            config.validate()

    def test_validation_catches_bad_decay(self):
        config = SageConfig(resonance_decay_init=1.5)
        with pytest.raises(AssertionError, match="resonance_decay_init"):
            config.validate()

    def test_validation_catches_bad_dropout(self):
        config = SageConfig(dropout=1.0)
        with pytest.raises(AssertionError, match="dropout"):
            config.validate()

    def test_validation_catches_bad_sparse_ratio(self):
        config = SageConfig(sparse_k_ratio=0.0)
        with pytest.raises(AssertionError, match="sparse_k_ratio"):
            config.validate()

    def test_validation_catches_bad_routing_capacity(self):
        config = SageConfig(routing_capacity=0.0)
        with pytest.raises(AssertionError, match="routing_capacity"):
            config.validate()

    def test_new_config_fields_exist(self):
        config = get_config("alpha")
        assert hasattr(config, "resonance_n_slots")
        assert hasattr(config, "resonance_mem_dim")
        assert hasattr(config, "resonance_decay_init")
        assert hasattr(config, "sparse_k_ratio")
        assert hasattr(config, "phase_encoding")
        assert hasattr(config, "predictive_coding")
        assert hasattr(config, "cognitive_routing")
        assert hasattr(config, "routing_capacity")
        assert hasattr(config, "max_train_iterations")
        assert hasattr(config, "dropout")
        assert hasattr(config, "gradient_checkpointing")
        assert hasattr(config, "layer_scale_init")
        assert hasattr(config, "context_length")

    def test_removed_fields_absent(self):
        config = get_config("alpha")
        assert not hasattr(config, "core_n_heads")
        assert not hasattr(config, "core_dropout")
        assert not hasattr(config, "vision_patch_dim")
        assert not hasattr(config, "audio_frame_dim")


# --- Model Tests ---


class TestModel:
    def test_instantiation(self, model: SageModel):
        assert model is not None

    def test_forward_pass(self, model: SageModel, small_config: SageConfig):
        B, L = 2, 16
        tokens = torch.randint(0, small_config.text_vocab_size, (B, L))
        output = model(tokens)

        assert "logits" in output
        assert "loss" in output
        assert "metrics" in output
        assert output["logits"].shape == (B, L, small_config.text_vocab_size)
        assert output["loss"] is None

    def test_forward_with_targets(self, model: SageModel, small_config: SageConfig):
        B, L = 2, 16
        tokens = torch.randint(0, small_config.text_vocab_size, (B, L))
        targets = torch.randint(0, small_config.text_vocab_size, (B, L))
        output = model(tokens, targets=targets)

        assert output["loss"] is not None
        assert output["loss"].ndim == 0
        assert output["loss"].item() > 0

    def test_weight_tying(self, model: SageModel):
        assert model.core.lm_head.weight is model.graph.node_embeddings.weight

    def test_no_weight_tying(self, small_config: SageConfig):
        small_config.weight_tying = False
        m = SageModel(small_config)
        assert m.core.lm_head.weight is not m.graph.node_embeddings.weight

    def test_parameter_count(self, model: SageModel):
        counts = model.count_parameters()
        expected_keys = {
            "graph_substrate", "sensory_cortex", "phase_encoding",
            "reasoning_core", "metacognitive", "total", "active_per_token",
        }
        assert set(counts.keys()) == expected_keys
        assert counts["total"] > 0
        assert counts["sensory_cortex"] == 0

    def test_metrics(self, model: SageModel, small_config: SageConfig):
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        output = model(tokens)
        assert output["metrics"]["iterations_used"] >= small_config.min_think_iterations

    def test_gradient_checkpointing(self, small_config: SageConfig):
        small_config.gradient_checkpointing = True
        m = SageModel(small_config)
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        targets = torch.randint(0, small_config.text_vocab_size, (1, 8))
        output = m(tokens, targets=targets)
        assert output["loss"] is not None
        output["loss"].backward()

    def test_dropout_config(self, small_config: SageConfig):
        small_config.dropout = 0.2
        m = SageModel(small_config)
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        m.train()
        output = m(tokens)
        assert output["logits"].shape[-1] == small_config.text_vocab_size

    def test_multi_iteration_training(self, small_config: SageConfig):
        small_config.max_train_iterations = 3
        m = SageModel(small_config)
        m.train()
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        targets = torch.randint(0, small_config.text_vocab_size, (1, 8))
        output = m(tokens, targets=targets)
        assert output["loss"] is not None
        output["loss"].backward()


# --- Generation Utility Tests ---


class TestGeneration:
    def test_top_p_sample_shape(self):
        logits = torch.randn(2, 100)
        result = top_p_sample(logits, temperature=0.8, top_p=0.9)
        assert result.shape == (2, 1)
        assert result.dtype == torch.int64

    def test_top_p_sample_in_range(self):
        vocab_size = 50
        logits = torch.randn(4, vocab_size)
        result = top_p_sample(logits, temperature=1.0, top_p=0.95)
        assert (result >= 0).all()
        assert (result < vocab_size).all()

    def test_low_temperature_deterministic(self):
        logits = torch.zeros(1, 10)
        logits[0, 7] = 100.0
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


# --- Component Tests ---


class TestComponents:
    def test_graph_substrate(self, small_config: SageConfig):
        from sage.graph_store import GraphSubstrate
        graph = GraphSubstrate(small_config)
        assert graph.node_embeddings.weight.shape == (
            small_config.n_nodes, small_config.core_dim,
        )

    def test_graph_forward(self, small_config: SageConfig):
        from sage.graph_store import GraphSubstrate
        graph = GraphSubstrate(small_config)
        ids = torch.randint(0, small_config.n_nodes, (2, 8))
        embs = graph(ids)
        assert embs.shape == (2, 8, small_config.core_dim)

    def test_graph_normalize(self, small_config: SageConfig):
        from sage.graph_store import GraphSubstrate
        small_config.normalize_embeddings = True
        graph = GraphSubstrate(small_config)
        ids = torch.randint(0, small_config.n_nodes, (1, 4))
        embs = graph(ids)
        norms = embs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_sensory_cortex(self, small_config: SageConfig):
        from sage.sensory_cortex import SensoryCortex
        cortex = SensoryCortex(small_config)
        tokens = torch.randint(0, small_config.text_vocab_size, (2, 8))
        ids, energies, positions = cortex.ground_text(tokens)
        assert ids.shape == (2, 8)
        assert energies.shape == (2, 8)

    def test_sensory_cortex_oov_warning(self, small_config: SageConfig):
        from sage.sensory_cortex import SensoryCortex
        cortex = SensoryCortex(small_config)
        tokens = torch.tensor([[999, 0, 1]])
        with pytest.warns(RuntimeWarning, match="out of range"):
            ids, _, _ = cortex.ground_text(tokens)
        assert ids[0, 0].item() == small_config.text_vocab_size - 1

    def test_phase_encoding(self, small_config: SageConfig):
        from sage.phase_encoding import PhaseEncoding
        phase = PhaseEncoding(small_config)
        x = torch.randn(2, 8, small_config.core_dim)
        out = phase(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)

    def test_phase_encoding_different_positions(self, small_config: SageConfig):
        from sage.phase_encoding import PhaseEncoding
        phase = PhaseEncoding(small_config)
        x = torch.ones(1, 16, small_config.core_dim)
        out = phase(x)
        assert not torch.allclose(out[:, 0], out[:, 1], atol=1e-5)

    def test_harmonic_wave_mixer(self, small_config: SageConfig):
        from sage.reasoning_core import HarmonicWaveMixer
        wave = HarmonicWaveMixer(small_config.core_dim, gamma_k=3, beta_k=7, theta_k=15)
        x = torch.randn(2, 8, small_config.core_dim)
        out = wave(x)
        assert out.shape == x.shape

    def test_hebbian_resonance_memory(self, small_config: SageConfig):
        from sage.reasoning_core import HebbianResonanceMemory
        mem = HebbianResonanceMemory(
            dim=small_config.core_dim,
            n_slots=small_config.resonance_n_slots,
            mem_dim=small_config.resonance_mem_dim,
            decay_init=0.9,
        )
        x = torch.randn(1, 16, small_config.core_dim)
        out, state = mem(x)
        assert out.shape == x.shape
        K, M = small_config.resonance_n_slots, small_config.resonance_mem_dim
        assert state.shape == (1, K, M, M)

    def test_hebbian_memory_with_state(self, small_config: SageConfig):
        from sage.reasoning_core import HebbianResonanceMemory
        mem = HebbianResonanceMemory(
            dim=small_config.core_dim,
            n_slots=small_config.resonance_n_slots,
            mem_dim=small_config.resonance_mem_dim,
        )
        x1 = torch.randn(1, 8, small_config.core_dim)
        x2 = torch.randn(1, 8, small_config.core_dim)
        _, state = mem(x1)
        out, state2 = mem(x2, state=state)
        assert out.shape == x2.shape
        assert not torch.equal(state, state2)

    def test_sparse_cortical_mlp(self, small_config: SageConfig):
        from sage.reasoning_core import SparseCorticalMLP
        mlp = SparseCorticalMLP(
            small_config.core_dim, small_config.core_mlp_dim,
            k_ratio=small_config.sparse_k_ratio,
        )
        x = torch.randn(2, 8, small_config.core_dim)
        out = mlp(x)
        assert out.shape == x.shape

    def test_cortical_block(self, small_config: SageConfig):
        from sage.reasoning_core import CorticalBlock
        block = CorticalBlock(small_config, layer_idx=0)
        x = torch.randn(1, 8, small_config.core_dim)
        out, state, prediction = block(x)
        assert out.shape == x.shape
        assert state is not None

    def test_cortical_block_layer_scale(self, small_config: SageConfig):
        from sage.reasoning_core import CorticalBlock
        block = CorticalBlock(small_config, layer_idx=0)
        assert hasattr(block, "wave_scale")
        assert hasattr(block, "resonance_scale")
        assert hasattr(block, "mlp_scale")
        assert block.wave_scale.item() == pytest.approx(small_config.layer_scale_init, abs=1e-8)

    def test_predictive_coding(self, small_config: SageConfig):
        from sage.reasoning_core import CorticalBlock
        block = CorticalBlock(small_config, layer_idx=0)
        assert block.predictor is not None
        assert block.prediction_gate is not None
        x = torch.randn(1, 8, small_config.core_dim)
        _, _, prediction = block(x)
        assert prediction is not None
        assert prediction.shape == x.shape

    def test_predictive_coding_last_layer_no_predictor(self, small_config: SageConfig):
        from sage.reasoning_core import CorticalBlock
        block = CorticalBlock(small_config, layer_idx=small_config.core_n_layers - 1)
        assert block.predictor is None

    def test_reasoning_core_returns_states(self, small_config: SageConfig):
        from sage.reasoning_core import ReasoningCore
        core = ReasoningCore(small_config)
        x = torch.randn(1, 8, small_config.core_dim)
        out, states = core(x)
        assert out.shape == x.shape
        assert len(states) == small_config.core_n_layers

    def test_metacognitive_assess(self, small_config: SageConfig):
        from sage.metacognitive import MetacognitiveController
        metacog = MetacognitiveController(small_config)
        current = torch.randn(2, 8, small_config.core_dim)
        previous = torch.randn(2, 8, small_config.core_dim)
        result = metacog.assess(current, previous, 0)
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["needs_retrieval"], bool)
        assert result["retrieval_vector"].shape == (2, small_config.core_dim)
        assert 0.0 <= result["estimated_difficulty"] <= 1.0
        assert result["per_token_difficulty"].shape == (2, 8)

    def test_metacognitive_refinement_loss(self, small_config: SageConfig):
        from sage.metacognitive import MetacognitiveController
        metacog = MetacognitiveController(small_config)
        logits_prev = torch.randn(1, 8, small_config.text_vocab_size)
        logits_curr = torch.randn(1, 8, small_config.text_vocab_size)
        targets = torch.randint(0, small_config.text_vocab_size, (1, 8))
        loss = metacog.refinement_loss(logits_prev, logits_curr, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_cognitive_routing_disabled(self, small_config: SageConfig):
        small_config.cognitive_routing = False
        m = SageModel(small_config)
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        output = m(tokens)
        assert output["logits"].shape[-1] == small_config.text_vocab_size

    def test_predictive_coding_disabled(self, small_config: SageConfig):
        small_config.predictive_coding = False
        m = SageModel(small_config)
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        output = m(tokens)
        assert output["logits"].shape[-1] == small_config.text_vocab_size

    def test_phase_encoding_disabled(self, small_config: SageConfig):
        small_config.phase_encoding = False
        m = SageModel(small_config)
        tokens = torch.randint(0, small_config.text_vocab_size, (1, 8))
        output = m(tokens)
        assert output["logits"].shape[-1] == small_config.text_vocab_size

    def test_causal_conv_step(self, small_config: SageConfig):
        from sage.reasoning_core import CausalConv1d
        conv = CausalConv1d(32, kernel_size=5)
        x = torch.randn(1, 10, 32)
        full_out = conv(x)
        buf = None
        step_outs = []
        for t in range(10):
            out, buf = conv.forward_step(x[:, t:t+1, :], buf)
            step_outs.append(out)
        step_out = torch.cat(step_outs, dim=1)
        assert torch.allclose(full_out, step_out, atol=1e-5)

    def test_wave_mixer_step(self, small_config: SageConfig):
        from sage.reasoning_core import HarmonicWaveMixer
        wave = HarmonicWaveMixer(small_config.core_dim, gamma_k=3, beta_k=7, theta_k=15)
        x = torch.randn(1, 8, small_config.core_dim)
        full_out = wave(x)
        state = None
        step_outs = []
        for t in range(8):
            out, state = wave.forward_step(x[:, t:t+1, :], state)
            step_outs.append(out)
        step_out = torch.cat(step_outs, dim=1)
        assert torch.allclose(full_out, step_out, atol=1e-4)

    def test_hebbian_step(self, small_config: SageConfig):
        from sage.reasoning_core import HebbianResonanceMemory
        mem = HebbianResonanceMemory(
            dim=small_config.core_dim,
            n_slots=small_config.resonance_n_slots,
            mem_dim=small_config.resonance_mem_dim,
        )
        x = torch.randn(1, 1, small_config.core_dim)
        out, state, norm_state = mem.forward_step(x)
        assert out.shape == x.shape
        K, M = small_config.resonance_n_slots, small_config.resonance_mem_dim
        assert state.shape == (1, K, M, M)
        assert norm_state.shape == (1, K, M)

    def test_cortical_block_step(self, small_config: SageConfig):
        from sage.reasoning_core import CorticalBlock
        block = CorticalBlock(small_config, layer_idx=0)
        x = torch.randn(1, 1, small_config.core_dim)
        out, state = block.forward_step(x, None)
        assert out.shape == x.shape
        assert "wave" in state
        assert "resonance" in state

    def test_reasoning_core_step(self, small_config: SageConfig):
        from sage.reasoning_core import ReasoningCore
        core = ReasoningCore(small_config)
        x = torch.randn(1, 1, small_config.core_dim)
        out, states = core.forward_step(x, None)
        assert out.shape == x.shape
        assert len(states) == small_config.core_n_layers

    def test_model_forward_step(self, model: SageModel, small_config: SageConfig):
        token = torch.randint(0, small_config.text_vocab_size, (1, 1))
        logits, state = model.forward_step(token, step_idx=0)
        assert logits.shape == (1, 1, small_config.text_vocab_size)
        assert "core_state" in state

    def test_model_generate_fast(self, model: SageModel, small_config: SageConfig):
        prompt = torch.randint(0, small_config.text_vocab_size, (1, 4))
        generated = model.generate_fast(prompt, max_new_tokens=5, temperature=1.0)
        assert len(generated) <= 5

    def test_parallel_scan(self, small_config: SageConfig):
        from sage.reasoning_core import parallel_scan
        B, L, K, M = 1, 8, 2, 4
        decays = torch.rand(B, L, K, 1, 1) * 0.5 + 0.5
        updates = torch.randn(B, L, K, M, M) * 0.1
        result = parallel_scan(decays, updates)
        assert result.shape == (B, L, K, M, M)
        state = torch.zeros(B, K, M, M)
        for t in range(L):
            state = decays[:, t] * state + updates[:, t]
            assert torch.allclose(result[:, t], state, atol=1e-5)

    def test_phase_encoding_step(self, small_config: SageConfig):
        from sage.phase_encoding import PhaseEncoding
        phase = PhaseEncoding(small_config)
        x = torch.randn(1, 8, small_config.core_dim)
        full_out = phase(x)
        for t in range(8):
            step_out = phase.forward_step(x[:, t:t+1, :], t)
            assert torch.allclose(full_out[:, t:t+1, :], step_out, atol=1e-5)

    def test_cross_band_mixing(self, small_config: SageConfig):
        from sage.reasoning_core import HarmonicWaveMixer
        wave = HarmonicWaveMixer(small_config.core_dim)
        assert hasattr(wave, "cross_band_mix")


# --- Version Test ---


class TestVersion:
    def test_version(self):
        import sage
        assert sage.__version__ == "6.0.0"

    def test_exports(self):
        from sage import SageModel, SageConfig, get_config, generate_tokens
        assert SageModel is not None
        assert SageConfig is not None
        assert get_config is not None
        assert generate_tokens is not None
