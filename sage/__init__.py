"""
Sage 6.0 — Brain-Inspired Ultra-Efficient Language Architecture

A fundamentally new approach to language modeling inspired by neuroscience,
replacing attention with harmonic wave propagation, Hebbian resonance memory,
and sparse cortical activation for linear-time sequence processing with
constant-memory inference.

Key components:
    - GraphSubstrate: Sparse knowledge storage (scalable to NVMe)
    - SensoryCortex: Input grounding with OOV detection
    - PhaseEncoding: Hippocampal theta phase position encoding
    - ReasoningCore: Harmonic Waves + Hebbian Memory + Sparse Cortex
    - MetacognitiveController: Adaptive cognitive load routing
    - SageModel: Full model orchestrator

Example:
    >>> from sage import SageModel, SageConfig, get_config
    >>> config = get_config("alpha")
    >>> model = SageModel(config)
    >>> import torch
    >>> tokens = torch.randint(0, config.text_vocab_size, (1, 128))
    >>> output = model(tokens)
    >>> output["logits"].shape
    torch.Size([1, 128, 32000])
"""

__version__ = "7.0.0"
__author__ = "Gaurav Batule"

from .config import SageConfig, get_config
from .generation import generate_tokens
from .sage import SageModel

__all__ = ["SageConfig", "SageModel", "get_config", "generate_tokens"]
