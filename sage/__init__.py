"""
Sage 4.0 — Hybrid Graph-Cortex Language Model Architecture

A fundamentally new approach to language modeling that replaces attention
with Wave Propagation (multi-scale causal convolutions) and Resonance Memory
(compressed cumulative context), achieving linear-time sequence processing.

Key components:
    - GraphSubstrate: Sparse knowledge storage (scalable to NVMe)
    - SensoryCortex: Multimodal input grounding
    - TemporalBinding: Positional pass-through (position is implicit in causal convolutions)
    - ReasoningCore: Wave Propagation + Resonance Memory stack
    - MetacognitiveController: Adaptive thinking depth per token
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

__version__ = "4.0.0"
__author__ = "Gaurav Batule"

from .config import SageConfig, get_config
from .sage import SageModel

__all__ = ["SageConfig", "SageModel", "get_config"]
