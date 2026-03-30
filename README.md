# Sage 4.0 — Hybrid Graph-Cortex Language Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A fundamentally new approach to language modeling that replaces attention with Wave Propagation and Resonance Memory, achieving linear-time sequence processing with constant memory inference.**

## What is Sage?

Sage is a **Hybrid Graph-Cortex** architecture that separates **knowledge storage** (a sparse topological graph) from **reasoning** (a compact dense core). Instead of the Transformer's O(n²) attention mechanism, Sage uses two complementary mechanisms:

1. **Causal Wave Propagation** — Multi-scale causal convolutions for local syntax and grammar understanding. Information flows causally through the sequence, with each position integrating signals from its local neighborhood at multiple scales.

2. **Resonance Memory** — A compressed neural whiteboard with K memory slots. Each position writes important information and reads relevant context via cumulative accumulation. This provides global context access in O(n·K·D) — linear in sequence length.

### Key Properties

- **Knowledge** lives in a sparse graph on disk (scalable to billions of nodes on NVMe)
- **Reasoning** happens in a small, reusable neural core that only loads active concepts
- **Memory** is **constant** — no KV cache, no growth with context length
- **Compute** is **adaptive** — easy tokens get 1 pass, complex reasoning gets up to 16

```
Token IDs → Graph Embedding → [Wave Mixing → Resonance Memory → MLP] × N → Output
                                         ↑ Repeated via Metacognitive Control
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: Graph Substrate                                        │
│  • Embedding table mapping token/concept IDs to core_dim vectors │
│  • At scale: memory-mapped from NVMe, only active subgraph loads │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 2: Causal Wave Propagation (local understanding)          │
│  • Multi-scale depthwise-separable causal convolutions           │
│  • Short (k=3), medium (k=5-17), long (k=11-43) receptive fields│
│  • Gated output with learned value projection                    │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 3: Resonance Memory (global understanding)                │
│  • K memory slots with cumulative write/read (fully parallelizable)│
│  • Position i's memory = compressed summary of positions 0..i    │
│  • Gated integration with current representation                 │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 4: SwiGLU MLP (per-position reasoning)                   │
│  • Standard feed-forward with SiLU-gated linear unit             │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 5: Metacognitive Controller                               │
│  • Adaptive compute: easy tokens = 1 pass, hard tokens = 8+     │
│  • Stagnation detection triggers graph re-retrieval when stuck   │
│  • Confidence-based early exit at inference time                  │
└──────────────────────────────────────────────────────────────────┘
```

## Quickstart

### Install

```bash
git clone https://github.com/gauravbatule/sage-architecture.git
cd sage-architecture
pip install -e .
```

### Train a Character-Level Demo

```bash
python -m sage.train --epochs 10 --batch-size 32
```

This downloads a Shakespeare corpus and trains a character-level Sage model. You'll see loss decrease and sample text generation.

### Train a Chat Model (BPE)

```bash
python train_chat.py --epochs 15 --batch-size 8
```

Trains on the Stanford Alpaca instruction dataset with tiktoken BPE tokenization. Optimized for consumer GPUs (4GB VRAM).

### Run the Chat Server

```bash
python serve.py
# Open http://localhost:8888
```

### Use in Code

```python
from sage import SageModel, SageConfig, get_config

# Create a model
config = get_config("alpha")  # or "beta", "omega"
model = SageModel(config)

# Forward pass
import torch
tokens = torch.randint(0, config.text_vocab_size, (1, 128))
output = model(tokens)
print(output["logits"].shape)  # (1, 128, vocab_size)
```

## Why Not Attention?

| Property | Transformer | Mamba | **Sage 4.0** |
|:---|:---|:---|:---|
| Sequence mixing | O(n²) attention | O(n) selective scan | **O(n·k) causal conv + O(n·K·D) resonance** |
| Inference memory | KV cache grows with context | Fixed state | **Fixed (constant)** |
| Params activated/token | 100% | 100% | **5-15% (sparse graph)** |
| Adaptive compute | ❌ Fixed depth | ❌ Fixed depth | **✅ 1-16 iterations** |
| Position encoding | Learned / RoPE | Implicit in recurrence | **Implicit in causal conv** |
| Parallelizable | ✅ | ⚠️ Sequential scan | **✅ Fully parallel** |

### Predefined Scales

| Config | Graph Nodes | Core Params | Inference VRAM | Use Case |
|:---|:---|:---|:---|:---|
| Alpha | 1M | ~15M | <100MB | Research & prototyping |
| Beta | 100M | ~200M | ~2GB | Mid-scale experiments |
| Omega | 10B | ~1B | ~16GB | Full-scale training |

## Project Structure

```
sage-architecture/
├── sage/                       # Core package
│   ├── __init__.py             # Package exports, version
│   ├── config.py               # Hyperparameters and predefined scales
│   ├── graph_store.py          # Graph substrate (embedding store)
│   ├── sensory_cortex.py       # Multimodal input grounding
│   ├── temporal_binding.py     # Position pass-through (causal conv handles it)
│   ├── reasoning_core.py       # Wave Propagation + Resonance Memory
│   ├── metacognitive.py        # Adaptive thinking depth controller
│   ├── sage.py                 # Full model orchestrator
│   ├── generation.py           # Text generation utilities
│   └── train.py                # Character-level training pipeline
├── tests/                      # Test suite
│   └── test_model.py           # Model, config, and component tests
├── web/                        # Chat UI
│   └── index.html              # Single-file web interface
├── serve.py                    # HTTP chat server
├── train_chat.py               # BPE conversational training
├── pyproject.toml              # Python packaging & tool config
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Version history
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Key Innovations

1. **Wave Propagation**: Multi-scale causal convolutions replace attention for local sequence mixing. Deeper layers get wider receptive fields automatically. Inherently causal — no masking needed.

2. **Resonance Memory**: A compressed cumulative memory system provides global context. Each position writes to K shared slots and reads from the accumulated state. O(n) via `cumsum`, fully parallelizable on GPU.

3. **Metacognitive Control**: The model decides how hard to think per token. Confidence estimation and stagnation detection enable adaptive compute depth, saving resources on easy tokens.

4. **Weight Tying**: Embedding and output head share parameters, reducing model size by ~30% with no quality loss.

## Current Status

> [!NOTE]
> Sage is an experimental research architecture. It has not been trained at scale on trillion-token datasets. The architecture is novel and the scaling properties are theoretical projections based on complexity analysis, not empirical benchmarks at scale.

**What works:**
- ✅ Architecture compiles and trains
- ✅ Character-level Shakespeare demo converges
- ✅ BPE conversational training on Alpaca
- ✅ Interactive web chat UI

**What's needed:**
- Large-scale training validation (100B+ tokens)
- Learned graph topology (currently static embedding)
- ANN retrieval for billion-node graphs (replace brute-force similarity)
- Formal ablation studies (Wave vs. Attention, Resonance vs. KV cache)

## Citation

If you use Sage in your research, please cite:

```bibtex
@software{batule2026sage,
  title  = {Sage: Hybrid Graph-Cortex Language Model with Wave Propagation and Resonance Memory},
  author = {Batule, Gaurav},
  year   = {2026},
  url    = {https://github.com/gauravbatule/sage-architecture},
}
```

## License

MIT License — free for research and commercial use. See [LICENSE](LICENSE).

## Author

**Gaurav Batule**

---

*Sage explores the frontier between symbolic AI and neural networks — replacing attention with wave propagation and resonance memory for linear-time language modeling.*
