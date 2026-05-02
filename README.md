# Sage 5.0 — Hybrid Graph-Cortex Language Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/gauravbatule/SAGE/actions/workflows/ci.yml/badge.svg)](https://github.com/gauravbatule/SAGE/actions/workflows/ci.yml)

> **A fundamentally new approach to language modeling that replaces attention with Wave Propagation and Resonance Memory, achieving linear-time sequence processing with constant memory inference.**

## What is Sage?

Sage is a **Hybrid Graph-Cortex** architecture that separates **knowledge storage** (a sparse topological graph) from **reasoning** (a compact dense core). Instead of the Transformer's O(n²) attention mechanism, Sage uses two complementary mechanisms:

1. **Causal Wave Propagation** — Multi-scale causal convolutions for local syntax and grammar understanding. Information flows causally through the sequence, with each position integrating signals from its local neighborhood at multiple scales.

2. **Resonance Memory** — A compressed neural whiteboard with K memory slots and exponential decay. Each position writes important information and reads relevant context via cumulative accumulation. This provides global context access in O(n·K·D) — linear in sequence length.

### Key Properties

- **Knowledge** lives in a sparse graph on disk (scalable to billions of nodes on NVMe)
- **Reasoning** happens in a small, reusable neural core that only loads active concepts
- **Memory** is **constant** — no KV cache, no growth with context length
- **Compute** is **adaptive** — easy tokens get 1 pass, complex reasoning gets up to 16
- **Training** supports mixed precision (AMP), gradient accumulation, and checkpoint resume

```
Token IDs → Graph Embedding → [Wave Mixing → Resonance Memory → MLP] × N → Output
                                         ↑ Repeated via Metacognitive Control
```

## What's New in 5.0

- **Exponential Decay in Resonance Memory** — Older context decays gracefully, preventing stale information from dominating
- **Per-Layer Learnable Scaling** — Layer-scale initialization for training stability at depth
- **Gradient Checkpointing** — Configurable memory-compute tradeoff for larger models
- **Dropout Throughout** — Configurable regularization in wave, resonance, and MLP blocks
- **Fixed Metacognitive Controller** — Iteration embeddings are now properly used for confidence estimation; added difficulty predictor
- **SSE Streaming Server** — Real-time token-by-token streaming via Server-Sent Events
- **Multi-turn Chat API** — Conversation context maintained across messages
- **Production Web UI** — Markdown rendering, code highlighting, theme toggle, conversation export, stop generation, keyboard shortcuts
- **GitHub Actions CI** — Automated linting and testing on Python 3.10-3.12
- **Docker Support** — Multi-stage build with CPU-only PyTorch
- **Benchmarking Tool** — Measure throughput, latency percentiles, and generation speed
- **Training Improvements** — Gradient accumulation, mixed precision AMP, cosine warmup LR, checkpoint resume, optional W&B logging

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
│  • Gated output with learned value projection + layer scale      │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 3: Resonance Memory (global understanding)                │
│  • K memory slots with cumulative write/read + exponential decay │
│  • Position i's memory = decayed summary of positions 0..i       │
│  • Gated integration with current representation + layer scale   │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 4: SwiGLU MLP (per-position reasoning)                   │
│  • Standard feed-forward with SiLU-gated linear unit             │
│  • Dropout + layer scale for training stability                  │
├──────────────────────────────────────────────────────────────────┤
│  LAYER 5: Metacognitive Controller                               │
│  • Adaptive compute: easy tokens = 1 pass, hard tokens = 8+     │
│  • Iteration-aware confidence estimation                         │
│  • Difficulty prediction for compute budgeting                   │
└──────────────────────────────────────────────────────────────────┘
```

## Quickstart

### Install

```bash
git clone https://github.com/gauravbatule/SAGE.git
cd SAGE
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

### Advanced Training

```bash
# Gradient accumulation (effective batch = 8 * 4 = 32)
python train_chat.py --batch-size 8 --grad-accum-steps 4

# Resume training from checkpoint
python train_chat.py --resume

# Enable W&B logging
python train_chat.py --wandb
```

### Run the Chat Server

```bash
python serve.py
# Open http://localhost:8888
```

### Run Benchmarks

```bash
python -m sage.benchmark --config alpha --device cpu
python -m sage.benchmark --config beta --output results.json
```

### Docker

```bash
docker build -t sage .
docker run -p 8888:8888 sage                    # Serve
docker run sage python -m sage.train            # Train
docker run sage python -m sage.benchmark        # Benchmark
```

### Use in Code

```python
from sage import SageModel, SageConfig, get_config

config = get_config("alpha")  # or "beta", "omega"
model = SageModel(config)

import torch
tokens = torch.randint(0, config.text_vocab_size, (1, 128))
output = model(tokens)
print(output["logits"].shape)  # (1, 128, vocab_size)
```

## Why Not Attention?

| Property | Transformer | Mamba | **Sage 5.0** |
|:---|:---|:---|:---|
| Sequence mixing | O(n²) attention | O(n) selective scan | **O(n·k) causal conv + O(n·K·D) resonance** |
| Inference memory | KV cache grows with context | Fixed state | **Fixed (constant)** |
| Params activated/token | 100% | 100% | **5-15% (sparse graph)** |
| Adaptive compute | Fixed depth | Fixed depth | **1-16 iterations** |
| Position encoding | Learned / RoPE | Implicit in recurrence | **Implicit in causal conv** |
| Parallelizable | Yes | Sequential scan | **Fully parallel** |
| Memory decay | N/A | Via state transitions | **Exponential decay (configurable)** |

### Predefined Scales

| Config | Graph Nodes | Core Params | Inference VRAM | Use Case |
|:---|:---|:---|:---|:---|
| Alpha | 1M | ~15M | <100MB | Research & prototyping |
| Beta | 100M | ~200M | ~2GB | Mid-scale experiments |
| Omega | 10B | ~1B | ~16GB | Full-scale training |

## API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/` | GET | Web chat UI |
| `/api/info` | GET | Model metadata |
| `/api/health` | GET | Health check |
| `/api/generate` | POST | Standard generation |
| `/api/stream` | POST | SSE streaming generation |
| `/api/chat` | POST | Multi-turn conversation |

## Project Structure

```
SAGE/
├── sage/                       # Core package
│   ├── __init__.py             # Package exports, version
│   ├── config.py               # Hyperparameters and predefined scales
│   ├── graph_store.py          # Graph substrate (embedding store)
│   ├── sensory_cortex.py       # Multimodal input grounding
│   ├── temporal_binding.py     # Position pass-through
│   ├── reasoning_core.py       # Wave Propagation + Resonance Memory
│   ├── metacognitive.py        # Adaptive thinking depth controller
│   ├── sage.py                 # Full model orchestrator
│   ├── generation.py           # Text generation utilities
│   ├── train.py                # Character-level training pipeline
│   └── benchmark.py            # Performance benchmarking
├── tests/                      # Test suite (37 tests)
│   └── test_model.py           # Model, config, and component tests
├── web/                        # Chat UI
│   └── index.html              # Single-file web interface
├── .github/workflows/ci.yml    # GitHub Actions CI
├── serve.py                    # HTTP chat server with streaming
├── train_chat.py               # BPE conversational training
├── Dockerfile                  # Docker support
├── pyproject.toml              # Python packaging & tool config
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Version history
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Key Innovations

1. **Wave Propagation**: Multi-scale causal convolutions replace attention for local sequence mixing. Deeper layers get wider receptive fields automatically. Inherently causal — no masking needed.

2. **Resonance Memory with Decay**: A compressed cumulative memory system with exponential decay provides global context. Each position writes to K shared slots and reads from the accumulated state. O(n) via `cumsum`, fully parallelizable on GPU. Decay prevents stale context from dominating.

3. **Metacognitive Control**: The model decides how hard to think per token. Iteration-aware confidence estimation and stagnation detection enable adaptive compute depth, saving resources on easy tokens. Difficulty prediction estimates needed iterations.

4. **Layer-Scale Initialization**: Per-layer learnable scaling factors initialized near zero enable stable training of deep models — a technique from vision transformers adapted for sequence processing.

5. **Weight Tying**: Embedding and output head share parameters, reducing model size by ~30% with no quality loss.

## Current Status

> [!NOTE]
> Sage is an experimental research architecture. It has not been trained at scale on trillion-token datasets. The architecture is novel and the scaling properties are theoretical projections based on complexity analysis, not empirical benchmarks at scale.

**What works:**
- Architecture compiles, trains, and passes 37 automated tests
- Character-level Shakespeare demo converges
- BPE conversational training on Alpaca with AMP and gradient accumulation
- Interactive web chat UI with markdown, code highlighting, and streaming
- SSE streaming server with multi-turn conversation support
- Docker containerization and GitHub Actions CI

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
  url    = {https://github.com/gauravbatule/SAGE},
}
```

## License

MIT License — free for research and commercial use. See [LICENSE](LICENSE).

## Author

**Gaurav Batule**

---

*Sage explores the frontier between symbolic AI and neural networks — replacing attention with wave propagation and resonance memory for linear-time language modeling.*
