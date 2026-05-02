# Changelog

All notable changes to the Sage Architecture project are documented here.

## [5.0.0] — 2026-05-02

### Architecture
- **Exponential decay in Resonance Memory** — configurable `resonance_decay` factor prevents stale context from dominating. Older writes decay by `decay^distance`.
- **Configurable resonance dimensions** — `resonance_slots` (default 32) and `resonance_mem_dim` (default 64) are now config-driven instead of hardcoded.
- **Per-layer learnable scaling** — `layer_scale_init` enables stable training of deep models via near-zero initialized per-layer scales.
- **Dropout throughout** — configurable `dropout` applied after wave mixer, resonance memory, and MLP in each block.
- **Gradient checkpointing** — `gradient_checkpointing` config flag enables memory-compute tradeoff for training larger models.
- **Fixed Metacognitive Controller** — iteration embeddings are now properly projected and used for confidence estimation (were computed but unused in v4.0). Added difficulty predictor.
- **Graph substrate forward method** — added `forward()` convenience method with optional embedding normalization.
- **Sensory cortex OOV warning** — out-of-vocabulary tokens now trigger a RuntimeWarning instead of silently clamping.
- Fixed duplicate docstring in ResonanceMemory (steps 3-4 were listed twice).

### Training
- **Gradient accumulation** — `--grad-accum-steps` for effective larger batch sizes on consumer GPUs.
- **Mixed precision (AMP)** — auto-detects CUDA and uses `torch.amp.autocast` + `GradScaler`.
- **Cosine LR with linear warmup** — replaces fixed `CosineAnnealingLR(T_max=1000)` with proper warmup phase (10% of total steps).
- **Checkpoint resume** — `--resume` flag loads model, optimizer, scaler state, and training progress.
- **Fixed double loss computation** — `train_chat.py` no longer computes `cross_entropy` twice. Uses `-100` padding in targets (standard ignore_index).
- **ETA display** — training loop shows estimated time remaining.
- **Optional W&B logging** — `--wandb` flag for Weights & Biases integration.

### Server
- **SSE streaming** — `POST /api/stream` returns token-by-token Server-Sent Events.
- **Multi-turn chat** — `POST /api/chat` accepts conversation history with role/content messages.
- **Health check** — `GET /api/health` returns model status.
- **CORS support** — proper preflight handling with `OPTIONS` method.
- **Thread safety** — `threading.Lock` around model inference.
- **Request validation** — JSON error responses with appropriate status codes.

### Web UI
- **Markdown rendering** — responses rendered with marked.js.
- **Code highlighting** — syntax highlighting via highlight.js with copy-to-clipboard.
- **Theme toggle** — dark/light mode with localStorage persistence.
- **Stop generation** — abort button with AbortController support.
- **Copy message** — per-message copy button.
- **Conversation export** — download as JSON or Markdown.
- **Connection status** — live indicator showing server connectivity.
- **Keyboard shortcuts** — Ctrl+Enter send, Ctrl+N new chat, Esc stop, etc.
- **Smart auto-scroll** — doesn't auto-scroll when user scrolls up.
- **Mobile responsive** — fully responsive on all screen sizes.

### Infrastructure
- **GitHub Actions CI** — automated linting (ruff) and testing (pytest) on Python 3.10-3.12.
- **Dockerfile** — multi-stage build with CPU-only PyTorch, health check.
- **Benchmarking tool** — `python -m sage.benchmark` measures throughput, latency percentiles, and generation speed.
- Updated `pyproject.toml` to v5.0.0 with correct repository URLs.
- Added Python 3.13 classifier.

### Testing
- Expanded test suite from 21 to **37 tests**.
- Added tests for: gradient checkpointing, dropout, layer scale, resonance decay, graph forward/normalize, OOV warnings, metacognitive difficulty, version exports.

### Removed
- `rope_theta` config field (not used — this isn't a transformer).
- Renamed `max_seq_len` to `context_length` for clarity.

## [4.0.0] — 2026-03-30

### Changed
- **Replaced attention with Wave Propagation + Resonance Memory** — the core architectural innovation.
  - Multi-scale causal convolutions for local syntax understanding (O(n·k) per layer).
  - Resonance Memory for global context via cumulative write/read (O(n·K·D) per layer).
  - Total complexity: linear in sequence length. No attention. No recurrence.
- Upgraded tokenizer from character-level to BPE via `tiktoken` (cl100k_base, 100K vocab).
- Implemented weight tying between embedding and LM head (~30% parameter savings).
- Removed dead graph edge parameters (`edge_targets`, `edge_weights`, `edge_mask`).
- Direct embedding to `core_dim` (eliminated lossy `node_dim` bottleneck).
- Temporal Binding simplified to pass-through (position is implicit in causal convolutions).
- Added mixed-precision (AMP) training support for consumer GPUs.

### Added
- `sage/generation.py` — consolidated generation utilities.
- `pyproject.toml` — proper Python packaging.
- `tests/` — automated test suite.
- `CONTRIBUTING.md` — contribution guidelines.
- Web chat UI for interactive demos.

### Removed
- Oscillatory phase codes (replaced by implicit position in causal convolutions).
- `node_dim` config parameter (direct core_dim embedding instead).
- Explicit positional encoding modules.

## [3.5.0] — 2026-03-15

### Added
- Initial Hybrid Graph-Cortex architecture.
- Character-level Shakespeare training demo.
- Metacognitive controller with adaptive thinking depth.
- Topological knowledge graph substrate.
- Conversational training on Alpaca dataset.
