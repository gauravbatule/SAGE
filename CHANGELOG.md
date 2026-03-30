# Changelog

All notable changes to the Sage Architecture project are documented here.

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
