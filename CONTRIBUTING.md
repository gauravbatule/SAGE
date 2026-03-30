# Contributing to Sage Architecture

Thank you for your interest in contributing to Sage! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/gauravbatule/sage-architecture.git
cd sage-architecture

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting. Run before submitting:

```bash
ruff check sage/
ruff format sage/
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Ensure all tests pass** before submitting.
4. **Update documentation** if you change public APIs.
5. **Keep commits atomic** — one logical change per commit.

## Architecture Overview

Before contributing, familiarize yourself with the core components:

- **`sage/reasoning_core.py`** — Wave Propagation + Resonance Memory (the core innovation)
- **`sage/sage.py`** — Model orchestrator
- **`sage/config.py`** — All hyperparameters
- **`sage/generation.py`** — Text generation utilities

## Reporting Issues

- Use [GitHub Issues](https://github.com/gauravbatule/sage-architecture/issues)
- Include your Python version, PyTorch version, and hardware
- Provide a minimal reproduction script if possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
