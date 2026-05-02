"""
SAGE Benchmark — throughput, latency, memory, and generation speed.

Usage::

    python -m sage.benchmark                        # alpha config, cpu
    python -m sage.benchmark --config beta          # beta config
    python -m sage.benchmark --device cuda          # GPU benchmark
    python -m sage.benchmark --output results.json  # save JSON
"""

from __future__ import annotations

__all__ = ["run_benchmark", "main"]

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

from .config import SageConfig, get_config
from .sage import SageModel


@dataclass
class SequenceResult:
    seq_len: int
    throughput_tps: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float


@dataclass
class BenchmarkResult:
    config_name: str
    device: str
    param_breakdown: Dict[str, int]
    total_params: int
    peak_memory_mb: float
    sequence_results: List[SequenceResult] = field(default_factory=list)
    generation_tps: Optional[float] = None


def _percentile(values: List[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def run_benchmark(
    config: SageConfig,
    device: str = "cpu",
    seq_lengths: Optional[List[int]] = None,
    n_warmup: int = 3,
    n_trials: int = 20,
    gen_tokens: int = 50,
) -> BenchmarkResult:
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256, 512, 1024]

    model = SageModel(config).to(device)
    model.eval()

    param_breakdown = model.count_parameters()
    total_params = param_breakdown["total"]

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    peak_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    seq_results: List[SequenceResult] = []
    for seq_len in seq_lengths:
        if seq_len > config.n_active_limit:
            continue

        ids = torch.randint(0, config.text_vocab_size, (1, seq_len), device=device)
        latencies: List[float] = []

        with torch.no_grad():
            for _ in range(n_warmup):
                _sync(device)
                model(ids)
                _sync(device)

            for _ in range(n_trials):
                _sync(device)
                t0 = time.perf_counter()
                model(ids)
                _sync(device)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        tps = seq_len / (p50 / 1000.0)

        if device.startswith("cuda"):
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated(device) / (1024 ** 2))

        seq_results.append(SequenceResult(
            seq_len=seq_len,
            throughput_tps=round(tps, 1),
            latency_ms_p50=round(p50, 2),
            latency_ms_p95=round(p95, 2),
            latency_ms_p99=round(p99, 2),
        ))

    gen_tps = None
    try:
        prompt = torch.randint(0, config.text_vocab_size, (1, 16), device=device)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=gen_tokens, temperature=1.0, top_p=0.9)
        _sync(device)
        gen_tps = round(gen_tokens / max(time.perf_counter() - t0, 1e-9), 2)
    except Exception:
        pass

    return BenchmarkResult(
        config_name=config.name,
        device=device,
        param_breakdown=param_breakdown,
        total_params=total_params,
        peak_memory_mb=round(peak_mb, 2),
        sequence_results=seq_results,
        generation_tps=gen_tps,
    )


def _fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def print_results(r: BenchmarkResult) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  SAGE Benchmark — {r.config_name}  |  device: {r.device}")
    print(sep)

    print("\nParameter Breakdown:")
    for k, v in r.param_breakdown.items():
        print(f"    {k:<22} {_fmt(v):>10}")

    label = "Peak GPU Memory" if r.device.startswith("cuda") else "Estimated Memory"
    print(f"\n{label}: {r.peak_memory_mb:.1f} MiB")

    print(f"\n{'Seq Len':>10} {'Tok/s':>12} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10}")
    print("  " + "-" * 56)
    for s in r.sequence_results:
        print(f"  {s.seq_len:>8} {s.throughput_tps:>12,.1f} {s.latency_ms_p50:>10.2f} "
              f"{s.latency_ms_p95:>10.2f} {s.latency_ms_p99:>10.2f}")

    if r.generation_tps is not None:
        print(f"\nAutoregressive Generation: {r.generation_tps:,.2f} tok/s")

    print(f"\n{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m sage.benchmark",
        description="Benchmark SAGE model throughput, latency, and memory.",
    )
    parser.add_argument("--config", default="alpha")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-lengths", default="32,64,128,256,512,1024")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--gen-tokens", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seq_lengths = [int(s.strip()) for s in args.seq_lengths.split(",") if s.strip()]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu.", file=sys.stderr)
        device = "cpu"

    config = get_config(args.config)

    print(f"Running benchmark: config={args.config}, device={device}")
    result = run_benchmark(config, device, seq_lengths, args.warmup, args.trials, args.gen_tokens)
    print_results(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"JSON results written to: {args.output}")


if __name__ == "__main__":
    main()
