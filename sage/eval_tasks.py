"""
Sage 6.0 — Synthetic Evaluation Tasks

Benchmarks specifically designed to test the capabilities that matter for
non-attention architectures: associative recall, long-context retrieval,
and in-context learning.

These tasks answer the hard question: can Hebbian resonance memory recover
the associative recall that made attention dominant?

Tasks:
  1. Associative Recall — Given key-value pairs, recall value for query key
  2. Selective Copy — Copy only marked tokens from a sequence
  3. Induction Head — Detect and continue [A][B]...[A] → [B] patterns
  4. Needle in a Haystack — Find a planted fact in a long distractor context

Usage:
    python -m sage.eval_tasks --checkpoint sage_best.pt --device cpu
"""

__all__ = [
    "AssociativeRecallTask",
    "SelectiveCopyTask",
    "InductionHeadTask",
    "run_eval_suite",
]

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import SageConfig
from .sage import SageModel

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    task_name: str
    accuracy: float
    num_examples: int
    avg_loss: float
    details: Dict


class AssociativeRecallTask:
    """
    Tests whether Hebbian resonance memory can perform key-value associative recall.

    Format: key1=val1, key2=val2, ..., keyN=valN, query=keyQ, answer=
    Model must output valQ.

    This is the critical test — attention excels at this because it can
    directly look up any previous position. Resonance memory must store
    associations in its matrix state and retrieve via interference (M @ q).
    """

    def __init__(self, vocab_size: int, num_pairs: int = 8,
                 key_range: int = 50, val_range: int = 50,
                 num_examples: int = 200):
        self.vocab_size = vocab_size
        self.num_pairs = num_pairs
        self.key_range = min(key_range, vocab_size // 3)
        self.val_range = min(val_range, vocab_size // 3)
        self.num_examples = num_examples
        self.separator = min(vocab_size - 1, self.key_range + self.val_range + 1)
        self.query_marker = min(vocab_size - 1, self.separator + 1)

    def generate_example(self) -> Tuple[torch.Tensor, int]:
        keys = random.sample(range(1, self.key_range), self.num_pairs)
        vals = [random.randint(self.key_range, self.key_range + self.val_range - 1) for _ in keys]

        tokens = []
        for k, v in zip(keys, vals):
            tokens.extend([k, v, self.separator])

        query_idx = random.randint(0, len(keys) - 1)
        tokens.extend([self.query_marker, keys[query_idx]])
        target = vals[query_idx]

        return torch.tensor(tokens, dtype=torch.long), target

    @torch.no_grad()
    def evaluate(self, model: SageModel, device: str = "cpu") -> EvalResult:
        model.eval()
        model.to(device)
        correct = 0
        total_loss = 0.0

        for _ in range(self.num_examples):
            tokens, target = self.generate_example()
            tokens = tokens.unsqueeze(0).to(device)

            output = model(tokens)
            last_logits = output["logits"][0, -1, :]

            predicted = last_logits.argmax().item()
            if predicted == target:
                correct += 1

            target_t = torch.tensor([target], device=device)
            loss = F.cross_entropy(last_logits.unsqueeze(0), target_t)
            total_loss += loss.item()

        accuracy = correct / self.num_examples
        return EvalResult(
            task_name="associative_recall",
            accuracy=accuracy,
            num_examples=self.num_examples,
            avg_loss=total_loss / self.num_examples,
            details={
                "num_pairs": self.num_pairs,
                "correct": correct,
            },
        )


class SelectiveCopyTask:
    """
    Tests selective information routing through the architecture.

    A sequence of tokens with some marked (via a marker token). The model
    must output the marked tokens in order at the end.

    Tests whether cognitive routing + resonance memory can selectively
    attend to and remember specific positions.
    """

    def __init__(self, vocab_size: int, seq_len: int = 32,
                 num_marked: int = 4, num_examples: int = 200):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_marked = num_marked
        self.num_examples = num_examples
        self.marker = min(vocab_size - 1, vocab_size - 2)
        self.end_marker = min(vocab_size - 1, vocab_size - 3)
        self.content_range = max(2, vocab_size - 5)

    def generate_example(self) -> Tuple[torch.Tensor, List[int]]:
        positions = sorted(random.sample(range(self.seq_len), self.num_marked))

        tokens = []
        targets = []
        for i in range(self.seq_len):
            tok = random.randint(1, self.content_range)
            if i in positions:
                tokens.extend([self.marker, tok])
                targets.append(tok)
            else:
                tokens.append(tok)

        tokens.append(self.end_marker)
        for t in targets[:-1]:
            tokens.append(t)

        return torch.tensor(tokens, dtype=torch.long), targets[-1]

    @torch.no_grad()
    def evaluate(self, model: SageModel, device: str = "cpu") -> EvalResult:
        model.eval()
        model.to(device)
        correct = 0
        total_loss = 0.0

        for _ in range(self.num_examples):
            tokens, target = self.generate_example()
            tokens = tokens.unsqueeze(0).to(device)

            output = model(tokens)
            last_logits = output["logits"][0, -1, :]

            predicted = last_logits.argmax().item()
            if predicted == target:
                correct += 1

            target_t = torch.tensor([target], device=device)
            loss = F.cross_entropy(last_logits.unsqueeze(0), target_t)
            total_loss += loss.item()

        accuracy = correct / self.num_examples
        return EvalResult(
            task_name="selective_copy",
            accuracy=accuracy,
            num_examples=self.num_examples,
            avg_loss=total_loss / self.num_examples,
            details={"num_marked": self.num_marked, "correct": correct},
        )


class InductionHeadTask:
    """
    Tests in-context learning via induction head detection.

    Pattern: [A][B] ... filler ... [A] → model should predict [B]

    This is the canonical test for in-context learning. Transformers
    form "induction heads" that detect [A]→[B] patterns and complete them.
    Can Sage's harmonic waves + Hebbian memory do the same?
    """

    def __init__(self, vocab_size: int, filler_len: int = 16,
                 num_examples: int = 200):
        self.vocab_size = vocab_size
        self.filler_len = filler_len
        self.num_examples = num_examples
        self.content_range = max(2, vocab_size - 2)

    def generate_example(self) -> Tuple[torch.Tensor, int]:
        a = random.randint(1, self.content_range)
        b = random.randint(1, self.content_range)
        while b == a:
            b = random.randint(1, self.content_range)

        filler = [random.randint(1, self.content_range) for _ in range(self.filler_len)]
        while a in filler:
            filler = [random.randint(1, self.content_range) for _ in range(self.filler_len)]

        tokens = [a, b] + filler + [a]
        return torch.tensor(tokens, dtype=torch.long), b

    @torch.no_grad()
    def evaluate(self, model: SageModel, device: str = "cpu") -> EvalResult:
        model.eval()
        model.to(device)
        correct = 0
        total_loss = 0.0

        for _ in range(self.num_examples):
            tokens, target = self.generate_example()
            tokens = tokens.unsqueeze(0).to(device)

            output = model(tokens)
            last_logits = output["logits"][0, -1, :]

            predicted = last_logits.argmax().item()
            if predicted == target:
                correct += 1

            target_t = torch.tensor([target], device=device)
            loss = F.cross_entropy(last_logits.unsqueeze(0), target_t)
            total_loss += loss.item()

        accuracy = correct / self.num_examples
        return EvalResult(
            task_name="induction_head",
            accuracy=accuracy,
            num_examples=self.num_examples,
            avg_loss=total_loss / self.num_examples,
            details={
                "filler_len": self.filler_len,
                "correct": correct,
            },
        )


def run_eval_suite(
    model: SageModel,
    device: str = "cpu",
    num_examples: int = 200,
) -> List[EvalResult]:
    vocab = model.config.text_vocab_size
    results = []

    tasks = [
        AssociativeRecallTask(vocab, num_pairs=4, num_examples=num_examples),
        AssociativeRecallTask(vocab, num_pairs=8, num_examples=num_examples),
        SelectiveCopyTask(vocab, seq_len=32, num_marked=4, num_examples=num_examples),
        InductionHeadTask(vocab, filler_len=8, num_examples=num_examples),
        InductionHeadTask(vocab, filler_len=32, num_examples=num_examples),
    ]

    for task in tasks:
        t0 = time.time()
        result = task.evaluate(model, device=device)
        elapsed = time.time() - t0

        label = result.task_name
        if hasattr(task, "num_pairs"):
            label += f" (pairs={task.num_pairs})"
        elif hasattr(task, "filler_len"):
            label += f" (filler={task.filler_len})"

        logger.info(
            "  %-35s  acc=%.1f%%  loss=%.3f  (%.1fs)",
            label, result.accuracy * 100, result.avg_loss, elapsed,
        )
        results.append(result)

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Sage 6.0 — Synthetic Evaluation Suite"
    )
    parser.add_argument("--checkpoint", type=str, default="sage_best.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if config is None:
        from .config import get_config
        config = get_config("alpha")

    model = SageModel(config)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    model.to(device)

    logger.info("=" * 60)
    logger.info("  Sage 6.0 — Synthetic Evaluation Suite")
    logger.info("  Tests: associative recall, selective copy, induction heads")
    logger.info("  Device: %s | Examples per task: %d", device, args.num_examples)
    logger.info("=" * 60)

    results = run_eval_suite(model, device=device, num_examples=args.num_examples)

    logger.info("-" * 60)
    avg_acc = sum(r.accuracy for r in results) / len(results) * 100
    logger.info("  Average accuracy: %.1f%%", avg_acc)
    logger.info("-" * 60)

    if args.output:
        out = {
            "model": config.name,
            "results": [
                {
                    "task": r.task_name,
                    "accuracy": r.accuracy,
                    "loss": r.avg_loss,
                    "details": r.details,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
