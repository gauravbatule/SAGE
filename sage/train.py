"""
Sage 4.0 — Training Pipeline

Downloads a small text corpus (Shakespeare), trains a character-level Sage model,
and demonstrates that the architecture learns language patterns.

Usage::

    python -m sage.train                    # Train with defaults
    python -m sage.train --epochs 20        # More epochs
    python -m sage.train --batch-size 64    # Larger batches
"""

__all__ = ["CharTokenizer", "TextChunkDataset", "Trainer", "generate_text"]

import argparse
import logging
import math
import os
import time
import urllib.request
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import SageConfig
from .generation import top_p_sample
from .sage import SageModel

logger = logging.getLogger(__name__)

# ─── Dataset ──────────────────────────────────────────────────────────────────

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_data(path: str = "data/shakespeare.txt") -> str:
    """Download Shakespeare corpus if not cached."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        logger.info("Downloading Shakespeare corpus...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        logger.info("Saved to %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Character-level tokenizer — simplest possible, no external deps."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(i, "?") for i in ids)


class TextChunkDataset(Dataset):
    """Chunks tokenized text into fixed-length sequences for training."""

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end].long()
        return chunk[:-1], chunk[1:]  # (input, target)


# ─── Trainer ──────────────────────────────────────────────────────────────────


class Trainer:
    """Training loop for Sage 4.0."""

    def __init__(
        self,
        model: SageModel,
        train_dataset: TextChunkDataset,
        val_dataset: Optional[TextChunkDataset] = None,
        lr: float = 3e-4,
        batch_size: int = 16,
        grad_clip: float = 1.0,
        device: str = "auto",
    ) -> None:
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "mps" if device == "auto" and torch.backends.mps.is_available()
            else "cpu" if device == "auto" else device
        )
        self.model = model.to(self.device)
        self.grad_clip = grad_clip

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=0,
        )
        self.val_loader = (
            DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                drop_last=True, num_workers=0,
            )
            if val_dataset else None
        )

        # Separate parameter groups: graph learns slower (knowledge stability)
        self.optimizer = torch.optim.AdamW([
            {"params": model.graph.parameters(), "lr": lr * 0.3},
            {"params": model.core.parameters(), "lr": lr},
            {"params": model.binder.parameters(), "lr": lr},
            {"params": model.metacog.parameters(), "lr": lr},
            {"params": model.senses.parameters(), "lr": lr},
        ], weight_decay=0.1)

        # Cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.1,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute validation loss."""
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss, count = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x, targets=y)
            total_loss += out["loss"].item()
            count += 1
        return total_loss / max(count, 1)

    def train(self, epochs: int = 10, log_interval: int = 20) -> None:
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("  Training on %s", self.device.upper())
        logger.info("  Batches/epoch: %d", len(self.train_loader))
        logger.info("  Epochs: %d", epochs)
        logger.info("=" * 60)

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            t0 = time.time()

            for step, (x, y) in enumerate(self.train_loader, 1):
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x, targets=y)
                loss = out["loss"]

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_tokens += x.numel()
                global_step += 1

                if step % log_interval == 0:
                    avg = epoch_loss / step
                    elapsed = time.time() - t0
                    tok_s = epoch_tokens / max(elapsed, 1e-6)
                    ppl = math.exp(min(avg, 20))
                    lr_now = self.optimizer.param_groups[1]["lr"]
                    logger.info(
                        "  Epoch %2d | Step %4d/%d | Loss %.4f | PPL %7.1f | "
                        "Tok/s %6.0f | LR %.2e",
                        epoch, step, len(self.train_loader), avg, ppl, tok_s, lr_now,
                    )

            # End of epoch
            avg_train = epoch_loss / len(self.train_loader)
            val_loss = self.evaluate()
            elapsed = time.time() - t0

            logger.info(
                "  ---- Epoch %2d done in %.1fs | Train Loss %.4f | Val Loss %.4f | "
                "Train PPL %.1f | Val PPL %.1f",
                epoch, elapsed, avg_train, val_loss,
                math.exp(min(avg_train, 20)), math.exp(min(val_loss, 20)),
            )

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "sage_best.pt")
                logger.info("  * Saved best model (val_loss=%.4f)", val_loss)

        logger.info("=" * 60)
        logger.info("  Training complete. Best val loss: %.4f", best_val_loss)
        logger.info("=" * 60)


# ─── Generation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_text(
    model: SageModel,
    tokenizer: CharTokenizer,
    prompt: str = "ROMEO:",
    max_tokens: int = 200,
    temperature: float = 0.8,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt using character-level tokenizer."""
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        if input_ids.shape[1] > model.config.n_active_limit:
            input_ids = input_ids[:, -model.config.n_active_limit:]

        model.metacog.reset()
        out = model(input_ids)
        next_token = top_p_sample(out["logits"][:, -1, :], temperature)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m sage.train``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train Sage 4.0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  SAGE 4.0 — Wave Propagation + Resonance Memory")
    logger.info("  Character-Level Training Demo")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    raw = download_data()
    tokenizer = CharTokenizer(raw)
    logger.info("Dataset: %s characters, vocab size: %d", f"{len(raw):,}", tokenizer.vocab_size)

    all_ids = torch.tensor(tokenizer.encode(raw), dtype=torch.long)
    split = int(len(all_ids) * 0.9)
    train_ds = TextChunkDataset(all_ids[:split], args.seq_len)
    val_ds = TextChunkDataset(all_ids[split:], args.seq_len)
    logger.info("Train chunks: %s | Val chunks: %s", f"{len(train_ds):,}", f"{len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────
    config = SageConfig(
        name="sage-shakespeare",
        n_nodes=tokenizer.vocab_size + 1000,
        n_active_limit=256,
        text_vocab_size=tokenizer.vocab_size,
        core_dim=256,
        core_n_heads=4,
        core_n_layers=4,
        core_mlp_ratio=2.667,
        max_seq_len=args.seq_len * 4,
        max_think_iterations=2,
        min_think_iterations=1,
        metacog_dim=64,
        init_std=0.02,
    )

    model = SageModel(config)
    params = model.count_parameters()

    logger.info("-" * 50)
    logger.info("  Model: %s", config.name)
    for k, v in params.items():
        logger.info("  %-25s: %12s", k, f"{v:,}")
    logger.info("-" * 50)

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model, train_ds, val_ds,
        lr=args.lr, batch_size=args.batch_size, device=args.device,
    )
    trainer.train(epochs=args.epochs)

    # ── Generate ──────────────────────────────────────────────────────
    device = trainer.device
    prompts = ["ROMEO:", "To be or ", "The king "]

    logger.info("=" * 60)
    logger.info("  Sample Generations:")
    logger.info("=" * 60)
    for prompt in prompts:
        text = generate_text(model, tokenizer, prompt, max_tokens=150, device=device)
        logger.info("  Prompt: '%s'", prompt)
        logger.info("  Output: %s", text[:300])
        logger.info("-" * 50)

    logger.info("Done! Model saved to sage_best.pt")


if __name__ == "__main__":
    main()
