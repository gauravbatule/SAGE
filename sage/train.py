"""
Sage 6.0 — Character-Level Training Pipeline

Downloads a small text corpus (Shakespeare), trains a character-level Sage model,
and demonstrates that the architecture learns language patterns.

Usage::

    python -m sage.train                         # Train with defaults
    python -m sage.train --epochs 20             # More epochs
    python -m sage.train --batch-size 64         # Larger batches
    python -m sage.train --grad-accum-steps 4    # Gradient accumulation
    python -m sage.train --resume                # Resume from sage_best.pt
    python -m sage.train --wandb                 # Enable W&B logging
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


# ─── LR Schedule ──────────────────────────────────────────────────────────────


def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine annealing schedule with linear warmup phase."""
    if step < warmup_steps:
        # Linear warmup from 0 to max_lr
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ─── Trainer ──────────────────────────────────────────────────────────────────


class Trainer:
    """Training loop for Sage 6.0 with warmup, AMP, gradient accumulation, and resume."""

    def __init__(
        self,
        model: SageModel,
        train_dataset: TextChunkDataset,
        val_dataset: Optional[TextChunkDataset] = None,
        lr: float = 3e-4,
        batch_size: int = 16,
        grad_clip: float = 1.0,
        device: str = "auto",
        grad_accum_steps: int = 1,
    ) -> None:
        # Device resolution
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, grad_accum_steps)

        # Mixed precision — DISABLED: Hebbian memory produces NaN gradients in float16
        self.use_amp = False
        self.scaler = None

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

        # Collect parameters, avoiding duplicates from weight tying
        seen = set()
        param_groups = []
        for name, group_lr in [
            ("graph", lr * 0.3),
            ("core", lr),
            ("phase", lr),
            ("metacog", lr),
            ("senses", lr),
        ]:
            params = []
            for p in getattr(model, name).parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
            if params:
                param_groups.append({"params": params, "lr": group_lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)

        self.base_lr = lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute validation loss."""
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss, count = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self.device, enabled=self.use_amp):
                out = self.model(x, targets=y)
            total_loss += out["loss"].item()
            count += 1
        return total_loss / max(count, 1)

    def _set_lr(self, lr: float) -> None:
        """Set LR on all param groups, respecting the graph's slower rate."""
        for i, pg in enumerate(self.optimizer.param_groups):
            # Group 0 is graph — keep its 0.3x ratio
            pg["lr"] = lr * 0.3 if i == 0 else lr

    def train(
        self,
        epochs: int = 10,
        log_interval: int = 20,
        wandb_run=None,
    ) -> None:
        """Full training loop with warmup, AMP, and gradient accumulation."""
        batches_per_epoch = len(self.train_loader)
        # Effective optimizer steps per epoch (divided by accum)
        opt_steps_per_epoch = max(1, batches_per_epoch // self.grad_accum_steps)
        total_opt_steps = epochs * opt_steps_per_epoch
        warmup_steps = max(1, total_opt_steps // 10)  # 10% warmup

        logger.info("=" * 60)
        logger.info("  Training on %s", self.device.upper())
        logger.info("  Batches/epoch : %d", batches_per_epoch)
        logger.info("  Grad accum    : %d  (eff batch x%d)", self.grad_accum_steps, self.grad_accum_steps)
        logger.info("  Epochs        : %d", epochs)
        logger.info("  Total opt steps: %d  (warmup: %d)", total_opt_steps, warmup_steps)
        logger.info("  Mixed precision: %s", "ON (CUDA AMP)" if self.use_amp else "OFF")
        logger.info("=" * 60)

        global_opt_step = 0
        best_val_loss = float("inf")
        training_start = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            for step, (x, y) in enumerate(self.train_loader, 1):
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass under AMP
                with torch.amp.autocast(self.device, enabled=self.use_amp):
                    out = self.model(x, targets=y)
                    # Scale loss by accumulation steps so gradients average correctly
                    loss = out["loss"] / self.grad_accum_steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.grad_accum_steps  # unscaled for logging
                epoch_tokens += x.numel()

                # Optimizer step every grad_accum_steps micro-batches
                is_accum_boundary = (step % self.grad_accum_steps == 0) or (step == batches_per_epoch)
                if is_accum_boundary:
                    # Update LR before the step
                    lr_now = cosine_lr_with_warmup(
                        global_opt_step, warmup_steps, total_opt_steps,
                        self.base_lr, self.base_lr * 0.1,
                    )
                    self._set_lr(lr_now)

                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    global_opt_step += 1

                # Logging
                if step % log_interval == 0:
                    avg = epoch_loss / step
                    elapsed = time.time() - t0
                    tok_s = epoch_tokens / max(elapsed, 1e-6)
                    ppl = math.exp(min(avg, 20))
                    lr_display = self.optimizer.param_groups[1]["lr"]

                    # Estimated time remaining
                    steps_done = (epoch - 1) * batches_per_epoch + step
                    steps_total = epochs * batches_per_epoch
                    total_elapsed = time.time() - training_start
                    eta_s = total_elapsed / steps_done * (steps_total - steps_done)
                    eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"

                    logger.info(
                        "  Epoch %2d | Step %4d/%d | Loss %.4f | PPL %7.1f | "
                        "Tok/s %6.0f | LR %.2e | ETA %s",
                        epoch, step, batches_per_epoch, avg, ppl, tok_s, lr_display, eta_str,
                    )

                    if wandb_run is not None:
                        wandb_run.log({
                            "train/loss": avg,
                            "train/ppl": ppl,
                            "train/lr": lr_display,
                            "train/tok_per_sec": tok_s,
                            "step": global_opt_step,
                        })

            # End of epoch
            avg_train = epoch_loss / len(self.train_loader)
            val_loss = self.evaluate()
            elapsed = time.time() - t0

            val_ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")
            logger.info(
                "  ---- Epoch %2d done in %.1fs | Train Loss %.4f | Val Loss %.4f | "
                "Train PPL %.1f | Val PPL %.1f",
                epoch, elapsed, avg_train, val_loss,
                math.exp(min(avg_train, 20)), val_ppl,
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "epoch/train_loss": avg_train,
                    "epoch/val_loss": val_loss,
                    "epoch/val_ppl": val_ppl,
                })

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "global_opt_step": global_opt_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                }, "sage_best.pt")
                logger.info("  * Saved best model (val_loss=%.4f)", val_loss)

        total_time = time.time() - training_start
        logger.info("=" * 60)
        logger.info("  Training complete in %.1fs. Best val loss: %.4f", total_time, best_val_loss)
        logger.info("=" * 60)

    def load_checkpoint(self, path: str = "sage_best.pt") -> int:
        """Load checkpoint from path. Returns the starting epoch."""
        if not os.path.exists(path):
            logger.warning("Checkpoint %s not found — starting from scratch.", path)
            return 1
        ckpt = torch.load(path, map_location=self.device)
        # Support both old (bare state_dict) and new (full dict) formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.scaler and ckpt.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("best_val_loss", float("inf"))
            logger.info(
                "Resumed from %s (epoch %d, best_val=%.4f)",
                path, ckpt.get("epoch", 0), best_val,
            )
        else:
            # Old format: bare state dict saved directly
            self.model.load_state_dict(ckpt)
            start_epoch = 1
            logger.info("Loaded weights from %s (old format — optimizer state not restored).", path)
        return start_epoch


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

    parser = argparse.ArgumentParser(
        description=(
            "Train Sage 6.0 on Shakespeare (character-level). "
            "Supports gradient accumulation, AMP, cosine-with-warmup LR, "
            "checkpoint resume, and optional W&B logging."
        )
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch size per step")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for training chunks")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto | cpu | cuda | mps")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Number of micro-batches to accumulate before an optimizer step (default: 1 = no accum)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from sage_best.pt if it exists",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging (requires wandb to be installed)",
    )
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  SAGE 6.0 — Harmonic Waves + Hebbian Resonance Memory")
    logger.info("  Character-Level Training Demo")
    logger.info("=" * 60)

    # ── Optional W&B ──────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(
                project="sage-char",
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "lr": args.lr,
                    "grad_accum_steps": args.grad_accum_steps,
                },
            )
            logger.info("W&B run: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging. pip install wandb")

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
        core_n_layers=4,
        core_mlp_ratio=2.667,
        context_length=args.seq_len * 4,
        resonance_n_slots=4,
        resonance_mem_dim=32,
        resonance_decay_init=0.95,
        sparse_k_ratio=0.3,
        dropout=0.1,
        layer_scale_init=1.0,
        max_think_iterations=1,
        min_think_iterations=1,
        max_train_iterations=1,
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

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model, train_ds, val_ds,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        grad_accum_steps=args.grad_accum_steps,
    )

    # ── Resume ────────────────────────────────────────────────────────
    if args.resume:
        trainer.load_checkpoint("sage_best.pt")

    # ── Train ─────────────────────────────────────────────────────────
    trainer.train(epochs=args.epochs, wandb_run=wandb_run)

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

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Done! Model saved to sage_best.pt")


if __name__ == "__main__":
    main()
