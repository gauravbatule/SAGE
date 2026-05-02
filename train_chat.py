"""
Sage 6.0 — Conversational Training with BPE Tokenizer

Trains the Sage model on the Stanford Alpaca instruction-following dataset
using tiktoken BPE tokenization, mixed-precision training, cosine learning
rate scheduling with warmup, gradient accumulation, and checkpoint resume.

Usage::

    python train_chat.py                         # Train with defaults
    python train_chat.py --epochs 20             # More epochs
    python train_chat.py --batch-size 16         # Larger batches
    python train_chat.py --grad-accum-steps 4    # Gradient accumulation
    python train_chat.py --resume                # Resume from checkpoint
    python train_chat.py --wandb                 # Enable W&B logging
"""

import argparse
import json
import logging
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import tiktoken

from sage.config import SageConfig
from sage.generation import extract_assistant_response, top_p_sample
from sage.sage import SageModel

logger = logging.getLogger(__name__)

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

# Padding sentinel: -100 is the standard ignore_index used by cross_entropy
# (matches the model's own ignore_index in sage.py)
PAD_TARGET_ID = -100


def download_alpaca(path: str = "data/alpaca_data.json") -> list:
    """Download Alpaca dataset if not cached."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(path):
        logger.info("Downloading Alpaca dataset...")
        urllib.request.urlretrieve(ALPACA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_conversations(raw_data: list) -> list:
    """Format Alpaca data into chat turns."""
    conversations = []
    for item in raw_data:
        instruction = item["instruction"].strip()
        inp = item.get("input", "").strip()
        output = item["output"].strip()

        if len(output) < 10 or len(instruction) < 5:
            continue

        user = f"{instruction} {inp}" if inp else instruction
        conversations.append(f"User: {user}\nAssistant: {output}")
    return conversations


class ChatDataset(Dataset):
    """
    BPE-tokenized conversation dataset.

    Padding positions in the *target* tensor are set to PAD_TARGET_ID (-100)
    so that the model's built-in cross_entropy (ignore_index=-100) skips them —
    eliminating the need for a separate manual loss computation.
    """

    def __init__(
        self,
        conversations: list,
        enc: tiktoken.Encoding,
        max_len: int = 256,
    ) -> None:
        self.samples: list = []
        logger.info("Encoding %d conversations with BPE...", len(conversations))
        eot = enc.eot_token

        for i, conv in enumerate(conversations):
            if i % 10000 == 0 and i > 0:
                logger.info("  %d/%d encoded...", i, len(conversations))

            ids = enc.encode(conv, allowed_special=set())
            ids = ids + [eot]

            if len(ids) < 8 or len(ids) > max_len:
                continue

            # Pad input with eot token (won't affect targets)
            pad_len = max_len + 1 - len(ids)
            padded_input = ids + [eot] * pad_len
            padded_input = padded_input[:max_len + 1]

            # Target: shifted by 1, padding positions use PAD_TARGET_ID=-100
            padded_target = ids[1:] + [PAD_TARGET_ID] * (max_len - len(ids) + 1)
            padded_target = padded_target[:max_len]

            x = torch.tensor(padded_input[:max_len], dtype=torch.long)
            y = torch.tensor(padded_target, dtype=torch.long)
            self.samples.append((x, y))

        logger.info("Dataset ready: %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


@torch.no_grad()
def generate_response(
    model: SageModel,
    enc: tiktoken.Encoding,
    prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.7,
    device: str = "cpu",
) -> str:
    """Generate a single assistant response for evaluation."""
    model.eval()
    full_prompt = f"User: {prompt}\nAssistant:"
    ids = enc.encode(full_prompt, allowed_special=set())
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = len(ids)

    for _ in range(max_tokens):
        if input_ids.shape[1] > model.config.n_active_limit:
            input_ids = input_ids[:, -model.config.n_active_limit:]

        model.metacog.reset()
        out = model(input_ids)
        next_token = top_p_sample(out["logits"][:, -1, :], temperature, top_p=0.9)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        tok_id = next_token.item()
        if tok_id == enc.eot_token:
            break

        # Early stop if the model begins a new user turn
        if input_ids.shape[1] > prompt_len + 3:
            tail = enc.decode(input_ids[0, -10:].tolist())
            if "User:" in tail:
                break

    full = enc.decode(input_ids[0].tolist())
    return extract_assistant_response(full, len(full_prompt))


def cosine_lr(
    step: int,
    warmup: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine annealing schedule with linear warmup."""
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    """Entry point for conversational training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Sage 6.0 — Conversational Training with BPE. "
            "Trains on Stanford Alpaca with cosine-warmup LR, AMP, gradient "
            "accumulation, and checkpoint resume."
        )
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size (8 for 4 GB VRAM)")
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--n-conversations", type=int, default=15000, help="Max conversations to use")
    parser.add_argument("--output", type=str, default="sage_chat_model.pt", help="Output checkpoint path")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from --output checkpoint if it exists",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging (requires wandb to be installed)",
    )
    args = parser.parse_args()

    grad_accum_steps = max(1, args.grad_accum_steps)

    logger.info("=" * 60)
    logger.info("  SAGE 6.0 — Harmonic Waves + Hebbian Resonance Memory")
    logger.info("  BPE Tokenizer + Conversational Training")
    logger.info("=" * 60)

    # ── Optional W&B ──────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(
                project="sage-chat",
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "effective_batch": args.batch_size * grad_accum_steps,
                    "max_len": args.max_len,
                    "lr": args.lr,
                },
            )
            logger.info("W&B run: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging. pip install wandb")

    # ── BPE tokenizer ─────────────────────────────────────────────────
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab
    logger.info("BPE Tokenizer: %s tokens (cl100k_base)", f"{vocab_size:,}")

    # ── Data ──────────────────────────────────────────────────────────
    raw = download_alpaca()
    conversations = format_conversations(raw)
    logger.info("Formatted %d conversations", len(conversations))

    conversations = conversations[:args.n_conversations]
    logger.info("Using %d conversations", len(conversations))

    # ChatDataset uses PAD_TARGET_ID=-100 so model.forward handles the ignore_index
    dataset = ChatDataset(conversations, enc, max_len=args.max_len)
    split = int(len(dataset) * 0.95)
    train_set = torch.utils.data.Subset(dataset, range(split))
    val_set = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ── Model ─────────────────────────────────────────────────────────
    config = SageConfig(
        name="sage-chat-v6",
        n_nodes=vocab_size,
        n_active_limit=args.max_len,
        text_vocab_size=vocab_size,
        core_dim=256,
        core_n_layers=6,
        core_mlp_ratio=2.667,
        context_length=args.max_len + 128,
        max_think_iterations=2,
        min_think_iterations=1,
        max_train_iterations=2,
        metacog_dim=64,
        weight_tying=True,
        init_std=0.02,
        resonance_n_slots=4,
        resonance_mem_dim=32,
        resonance_decay_init=0.95,
        sparse_k_ratio=0.3,
        dropout=0.1,
        layer_scale_init=1e-4,
    )

    model = SageModel(config)
    counts = model.count_parameters()
    logger.info("--- Parameter Budget ---")
    for k, v in counts.items():
        pct = (v / counts["total"] * 100) if counts.get("total", 0) > 0 else 0
        logger.info("  %-20s: %10s (%5.1f%%)", k, f"{v:,}", pct)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    logger.info("Device: %s | AMP: %s | Grad accum: %d", device, use_amp, grad_accum_steps)
    model = model.to(device)

    # ── Optimizer & LR schedule ───────────────────────────────────────
    max_lr = args.lr
    min_lr = 1e-5
    batches_per_epoch = len(train_loader)
    # Optimizer steps per epoch = batches / accumulation steps
    opt_steps_per_epoch = max(1, batches_per_epoch // grad_accum_steps)
    max_steps = opt_steps_per_epoch * args.epochs
    # Warmup for 10% of total optimizer steps (minimum 50 steps)
    warmup_steps = max(50, max_steps // 10)

    logger.info(
        "LR schedule: warmup %d steps → cosine to %d total opt steps",
        warmup_steps, max_steps,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=0.1, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 1
    best_val = float("inf")
    global_opt_step = 0

    if args.resume and os.path.exists(args.output):
        logger.info("Resuming from %s ...", args.output)
        ckpt = torch.load(args.output, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scaler and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("best_val", float("inf"))
            global_opt_step = ckpt.get("global_opt_step", 0)
            logger.info(
                "Resumed: epoch %d, best_val=%.4f, opt_step=%d",
                start_epoch - 1, best_val, global_opt_step,
            )
        else:
            logger.warning("Checkpoint format not recognized — starting fresh.")
    elif args.resume:
        logger.warning("--resume specified but %s not found — starting fresh.", args.output)

    logger.info("Training for %d epochs (%d opt steps)...", args.epochs, max_steps)
    logger.info("-" * 60)

    # ── Training loop ─────────────────────────────────────────────────
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            # Forward: pass targets so model computes loss with ignore_index=-100
            # (padding positions in y are already set to -100 by ChatDataset)
            with torch.amp.autocast(device, enabled=use_amp):
                out = model(x, targets=y)
                # out["loss"] is already computed with ignore_index=-100
                # Divide by accum steps so gradients average correctly
                loss = out["loss"] / grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * grad_accum_steps  # unscale for logging
            steps += 1

            is_accum_boundary = (batch_idx % grad_accum_steps == 0) or (batch_idx == batches_per_epoch)
            if is_accum_boundary:
                # Update LR
                lr = cosine_lr(global_opt_step, warmup_steps, max_steps, max_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_opt_step += 1

            if steps % 50 == 0:
                avg = total_loss / steps
                elapsed = time.time() - t0
                speed = steps / max(elapsed, 1e-6)
                ppl = math.exp(min(avg, 20))

                # ETA estimate
                steps_done = (epoch - start_epoch) * batches_per_epoch + batch_idx
                steps_total = (args.epochs - start_epoch + 1) * batches_per_epoch
                total_elapsed = time.time() - training_start
                eta_s = total_elapsed / max(steps_done, 1) * max(steps_total - steps_done, 0)
                eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
                lr_display = optimizer.param_groups[0]["lr"]

                logger.info(
                    "  E%d [%d/%d] loss=%.4f ppl=%.1f lr=%.5f %.1f step/s ETA %s",
                    epoch, steps, batches_per_epoch, avg, ppl, lr_display, speed, eta_str,
                )

                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": avg,
                        "train/ppl": ppl,
                        "train/lr": lr_display,
                        "step": global_opt_step,
                    })

        avg_train = total_loss / max(steps, 1)
        elapsed = time.time() - t0

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device, enabled=use_amp):
                    out = model(x, targets=y)
                    val_loss += out["loss"].item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        lr_display = optimizer.param_groups[0]["lr"]
        logger.info(
            "E%2d | %5.1fs | lr=%.5f | Train %.4f PPL %.1f | Val %.4f PPL %.1f",
            epoch, elapsed, lr_display,
            avg_train, math.exp(min(avg_train, 20)),
            avg_val, math.exp(min(avg_val, 20)),
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "epoch/train_loss": avg_train,
                "epoch/val_loss": avg_val,
                "epoch/val_ppl": math.exp(min(avg_val, 20)),
            })

        # ── Save best checkpoint ───────────────────────────────────────
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "epoch": epoch,
                "global_opt_step": global_opt_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "config": config.__dict__,
                "tokenizer_name": "cl100k_base",
                "best_val": best_val,
            }, args.output)
            logger.info("   * Saved best (val=%.4f) → %s", avg_val, args.output)

    total_time = time.time() - training_start
    logger.info("=" * 60)
    logger.info("  Training complete in %.1fs. Best val: %.4f", total_time, best_val)
    logger.info("=" * 60)

    # ── Sample conversations ───────────────────────────────────────────
    logger.info("\n--- Sample Conversations ---\n")
    test_prompts = [
        "What is machine learning?",
        "Write a short poem about the ocean.",
        "How do I make pasta?",
        "Explain gravity to a child.",
        "What are three tips for staying healthy?",
    ]

    for prompt in test_prompts:
        response = generate_response(model, enc, prompt, max_tokens=120, device=device)
        logger.info("User : %s", prompt)
        logger.info("Sage : %s\n", response[:400] if response else "(no response)")

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Model saved to %s", args.output)
    logger.info("Run: python serve.py")


if __name__ == "__main__":
    main()
