"""
Sage 4.0 — Conversational Training with BPE Tokenizer

Trains the Sage model on the Stanford Alpaca instruction-following dataset
using tiktoken BPE tokenization, mixed-precision training, and cosine
learning rate scheduling with warmup.

Usage::

    python train_chat.py                    # Train with defaults
    python train_chat.py --epochs 20        # More epochs
    python train_chat.py --batch-size 16    # Larger batches
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


def download_alpaca(path: str = "data/alpaca_data.json") -> list:
    """Download Alpaca dataset if not cached."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(path):
        logger.info("Downloading Alpaca dataset...")
        urllib.request.urlretrieve(ALPACA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_conversations(raw_data: list) -> list[str]:
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
    """BPE-tokenized conversation dataset with padding."""

    def __init__(
        self,
        conversations: list[str],
        enc: tiktoken.Encoding,
        max_len: int = 256,
        pad_id: int = 0,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        logger.info("Encoding %d conversations with BPE...", len(conversations))

        for i, conv in enumerate(conversations):
            if i % 10000 == 0 and i > 0:
                logger.info("  %d/%d encoded...", i, len(conversations))
            ids = enc.encode(conv, allowed_special=set())
            ids = ids + [enc.eot_token]
            if len(ids) < 8 or len(ids) > max_len:
                continue
            padded = ids + [pad_id] * (max_len + 1 - len(ids))
            padded = padded[:max_len + 1]
            t = torch.tensor(padded, dtype=torch.long)
            self.samples.append((t[:-1], t[1:]))

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
    max_tokens: int = 100,
    temperature: float = 0.7,
    device: str = "cpu",
) -> str:
    """Generate a single assistant response for evaluation."""
    model.eval()
    full_prompt = f"User: {prompt}\nAssistant:"
    ids = enc.encode(full_prompt, allowed_special=set())
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

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

        if input_ids.shape[1] > len(ids) + 3:
            tail = enc.decode(input_ids[0, -10:].tolist())
            if "User:" in tail:
                break

    full = enc.decode(input_ids[0].tolist())
    return extract_assistant_response(full, len(full_prompt))


def cosine_lr(
    step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float,
) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup:
        return max_lr * step / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main() -> None:
    """Entry point for conversational training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Sage 4.0 — Conversational Training")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (8 for 4GB VRAM)")
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--n-conversations", type=int, default=15000)
    parser.add_argument("--output", type=str, default="sage_chat_model.pt")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  SAGE 4.0 — Wave Propagation + Resonance Memory")
    logger.info("  BPE Tokenizer + Conversational Training")
    logger.info("=" * 60)

    # BPE tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab
    logger.info("BPE Tokenizer: %s tokens (cl100k_base)", f"{vocab_size:,}")

    # Data
    raw = download_alpaca()
    conversations = format_conversations(raw)
    logger.info("Formatted %d conversations", len(conversations))

    conversations = conversations[:args.n_conversations]
    logger.info("Using %d conversations", len(conversations))

    dataset = ChatDataset(conversations, enc, max_len=args.max_len, pad_id=0)
    split = int(len(dataset) * 0.95)
    train_set = torch.utils.data.Subset(dataset, range(split))
    val_set = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model
    config = SageConfig(
        name="sage-chat-v4",
        n_nodes=vocab_size,
        n_active_limit=args.max_len,
        text_vocab_size=vocab_size,
        core_dim=256,
        core_n_heads=4,
        core_n_layers=6,
        core_mlp_ratio=2.667,
        max_seq_len=args.max_len + 128,
        max_think_iterations=2,
        min_think_iterations=1,
        metacog_dim=64,
        weight_tying=True,
        init_std=0.02,
    )

    model = SageModel(config)
    counts = model.count_parameters()
    logger.info("--- Parameter Budget ---")
    for k, v in counts.items():
        pct = (v / counts["total"] * 100) if counts["total"] > 0 else 0
        logger.info("  %-20s: %10s (%5.1f%%)", k, f"{v:,}", pct)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    model = model.to(device)

    # Optimizer
    max_lr = args.lr
    min_lr = 1e-5
    warmup_steps = 200
    max_steps = len(train_loader) * args.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=0.1, betas=(0.9, 0.95),
    )

    # Training
    best_val = float("inf")
    global_step = 0

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    use_amp = device == "cuda"

    logger.info("Training for %d epochs (%d steps)...", args.epochs, max_steps)
    logger.info("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()
        n_batches = len(train_loader)

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            lr = cosine_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x, targets=y)
                logits = out["logits"]
                loss = F.cross_entropy(
                    logits.view(-1, config.text_vocab_size),
                    y.view(-1),
                    ignore_index=0,
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            steps += 1
            global_step += 1

            if steps % 50 == 0:
                avg = total_loss / steps
                elapsed = time.time() - t0
                speed = steps / elapsed
                logger.info(
                    "  E%d [%d/%d] loss=%.4f lr=%.5f %.1f step/s",
                    epoch, steps, n_batches, avg, lr, speed,
                )

        avg_train = total_loss / steps
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(x, targets=y)
                    logits = out["logits"]
                    loss = F.cross_entropy(
                        logits.view(-1, config.text_vocab_size),
                        y.view(-1),
                        ignore_index=0,
                    )
                val_loss += loss.item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        logger.info(
            "E%2d | %5.1fs | lr=%.5f | Train %.4f PPL %.1f | Val %.4f PPL %.1f",
            epoch, elapsed, lr, avg_train, math.exp(min(avg_train, 20)),
            avg_val, math.exp(min(avg_val, 20)),
        )

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model": model.state_dict(),
                "config": config.__dict__,
                "tokenizer_name": "cl100k_base",
            }, args.output)
            logger.info("   * Saved best (val=%.4f)", avg_val)

    logger.info("=" * 60)
    logger.info("  Training complete. Best val: %.4f", best_val)
    logger.info("=" * 60)

    # Test generation
    logger.info("\n--- Sample Conversations ---\n")
    test_prompts = [
        "What is machine learning?",
        "Write a short poem about the ocean.",
        "How do I make pasta?",
        "Explain gravity to a child.",
        "What are three tips for staying healthy?",
    ]

    for prompt in test_prompts:
        response = generate_response(model, enc, prompt, max_tokens=80, device=device)
        logger.info("User: %s", prompt)
        logger.info("Sage: %s\n", response)

    logger.info("Model saved to %s", args.output)
    logger.info("Run: python serve.py")


if __name__ == "__main__":
    main()
