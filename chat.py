"""
Sage 6.0 — Interactive Chat CLI (Streaming)

Loads the trained model and lets you chat with it directly in the terminal.
Auto-detects whether the checkpoint is character-level or BPE.
Tokens are printed in real-time as they are generated (streaming mode).

Usage:
    python chat.py                          # Load sage_best.pt
    python chat.py --model sage_chat_model.pt  # Load chat model
"""

import argparse
import sys
import time
import warnings

import torch

from sage.config import SageConfig
from sage.sage import SageModel
from sage.generation import top_p_sample

warnings.filterwarnings("ignore")


def detect_and_load(path: str, device: str = "cpu"):
    """Load checkpoint and auto-detect tokenizer type."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        print("Error: unrecognized checkpoint format.")
        sys.exit(1)

    # Detect BPE (chat) vs character-level
    is_bpe = "tokenizer_name" in ckpt or "config" in ckpt and isinstance(ckpt["config"], dict)

    if is_bpe and "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ckpt["config"]
        config = SageConfig(**{k: v for k, v in cfg.items() if hasattr(SageConfig, k)})
    else:
        # Character-level: infer from state dict
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        if isinstance(sd, dict) and "graph.node_embeddings.weight" in sd:
            n_nodes, core_dim = sd["graph.node_embeddings.weight"].shape
            n_layers = sum(1 for k in sd if k.endswith(".wave_norm.weight") and k.startswith("core.layers."))
            decay_k = "core.layers.0.resonance.decay_proj.weight"
            n_slots = sd[decay_k].shape[0] if decay_k in sd else 4
            wk = "core.layers.0.resonance.write_key.weight"
            mem_dim = sd[wk].shape[0] // n_slots if wk in sd else 32
            mc = "metacog.confidence_net.0.weight"
            metacog_dim = sd[mc].shape[0] if mc in sd else 64

            # Detect old vs new architecture by checking for new keys
            has_cross_band = any("cross_band_mix" in k for k in sd)
            has_new_predictor = any("predictor.0.weight" in k for k in sd)

            config = SageConfig(
                name="sage-loaded",
                n_nodes=n_nodes,
                n_active_limit=128,
                text_vocab_size=65,
                core_dim=core_dim,
                core_n_layers=n_layers,
                core_mlp_ratio=2.667,
                context_length=256,
                max_think_iterations=1,
                min_think_iterations=1,
                max_train_iterations=2,
                metacog_dim=metacog_dim,
                weight_tying=True,
                init_std=0.02,
                resonance_n_slots=n_slots,
                resonance_mem_dim=mem_dim,
                resonance_decay_init=0.95,
                sparse_k_ratio=0.4,
                dropout=0.0,
                layer_scale_init=1e-4,
            )
        else:
            print("Error: cannot infer config from checkpoint.")
            sys.exit(1)

    model = SageModel(config)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "model" if "model" in ckpt else None
    if sd_key:
        ckpt_sd = ckpt[sd_key]
        model_sd = model.state_dict()
        filtered = {}
        for k, v in ckpt_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
        model.load_state_dict(filtered, strict=False)
    # Speed optimization: use 1 iteration for fast inference
    config.max_think_iterations = 1
    config.min_think_iterations = 1

    model.to(device)
    model.eval()

    # Build tokenizer
    if is_bpe and ckpt.get("tokenizer_name"):
        import tiktoken
        enc = tiktoken.get_encoding(ckpt["tokenizer_name"])
        tok_type = "bpe"
        return model, config, enc, tok_type
    else:
        from sage.train import download_data, CharTokenizer
        text = download_data()
        tok = CharTokenizer(text)
        config.text_vocab_size = tok.vocab_size
        tok_type = "char"
        return model, config, tok, tok_type


def stream_response(model, config, tokenizer, tok_type, prompt, device,
                    max_tokens=300, temperature=0.8, top_p=0.9):
    """
    Generate and stream tokens one at a time, printing each as it arrives.

    Returns the number of tokens generated and elapsed time (for stats).
    """
    if tok_type == "bpe":
        full = f"User: {prompt}\nAssistant:"
        ids = tokenizer.encode(full, allowed_special=set())
    else:
        ids = tokenizer.encode(prompt)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    L = input_ids.shape[1]

    n_generated = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        # --- Phase 1: Prefill ---
        model.eval()
        model.metacog.reset()
        prefill_out = model(input_ids)
        first_logits = prefill_out["logits"][:, -1, :]
        if config.weight_tying and first_logits.shape[-1] > config.text_vocab_size:
            first_logits = first_logits[:, :config.text_vocab_size]
        next_token = top_p_sample(first_logits, temperature, top_p)

        # --- Seed recurrent state from the last prompt token ---
        last_prompt_token = input_ids[:, -1:]
        _, recurrent_state = model.forward_step(
            last_prompt_token, step_idx=L - 1, recurrent_state=None
        )

        # --- Phase 2: Streaming decode ---
        step_idx = L
        generated = []

        while len(generated) < max_tokens:
            token_id = next_token.item()

            # BPE stop check
            if tok_type == "bpe" and token_id == tokenizer.eot_token:
                break

            generated.append(token_id)
            n_generated += 1

            # Decode and print immediately
            if tok_type == "char":
                char = tokenizer.decode([token_id])
            else:
                char = tokenizer.decode([token_id])
            print(char, end="", flush=True)

            # BPE stop-string check (after printing)
            if tok_type == "bpe" and n_generated > 3:
                tail_ids = generated[-12:]
                tail_text = tokenizer.decode(tail_ids)
                if "User:" in tail_text:
                    # Trim trailing "User:" fragment — already printed, but stop here
                    break

            logits, recurrent_state = model.forward_step(
                next_token, step_idx=step_idx, recurrent_state=recurrent_state
            )
            if config.weight_tying and logits.shape[-1] > config.text_vocab_size:
                logits = logits[:, :, :config.text_vocab_size]

            next_token = top_p_sample(logits[:, -1, :], temperature, top_p)
            step_idx += 1

    elapsed = time.perf_counter() - t_start
    return n_generated, elapsed


def main():
    parser = argparse.ArgumentParser(description="Sage 6.0 — Interactive Chat (Streaming)")
    parser.add_argument("--model", type=str, default="sage_best.pt", help="Checkpoint path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGE 6.0 — Interactive Chat  [streaming]")
    print("=" * 60)
    print(f"  Loading {args.model} on {args.device}...")

    model, config, tokenizer, tok_type = detect_and_load(args.model, args.device)
    params = sum(p.numel() for p in model.parameters())

    if tok_type == "bpe":
        print(f"  Model: {config.name} | {params:,} params | BPE tokenizer")
        print(f"  Type your message and press Enter. Type 'quit' to exit.")
    else:
        print(f"  Model: {config.name} | {params:,} params | Character-level")
        print(f"  This is a Shakespeare model — give it Shakespeare-style prompts.")
        print(f"  Example: 'ROMEO:\\nO, ' or 'HAMLET:\\nTo be, '")
        print(f"  Type 'quit' to exit.")

    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Allow escaped newlines in prompt
        user_input = user_input.replace("\\n", "\n")

        if tok_type == "bpe":
            print("Sage: ", end="", flush=True)
        else:
            print()  # blank line before char-level output

        try:
            n_tok, elapsed = stream_response(
                model, config, tokenizer, tok_type, user_input,
                device=args.device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue

        # Newline after streamed content + timing stats
        tok_per_sec = n_tok / elapsed if elapsed > 0 else float("inf")
        print(f"\n[{n_tok} tokens | {elapsed:.2f}s | {tok_per_sec:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
