"""
Sage 6.0 — Interactive Chat CLI

Loads the trained model and lets you chat with it directly in the terminal.
Auto-detects whether the checkpoint is character-level or BPE.

Usage:
    python chat.py                          # Load sage_best.pt
    python chat.py --model sage_chat_model.pt  # Load chat model
"""

import argparse
import sys
import warnings

import torch

from sage.config import SageConfig
from sage.sage import SageModel
from sage.generation import generate_tokens, top_p_sample

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
            decay_k = f"core.layers.0.resonance.decay_proj.weight"
            n_slots = sd[decay_k].shape[0] if decay_k in sd else 4
            wk = "core.layers.0.resonance.write_key.weight"
            mem_dim = sd[wk].shape[0] // n_slots if wk in sd else 32
            mc = "metacog.confidence_net.0.weight"
            metacog_dim = sd[mc].shape[0] if mc in sd else 64

            config = SageConfig(
                name="sage-loaded",
                n_nodes=n_nodes,
                n_active_limit=128,
                text_vocab_size=65,
                core_dim=core_dim,
                core_n_layers=n_layers,
                core_mlp_ratio=2.667,
                context_length=256,
                max_think_iterations=2,
                min_think_iterations=1,
                max_train_iterations=2,
                metacog_dim=metacog_dim,
                weight_tying=True,
                init_std=0.02,
                resonance_n_slots=n_slots,
                resonance_mem_dim=mem_dim,
                resonance_decay_init=0.95,
                sparse_k_ratio=0.3,
                dropout=0.1,
                layer_scale_init=1e-4,
            )
        else:
            print("Error: cannot infer config from checkpoint.")
            sys.exit(1)

    model = SageModel(config)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "model" if "model" in ckpt else None
    if sd_key:
        model.load_state_dict(ckpt[sd_key])
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


def generate_response(model, config, tokenizer, tok_type, prompt, device, max_tokens=300, temperature=0.8):
    """Generate a response given a prompt."""
    if tok_type == "bpe":
        full = f"User: {prompt}\nAssistant:"
        ids = tokenizer.encode(full, allowed_special=set())
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = generate_tokens(
            model, input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop_token_id=tokenizer.eot_token,
            stop_strings=["User:"],
            decode_fn=lambda x: tokenizer.decode(x),
        )
        response = tokenizer.decode(generated)
        if "User:" in response:
            response = response[:response.find("User:")]
        return response.strip()
    else:
        # Character-level Shakespeare model
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = generate_tokens(
            model, input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Sage 6.0 — Interactive Chat")
    parser.add_argument("--model", type=str, default="sage_best.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGE 6.0 — Interactive Chat")
    print("=" * 60)
    print(f"  Loading {args.model}...")

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

        # For char model, allow \n in input
        user_input = user_input.replace("\\n", "\n")

        response = generate_response(
            model, config, tokenizer, tok_type, user_input,
            args.device, args.max_tokens, args.temperature,
        )

        if tok_type == "bpe":
            print(f"Sage: {response}")
        else:
            print(f"\n{response}\n")
        print()


if __name__ == "__main__":
    main()
