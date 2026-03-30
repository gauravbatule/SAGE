"""
Sage 4.0 — Generation Utilities

Consolidated text generation logic used across training, serving, and inference.
Eliminates code duplication by providing a single, well-tested generation API.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

__all__ = ["top_p_sample", "generate_tokens", "extract_assistant_response"]


def top_p_sample(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling from logits.

    Args:
        logits: Raw logits of shape ``(B, vocab_size)``.
        temperature: Sampling temperature. Lower = more deterministic.
        top_p: Cumulative probability threshold for nucleus sampling.

    Returns:
        Sampled token IDs of shape ``(B, 1)``.
    """
    logits = logits / max(temperature, 1e-6)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    removal_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
    sorted_logits[removal_mask] = float("-inf")

    probs = F.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, sampled_idx)


@torch.no_grad()
def generate_tokens(
    model: "SageModel",  # noqa: F821 — forward reference to avoid circular import
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop_token_id: Optional[int] = None,
    stop_strings: Optional[List[str]] = None,
    decode_fn: Optional[callable] = None,
) -> List[int]:
    """
    Autoregressive token generation with top-p sampling.

    Args:
        model: A ``SageModel`` instance (must be in eval mode).
        input_ids: Prompt token IDs of shape ``(1, L)``.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        stop_token_id: Stop generation when this token is produced.
        stop_strings: Stop generation if any of these strings appear in the tail.
        decode_fn: Optional decoder function for stop-string detection.

    Returns:
        List of generated token IDs (excluding the prompt).
    """
    model.eval()
    generated: List[int] = []
    current_ids = input_ids

    for _ in range(max_new_tokens):
        # Truncate to active limit
        if current_ids.shape[1] > model.config.n_active_limit:
            current_ids = current_ids[:, -model.config.n_active_limit:]

        model.metacog.reset()
        output = model(current_ids)
        next_token = top_p_sample(output["logits"][:, -1, :], temperature, top_p)

        token_id = next_token.item()
        generated.append(token_id)
        current_ids = torch.cat([current_ids, next_token], dim=1)

        # Stop conditions
        if stop_token_id is not None and token_id == stop_token_id:
            break

        if stop_strings and decode_fn and len(generated) > 3:
            tail_text = decode_fn(current_ids[0, -12:].tolist())
            if any(s in tail_text for s in stop_strings):
                break

    return generated


def extract_assistant_response(full_text: str, prompt_length: int = 0) -> str:
    """
    Extract the assistant's response from a generated conversation string.

    Args:
        full_text: The complete generated text including prompt.
        prompt_length: Length of the original prompt for fallback extraction.

    Returns:
        Cleaned assistant response string.
    """
    if "Assistant:" in full_text:
        response = full_text[full_text.rfind("Assistant:") + len("Assistant:"):]
        if "User:" in response:
            response = response[:response.find("User:")]
        return response.strip()

    if prompt_length > 0:
        return full_text[prompt_length:].strip()

    return full_text.strip()
