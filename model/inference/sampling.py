"""Next-token sampling on a 1-D logits vector (all functions are pure; none mutate their input)."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from model.tokenizer.bpe import BOS_ID, PAD_ID, UNK_ID

# Tokens the model must never emit as text (they only exist as inputs / padding).
DEFAULT_BANNED = (PAD_ID, UNK_ID, BOS_ID)


def apply_repetition_penalty(logits: torch.Tensor, history: Sequence[int], penalty: float) -> torch.Tensor:
    """CTRL-style: make already-seen tokens less likely (penalty > 1)."""
    if penalty == 1.0 or not history:
        return logits
    out = logits.clone()
    idx = torch.tensor(sorted(set(history)), dtype=torch.long, device=out.device)
    seen = out[idx]
    out[idx] = torch.where(seen > 0, seen / penalty, seen * penalty)
    return out


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability >= p (always keeps the best one)."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove_sorted = cumulative - torch.softmax(sorted_logits, dim=-1) >= p  # drop tokens that start after mass p
    remove = torch.zeros_like(remove_sorted).scatter(0, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    history: Sequence[int] = (),
    banned: Sequence[int] = DEFAULT_BANNED,
    generator: torch.Generator | None = None,
) -> int:
    logits = logits.float().cpu()
    if banned:
        logits = logits.index_fill(0, torch.tensor(list(banned), dtype=torch.long), float("-inf"))
    logits = apply_repetition_penalty(logits, history, repetition_penalty)
    if temperature <= 0.0:  # greedy
        return int(torch.argmax(logits))
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
