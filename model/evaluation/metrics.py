"""Evaluation helpers: held-out loss / perplexity and degeneration (repetition) metrics."""
from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence

import torch

from model.architecture.transformer import IGNORE_INDEX, Transformer
from model.data.dataset import TokenDataset


@torch.no_grad()
def evaluate_loss(
    model: Transformer,
    dataset: TokenDataset,
    batch_size: int,
    max_batches: int | None,
    device: torch.device | str,
    autocast_ctx=None,
) -> dict[str, float]:
    """Mean next-token cross-entropy over fixed held-out windows (identical every call)."""
    was_training = model.training
    model.eval()
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    total, tokens, batches = 0.0, 0, 0
    for x, y in dataset.eval_batches(batch_size, max_batches):
        with ctx:
            out = model(x.to(device), targets=y.to(device))
        n = int((y != IGNORE_INDEX).sum())  # token-weighted: batches with ignored/padded targets count less
        total += float(out.loss) * n
        tokens += n
        batches += 1
    model.train(was_training)
    if tokens == 0:
        raise ValueError("evaluation set produced no scorable tokens")
    loss = total / tokens
    return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "tokens": tokens}


def repetition_rate(ids: Sequence[int], n: int = 3) -> float:
    """Fraction of n-grams that repeat an earlier n-gram (0 = none, ->1 = degenerate loops)."""
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)
