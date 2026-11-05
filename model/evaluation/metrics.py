"""Evaluation helpers: held-out loss / perplexity and degeneration (repetition) metrics."""
from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence

import torch

from model.architecture.transformer import Transformer
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
    total, batches = 0.0, 0
    for x, y in dataset.eval_batches(batch_size, max_batches):
        with ctx:
            out = model(x.to(device), targets=y.to(device))
        total += float(out.loss)
        batches += 1
    model.train(was_training)
    loss = total / batches
    return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "tokens": batches * batch_size * dataset.seq_len}


def repetition_rate(ids: Sequence[int], n: int = 3) -> float:
    """Fraction of n-grams that repeat an earlier n-gram (0 = none, ->1 = degenerate loops)."""
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)
