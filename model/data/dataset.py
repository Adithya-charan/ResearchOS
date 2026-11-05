"""Packed token streams and deterministic batch sampling."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch


def write_token_file(path: str | Path, ids: Sequence[int] | np.ndarray) -> int:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.uint16).max):
        raise ValueError("token ids do not fit in uint16")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.uint16).tofile(path)
    return int(arr.size)


class TokenDataset:
    """Flat stream of token ids (documents concatenated with <DOC> ... <EOS> markers).

    Training batches are random windows chosen by a generator seeded from
    (seed, step, micro_step), so any step can be reproduced exactly - which is
    what makes resuming from a checkpoint deterministic.
    """

    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        self.seq_len = seq_len
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) < seq_len + 2:
            raise ValueError(f"{self.path} has {len(self.tokens)} tokens; need > {seq_len + 1}")

    def __len__(self) -> int:
        return len(self.tokens)

    def _window(self, start: int) -> tuple[np.ndarray, np.ndarray]:
        chunk = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]

    def train_batch(self, seed: int, step: int, micro: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng([seed, step, micro])
        starts = rng.integers(0, len(self.tokens) - self.seq_len - 1, size=batch_size)
        xs, ys = zip(*(self._window(int(s)) for s in starts))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def eval_batches(self, batch_size: int, max_batches: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Fixed, non-overlapping windows from the start of the stream (same every call)."""
        n_windows = (len(self.tokens) - 1) // self.seq_len
        n_batches = n_windows // batch_size
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)
        if n_batches == 0:
            raise ValueError("evaluation split too small for one batch")
        for b in range(n_batches):
            pairs = [self._window((b * batch_size + i) * self.seq_len) for i in range(batch_size)]
            xs, ys = zip(*pairs)
            yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
