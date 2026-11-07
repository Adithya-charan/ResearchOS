"""Small shared helpers: seeding, hashing, atomic file writes."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the project uses."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write `text` to `path` so readers never observe a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(6):  # Windows refuses to replace a file another process has open for a moment: retry briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
