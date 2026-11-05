"""Text cleaning, quality filtering, and exact + near-duplicate detection."""
from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib

import numpy as np

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLANKS_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip() + "\n"


def quality_check(
    text: str,
    min_chars: int = 200,
    min_alpha_ratio: float = 0.5,
    max_repeated_line_ratio: float = 0.5,
) -> tuple[bool, str]:
    """Cheap heuristics that reject empty, binary-looking, or boilerplate-dominated documents."""
    if len(text) < min_chars:
        return False, "too_short"
    alpha = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha < min_alpha_ratio:
        return False, "low_alpha_ratio"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines and 1 - len(set(lines)) / len(lines) > max_repeated_line_ratio:
        return False, "repeated_lines"
    return True, "ok"


def exact_hash(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class NearDuplicateIndex:
    """MinHash + LSH over word shingles. Deterministic (no Python hash randomisation)."""

    def __init__(self, num_perm: int = 64, bands: int = 16, shingle: int = 5, threshold: float = 0.8, seed: int = 1234):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        self.shingle, self.threshold = shingle, threshold
        self._buckets: dict[tuple[int, bytes], list[str]] = {}
        self._sigs: dict[str, np.ndarray] = {}

    def signature(self, text: str) -> np.ndarray:
        words = text.lower().split()
        if len(words) < self.shingle:
            grams = [" ".join(words)]
        else:
            grams = [" ".join(words[i : i + self.shingle]) for i in range(len(words) - self.shingle + 1)]
        h = np.fromiter((zlib.crc32(g.encode("utf-8")) for g in set(grams)), dtype=np.uint64)
        with np.errstate(over="ignore"):
            mixed = h[None, :] * self.a[:, None] + self.b[:, None]  # wraps mod 2**64
        return mixed.min(axis=1)

    def check_and_add(self, doc_id: str, text: str) -> str | None:
        """Return the id of an already-indexed near-duplicate, else index this doc and return None."""
        sig = self.signature(text)
        keys = [(b, sig[b * self.rows : (b + 1) * self.rows].tobytes()) for b in range(self.bands)]
        for key in keys:
            for other in self._buckets.get(key, ()):
                if float(np.mean(sig == self._sigs[other])) >= self.threshold:
                    return other
        self._sigs[doc_id] = sig
        for key in keys:
            self._buckets.setdefault(key, []).append(doc_id)
        return None
