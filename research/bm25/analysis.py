"""Lexical analysis shared by indexing, querying and the agents' term bookkeeping.

A deliberately light stemmer is used: it only has to be *consistent* between query and document, not linguistically
perfect ("electrolyte"/"electrolytes", "increase"/"increased"/"increases" all map to one stem).
"""
from __future__ import annotations

import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has have how in into is it its of on "
    "or such than that the their then there these this those to was were what when where which while who why will "
    "with would we our us not no yes also may might should whether between within about over under".split()
)


@lru_cache(maxsize=200_000)  # pure function over a small vocabulary: caching makes indexing ~3x faster
def stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix) and not word.endswith("ss"):
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def raw_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def analyze(text: str) -> list[str]:
    """Stemmed, stopword-free tokens (the representation stored in the index)."""
    return [stem(t) for t in raw_tokens(text) if t not in STOPWORDS]


def term_surface_forms(text: str, limit: int | None = None) -> list[tuple[str, str]]:
    """Unique (stem, first surface word) pairs in order of first appearance.

    Stems are for *matching* (compare against `analyze(passage)`); queries must be built from the surface words,
    because stemming is not idempotent and a stemmed word would be stemmed a second time at search time."""
    seen: dict[str, str] = {}
    for tok in raw_tokens(text):
        if tok in STOPWORDS:
            continue
        s = stem(tok)
        if len(s) >= 2:
            seen.setdefault(s, tok)
    pairs = list(seen.items())
    return pairs[:limit] if limit else pairs


def key_terms(text: str, limit: int | None = None) -> list[str]:
    """Unique analysed (stemmed) terms in order of first appearance."""
    return [s for s, _ in term_surface_forms(text, limit)]


def phrase_chunks(text: str) -> list[list[str]]:
    """Split a question into noun-phrase-like chunks by cutting at stopwords ('how does X affect Y of Z' -> X | Y | Z)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    for t in raw_tokens(text):
        if t in STOPWORDS:
            if current:
                chunks.append(current)
                current = []
        else:
            current.append(t)
    if current:
        chunks.append(current)
    return chunks
