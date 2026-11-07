"""Train / validation / test leakage checks.

`prepare_dataset` already splits by document before chunking and removes exact and near duplicates across the whole
corpus. This module is the independent verification of that outcome: it re-derives the facts from the split lists and
the cleaned texts rather than trusting the code that produced them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from model.data.cleaning import exact_hash


class LeakageError(ValueError):
    """Raised when documents appear in more than one split."""


def find_overlaps(splits: Mapping[str, Sequence[tuple[str, str]]]) -> list[str]:
    """`splits` maps a split name to `(doc_id, text)` pairs. Returns human-readable problems (empty = clean):
    * the same doc_id in two splits;
    * byte-identical (after cleaning) text in two splits under different ids."""
    problems: list[str] = []
    names = list(splits)
    ids = {n: {d for d, _ in splits[n]} for n in names}
    hashes: dict[str, dict[str, str]] = {n: {exact_hash(t): d for d, t in splits[n]} for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for d in sorted(ids[a] & ids[b]):
                problems.append(f"document {d} is in both {a} and {b}")
            for h in sorted(set(hashes[a]) & set(hashes[b])):
                problems.append(f"identical text in {a}:{hashes[a][h]} and {b}:{hashes[b][h]}")
    return problems


def assert_no_leakage(splits: Mapping[str, Sequence[tuple[str, str]]]) -> None:
    problems = find_overlaps(splits)
    if problems:
        raise LeakageError("train/validation/test leakage detected:\n  " + "\n  ".join(problems[:20]))
