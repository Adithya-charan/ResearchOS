"""In-memory BM25 passage index (Okapi BM25, Lucene-style IDF).

Phase 6 uses this as the Discovery agent's search tool. Phase 7 builds the persistent, page-aware version
(PDF pages, file hashes, extraction quality) on top of the same `Passage` / `search` shape.
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from research.bm25.analysis import analyze


@dataclass(frozen=True)
class Passage:
    passage_id: str
    document_id: str
    text: str
    page_number: int | None = None
    section: str = ""
    quality: str = "ok"  # extraction quality of the source page: ok | low (see research.corpus.quality)


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float
    matched_terms: tuple[str, ...]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.clear()

    def clear(self) -> None:
        """Empty the index in place (holders of a reference to it - e.g. agent tools - keep seeing the live index)."""
        self._passages: list[Passage] = []
        self._by_id: dict[str, int] = {}
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(passage idx, tf)]
        self._documents: set[str] = set()
        self._total_len = 0

    # ------------------------------------------------------------------ building
    def add_passages(self, passages: Iterable[Passage]) -> None:
        for p in passages:
            if p.passage_id in self._by_id:
                raise ValueError(f"duplicate passage id {p.passage_id}")
            terms = analyze(p.text)
            idx = len(self._passages)
            self._passages.append(p)
            self._by_id[p.passage_id] = idx
            self._lengths.append(len(terms))
            self._total_len += len(terms)
            self._documents.add(p.document_id)
            for term, tf in Counter(terms).items():
                self._postings[term].append((idx, tf))

    @property
    def num_passages(self) -> int:
        return len(self._passages)

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    def get(self, passage_id: str) -> Passage | None:
        idx = self._by_id.get(passage_id)
        return self._passages[idx] if idx is not None else None

    def document_passages(self, document_id: str) -> list[Passage]:
        """The document's indexed passages in the order they were added (reading order)."""
        return [p for p in self._passages if p.document_id == document_id]

    def has_document(self, document_id: str) -> bool:
        return document_id in self._documents

    # ------------------------------------------------------------------ searching
    def idf(self, term: str) -> float:
        n = len(self._postings.get(term, ()))
        return math.log(1.0 + (len(self._passages) - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 10, document_ids: set[str] | None = None,
               predicate: Callable[[Passage], bool] | None = None) -> list[SearchHit]:
        """Top-k passages. `document_ids` restricts to those documents; `predicate` is an extra per-passage filter."""
        if not self._passages:
            return []
        terms = list(dict.fromkeys(analyze(query)))
        avg_len = max(self._total_len / len(self._passages), 1e-9)
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, list[str]] = defaultdict(list)
        for term in terms:
            idf = self.idf(term)
            for idx, tf in self._postings.get(term, ()):
                p = self._passages[idx]
                if document_ids is not None and p.document_id not in document_ids:
                    continue
                if predicate is not None and not predicate(p):
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[idx] / avg_len)
                scores[idx] += idf * tf * (self.k1 + 1) / norm
                matched[idx].append(term)
        # deterministic: highest score first, ties broken by passage id
        best = heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], self._passages[kv[0]].passage_id))
        return [SearchHit(self._passages[i], s, tuple(matched[i])) for i, s in best]


_HEADING_RE = re.compile(r"^(?:\d+\.\s+\S.*|Abstract|Conclusion\.?)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def passages_from_text(document_id: str, text: str, max_words: int = 110, page_number: int | None = None) -> list[Passage]:
    """Split plain text into passages: one per paragraph block, long blocks cut at sentence boundaries.
    A single-line block that looks like a heading ('Abstract', '3. Results') becomes the section label."""
    out: list[Passage] = []
    section = ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if "\n" not in block and _HEADING_RE.match(block):
            section = block
            continue
        sentences, current, count = _SENTENCE_RE.split(block), [], 0
        pieces: list[str] = []
        for s in sentences:
            w = len(s.split())
            if current and count + w > max_words:
                pieces.append(" ".join(current))
                current, count = [], 0
            current.append(s)
            count += w
        if current:
            pieces.append(" ".join(current))
        for piece in pieces:
            out.append(Passage(f"{document_id}:p{len(out):03d}", document_id, piece, page_number, section))
    return out
