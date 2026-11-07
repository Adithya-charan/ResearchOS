"""Bounded novelty assessment of a gap or hypothesis against the passages retrieved from ONE indexed corpus.

The labels are deliberately cautious and always scoped to the corpus that was searched. This module can say "similar prior work was
identified in these documents"; it can never say an idea is new, and "Potentially novel within searched corpus" means only that
nothing close was found by a lexical search of a corpus that was large enough to be worth searching."""
from __future__ import annotations

import re

from research.bm25.analysis import analyze
from research.bm25.index import _SENTENCE_RE
from research.evidence.types import Candidate, NoveltyLabel

MIN_CORPUS_DOCUMENTS = 5   # below this a "nothing found" result says nothing about novelty
MIN_STATEMENT_STEMS = 3
SIMILAR_COVERAGE = 0.6     # one sentence covers this share of the statement's content stems
PARTIAL_COVERAGE = 0.3
MAX_CLOSEST = 5
_AIM = re.compile(r"\b(?:we (?:study|investigate|examine|ask|test|measure|evaluate|compare|aim)|this (?:report|paper|study) "
                  r"(?:studies|examines|tests|links|compares)|whether|research question|aim(?:s)? to)\b|\?", re.I)

LIMITATIONS = [
    "Lexical comparison only: prior work that describes the same idea in different words is not detected (no semantic matching).",
    "Only the documents in this corpus were searched; a result says nothing about other databases, unindexed documents, "
    "preprints, patents or newer work.",
    "Similarity here means shared content words in a sentence; it does not show that two works ask the same question or reach the "
    "same conclusion - a human must read the listed documents.",
    "A label is not a novelty claim. 'Potentially novel within searched corpus' is not evidence that the idea is new.",
]
UNSEARCHED = ["Literature outside the indexed corpus: other databases, unindexed documents, and unpublished or newer work.",
              "Work that expresses the same idea in different vocabulary (synonyms, other languages, abbreviations)."]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(" ".join(text.split())) if len(s.split()) >= 3]


def assess_novelty(statement: str, *, passages: list[Candidate], corpus: dict, queries: list[str],
                   doc_info: dict[str, dict] | None = None, statement_kind: str = "hypothesis") -> dict:
    """`passages` are the retrieved candidates (text read from the corpus); `corpus` = {"documents": n, "passages": n}."""
    doc_info = doc_info or {}
    stems = set(analyze(statement))
    per_doc: dict[str, dict] = {}
    for c in passages:
        if not c.text or c.resolved is False or c.quality != "ok":
            continue
        for s in _sentences(c.text):
            cov = len(stems & set(analyze(s))) / len(stems) if stems else 0.0
            d = per_doc.setdefault(c.document_id, {"document_id": c.document_id, "best_coverage": 0.0, "passage_id": None,
                                                   "page_number": None, "quote": "", "matched_terms": [], "aims": []})
            if cov > d["best_coverage"]:
                d.update(best_coverage=round(cov, 4), passage_id=c.passage_id, page_number=c.page_number, quote=s,
                         matched_terms=sorted(stems & set(analyze(s))))
            if cov >= 0.4 and _AIM.search(s) and len(d["aims"]) < 2:
                d["aims"].append({"passage_id": c.passage_id, "page_number": c.page_number, "quote": s, "coverage": round(cov, 4)})
    closest = sorted(per_doc.values(), key=lambda d: (-d["best_coverage"], d["document_id"]))[:MAX_CLOSEST]
    for d in closest:
        info = doc_info.get(d["document_id"]) or {}
        d.update(title=info.get("title"), year=info.get("year"), doi=info.get("doi"))
    similar = [d for d in closest if d["best_coverage"] >= SIMILAR_COVERAGE]
    covered_by_closest = set()
    for d in closest[:3]:
        covered_by_closest |= set(d["matched_terms"])
    differences = sorted(stems - covered_by_closest)

    n_docs = int(corpus.get("documents", 0))
    reasons: list[str] = []
    if len(stems) < MIN_STATEMENT_STEMS:
        label = NoveltyLabel.UNCLEAR
        reasons.append("the statement has too few content words to compare")
    elif n_docs < MIN_CORPUS_DOCUMENTS or not queries:
        label = NoveltyLabel.INSUFFICIENT_COVERAGE
        reasons.append(f"the corpus holds {n_docs} document(s) (minimum {MIN_CORPUS_DOCUMENTS} for a meaningful comparison)"
                       if n_docs < MIN_CORPUS_DOCUMENTS else "no search was run for this statement")
    elif similar:
        label = NoveltyLabel.SIMILAR_PRIOR_WORK
        reasons.append(f"{len(similar)} document(s) contain a sentence covering at least {int(SIMILAR_COVERAGE * 100)}% of the "
                       "statement's content words")
    elif closest and closest[0]["best_coverage"] >= PARTIAL_COVERAGE:
        label = NoveltyLabel.UNCLEAR
        reasons.append("some documents overlap partly with the statement; lexical overlap cannot settle whether they cover it")
    else:
        label = NoveltyLabel.POTENTIALLY_NOVEL
        reasons.append("no sentence in the searched corpus overlaps substantially with the statement")
    return {
        "statement": statement, "statement_kind": statement_kind, "label": label.value, "reasons": reasons,
        "similar_existing_research": [{k: d[k] for k in ("document_id", "title", "year", "doi", "best_coverage", "passage_id",
                                                         "page_number", "quote", "matched_terms")} for d in similar],
        "closest_documents": [{k: d[k] for k in ("document_id", "title", "year", "doi", "best_coverage", "passage_id", "page_number",
                                                 "quote", "matched_terms")} for d in closest],
        "overlapping_research_questions": [{"document_id": d["document_id"], **a} for d in closest for a in d["aims"]],
        "differences_from_closest": {"statement_terms_not_found_in_closest_documents": differences,
                                     "note": "terms of the statement that none of the three closest documents' best sentences contain"},
        "search_coverage": {"corpus_documents": n_docs, "corpus_passages": int(corpus.get("passages", 0)), "queries": list(queries),
                            "documents_with_retrieved_passages": len(per_doc),
                            "passages_read": sum(1 for c in passages if c.text and c.resolved is not False and c.quality == "ok")},
        "unsearched": UNSEARCHED, "limitations": LIMITATIONS,
        "disclaimer": "This is a bounded, corpus-scoped comparison. It does not establish that the idea is new or that it is not.",
    }
