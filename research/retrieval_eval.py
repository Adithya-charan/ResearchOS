"""Retrieval evaluation: ranking metrics over a small labelled set, plus checks that deduplication, declared metadata and citations
behave. It exercises the real ingestion pipeline and the real BM25 index - no mocks - and reports weak spots (vocabulary mismatch)
instead of hiding them."""
from __future__ import annotations

import json
from pathlib import Path

from research.corpus import Corpus, SearchFilters

KS = (1, 3, 5)
LIMITATIONS = [
    "BM25 matches words, not meaning: a query that shares no vocabulary with the relevant document finds nothing ('synonym' queries below).",
    "Stemming is English-only and light; spelling variants, abbreviations and other languages are not bridged.",
    "Scores are relative to the indexed corpus (IDF depends on it) and are not comparable across libraries.",
    "This set is tiny (invented documents, one relevant document per query in most cases); its numbers are a sanity check, "
    "not a measure of quality on real literature.",
    "Phrase search requires exact contiguous words (no stemming, no stopword removal inside the phrase).",
]


# ------------------------------------------------------------------ metrics
def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant. Fewer than k results count the missing slots as non-relevant."""
    return sum(1 for r in ranked[:k] if r in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for r in ranked[:k] if r in relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _filters(spec: dict | None) -> SearchFilters | None:
    if not spec:
        return None
    return SearchFilters(document_types=frozenset(spec["document_types"]) if spec.get("document_types") else None,
                         author=spec.get("author"), year_min=spec.get("year_min"), year_max=spec.get("year_max"))


# ------------------------------------------------------------------ evaluation
def run_evaluation(dataset_dir: str | Path, workdir: str | Path, ks: tuple[int, ...] = KS) -> dict:
    dataset_dir = Path(dataset_dir)
    declared = json.loads((dataset_dir / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((dataset_dir / "queries.json").read_text(encoding="utf-8"))
    files = sorted((dataset_dir / "docs").glob("*.md"))

    corpus = Corpus(workdir, isolate=False)
    try:
        ingest = {f.name: corpus.ingest(f, metadata=declared.get(f.name), source="retrieval-eval", license="CC0-1.0") for f in files}
        by_filename = {d.filename: d for d in corpus.list_documents()}
        name_of = {d.id: d.filename for d in corpus.list_documents()}

        dedup = {
            "files_offered": len(files), "documents_stored": len(by_filename),
            "duplicates_detected": sorted(n for n, r in ingest.items() if r.status == "duplicate"),
            "duplicate_points_at": {n: name_of.get(r.document_id) for n, r in ingest.items() if r.status == "duplicate"},
            "failed_or_rejected": sorted(n for n, r in ingest.items() if r.status in ("failed", "rejected")),
        }

        mismatches = []
        for fname, want in declared.items():
            rec = by_filename.get(fname)
            if rec is None:
                continue  # a duplicate or rejected file; reported under dedup
            got = {"title": rec.title, "authors": rec.authors, "publication_date": rec.publication_date, "url": rec.url,
                   "document_type": rec.document_type}
            for key, value in want.items():
                if got[key] != value:
                    mismatches.append({"file": fname, "field": key, "declared": value, "stored": got[key]})
            if rec.year != int(want["publication_date"][:4]):
                mismatches.append({"file": fname, "field": "year", "declared": want["publication_date"][:4], "stored": rec.year})
        metadata = {"documents_checked": sum(1 for f in declared if f in by_filename), "mismatches": mismatches}

        per_query, citations_total, citations_ok, unresolved = [], 0, 0, []
        for case in queries:
            rel = set(case["relevant"])
            flt = _filters(case.get("filters"))
            docs = corpus.search_documents(case["query"], k=max(ks), filters=flt)
            ranked = [name_of[d["document"]["id"]] for d in docs]
            chunks = corpus.search(case["query"], k=max(ks), filters=flt)
            chunk_names = [name_of[h.passage.document_id] for h in chunks]
            for h in chunks:  # every returned result must cite a real, verifiable stored passage
                c = h.citation()
                citations_total += 1
                stored = corpus.store.get_passage(c["chunk_id"])
                v = corpus.verify_passage(c["chunk_id"])
                ok = (stored is not None and stored[0].document_id == c["document_id"] and v.ok
                      and c["excerpt"].rstrip(".").strip() in " ".join(stored[0].text.split()) and bool(c["title"]))
                citations_ok += ok
                if not ok:
                    unresolved.append({"query": case["id"], "chunk_id": c["chunk_id"]})
            per_query.append({
                "id": case["id"], "kind": case["kind"], "query": case["query"], "relevant": sorted(rel), "ranked": ranked,
                "precision": {k: round(precision_at_k(ranked, rel, k), 4) for k in ks},
                "recall": {k: round(recall_at_k(ranked, rel, k), 4) for k in ks},
                "reciprocal_rank": round(reciprocal_rank(ranked, rel), 4),
                "chunk_precision": {k: round(precision_at_k(chunk_names, rel, k), 4) for k in ks},
            })

        def aggregate(rows: list[dict]) -> dict:
            return {"queries": len(rows), "mrr": _mean([r["reciprocal_rank"] for r in rows]),
                    **{f"precision@{k}": _mean([r["precision"][k] for r in rows]) for k in ks},
                    **{f"recall@{k}": _mean([r["recall"][k] for r in rows]) for k in ks},
                    **{f"chunk_precision@{k}": _mean([r["chunk_precision"][k] for r in rows]) for k in ks}}

        kinds = sorted({r["kind"] for r in per_query})
        return {
            "dataset": {"documents": len(declared), "queries": len(queries), "ks": list(ks)},
            "overall": aggregate(per_query), "by_kind": {k: aggregate([r for r in per_query if r["kind"] == k]) for k in kinds},
            "per_query": per_query, "deduplication": dedup, "metadata": metadata,
            "citations": {"results_checked": citations_total, "resolved": citations_ok, "unresolved": unresolved},
            "limitations": LIMITATIONS,
        }
    finally:
        corpus.close()
