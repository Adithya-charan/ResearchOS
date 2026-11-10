"""P11.1 retrieval benchmark: the unchanged BM25 baseline against each advanced feature on a labelled set, reporting Recall@k,
Precision@k, MRR, the duplicate rate and the behaviour on queries that have no relevant document.

Everything runs the real index code on the committed dataset `datasets/p11_eval` (INVENTED documents, CC0: a regression and ablation
set, not evidence about real literature). Ranking is document-level (a document's score is its best passage)."""
from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

from research.bm25.advanced import BASELINE, AdvancedConfig, AdvancedSearcher
from research.bm25.index import BM25Index, passages_from_text

KS = (1, 3, 5)
DATASET = Path(__file__).resolve().parents[1] / "datasets" / "p11_eval"

# name -> configuration; each row switches on ONE feature over the identity configuration, then combinations
CONFIGS: dict[str, AdvancedConfig] = {
    "identity (all off)": BASELINE,
    "normalize": AdvancedConfig(normalize=True),
    "phrase_bonus": AdvancedConfig(normalize=False, phrase_bonus=0.5),
    "title_boost": AdvancedConfig(normalize=False, title_boost=1.0),
    "heading_boost": AdvancedConfig(normalize=False, heading_boost=1.0),
    "variants": AdvancedConfig(normalize=False, variants=True),
    "prf(5 terms)": AdvancedConfig(normalize=False, prf_terms=5),
    "dedupe": AdvancedConfig(normalize=False, suppress_duplicates=True),
    "max_per_document=2": AdvancedConfig(normalize=False, max_per_document=2),
    "prf+dedupe": AdvancedConfig(normalize=False, prf_terms=5, suppress_duplicates=True),
    "normalize+prf+dedupe": AdvancedConfig(normalize=True, prf_terms=5, suppress_duplicates=True),
    "normalize+title+variants+dedupe": AdvancedConfig(normalize=True, title_boost=1.0, variants=True, suppress_duplicates=True),
    "all features": AdvancedConfig(normalize=True, phrase_bonus=0.5, title_boost=1.0, heading_boost=0.5, variants=True, prf_terms=5,
                                   suppress_duplicates=True, max_per_document=2),
}


def load(dataset: Path = DATASET) -> tuple[BM25Index, dict[str, str], dict[str, str], list[dict]]:
    corpus = json.loads((dataset / "corpus.json").read_text(encoding="utf-8"))
    idx = BM25Index()
    titles, dup_of = {}, {}
    for d in corpus["docs"]:
        idx.add_passages(passages_from_text(d["id"], d["text"]))
        titles[d["id"]] = d["title"]
        if d.get("duplicate_of"):
            dup_of[d["id"]] = d["duplicate_of"]
    return idx, titles, dup_of, json.loads((dataset / "queries.json").read_text(encoding="utf-8"))


def _p(ranked: list[str], rel: set[str], k: int) -> float:
    return sum(1 for r in ranked[:k] if r in rel) / k


def _r(ranked: list[str], rel: set[str], k: int) -> float:
    return sum(1 for r in ranked[:k] if r in rel) / len(rel) if rel else 0.0


def _rr(ranked: list[str], rel: set[str]) -> float:
    return next((1.0 / i for i, r in enumerate(ranked, 1) if r in rel), 0.0)


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 4) if xs else 0.0


def evaluate(config: AdvancedConfig | None, idx: BM25Index, titles: dict[str, str], dup_of: dict[str, str], queries: list[dict]) -> dict:
    """`config=None` runs the original `BM25Index.search_documents` (the true baseline, not the advanced code path)."""
    searcher = AdvancedSearcher(idx, titles)
    rows, missing_hits = [], []
    for q in queries:
        if config is None:
            ranked = [h.document_id for h in idx.search_documents(q["query"], k=10)]
        else:
            ranked = [d for d, _ in searcher.rank_documents(q["query"], k=10, config=config)]
        rel = set(q["relevant"])
        raw_ranked = ranked
        # a declared duplicate is the same document for relevance purposes (it is reported separately as redundancy)
        ranked = list(dict.fromkeys(dup_of.get(d, d) for d in raw_ranked))
        if not rel:
            missing_hits.append({"id": q["id"], "results": len(raw_ranked)})
            continue
        top5 = raw_ranked[:5]
        dup_pairs = sum(1 for d in top5 if dup_of.get(d) in top5)
        rows.append({"id": q["id"], "kind": q["kind"], "rr": _rr(ranked, rel), "dup": dup_pairs,
                     **{f"p{k}": _p(ranked, rel, k) for k in KS}, **{f"r{k}": _r(ranked, rel, k) for k in KS}})

    def agg(rs: list[dict]) -> dict:
        return {"queries": len(rs), "mrr": _mean([r["rr"] for r in rs]),
                **{f"precision@{k}": _mean([r[f"p{k}"] for r in rs]) for k in KS},
                **{f"recall@{k}": _mean([r[f"r{k}"] for r in rs]) for k in KS},
                "duplicate_pairs_in_top5": sum(r["dup"] for r in rs)}

    kinds = sorted({r["kind"] for r in rows})
    return {"overall": agg(rows), "by_kind": {k: agg([r for r in rows if r["kind"] == k]) for k in kinds},
            "no_relevant_document_queries": {"queries": len(missing_hits), "returned_any_result": sum(1 for m in missing_hits if m["results"]),
                                             "note": "these queries have no relevant document; any result is a false lead and a ranker cannot tell"}}


def run(dataset: Path = DATASET) -> dict:
    idx, titles, dup_of, queries = load(dataset)
    results = {"baseline BM25 (original search_documents)": evaluate(None, idx, titles, dup_of, queries)}
    for name, cfg in CONFIGS.items():
        results[name] = evaluate(cfg, idx, titles, dup_of, queries)
    return {
        "dataset": {"documents": len({p.document_id for p in idx._passages}), "passages": idx.num_passages, "queries": len(queries),
                    "kinds": sorted({q["kind"] for q in queries}), "synthetic": True},
        "ks": list(KS), "results": results,
        "limitations": [
            "The corpus is 18 invented documents (CC0) written for this test; scores say how the methods behave on this set, not on real literature.",
            "Relevance labels were written by the same author as the documents and queries; a human second annotator has not reviewed them.",
            "All methods are lexical. Expansion uses only the corpus's own vocabulary and top-passage co-occurrence; it does not know synonyms.",
            "Document-level ranking uses the best passage; duplicates count against precision unless suppressed.",
            "Queries with no relevant document are reported separately; they are not part of Recall/Precision/MRR.",
        ],
    }


def config_dict(cfg: AdvancedConfig) -> dict:
    return replace(cfg).__dict__
