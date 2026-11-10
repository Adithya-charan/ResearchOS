"""P11.7 reproducible evaluation of the research-quality features, on the INVENTED p11 corpus (datasets/p11_eval, CC0-1.0).

Nothing here is scientific validation: documents, claims and expected labels are authored for testing, and the system is lexical. The numbers say
how the implemented rules behave on clean, small, author-labelled data; they are an upper bound on real-world behaviour. Run it twice and you
get the same numbers (no randomness, no network, no model).

  evidence   BM25 top-5 -> verify_claim; status accuracy per case kind, support precision, unsupported-claim rate, contradiction detection
  citation   every passage cited correctly and with each kind of defect injected; per-defect detection rate and false-alarm rate
  gaps       conflicting pairs and stated limitations against author expectations; traceability of every gap
  retrieval  the P11.1 benchmark's baseline vs recommended summary (full tables are in retrieval_p11.json)

End-to-end report completeness and execution reliability need the API and are measured by scripts/eval_p11.py."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from research.bm25.index import BM25Index
from research.evidence import Candidate, verify_claim
from research.evidence.citations import check_citation
from research.gaps import analyze_corpus_gaps, validate_gap_report
from research.retrieval_benchmark import load, run as run_retrieval

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "p11_eval"
SEARCH_K = 5
LIMITATIONS = [
    "INVENTED corpus and author-written labels: this checks that the rules behave as designed, not that they work on real literature.",
    "The corpus is small (18 documents, 4 topics); precision and recall on it are far more favourable than on a real library.",
    "Expected statuses were assigned by meaning before the system ran, but the same author wrote the claims and the documents; shared wording "
    "makes lexical matching easier than it will be in practice.",
    "Paraphrase cases are included because the verifier is lexical and is expected to miss some; those misses are reported, not hidden.",
    "Citation defects are injected programmatically, so detection rates show the checker finds those defect types, not that real-world "
    "citation errors are limited to them.",
]


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def evidence_eval(idx: BM25Index, cases: list[dict] | None = None) -> dict:
    cases = cases if cases is not None else json.loads((DATASET / "evidence_cases.json").read_text("utf-8"))["cases"]
    rows = []
    for c in cases:
        cands = [Candidate(passage_id=h.passage.passage_id, document_id=h.passage.document_id, text=h.passage.text, resolved=True,
                           page_number=h.passage.page_number, section=h.passage.section, retrieval_method="bm25_search", retrieval_score=h.score)
                 for h in idx.search(c["claim"], k=SEARCH_K)]
        r = verify_claim(c["claim"], "result", cands, searched=True)
        used_docs = sorted({e["document_id"] for e in r["evidence"]})
        rows.append({"id": c["id"], "kind": c["kind"], "claim": c["claim"], "expected": c["expected"], "predicted": r["status"],
                     "correct": r["status"] == c["expected"], "conflicting": bool(r.get("conflicting")), "evidence_documents": used_docs,
                     "expected_documents_found": (None if not c["documents"] else all(d in used_docs for d in c["documents"])),
                     "reasons": [e["reason"] for e in r["evidence"]][:3]})
    by_kind: dict[str, dict] = defaultdict(lambda: {"cases": 0, "correct": 0})
    for r in rows:
        by_kind[r["kind"]]["cases"] += 1
        by_kind[r["kind"]]["correct"] += r["correct"]
    for v in by_kind.values():
        v["accuracy"] = _rate(v["correct"], v["cases"])
    pred_sup = [r for r in rows if r["predicted"] == "SUPPORTED"]
    true_sup = [r for r in rows if r["expected"] == "SUPPORTED"]
    pred_con = [r for r in rows if r["predicted"] == "CONTRADICTED"]
    true_con = [r for r in rows if r["expected"] == "CONTRADICTED"]
    sup_ok = sum(r["expected"] == "SUPPORTED" for r in pred_sup)
    con_ok = sum(r["expected"] == "CONTRADICTED" for r in pred_con)
    contested = [r for r in rows if r["expected"] == "PARTIALLY_SUPPORTED" and r["kind"] == "multi_document"]
    return {
        "cases": len(rows), "status_accuracy": _rate(sum(r["correct"] for r in rows), len(rows)), "by_kind": dict(by_kind),
        "evidence_support_precision": _rate(sup_ok, len(pred_sup)), "evidence_support_recall": _rate(sum(r["predicted"] == "SUPPORTED" for r in true_sup), len(true_sup)),
        "unsupported_claim_rate": _rate(len(pred_sup) - sup_ok, len(pred_sup)),
        "unsupported_claim_rate_definition": "of claims labelled SUPPORTED, the share whose expected status is not SUPPORTED (a false 'supported')",
        "contradiction_detection": {"precision": _rate(con_ok, len(pred_con)), "recall": _rate(sum(r["predicted"] == "CONTRADICTED" for r in true_con), len(true_con)),
                                    "predicted": len(pred_con), "expected": len(true_con)},
        "insufficient_correct": _rate(sum(r["correct"] for r in rows if r["expected"] == "INSUFFICIENT_EVIDENCE"), sum(r["expected"] == "INSUFFICIENT_EVIDENCE" for r in rows)),
        "conflicting_sources_surfaced": _rate(sum(r["conflicting"] for r in contested), len(contested)),
        "wrong": [{k: r[k] for k in ("id", "kind", "expected", "predicted")} for r in rows if not r["correct"]],
        "rows": rows,
    }


def citation_eval(idx: BM25Index) -> dict:
    """Cite every passage correctly, then once per injected defect; the checker must flag the defect and nothing else."""
    docs = sorted({p.document_id for d in {p.document_id for p in _all(idx)} for p in idx.document_passages(d)})
    other = {d: docs[(i + 1) % len(docs)] for i, d in enumerate(docs)}
    totals: dict[str, Counter] = defaultdict(Counter)
    for p in _all(idx):
        words = p.text.split()
        if len(words) < 8:
            continue
        good = {"document_id": p.document_id, "passage_id": p.passage_id, "page_number": p.page_number, "quote": " ".join(words[:8]), "source_text": p.text}
        variants = {
            "valid": (good, None),
            "fabricated_passage_id": ({**good, "passage_id": f"{p.document_id}:p999"}, "passage_missing"),
            "missing_passage_id": ({**good, "passage_id": None}, "no_passage_id"),
            "wrong_document": ({**good, "document_id": other[p.document_id]}, "document_mismatch"),
            "wrong_page": ({**good, "page_number": 99}, "page_mismatch"),
            "altered_quote": ({**good, "quote": " ".join(words[:4] + ["zzqx"] + words[4:8])}, "quote_not_in_passage"),
            "changed_source": ({**good, "source_text": p.text + " (edited later)"}, "source_changed"),
        }
        for name, (cit, expected) in variants.items():
            issues, _ = check_citation(cit, idx.get)
            t = totals[name]
            t["n"] += 1
            if expected is None:
                t["flagged"] += bool(issues)  # a false alarm
            else:
                t["detected"] += expected in issues
                t["exact"] += issues == [expected]
    out = {}
    for name, t in totals.items():
        out[name] = ({"citations": t["n"], "false_alarm_rate": _rate(t["flagged"], t["n"])} if name == "valid"
                     else {"citations": t["n"], "detection_rate": _rate(t["detected"], t["n"]), "flagged_only_that_issue": _rate(t["exact"], t["n"])})
    bad = [v["detection_rate"] for k, v in out.items() if k != "valid"]
    return {"by_case": out, "citation_correctness": {
        "valid_citations_accepted": _rate(totals["valid"]["n"] - totals["valid"]["flagged"], totals["valid"]["n"]),
        "defective_citations_flagged": _rate(sum(totals[k]["detected"] for k in totals if k != "valid"), sum(totals[k]["n"] for k in totals if k != "valid")),
        "lowest_defect_detection_rate": min(bad) if bad else None},
        "scope": "Checks existence, document, page, quote and source-text agreement. Whether the passage supports the claim is the verifier's job (see evidence)."}


def _all(idx: BM25Index):
    seen = []
    for i in range(idx.num_passages):
        seen.append(idx.passage_at(i))
    return seen


# Author expectations for the gap analysis (assigned by meaning from the invented documents)
EXPECT_CONFLICTS = [frozenset({"soil1", "soil2"}), frozenset({"bat4", "bat5"})]
EXPECT_LIMITATION_DOCS = {"bat1", "bat2", "bat3", "bat4", "bat5", "coral1", "coral2", "coral3", "coral4", "soil1", "soil2", "soil3", "soil4",
                          "fl1", "fl2", "fl3", "fl4"}  # dup1 is a copy without a limitations section


def gap_eval(idx: BM25Index, titles: dict[str, str]) -> dict:
    rep = analyze_corpus_gaps(idx, titles)
    found = [frozenset(g["documents"]) for g in rep["gaps"] if g["type"] == "conflicting_findings"]
    tp = sum(f in EXPECT_CONFLICTS for f in found)
    lim_docs = {g["documents"][0] for g in rep["gaps"] if g["type"] in ("methodological_limitation", "unanswered_question")}
    return {
        "gaps": len(rep["gaps"]), "counts": rep["counts"],
        "conflict_detection": {"expected_pairs": len(EXPECT_CONFLICTS), "reported_pairs": len(found), "precision": _rate(tp, len(found)),
                               "recall": _rate(sum(e in found for e in EXPECT_CONFLICTS), len(EXPECT_CONFLICTS))},
        "stated_limitation_recall": _rate(len(lim_docs & EXPECT_LIMITATION_DOCS), len(EXPECT_LIMITATION_DOCS)),
        "stated_limitations_outside_expectation": sorted(lim_docs - EXPECT_LIMITATION_DOCS),
        "traceability_problems": validate_gap_report(rep, idx),
        "every_gap_linked_to_a_passage": all(g["evidence"] for g in rep["gaps"]),
        "hypotheses_labelled": all(g["basis"] == "hypothesis" for g in rep["gaps"] if g["type"] == "missing_comparison"),
        "note": "Single-document themes and missing comparisons have no author ground truth; they are listed, not scored.",
    }


def retrieval_summary() -> dict:
    full = run_retrieval()
    res = full["results"]
    keep = ("recall@3", "recall@5", "precision@1", "precision@3", "mrr", "duplicate_pairs_in_top5")
    pick = lambda name: {k: res[name]["overall"].get(k) for k in keep}  # noqa: E731
    return {"queries": full["dataset"]["queries"], "baseline_bm25": pick("baseline BM25 (original search_documents)"),
            "recommended_opt_in": pick("normalize+prf+dedupe") if "normalize+prf+dedupe" in res else None,
            "full_tables": "artifacts/evaluation/retrieval_p11.json"}


def run_offline() -> dict:
    idx, titles, _, _ = load()
    return {"synthetic": True, "corpus": "datasets/p11_eval (invented, CC0-1.0)", "limitations": LIMITATIONS,
            "retrieval": retrieval_summary(), "evidence": evidence_eval(idx), "citations": citation_eval(idx), "gaps": gap_eval(idx, titles)}
