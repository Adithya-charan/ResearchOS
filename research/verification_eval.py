"""Evaluation of claim verification and novelty assessment on a small labelled set. It runs the real corpus pipeline and the real
verifier (no mocks) and reports where the lexical method is wrong - including cases chosen because it is expected to be wrong.

The bundled set is SYNTHETIC: invented documents, claims written by the author, expected statuses assigned by meaning. It checks that
the pipeline behaves as designed; it is not scientific validation. Real-source sets (same format) are reported under a separate key."""
from __future__ import annotations

import json
from pathlib import Path

from research.bm25.analysis import term_surface_forms
from research.corpus import Corpus
from research.evidence import STATUSES, Candidate, assess_novelty, verify_claim

SEARCH_K = 5
LIMITATIONS = [
    "SYNTHETIC set: invented documents, author-written claims and author-assigned expected statuses; not scientific validation.",
    "Claims share vocabulary with the documents on purpose, because the verifier is lexical; accuracy here is an upper bound for "
    "how a lexical check behaves on clean text.",
    "Statuses are lexical evidence-strength labels: a paraphrase is not connected to its source, swapped roles look identical, and "
    "'reduced' vs 'eliminated' are not distinguished. Those cases are in the set (known_limitation) and are reported separately.",
    "No real-source examples are bundled; none were fabricated. See datasets/verification_eval/README.md.",
]


def _load(dataset: Path) -> tuple[dict, list, dict]:
    return (json.loads((dataset / "corpus.json").read_text(encoding="utf-8")),
            json.loads((dataset / "claims.json").read_text(encoding="utf-8")),
            json.loads((dataset / "novelty_cases.json").read_text(encoding="utf-8")) if (dataset / "novelty_cases.json").exists() else
            {"cases": [], "tiny_corpus_docs": []})


def _build(dataset: Path, workdir: Path, declared: dict, only: list[str] | None = None) -> Corpus:
    c = Corpus(workdir, isolate=False)
    for f in sorted((dataset / "docs").glob("*.md")):
        if only is None or f.name in only:
            c.ingest(f, metadata=declared.get(f.name), license="CC0-1.0")
    return c


def _cand(corpus: Corpus, passage_id: str, method: str, score: float | None) -> Candidate:
    p = corpus.index.get(passage_id)
    if p is None:
        return Candidate(passage_id=passage_id, document_id="", resolved=False, retrieval_method=method, retrieval_score=score)
    return Candidate(passage_id=p.passage_id, document_id=p.document_id, text=p.text, page_number=p.page_number, quality=p.quality,
                     resolved=corpus.verify_passage(passage_id).ok, retrieval_method=method, retrieval_score=score)


def _evaluate_set(dataset: Path, workdir: Path) -> dict:
    declared, claims, novelty = _load(dataset)
    corpus = _build(dataset, workdir / "full", declared)
    tiny = _build(dataset, workdir / "tiny", declared, novelty.get("tiny_corpus_docs") or None) if novelty["cases"] else None
    try:
        rows = []
        for c in claims:
            cands: list[Candidate] = []
            if c.get("attach_invalid"):
                cands.append(Candidate(passage_id="doc_missing:p000", document_id="doc_missing", resolved=False, retrieval_method="attached"))
            searched = c.get("searched", True) and c["claim_type"] not in ("gap", "hypothesis")
            if searched:
                cands += [_cand(corpus, h.passage.passage_id, "bm25_search", round(h.score, 6)) for h in corpus.index.search(c["text"], k=SEARCH_K)]
            v = verify_claim(c["text"], c["claim_type"], cands, searched=searched)
            rows.append({"id": c["id"], "category": c["category"], "text": c["text"], "expected": c["expected"], "actual": v["status"],
                         "correct": v["status"] == c["expected"], "known_limitation": c["known_limitation"],
                         "conflicting": v["conflicting"], "strength": v["strength"]["level"], "explanation": v["explanation"][:2],
                         "note": c.get("note", "")})
        matrix = {e: {a: 0 for a in STATUSES} for e in STATUSES}
        for r in rows:
            matrix[r["expected"]][r["actual"]] += 1

        def acc(rs: list[dict]) -> dict:
            return {"claims": len(rs), "correct": sum(r["correct"] for r in rs), "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None}

        normal = [r for r in rows if not r["known_limitation"]]
        hard = [r for r in rows if r["known_limitation"]]
        by_status = {s: acc([r for r in normal if r["expected"] == s]) for s in STATUSES}
        nrows = []
        for n in novelty["cases"]:
            lib = tiny if n["corpus"] == "tiny" and tiny is not None else corpus
            q = " ".join(s for _, s in term_surface_forms(n["statement"], limit=12)) or n["statement"]
            cs = [_cand(lib, h.passage.passage_id, "bm25_search", round(h.score, 6)) for h in lib.index.search(q, k=8)]
            a = assess_novelty(n["statement"], passages=cs, corpus={"documents": lib.index.num_documents, "passages": lib.index.num_passages},
                               queries=[q])
            nrows.append({"id": n["id"], "statement": n["statement"], "corpus": n["corpus"], "expected": n["expected"], "actual": a["label"],
                          "correct": a["label"] == n["expected"], "known_limitation": n.get("known_limitation", False),
                          "closest": [d["document_id"] for d in a["closest_documents"][:2]], "note": n.get("note", "")})
        return {
            "claims": rows, "confusion_matrix": matrix, "accuracy_excluding_known_limitations": acc(normal),
            "accuracy_on_known_limitations": acc(hard), "accuracy_by_expected_status": by_status, "accuracy_all": acc(rows),
            "novelty": nrows,
            "novelty_accuracy_excluding_known_limitations": acc([r for r in nrows if not r["known_limitation"]]),
            "novelty_accuracy_on_known_limitations": acc([r for r in nrows if r["known_limitation"]]),
            "dataset": {"documents": len(declared), "claims": len(claims), "novelty_cases": len(novelty["cases"])},
        }
    finally:
        corpus.close()
        if tiny is not None:
            tiny.close()


def run_verification_eval(dataset_dir: str | Path, workdir: str | Path, extra_dir: str | Path | None = None) -> dict:
    dataset_dir, workdir = Path(dataset_dir), Path(workdir)
    report = {"synthetic": _evaluate_set(dataset_dir, workdir / "synthetic"), "limitations": LIMITATIONS,
              "labels": "synthetic: invented documents and author-assigned expected statuses; not scientific validation"}
    extra = Path(extra_dir) if extra_dir else dataset_dir / "real"
    if (extra / "claims.json").exists() and (extra / "docs").is_dir():
        report["real_source"] = _evaluate_set(extra, workdir / "real")
        report["real_source"]["labels"] = "real-source examples supplied by the user; expected statuses assigned by the user"
    return report
