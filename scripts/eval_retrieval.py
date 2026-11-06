"""Evaluate BM25 retrieval on the labelled set in datasets/retrieval_eval and write artifacts/evaluation/retrieval_eval.json.

    python scripts/eval_retrieval.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.retrieval_eval import run_evaluation  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="retrieval-eval-") as td:
        report = run_evaluation(ROOT / "datasets" / "retrieval_eval", Path(td) / "corpus")
    out = ROOT / "artifacts" / "evaluation" / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{report['dataset']['documents']} documents, {report['dataset']['queries']} queries")
    print(f"{'kind':10} {'n':>3} {'MRR':>6} {'P@1':>6} {'P@3':>6} {'R@3':>6} {'R@5':>6}")
    for name, a in {"ALL": report["overall"], **report["by_kind"]}.items():
        print(f"{name:10} {a['queries']:>3} {a['mrr']:>6} {a['precision@1']:>6} {a['precision@3']:>6} {a['recall@3']:>6} {a['recall@5']:>6}")
    d, m, c = report["deduplication"], report["metadata"], report["citations"]
    print(f"dedup: {d['documents_stored']} stored from {d['files_offered']} files; duplicates: {d['duplicates_detected']}")
    print(f"metadata: {len(m['mismatches'])} mismatch(es) in {m['documents_checked']} documents")
    print(f"citations: {c['resolved']}/{c['results_checked']} resolved")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
