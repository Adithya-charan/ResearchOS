"""Run the P11.1 retrieval benchmark (baseline BM25 vs advanced features) and write artifacts/evaluation/retrieval_p11.json.

    python scripts/eval_retrieval_p11.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.retrieval_benchmark import CONFIGS, run  # noqa: E402


def main() -> int:
    report = run()
    report["configurations"] = {k: v.__dict__ for k, v in CONFIGS.items()}
    out = ROOT / "artifacts" / "evaluation" / "retrieval_p11.json"
    out.write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    print(f"{'configuration':<42}{'MRR':>7}{'P@1':>7}{'P@3':>7}{'R@3':>7}{'R@5':>7}{'dups':>6}")
    for name, r in report["results"].items():
        o = r["overall"]
        print(f"{name:<42}{o['mrr']:>7}{o['precision@1']:>7}{o['precision@3']:>7}{o['recall@3']:>7}{o['recall@5']:>7}{o['duplicate_pairs_in_top5']:>6}")
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
