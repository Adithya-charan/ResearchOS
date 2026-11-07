"""Evaluate claim verification and novelty assessment on datasets/verification_eval and write artifacts/evaluation/verification_eval.json.

    python scripts/eval_verification.py [--extra DIR]

The bundled set is synthetic (see its README): this is a behaviour check, not scientific validation."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.verification_eval import run_verification_eval  # noqa: E402


def _print(title: str, s: dict) -> None:
    print(f"\n[{title}] {s['dataset']['claims']} claims, {s['dataset']['documents']} documents")
    a, h, al = s["accuracy_excluding_known_limitations"], s["accuracy_on_known_limitations"], s["accuracy_all"]
    print(f"  accuracy excluding known limitations: {a['correct']}/{a['claims']} = {a['accuracy']}")
    print(f"  accuracy on known-limitation cases:   {h['correct']}/{h['claims']} = {h['accuracy']}")
    print(f"  accuracy over everything:             {al['correct']}/{al['claims']} = {al['accuracy']}")
    for st, v in s["accuracy_by_expected_status"].items():
        print(f"    expected {st:22} {v['correct']}/{v['claims']}")
    n, nh = s["novelty_accuracy_excluding_known_limitations"], s["novelty_accuracy_on_known_limitations"]
    print(f"  novelty labels: {n['correct']}/{n['claims']} (excluding known limitations), {nh['correct']}/{nh['claims']} on known limitations")
    for r in s["claims"]:
        if not r["correct"]:
            print(f"    MISS {r['id']} expected {r['expected']} got {r['actual']}{' (known limitation)' if r['known_limitation'] else ''}: {r['text'][:70]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default=None, help="a real-source set in the same format (reported separately)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="verification-eval-") as td:
        report = run_verification_eval(ROOT / "datasets" / "verification_eval", Path(td), args.extra)
    out = ROOT / "artifacts" / "evaluation" / "verification_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report["labels"])
    _print("synthetic", report["synthetic"])
    if "real_source" in report:
        _print("real-source", report["real_source"])
    else:
        print("\nNo real-source set supplied: none are bundled and none were fabricated.")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
