"""Reproduction and human review of stored experiment runs.

Reproduced = same code hash, dataset hash, config and seed re-run in a sandbox gave the same metrics and the same artifact
hashes. It shows the run is repeatable on this machine; it does not show the experiment tests what its hypothesis claims."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from research.experiments.runner import load_manifest, run_experiment, verify_manifest
from research.experiments.sandbox import Sandbox
from research.experiments.spec import ExperimentSpec


def _close(a, b, rtol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a == b or (rtol > 0 and math.isclose(a, b, rel_tol=rtol, abs_tol=0.0))
    return a == b


def reproduce(run_dir: Path, sandbox: Sandbox, *, approved_by: str, rtol: float = 0.0) -> dict:
    """Re-run the stored spec in a scratch directory and compare with the original manifest."""
    run_dir = Path(run_dir)
    problems = verify_manifest(run_dir)
    if problems:
        return {"reproduced": False, "reason": "the stored run does not match its manifest", "differences": problems}
    original = load_manifest(run_dir)
    spec = ExperimentSpec.model_validate_json((run_dir / "spec.json").read_text(encoding="utf-8"))
    scratch = Path(tempfile.mkdtemp(prefix="repro-"))
    try:
        new_dir, new = run_experiment(spec, sandbox, scratch, approved_by=approved_by, run_id="repro")
        diffs: list[str] = []
        if new["status"] != original["status"]:
            diffs.append(f"status {original['status']} -> {new['status']}")
        for key in ("code_sha256", "dataset_sha256"):
            if new[key] != original[key]:
                diffs.append(f"{key} differs")
        for k in sorted(set(original["metrics"]) | set(new["metrics"])):
            if k not in original["metrics"] or k not in new["metrics"] or not _close(original["metrics"][k], new["metrics"][k], rtol):
                diffs.append(f"metric {k}: {original['metrics'].get(k)!r} -> {new['metrics'].get(k)!r}")
        old_a = {a["path"]: a["sha256"] for a in original["artifacts"]}
        new_a = {a["path"]: a["sha256"] for a in new["artifacts"]}
        for path in sorted(set(old_a) | set(new_a)):
            if old_a.get(path) != new_a.get(path):
                diffs.append(f"artifact {path} differs")
        report = {"reproduced": not diffs, "differences": diffs, "rtol": rtol, "sandbox": sandbox.name,
                  "checked_at": datetime.now(timezone.utc).isoformat(),
                  "note": "Repeatable on this machine and sandbox; says nothing about whether the experiment is valid."}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    (run_dir / "reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def read_review(run_dir: Path) -> dict:
    p = Path(run_dir) / "review.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"status": "pending", "reviewer": None, "notes": "", "reproduced": None}


def review_experiment(run_dir: Path, reviewer: str, decision: str, notes: str = "") -> dict:
    """Record a human decision. Accepting needs an intact manifest and a succeeded run; the reproduction state is stored with it."""
    if decision not in ("accepted", "rejected"):
        raise ValueError("decision must be 'accepted' or 'rejected'")
    if not reviewer.strip():
        raise ValueError("a named reviewer is required")
    run_dir = Path(run_dir)
    if decision == "accepted":
        problems = verify_manifest(run_dir)
        if problems:
            raise ValueError("cannot accept a run whose files do not match its manifest: " + "; ".join(problems))
        if load_manifest(run_dir)["status"] != "succeeded":
            raise ValueError("only a succeeded run can be accepted")
    repro = run_dir / "reproduction.json"
    reproduced = json.loads(repro.read_text(encoding="utf-8"))["reproduced"] if repro.is_file() else None
    review = {"status": decision, "reviewer": reviewer, "notes": notes, "reproduced": reproduced,
              "reviewed_at": datetime.now(timezone.utc).isoformat()}
    (run_dir / "review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    return review
