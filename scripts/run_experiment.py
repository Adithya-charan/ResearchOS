"""Run, reproduce and review sandboxed experiments (blueprint section 14).

    python scripts/run_experiment.py run --example --approved-by alice
    python scripts/run_experiment.py run spec.json --approved-by alice
    python scripts/run_experiment.py reproduce artifacts/experiment_runs/exp-XXXX --approved-by alice
    python scripts/run_experiment.py review artifacts/experiment_runs/exp-XXXX --reviewer bob --decision accepted
    python scripts/run_experiment.py show artifacts/experiment_runs/exp-XXXX

Isolation: Docker when available (no network, read-only root, non-root, capped CPU/memory/processes). Without Docker, runs
are refused unless you pass --allow-weak, which uses a plain subprocess with NO isolation (read the code first)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.experiments import (ExperimentSpec, NotApproved, QuotaExceeded, SandboxRefused, default_sandbox,  # noqa: E402
                                  load_manifest, read_review, reproduce, review_experiment, run_experiment, verify_manifest)
from research.experiments.examples import example_spec  # noqa: E402


def _root(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def show(run_dir: Path) -> None:
    m = load_manifest(run_dir)
    problems = verify_manifest(run_dir)
    r = read_review(run_dir)
    print(f"Experiment {m['id']} '{m['name']}': {m['status']}  (isolation: {m['isolation']['level']}, "
          f"network blocked: {m['isolation']['network_blocked']})")
    print(f"  code {m['code_sha256'][:12]}  data {m['dataset_sha256'][:12]}  seed {m['seed']}  config {json.dumps(m['config'])}")
    print(f"  metrics: {json.dumps(m['metrics'])}")
    print(f"  artifacts: {', '.join(a['path'] for a in m['artifacts']) or 'none'}   duration {m['result']['duration_s']}s")
    print(f"  integrity: {'OK - files match the manifest' if not problems else 'FAILED - ' + '; '.join(problems)}")
    print(f"  human review: {r['status']}" + (f" by {r['reviewer']}" if r.get("reviewer") else "")
          + (f"; reproduced: {r['reproduced']}" if r.get("reproduced") is not None else ""))
    for w in m["warnings"] + m["isolation"]["notes"]:
        print(f"  ! {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("spec", nargs="?")
    r.add_argument("--example", action="store_true")
    r.add_argument("--approved-by", required=True)
    r.add_argument("--root", default="artifacts/experiment_runs")
    r.add_argument("--allow-weak", action="store_true", help="allow the non-isolating subprocess fallback")
    rp = sub.add_parser("reproduce")
    rp.add_argument("run_dir")
    rp.add_argument("--approved-by", required=True)
    rp.add_argument("--allow-weak", action="store_true")
    rp.add_argument("--rtol", type=float, default=0.0)
    rv = sub.add_parser("review")
    rv.add_argument("run_dir")
    rv.add_argument("--reviewer", required=True)
    rv.add_argument("--decision", choices=["accepted", "rejected"], required=True)
    rv.add_argument("--notes", default="")
    sh = sub.add_parser("show")
    sh.add_argument("run_dir")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        if args.cmd == "run":
            if args.example == bool(args.spec):
                print("error: give either a spec file or --example", file=sys.stderr)
                return 2
            spec = example_spec() if args.example else ExperimentSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
            sandbox = default_sandbox(args.allow_weak)
            run_dir, m = run_experiment(spec, sandbox, _root(args.root), approved_by=args.approved_by)
            show(run_dir)
            print(f"\nrun directory: {run_dir}\nNext: reproduce it, then review it.")
            return 0 if m["status"] == "succeeded" else 1
        if args.cmd == "reproduce":
            rep = reproduce(_root(args.run_dir), default_sandbox(args.allow_weak), approved_by=args.approved_by, rtol=args.rtol)
            print(f"reproduced: {rep['reproduced']}")
            for d in rep.get("differences", []):
                print(f"  difference: {d}")
            return 0 if rep["reproduced"] else 1
        if args.cmd == "review":
            rv_ = review_experiment(_root(args.run_dir), args.reviewer, args.decision, args.notes)
            print(f"review recorded: {rv_['status']} by {rv_['reviewer']} (reproduced: {rv_['reproduced']})")
            return 0
        show(_root(args.run_dir))
        return 0
    except (SandboxRefused, NotApproved, QuotaExceeded, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
