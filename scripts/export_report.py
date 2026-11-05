"""Export the research report of a finished run from the run database (never re-runs anything).

    python scripts/export_report.py --db artifacts/agent_runs/runs.db --run RUN_ID --out artifacts/reports/my_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_engine.reporting import export_report, report_from_store  # noqa: E402
from agent_engine.state.store import Store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="artifacts/agent_runs/runs.db")
    ap.add_argument("--run", required=True, metavar="RUN_ID")
    ap.add_argument("--out", required=True, metavar="DIR")
    ap.add_argument("--graph", action="store_true", help="also write the knowledge graph (graph.json, graph.mmd)")
    ap.add_argument("--experiments", metavar="DIR", help="with --graph: include experiment runs stored under DIR")
    args = ap.parse_args(argv)
    db = Path(args.db) if Path(args.db).is_absolute() else ROOT / args.db
    if not db.is_file():
        print(f"error: no run database at {db}", file=sys.stderr)
        return 2
    store = Store(db)
    try:
        store.get_run(args.run)
    except KeyError:
        print(f"error: no run {args.run} in {db}", file=sys.stderr)
        return 2
    report = report_from_store(store, args.run)
    store.close()
    if report is None:
        print("error: this run has no report (it was not run with --workflow full, or did not reach synthesis)", file=sys.stderr)
        return 1
    for kind, path in export_report(report, args.out).items():
        print(f"wrote {kind}: {path}")
    if args.graph:
        import json
        from research.experiments import load_manifest, read_review
        from research.knowledge_graph import build_graph, to_mermaid
        exps = []
        if args.experiments and Path(args.experiments).is_dir():
            for d in sorted(Path(args.experiments).iterdir()):
                if (d / "manifest.json").is_file():
                    m = load_manifest(d)
                    exps.append({"id": m["id"], "name": m["name"], "status": m["status"],
                                 "hypothesis_id": m["hypothesis_id"], "review": read_review(d)["status"]})
        graph = build_graph(report, exps)
        out = Path(args.out)
        (out / "graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "graph.mmd").write_text(to_mermaid(graph), encoding="utf-8")
        print(f"wrote graph: {out / 'graph.json'} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
