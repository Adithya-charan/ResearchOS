"""Run the three-agent research workflow (Supervisor -> Discovery -> Critic) over a local corpus.

    python scripts/run_research.py --question "How does annealing rate affect soil organic matter?" --max-docs 1500
    python scripts/run_research.py --library artifacts/corpus --question "..." --year-min 2018

Without --library the corpus is a folder of .txt files (the synthetic demo corpus by default). With --library it is a
document library built by scripts/manage_corpus.py (PDF, DOCX, TXT, Markdown, HTML): evidence is then reported with document
title, page and section, and every cited passage is re-checked against the stored original file.

Everything runs locally: BM25 retrieval, deterministic agents, SQLite state. A trained RESEARCHOS checkpoint can be
attached with --checkpoint; it is then used only for short, budgeted, clearly-labelled *untrusted* notes.
A run is durable: if it is interrupted, `--resume RUN_ID --db <same db>` continues from the persisted state.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from agent_engine.contracts import RunLimits, RunStatus
from agent_engine.registry.defaults import build_engine
from agent_engine.reporting import export_report, report_from_store
from agent_engine.state.store import Store
from research.bm25.index import BM25Index, passages_from_text
from research.corpus import Corpus, SearchFilters

DEFAULT_QUESTION = "How does annealing rate affect soil organic matter in rice paddies?"
QUIET_EVENTS = {"tool_call", "tool_result", "memory_write", "task_created"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_index(corpus_dir: Path, max_docs: int) -> BM25Index:
    files = sorted(corpus_dir.glob("*.txt"))
    if max_docs > 0:
        files = files[:max_docs]
    index = BM25Index()
    for f in files:
        index.add_passages(passages_from_text(f.stem, f.read_text(encoding="utf-8", errors="replace")))
    return index


def load_backend(checkpoint: str, tokenizer: str, device: str):
    from agent_engine.model_gateway import LocalBackend
    from model.inference import ResearchOSModel

    rt = ResearchOSModel(device=device)
    rt.load_checkpoint(checkpoint, tokenizer_dir=tokenizer, prefer="best")
    return LocalBackend(rt)


def describe_event(ev: dict, titles: dict[str, str]) -> str:
    p = ev["payload"]
    task = titles.get(ev["task_id"] or "", "")
    detail = {
        "task_started": f"attempt {p.get('attempt')}", "task_succeeded": p.get("summary", "")[:90],
        "task_retry": f"retry in {p.get('delay_s')}s: {str(p.get('error', ''))[:80]}", "task_failed": str(p.get("error", ""))[:100],
        "proposal_rejected": f"{p.get('key')}: {p.get('reason')}", "run_finished": f"{p.get('status')} {p.get('stop_reason') or ''}",
        "model_call": f"{p.get('tokens_used')} tokens", "model_call_skipped": p.get("reason", ""),
    }.get(ev["type"], "")
    return f"  [{ev['id']:>4}] {ev['type']:<18} {task[:44]:<44} {detail}".rstrip()


async def follow(engine, store, run_id: str, verbose: bool) -> None:
    titles: dict[str, str] = {}
    async for ev in engine.events(run_id):
        if ev["task_id"] and ev["task_id"] not in titles:
            titles[ev["task_id"]] = store.get_task(ev["task_id"])["title"]
        if verbose or ev["type"] not in QUIET_EVENTS:
            print(describe_event(ev, titles), flush=True)


def check_citations(corpus: Corpus, summary: dict) -> dict[str, dict]:
    """Re-verify every cited passage against the stored original file (blueprint 13.1)."""
    out = {}
    for ref in summary.get("evidence", []):
        v = corpus.verify_passage(ref["passage_id"])
        out[ref["passage_id"]] = {"ok": v.ok, "message": v.message}
    return out


def print_summary(store: Store, index: BM25Index, run_id: str, synthetic: bool, corpus: Corpus | None = None,
                  citations: dict[str, dict] | None = None) -> None:
    run = store.get_run(run_id)
    s = run["summary"] or {}
    a = s.get("assessment") or {}
    print("\n" + "=" * 78)
    print(f"Run summary  [{run['status']}{' / ' + run['stop_reason'] if run['stop_reason'] else ''}]  "
          f"{run['elapsed_s']:.1f}s  tasks: {s.get('task_counts')}")
    print(f"Question: {run['question']}")
    if s.get("corpus"):
        print(f"Corpus searched: {s['corpus']['documents']} documents / {s['corpus']['passages']} passages (lexical BM25)")
    if synthetic:
        print("NOTE: the corpus is the synthetic template corpus - its text is generated, not real research.")
    verdict = "SUFFICIENT" if a.get("sufficient") else "INSUFFICIENT"
    print(f"\nEvidence check: {verdict}" + (f" - {'; '.join(a.get('reasons', []))}" if a.get("reasons") else ""))
    for ref in s.get("evidence", [])[:8]:
        p = index.get(ref["passage_id"])
        text = " ".join(p.text.split())[:150] if p else "(passage not in index)"
        page = f" p.{ref['page_number']}" if ref["page_number"] else ""
        if corpus is not None and p is not None:
            doc = corpus.get_document(ref["document_id"])
            check = (citations or {}).get(ref["passage_id"])
            status = "" if check is None else ("  [verified against original]" if check["ok"] else f"  [NOT VERIFIED: {check['message']}]")
            print(f"  - {doc.title if doc else ref['document_id']} ({doc.year if doc and doc.year else 'year unknown'}){page}"
                  + (f", {p.section[:40]}" if p.section else "") + f"  {ref['passage_id']}{status}\n      \"{text}\"")
        else:
            print(f"  - {ref['document_id']} {ref['passage_id']}{page}  \"{text}\"")
    if len(s.get("evidence", [])) > 8:
        print(f"  ... and {len(s['evidence']) - 8} more")
    if a.get("low_quality_matches"):
        print("  Matches on pages with unreliable text extraction (shown, not counted as evidence):")
        for m in a["low_quality_matches"][:5]:
            print(f"    ~ {m['document_id']} {m['passage_id']}" + (f" p.{m['page_number']}" if m["page_number"] else ""))
    print("\nUncertainties:")
    for u in s.get("uncertainties", []):
        print(f"  * {u}")
    for f in s.get("failed_tasks", []):
        print(f"  ! failed task {f['agent_type']}: {f['error']}")
    if s.get("model_notes"):
        print("\nModel notes (experimental small model - UNTRUSTED drafts, not evidence):")
        for n in s["model_notes"]:
            print(f"  [{n['agent_type']}] {n['text'][:160]!r}")
    print(f"\nModel tokens used: {s.get('model_tokens_used', 0)}")
    print("\nLimitations:")
    for line in s.get("limitations", []):
        print(f"  - {line}")
    print(f"\nrun_id: {run_id}")


async def execute(engine, store, run_id: str, verbose: bool):
    watcher = asyncio.create_task(follow(engine, store, run_id, verbose))
    try:
        result = await engine.run(run_id)
    finally:
        await asyncio.wait_for(watcher, 10)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--corpus", default="datasets/raw/synthetic_v1", help="directory of .txt documents")
    ap.add_argument("--max-docs", type=int, default=1500, help="index at most this many documents (0 = all)")
    ap.add_argument("--library", metavar="DIR", help="a document library made with scripts/manage_corpus.py "
                                                     "(used instead of --corpus; evidence is verified against the originals)")
    ap.add_argument("--year-min", type=int, help="library only: skip documents published before this year")
    ap.add_argument("--year-max", type=int, help="library only: skip documents published after this year")
    ap.add_argument("--license", action="append", help="library only: only documents with this licence (repeatable)")
    ap.add_argument("--exclude-low-quality", action="store_true", help="library only: leave out pages with unreliable text extraction")
    ap.add_argument("--db", default="artifacts/agent_runs/runs.db", help="SQLite file holding durable run state")
    ap.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted/paused run from --db")
    ap.add_argument("--checkpoint", help="trained checkpoint dir: enables budgeted, untrusted model notes")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--discovery-k", type=int, default=8, help="passages per first-round search")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--max-tasks", type=int, default=40)
    ap.add_argument("--max-runtime", type=float, default=300.0, help="seconds")
    ap.add_argument("--max-model-tokens", type=int, default=2000)
    ap.add_argument("--verbose", action="store_true", help="also show tool calls and memory writes")
    ap.add_argument("--json", metavar="PATH", help="write the run summary as JSON")
    ap.add_argument("--workflow", choices=["basic", "full"], default="basic",
                    help="basic = evidence check only; full = literature analysis, gaps, hypotheses, verification and a report")
    ap.add_argument("--report-dir", metavar="DIR", help="full workflow: write report.json and report.md here")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = repo_root()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else root / p  # noqa: E731
    corpus: Corpus | None = None
    try:
        t0 = time.perf_counter()
        if args.library:
            library = resolve(args.library)
            if not (library / "corpus.db").is_file():
                print(f"error: no document library at {library} (create one with scripts/manage_corpus.py ingest ...)",
                      file=sys.stderr)
                return 2
            corpus = Corpus(library)
            index = corpus.build_index(SearchFilters(
                year_min=args.year_min, year_max=args.year_max, licenses=frozenset(args.license) if args.license else None,
                exclude_low_quality=args.exclude_low_quality))
            if index.num_passages == 0:
                print("error: the library has no searchable passages for these filters", file=sys.stderr)
                return 2
            print(f"Library {library}: {index.num_documents} document(s) / {index.num_passages} passage(s) searchable "
                  f"({time.perf_counter() - t0:.1f}s)")
            synthetic = False
        else:
            corpus_dir = resolve(args.corpus)
            if not corpus_dir.is_dir() or not any(corpus_dir.glob("*.txt")):
                print(f"error: no .txt documents in {corpus_dir}", file=sys.stderr)
                return 2
            index = build_index(corpus_dir, args.max_docs)
            print(f"Indexed {index.num_documents} documents / {index.num_passages} passages from {corpus_dir} "
                  f"in {time.perf_counter() - t0:.1f}s")
            synthetic = "synthetic" in str(corpus_dir).lower()
        return run_workflow(args, resolve, index, corpus, synthetic)
    finally:
        if corpus is not None:
            corpus.close()


def run_workflow(args, resolve, index: BM25Index, corpus: Corpus | None, synthetic: bool) -> int:
    backend = None
    if args.checkpoint:
        backend = load_backend(str(resolve(args.checkpoint)), str(resolve(args.tokenizer)), args.device)
        print(f"Attached local model {args.checkpoint} (notes are untrusted; budget {args.max_model_tokens} tokens)")

    store = Store(resolve(args.db))
    catalog = verifier = None
    if corpus is not None:
        def catalog(doc_id: str):
            d = corpus.get_document(doc_id)
            return None if d is None else {"title": d.title, "year": d.year, "doi": d.doi, "license": d.license}

        def verifier(passage_id: str):
            v = corpus.verify_passage(passage_id)
            return v.ok, v.message
    engine = build_engine(store, index, backend=backend, discovery_k=args.discovery_k, workflow=args.workflow,
                          catalog=catalog, verifier=verifier)
    if args.resume:
        run_id = args.resume
        try:
            store.get_run(run_id)
        except KeyError:
            print(f"error: no run {run_id} in {resolve(args.db)}", file=sys.stderr)
            return 2
        print(f"Resuming run {run_id}")
    else:
        project_id = engine.create_project(args.question[:80], args.question)
        limits = RunLimits(max_iterations=args.max_iterations, max_tasks=args.max_tasks, max_runtime_s=args.max_runtime,
                           max_model_tokens=args.max_model_tokens)
        run_id = engine.create_run(project_id, args.question, limits)
        print(f"Created run {run_id}\n")

    try:
        result = asyncio.run(execute(engine, store, run_id, args.verbose))
    except KeyboardInterrupt:
        print(f"\nInterrupted. State is saved; continue with: --resume {run_id} --db {args.db}", file=sys.stderr)
        return 130

    citations = check_citations(corpus, result.summary) if corpus is not None else None
    print_summary(store, index, run_id, synthetic, corpus, citations)
    report = report_from_store(store, run_id) if args.workflow == "full" else None
    if args.workflow == "full":
        if report is None:
            print("\nNo report was produced (the workflow did not reach synthesis).")
        else:
            print(f"\nReport: {report['status']} - {sum(len(v) for v in report['findings'].values())} verified finding(s), "
                  f"{len(report['unsupported_claims'])} unsupported, {len(report['candidate_gaps'])} candidate gap(s), "
                  f"{len(report['hypotheses'])} hypothesis proposal(s); human review: {report['human_review']['status']}")
            if args.report_dir:
                for kind, path in export_report(report, resolve(args.report_dir)).items():
                    print(f"  wrote {kind}: {path}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        payload = {"run_id": run_id, "status": result.status.value, "stop_reason": result.stop_reason,
                   "summary": result.summary}
        if citations is not None:
            payload["citation_checks"] = citations
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    store.close()
    return 0 if result.status == RunStatus.COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())
