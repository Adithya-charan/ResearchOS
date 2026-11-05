"""Phase 9: sandboxed experiments, reproducibility manifest, review, knowledge graph, resource controls.
Docker is not required: its command line is checked as data, and runs use the (explicitly weak) process fallback."""
import json
import sys
from pathlib import Path

import pytest

from agent_engine.contracts import AgentOutput, RunStatus
from research.experiments import (DockerSandbox, ExperimentSpec, NotApproved, QuotaExceeded, SandboxRefused,
                                  SubprocessSandbox, default_sandbox, load_manifest, read_review, reproduce,
                                  review_experiment, run_experiment, screen_code, verify_manifest)
from research.experiments.examples import example_spec
from research.experiments.runner import check_quota, parse_metrics
from research.formal import check_smt2, z3_available
from research.knowledge_graph import build_graph, to_mermaid

WEAK = SubprocessSandbox()


def spec(code, **kw):
    return ExperimentSpec(name="t", code=code, **kw)


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "runs"


# ================================================================== a reproducible experiment, end to end
def test_example_runs_records_a_complete_manifest_and_verifies(root):
    run_dir, m = run_experiment(example_spec(), WEAK, root, approved_by="alice")
    assert m["status"] == "succeeded" and m["metrics"]["samples"] == 20000
    assert 3.0 < m["metrics"]["pi_estimate"] < 3.3
    for key in ("code_sha256", "dataset_sha256", "dependencies", "seed", "config", "hardware", "limits", "result",
                "metrics", "artifacts", "isolation", "manifest_sha256"):
        assert key in m
    assert m["dependencies"]["python"] and (run_dir / "stdout.log").is_file() and (run_dir / "spec.json").is_file()
    assert m["artifacts"][0]["path"] == "estimate.txt" and verify_manifest(run_dir) == []
    assert m["isolation"]["level"] == "process" and m["isolation"]["network_blocked"] is False  # honest labelling


def test_same_spec_reproduces_and_a_different_seed_does_not(root):
    run_dir, m = run_experiment(example_spec(seed=3), WEAK, root, approved_by="alice")
    rep = reproduce(run_dir, WEAK, approved_by="alice")
    assert rep["reproduced"] is True and rep["differences"] == []
    assert (run_dir / "reproduction.json").is_file()
    # a run that depends on the clock is detected as not reproducible
    flaky, _ = run_experiment(spec('import json, time\nprint("RESULT " + json.dumps({"t": time.time()}))'), WEAK, root,
                              approved_by="alice")
    bad = reproduce(flaky, WEAK, approved_by="alice")
    assert bad["reproduced"] is False and any("metric t" in d for d in bad["differences"])
    other = run_experiment(example_spec(seed=4), WEAK, root, approved_by="alice")[1]
    assert other["metrics"]["pi_estimate"] != m["metrics"]["pi_estimate"]


def test_tolerance_allows_small_numeric_drift(root):
    code = ('import json, os\nprint("RESULT " + json.dumps({"x": 1.0 + 1e-9 * (os.getpid() % 7)}))')
    run_dir, _ = run_experiment(spec(code), WEAK, root, approved_by="a")
    assert reproduce(run_dir, WEAK, approved_by="a", rtol=1e-6)["reproduced"] is True


# ================================================================== approval and review
def test_no_run_without_a_named_approver(root):
    with pytest.raises(NotApproved):
        run_experiment(example_spec(), WEAK, root, approved_by="  ")
    assert not root.exists() or not any(root.iterdir())


def test_review_needs_an_intact_succeeded_run_and_records_reproduction(root):
    run_dir, _ = run_experiment(example_spec(), WEAK, root, approved_by="alice")
    assert read_review(run_dir)["status"] == "pending"
    r = review_experiment(run_dir, "bob", "accepted", "looks right")
    assert r["status"] == "accepted" and r["reproduced"] is None  # accepted without a reproduction is visible as such
    reproduce(run_dir, WEAK, approved_by="alice")
    assert review_experiment(run_dir, "bob", "accepted")["reproduced"] is True
    with pytest.raises(ValueError):
        review_experiment(run_dir, "", "accepted")
    with pytest.raises(ValueError):
        review_experiment(run_dir, "bob", "maybe")

    failed, m = run_experiment(spec("raise SystemExit(3)"), WEAK, root, approved_by="alice")
    assert m["status"] == "failed" and m["result"]["exit_code"] == 3
    with pytest.raises(ValueError, match="succeeded"):
        review_experiment(failed, "bob", "accepted")
    assert review_experiment(failed, "bob", "rejected")["status"] == "rejected"


def test_tampering_after_the_run_is_detected_and_blocks_acceptance(root):
    run_dir, _ = run_experiment(example_spec(), WEAK, root, approved_by="alice")
    (run_dir / "out" / "estimate.txt").write_text("3.0\n")
    assert any("artifact changed" in p for p in verify_manifest(run_dir))
    with pytest.raises(ValueError, match="manifest"):
        review_experiment(run_dir, "bob", "accepted")
    assert reproduce(run_dir, WEAK, approved_by="alice")["reproduced"] is False

    d2, _ = run_experiment(example_spec(), WEAK, root, approved_by="alice")
    (d2 / "main.py").write_bytes(b"print('x')")
    assert any("main.py" in p for p in verify_manifest(d2))
    d3, _ = run_experiment(example_spec(), WEAK, root, approved_by="alice")
    m = json.loads((d3 / "manifest.json").read_text())
    m["metrics"]["pi_estimate"] = 3.14159
    (d3 / "manifest.json").write_text(json.dumps(m))
    assert any("manifest content" in p for p in verify_manifest(d3))


# ================================================================== sandbox controls
def test_timeout_kills_the_run_and_is_recorded(root):
    _, m = run_experiment(spec("import time\ntime.sleep(30)", timeout_s=1), WEAK, root, approved_by="a")
    assert m["status"] == "timed_out" and m["result"]["timed_out"] and m["result"]["duration_s"] < 10


def test_output_is_capped_and_flagged(root):
    code = 'print("x" * 5000)\nprint("RESULT {\\"ok\\": 1}")'
    _, m = run_experiment(spec(code, max_output_bytes=1000), WEAK, root, approved_by="a")
    assert m["result"]["truncated"] and any("truncated" in w for w in m["warnings"]) and m["resources"]["stdout_bytes"] <= 1000


def test_metrics_parsing_is_strict_and_reports_problems():
    assert parse_metrics('noise\nRESULT {"a": 1}\n') == ({"a": 1}, [])
    assert parse_metrics("RESULT [1]")[0] == {} and parse_metrics("RESULT {bad")[1]
    assert parse_metrics("nothing")[1]


def test_environment_is_scrubbed_and_cwd_is_the_run_directory(root):
    code = ('import json, os\nprint("RESULT " + json.dumps({"secret": os.environ.get("RESEARCHOS_SECRET_TEST"), '
            '"cwd": os.path.basename(os.getcwd())}))')
    import os
    os.environ["RESEARCHOS_SECRET_TEST"] = "hunter2"
    try:
        run_dir, m = run_experiment(spec(code), WEAK, root, approved_by="a")
    finally:
        del os.environ["RESEARCHOS_SECRET_TEST"]
    assert m["metrics"]["secret"] is None and m["metrics"]["cwd"] == run_dir.name


def test_static_screen_refuses_obvious_network_and_process_use_and_leaves_nothing_behind(root):
    assert screen_code("import socket") == ["socket"] and screen_code("from urllib import request") == ["urllib"]
    assert screen_code("exec('1')") == ["exec"] and screen_code("x = (") != []
    assert screen_code("import json, random") == []
    with pytest.raises(SandboxRefused):
        run_experiment(spec("import socket"), WEAK, root, approved_by="a")
    assert list(root.iterdir()) == []


def test_model_generated_code_is_refused_without_a_container(root):
    with pytest.raises(SandboxRefused, match="container"):
        run_experiment(spec("print(1)", generated_by_model=True), WEAK, root, approved_by="a")
    with pytest.raises(SandboxRefused):
        default_sandbox(allow_weak=False) if not DockerSandbox().available else (_ for _ in ()).throw(SandboxRefused("docker"))


def test_docker_command_applies_every_blueprint_control(tmp_path):
    cmd = DockerSandbox(docker="docker").command(spec("print(1)", memory_mb=256, cpus=2, seed=9), tmp_path, "c1")
    joined = " ".join(cmd)
    for flag in ("--rm", "--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "--user 65534:65534",
                 "--memory 256m", "--cpus 2", "--pids-limit 64", "noexec", "EXPERIMENT_SEED=9"):
        assert flag in joined
    assert "docker.sock" not in joined and ":/work:ro" in joined  # code dir read-only; only out/ writable


def test_spec_validation_rejects_unsafe_input():
    for bad in ({"files": {"../x": "a"}}, {"files": {"main.py": "a"}}, {"timeout_s": 10_000}, {"memory_mb": 1}, {"nope": 1}):
        with pytest.raises(ValueError):
            ExperimentSpec(name="t", code="print(1)", **bad)


def test_input_files_are_hashed_and_readable_by_the_code(root):
    code = 'import json\nn = len(open("data.txt").read().split())\nprint("RESULT " + json.dumps({"words": n}))'
    run_dir, m = run_experiment(spec(code, files={"data.txt": "a b c d"}), WEAK, root, approved_by="a")
    assert m["metrics"]["words"] == 4 and m["input_files"] == ["data.txt"] and verify_manifest(run_dir) == []
    other = run_experiment(spec(code, files={"data.txt": "a b c"}), WEAK, root, approved_by="a")[1]
    assert other["dataset_sha256"] != m["dataset_sha256"]


def test_quota_stops_new_runs(root):
    root.mkdir()
    for i in range(3):
        (root / f"r{i}").mkdir()
    with pytest.raises(QuotaExceeded):
        check_quota(root, max_runs=3)
    check_quota(root, max_runs=4)


# ================================================================== engine integration: approval gate
def test_engine_runs_an_experiment_only_after_human_approval(tmp_path):
    from agent_engine.contracts import ProposedTask, RunLimits
    from agent_engine.registry.defaults import build_engine
    from agent_engine.state.store import Store
    from research.bm25.index import BM25Index, Passage
    from tests.conftest_agents import run

    idx = BM25Index()
    idx.add_passages([Passage(passage_id="d:p001", document_id="d", text="alpha beta gamma", page_number=1, section="", quality="ok")])
    store = Store()
    engine = build_engine(store, idx, workflow="full", experiments=(WEAK, tmp_path / "runs"), poll_interval=0.01)
    # replace the Supervisor with one that proposes an experiment (humans/planners author the spec; agents never write code)
    from agent_engine.agents.base import Agent

    class Plan(Agent):
        agent_type = "supervisor"
        description = "test planner"

        async def run(self, ctx, task):
            return AgentOutput(task_id=task.id, agent_type="supervisor", status="ok", summary="plan",
                               proposed_tasks=[ProposedTask(key="exp", agent_type="experiment", title="run pi",
                                                            input={"spec": example_spec().model_dump()})])

    engine.registry._agents["supervisor"] = Plan() if hasattr(engine.registry, "_agents") else None
    rid = engine.create_run(engine.create_project("e", "q"), "q", RunLimits(retry_backoff_s=0, task_timeout_s=20, max_runtime_s=60))
    res = run(engine.run(rid))
    assert res.status == RunStatus.PAUSED and res.stop_reason == "awaiting_approval"
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())  # nothing ran yet

    approval = store.list_approvals(rid, pending_only=True)[0]
    engine.decide_approval(approval["id"], True, "reviewer-7", "code read, safe")
    res = run(engine.run(rid))
    assert res.status == RunStatus.COMPLETED
    task = [t for t in store.list_tasks(rid) if t["agent_type"] == "experiment"][0]
    out = AgentOutput.model_validate(task["output"])
    exp = out.data["experiment"]
    assert exp["status"] == "succeeded" and exp["review"] == "pending"
    assert load_manifest(Path(exp["run_dir"]))["approved_by"] == "reviewer-7"
    assert any("process fallback" in u for u in out.uncertainties)


def test_a_rejected_experiment_never_runs(tmp_path):
    from agent_engine.agents.base import Agent
    from agent_engine.contracts import ProposedTask, RunLimits
    from agent_engine.registry.defaults import build_engine
    from agent_engine.state.store import Store
    from research.bm25.index import BM25Index
    from tests.conftest_agents import run

    store = Store()
    engine = build_engine(store, BM25Index(), workflow="full", experiments=(WEAK, tmp_path / "runs"), poll_interval=0.01)

    class Plan(Agent):
        agent_type = "supervisor"
        description = "test planner"

        async def run(self, ctx, task):
            return AgentOutput(task_id=task.id, agent_type="supervisor", status="ok", summary="plan",
                               proposed_tasks=[ProposedTask(key="exp", agent_type="experiment", title="x",
                                                            input={"spec": example_spec().model_dump()})])

    engine.registry._agents["supervisor"] = Plan()
    rid = engine.create_run(engine.create_project("e", "q"), "q", RunLimits(retry_backoff_s=0, task_timeout_s=20, max_runtime_s=60))
    run(engine.run(rid))
    engine.decide_approval(store.list_approvals(rid, pending_only=True)[0]["id"], False, "reviewer", "no")
    run(engine.run(rid))
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


# ================================================================== knowledge graph and formal checks
def test_graph_links_claims_gaps_hypotheses_and_experiments_without_upgrading_labels():
    report = {
        "question": "q?", "titles": {"d1": "Doc One"},
        "findings": {"method": [], "limitation": [], "result": [
            {"claim_id": "c1", "text": "Cells retained ninety percent.", "section": "R",
             "citation": {"document_id": "d1", "page_number": 2, "passage_id": "d1:p002"}}]},
        "needs_review": [{"claim_id": "c2", "text": "Cells lost half.", "reasons": ["x"],
                          "citation": {"document_id": "d1", "page_number": 3, "passage_id": "d1:p003"}}],
        "unsupported_claims": [{"claim_id": "c3", "text": "bad", "reasons": ["y"]}],
        "tensions": [{"a": "c1", "b": "c2", "shared_terms": ["cell"]}],
        "candidate_gaps": [{"id": "g1", "kind": "stated_limitation", "statement": "gap",
                            "evidence": [{"document_id": "d1", "page_number": 3, "passage_id": "d1:p003"}]}],
        "hypotheses": [{"id": "h1", "statement": "hyp", "gap_id": "g1"}],
    }
    g = build_graph(report, [{"id": "exp-1", "name": "pi", "status": "succeeded", "hypothesis_id": "h1", "review": "pending"}])
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["c1"]["status"] == "verified" and by_id["c2"]["status"] == "needs_review" and by_id["c3"]["status"] == "unsupported"
    assert by_id["g1"]["status"] == "candidate_gap" and by_id["h1"]["status"] == "hypothesis"
    assert by_id["exp-1"]["review"] == "pending"
    kinds = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
    assert ("c1", "d1", "cited_from") in kinds and ("h1", "g1", "derived_from") in kinds and ("exp-1", "h1", "tests") in kinds
    assert ("c1", "c2", "possible_tension") in kinds and ("g1", "d1", "evidenced_by") in kinds
    assert all(e["source"] in by_id and e["target"] in by_id for e in g["edges"])
    assert to_mermaid(g).startswith("graph LR") and "possible_tension" in to_mermaid(g)


def test_graph_from_a_real_workflow_report_is_consistent(tmp_path):
    pytest.importorskip("pymupdf")
    from tests.test_research_workflow import corpus as _c  # noqa: F401  (fixture reuse not needed; build directly)
    from agent_engine.registry.defaults import build_engine
    from agent_engine.reporting import report_from_store
    from agent_engine.contracts import RunLimits
    from agent_engine.state.store import Store
    from research.corpus import Corpus
    from tests.conftest_agents import run
    from tests.corpus_fixtures import build_library

    c = Corpus(tmp_path / "c", isolate=False)
    for p in build_library(tmp_path / "pdfs").values():
        c.ingest(p)
    store = Store()
    engine = build_engine(store, c.build_index(), workflow="full", poll_interval=0.01)
    q = "What percent accuracy was retained?"
    rid = engine.create_run(engine.create_project("g", q), q, RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60))
    run(engine.run(rid))
    g = build_graph(report_from_store(store, rid))
    ids = {n["id"] for n in g["nodes"]}
    assert {n["type"] for n in g["nodes"]} >= {"question", "document", "claim", "gap", "hypothesis"}
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
    c.close()


def test_formal_adapter_degrades_gracefully():
    r = check_smt2("(declare-const x Int)(assert (> x 3))(assert (< x 2))")
    if z3_available():
        assert r["result"] == "unsat"
        assert check_smt2("(assert (")["result"] == "error"
    else:
        assert r["available"] is False and r["result"] == "unavailable"


def test_cli_run_reproduce_review_show(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_experiment as cli
    root = tmp_path / "runs"
    assert cli.main(["run", "--example", "--approved-by", "alice", "--root", str(root)]) == (0 if DockerSandbox().available else 2)
    err = capsys.readouterr()
    if not DockerSandbox().available:
        assert "Docker is not available" in err.err  # refuses by default rather than silently running unsandboxed
    assert cli.main(["run", "--example", "--approved-by", "alice", "--root", str(root), "--allow-weak"]) == 0
    out = capsys.readouterr().out
    assert "integrity: OK" in out and "NO ISOLATION" in out
    run_dir = next(root.iterdir())
    assert cli.main(["reproduce", str(run_dir), "--approved-by", "alice", "--allow-weak"]) == 0
    assert cli.main(["review", str(run_dir), "--reviewer", "bob", "--decision", "accepted"]) == 0
    capsys.readouterr()
    assert cli.main(["show", str(run_dir)]) == 0
    shown = capsys.readouterr().out
    assert "human review: accepted by bob; reproduced: True" in shown
    assert cli.main(["run", "--approved-by", "a"]) == 2
