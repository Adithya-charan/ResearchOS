"""P4: the nine-agent research workflow as a system - declared contracts checked against real outputs, dependency ordering,
agent isolation, replanning after a failed search, duplicate prevention and shared context. The corpus is the invented fixture
library; scripted agents (test doubles) are used only where a test must force a failure."""
import asyncio

import pytest

pytest.importorskip("pymupdf")

from agent_engine.agents.discovery import DiscoveryAgent  # noqa: E402
from agent_engine.agents.specs import SPECS, AgentSpec  # noqa: E402
from agent_engine.contracts import AgentOutput, ProposedTask, RunLimits, RunStatus  # noqa: E402
from agent_engine.policies.policy import default_policy  # noqa: E402
from agent_engine.planner.planner import expand  # noqa: E402
from agent_engine.registry.defaults import build_engine  # noqa: E402
from agent_engine.state.store import Store  # noqa: E402
from research.corpus import Corpus  # noqa: E402
from tests.conftest_agents import make_engine, ok, run, start  # noqa: E402
from tests.corpus_fixtures import build_library  # noqa: E402

QUESTION = "What percent accuracy was retained?"
REQUIRED_ROLES = {"Supervisor", "Literature Discovery", "Literature Analysis", "Research Gap", "Hypothesis", "Experiment Design",
                  "Critical Reviewer", "Verification", "Synthesis"}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    lib = build_library(tmp_path_factory.mktemp("pdfs"))
    c = Corpus(tmp_path_factory.mktemp("corpus"), isolate=False)
    for path in lib.values():
        assert c.ingest(path).status == "ingested"
    yield c
    c.close()


def engine_for(corpus, **kw):
    store = Store()
    return store, build_engine(store, corpus.build_index(), workflow="full", poll_interval=0.01, **kw)


def execute(engine, question=QUESTION, **limits):
    rid = engine.create_run(engine.create_project("p4", question), question,
                            RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60, **limits))
    return rid, run(engine.run(rid))


@pytest.fixture(scope="module")
def full(corpus):
    store, engine = engine_for(corpus)
    rid, result = execute(engine)
    return store, engine, rid, result


# ------------------------------------------------------------------ the agent system
def test_all_nine_required_roles_exist_with_a_complete_declared_contract(full):
    _, engine, _, _ = full
    described = {d["agent_type"]: d for d in engine.registry.describe(engine.policy)}
    roles = {d["spec"]["role"] for d in described.values() if "spec" in d}
    assert REQUIRED_ROLES <= roles
    for t, d in described.items():
        sp = d["spec"]
        assert sp["role"] and sp["responsibilities"] and sp["output_keys"] and sp["evaluation_criteria"], t
        assert isinstance(d["tools"], list) and isinstance(d["may_propose"], list)  # permissions are declared per agent
    assert set(SPECS) >= set(described)


def test_each_agents_real_inputs_and_outputs_match_its_declared_contract(full):
    store, _, rid, result = full
    assert result.status == RunStatus.COMPLETED
    seen = set()
    for t in store.list_tasks(rid):
        spec = SPECS.get(t["agent_type"])
        if spec is None or t["status"] != "succeeded":
            continue
        seen.add(t["agent_type"])
        for k in spec.input_keys:
            assert k in t["input"], f"{t['agent_type']} task input lacks '{k}'"
        out = AgentOutput.model_validate(t["output"])  # typed output
        assert out.task_id == t["id"] and out.agent_type == t["agent_type"]
        for k in spec.output_keys:
            assert k in out.data, f"{t['agent_type']} output lacks data['{k}']"
    assert {"supervisor", "discovery", "critic", "literature", "gap", "hypothesis", "experiment_design", "reviewer",
            "verifier", "synthesis"} <= seen


def test_every_task_started_only_after_its_dependencies_succeeded(full):
    store, _, rid, _ = full
    tasks = {t["id"]: t for t in store.list_tasks(rid)}
    for t in tasks.values():
        for dep in t["depends_on"]:
            assert tasks[dep]["status"] == "succeeded"
            assert tasks[dep]["completed_at"] <= t["started_at"], f"{t['title']} ran before {tasks[dep]['title']} finished"


def test_the_review_and_plans_come_before_synthesis_and_are_in_the_report(full):
    store, _, rid, _ = full
    order = [e["task_id"] for e in store.list_events(rid) if e["type"] == "task_succeeded"]
    by_type = {t["agent_type"]: t["id"] for t in store.list_tasks(rid)}
    assert order.index(by_type["experiment_design"]) < order.index(by_type["reviewer"]) < order.index(by_type["synthesis"])


def test_experiment_plans_contain_no_results_and_cite_only_real_passages(full, corpus):
    store, _, rid, _ = full
    plans = next(t for t in store.list_tasks(rid) if t["agent_type"] == "experiment_design")["output"]["data"]["plans"]
    assert plans
    for p in plans:
        assert p["status"] == "plan_only" and "No experiment has been run" in p["label"]
        assert p["reproducibility"]["repeats_min"] >= 3
        assert not {"results", "result", "measured", "outcome_value"} & set(p)
        for item in p["datasets"] + p["baselines"] + p["metrics"]:
            c = item["citation"]
            assert corpus.verify_passage(c["passage_id"]).ok  # citation resolves to a stored passage that matches its source file
        for missing in p["needs_human_input"]:
            assert missing in ("datasets", "baselines", "metrics", "dependent variable", "independent variable")


def test_the_reviewer_reports_severities_never_approval_and_checks_plans(full):
    store, _, rid, _ = full
    out = next(t for t in store.list_tasks(rid) if t["agent_type"] == "reviewer")["output"]
    review = out["data"]["review"]
    assert set(review["counts"]) == {"blocking", "major", "minor"}
    assert "not an approval" in review["disclaimer"]
    assert all(f["severity"] in ("blocking", "major", "minor") for f in review["findings"])
    assert any(f["category"] == "assumption" for f in review["findings"])  # untested assumptions are always called out
    assert not any("approved" in f["message"].lower() for f in review["findings"])


def test_shared_context_is_consistent_across_agents(full):
    store, _, rid, _ = full
    run_row = store.get_run(rid)
    scope = store.memory_get(run_row["project_id"], "scope")["value"]
    tasks = store.list_tasks(rid)
    sup = next(t for t in tasks if t["agent_type"] == "supervisor")
    assert scope["terms"] == sup["output"]["data"]["terms"]
    for t in tasks:
        if t["agent_type"] in ("critic", "literature", "gap", "hypothesis", "experiment_design", "reviewer", "verifier"):
            assert t["input"]["terms"] == scope["terms"] and t["input"]["question"] == scope["question"]


# ------------------------------------------------------------------ isolation, duplicates, per-agent limits
def test_an_agent_sees_only_the_outputs_of_its_declared_dependencies():
    seen = {}

    def spy(ctx, task):
        seen["deps"] = set(task.dependency_outputs)
        return ok(task)

    engine, store = make_engine({
        "supervisor": lambda c, t: ok(t, proposed_tasks=[ProposedTask(key="a", agent_type="worker", title="a"),
                                                         ProposedTask(key="b", agent_type="worker", title="b"),
                                                         ProposedTask(key="spy", agent_type="spy", title="spy", depends_on=["a"])]),
        "worker": lambda c, t: ok(t), "spy": spy})
    rid = start(engine)
    run(engine.run(rid))
    a = next(t for t in store.list_tasks(rid) if t["title"] == "a")
    assert seen["deps"] == {a["id"]}  # b ran too, but its output is not visible to the spy


def test_an_identical_task_is_not_queued_twice_while_the_first_is_waiting_but_may_repeat_after_it_finished():
    limits, pol = RunLimits(), default_policy()
    parent = {"id": "t0", "agent_type": "supervisor", "iteration": 0}
    p = ProposedTask(key="d", agent_type="discovery", title="Search", input={"query": "x", "k": 5})
    pending = [{"id": "t1", "agent_type": "discovery", "title": "Search", "input": {"query": "x", "k": 5}, "depends_on": [],
                "status": "pending"}]
    acc, rej = expand(parent, [p], pending, limits, pol, {"discovery"})
    assert not acc and "duplicate_task" in rej[0].reason
    done = [{**pending[0], "status": "succeeded"}]
    acc, rej = expand(parent, [p], done, limits, pol, {"discovery"})
    assert len(acc) == 1 and not rej
    acc, rej = expand(parent, [p, p.model_copy(update={"key": "d2"})], [], limits, pol, {"discovery"})
    assert len(acc) == 1 and "duplicate_task" in rej[0].reason  # twins inside one proposal batch too


def test_a_per_agent_timeout_and_retry_ceiling_tighten_the_run_limits(monkeypatch):
    monkeypatch.setitem(SPECS, "worker", AgentSpec("worker", "Slow worker", ("test",), (), ("x",), timeout_s=0.05, max_retries=0,
                                                   evaluation_criteria=("test",)))

    async def slow(ctx, task):
        await asyncio.sleep(2)
        return ok(task)

    engine, store = make_engine({
        "supervisor": lambda c, t: ok(t, proposed_tasks=[ProposedTask(key="w", agent_type="worker", title="w")]), "worker": slow})
    rid = start(engine, max_retries=5, task_timeout_s=30)
    run(engine.run(rid))
    w = next(t for t in store.list_tasks(rid) if t["agent_type"] == "worker")
    assert w["status"] == "failed" and "timeout" in w["error"] and w["retry_count"] == 0  # one attempt: ceiling 0 beats run's 5


# ------------------------------------------------------------------ supervisor replanning
class FlakyDiscovery(DiscoveryAgent):
    """Test double: refuses one specific search so the Supervisor has something to replan around."""
    def __init__(self, bad_label: str):
        self.bad_label = bad_label

    async def run(self, ctx, task):
        if task.input.get("label") == self.bad_label:
            raise ValueError("search backend unavailable (injected by the test)")
        return await super().run(ctx, task)


def test_the_supervisor_replans_once_after_a_failed_search_and_the_run_still_completes(corpus):
    store, engine = engine_for(corpus)
    engine.registry._agents["discovery"] = FlakyDiscovery("full question")
    rid, result = execute(engine, max_retries=0)
    assert result.status in (RunStatus.COMPLETED, RunStatus.FAILED)
    tasks = store.list_tasks(rid)
    failed = [t for t in tasks if t["status"] == "failed" and t["agent_type"] == "discovery"]
    assert len(failed) == 1
    replans = [t for t in tasks if t["agent_type"] == "supervisor" and t["input"].get("replan_for")]
    assert len(replans) == 1 and replans[0]["status"] == "succeeded" and replans[0]["input"]["replan_for"] == failed[0]["id"]
    assert any(e["type"] == "replan_requested" for e in store.list_events(rid))
    new_search = [t for t in tasks if t["title"].startswith("Replanned search")]
    assert len(new_search) == 1 and new_search[0]["status"] == "succeeded"
    assert new_search[0]["input"]["query"] != failed[0]["input"]["query"]  # not a blind repeat of the failed task
    # the original review was doomed by the failure; the replacement review took over and the workflow continued
    assert any(t["title"].startswith("Re-check evidence after replanned search") and t["status"] == "succeeded" for t in tasks)
    assert any(t["agent_type"] == "synthesis" and t["status"] == "succeeded" for t in tasks)


def test_replanning_is_bounded_by_max_replans(corpus):
    store, engine = engine_for(corpus)

    class AlwaysFails(DiscoveryAgent):
        async def run(self, ctx, task):
            raise ValueError("injected failure")

    engine.registry._agents["discovery"] = AlwaysFails()
    rid, result = execute(engine, max_retries=0, max_replans=1)
    assert result.status == RunStatus.FAILED
    replans = [t for t in store.list_tasks(rid) if t["agent_type"] == "supervisor" and t["input"].get("replan_for")]
    assert len(replans) == 1  # one replan, then the failure stands: no endless loop
    assert len([t for t in store.list_tasks(rid)]) < 20


def test_replanning_can_be_switched_off(corpus):
    store, engine = engine_for(corpus)
    engine.registry._agents["discovery"] = FlakyDiscovery("full question")
    rid, _ = execute(engine, max_retries=0, max_replans=0)
    assert not [t for t in store.list_tasks(rid) if t["input"].get("replan_for")]


def test_agents_endpoint_describes_every_role_and_requires_login(tmp_path):
    from fastapi.testclient import TestClient
    from tests.test_api_experiments import make_app, sign_in
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        assert c.get("/agents").status_code == 401
        sign_in(c)
        body = c.get("/agents").json()
        roles = {a["spec"]["role"] for a in body["agents"] if "spec" in a}
        assert REQUIRED_ROLES <= roles
        sup = next(a for a in body["agents"] if a["agent_type"] == "supervisor")
        assert "discovery" in sup["may_propose"] and sup["spec"]["timeout_s"] == 30
        assert next(a for a in body["agents"] if a["agent_type"] == "experiment_design")["tools"] == []  # design only, no tools
