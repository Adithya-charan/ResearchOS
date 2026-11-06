"""P3: validated state transitions, per-attempt execution records, the lifecycle events a run must persist, and task detail over
the API. Scripted agents are test doubles (their output is labelled as such by the harness), never model output."""
import pytest
from fastapi.testclient import TestClient

from agent_engine.contracts import ProposedTask, RunStatus
from agent_engine.state import transitions
from agent_engine.state.store import Store
from agent_engine.state.transitions import InvalidTransition
from tests.conftest_agents import make_engine, ok, run, start
from tests.test_api_experiments import make_app, project, sign_in


def P(key, agent="worker", deps=()):
    return ProposedTask(key=key, agent_type=agent, title=key, depends_on=list(deps), input={})


def supervisor(*proposals):
    return lambda ctx, task: ok(task, proposed_tasks=list(proposals))


# ------------------------------------------------------------------ transitions
def test_terminal_run_statuses_cannot_be_left_and_illegal_jumps_are_refused():
    s = Store()
    pid = s.create_project("t", "q")
    rid = s.create_run(pid, "q", {})
    with pytest.raises(InvalidTransition, match="pending' to 'completed"):
        s.update_run(rid, status="completed")  # must go through running
    s.update_run(rid, status="running")
    s.update_run(rid, status="paused")
    s.update_run(rid, status="running")
    s.update_run(rid, status="completed")
    for later in ("running", "failed", "cancelled", "paused"):
        with pytest.raises(InvalidTransition):
            s.update_run(rid, status=later)
    s.update_run(rid, status="completed")  # restating the current status is harmless
    assert s.get_run(rid)["status"] == "completed"


def test_task_transitions_are_enforced():
    s = Store()
    rid = s.create_run(s.create_project("t", "q"), "q", {})
    s.add_tasks(rid, [{"id": "t1", "agent_type": "x", "title": "T"}])
    with pytest.raises(InvalidTransition):
        s.update_task("t1", status="succeeded")  # pending -> succeeded skips execution
    s.update_task("t1", status="running")
    s.update_task("t1", status="succeeded")
    with pytest.raises(InvalidTransition):
        s.update_task("t1", status="running")  # no resurrection of finished work
    assert s.get_task("t1")["status"] == "succeeded"


def test_every_documented_job_transition_target_is_a_known_status():
    known = set(transitions.JOB)
    assert all(targets <= known for targets in transitions.JOB.values())
    transitions.check_job("queued", "running")
    transitions.check_job("running", "queued")  # retry / graceful hand-back
    with pytest.raises(InvalidTransition):
        transitions.check_job("done", "running")
    with pytest.raises(InvalidTransition):
        transitions.check_job("cancelled", "queued")


# ------------------------------------------------------------------ execution records and events
def test_every_attempt_gets_an_execution_record_with_outcome_and_duration():
    attempts = []

    def flaky(ctx, task):
        attempts.append(task.attempt)
        if task.attempt < 2:
            raise RuntimeError("transient")
        return ok(task)

    engine, store = make_engine({"supervisor": supervisor(P("w")), "worker": flaky})
    rid = start(engine, max_retries=2)
    assert run(engine.run(rid)).status == RunStatus.COMPLETED
    w = next(t for t in store.list_tasks(rid) if t["title"] == "w")
    ex = store.list_executions(rid, w["id"])
    assert [(e["attempt"], e["status"]) for e in ex] == [(1, "retry"), (2, "succeeded")]
    assert "transient" in ex[0]["error"] and ex[1]["error"] is None
    assert all(e["duration_s"] is not None and e["duration_s"] >= 0 and e["finished_at"] for e in ex)
    assert len(store.list_executions(rid)) == len(store.list_tasks(rid)) + 1  # one per task plus the extra retry attempt


def test_a_failed_task_records_a_failed_execution():
    def broken(ctx, task):
        raise ValueError("always")

    engine, store = make_engine({"supervisor": supervisor(P("w")), "worker": broken})
    rid = start(engine, max_retries=1)
    run(engine.run(rid))
    w = next(t for t in store.list_tasks(rid) if t["title"] == "w")
    assert [e["status"] for e in store.list_executions(rid, w["id"])] == ["retry", "failed"]


def test_executions_left_open_by_a_dead_worker_are_closed_as_abandoned_on_recovery():
    engine, store = make_engine({"supervisor": supervisor(P("w")), "worker": lambda c, t: ok(t)})
    rid = start(engine)
    first = store.list_tasks(rid)[0]
    store.update_task(first["id"], status="running")
    open_id = store.begin_execution(rid, first["id"], first["agent_type"], 1)  # the worker died before finishing it
    assert engine.recover(rid) == [first["id"]]
    row = next(e for e in store.list_executions(rid) if e["id"] == open_id)
    assert row["status"] == "abandoned" and row["finished_at"] and "worker lost" in row["error"]


def test_a_run_persists_its_lifecycle_events_in_order():
    engine, store = make_engine({"supervisor": supervisor(P("w")), "worker": lambda c, t: ok(t)})
    rid = start(engine)
    run(engine.run(rid))
    types = [e["type"] for e in store.list_events(rid)]
    for needed in ("run_created", "planning_started", "agent_started", "task_started", "task_succeeded", "run_finished"):
        assert needed in types, needed
    assert types.index("run_created") < types.index("run_started") < types.index("planning_started") \
        < types.index("agent_started") < types.index("task_started") < types.index("task_succeeded") < types.index("run_finished")
    assert types.count("planning_started") == 1  # only the first supervisor attempt plans from scratch
    ids = [e["id"] for e in store.list_events(rid)]
    assert ids == sorted(ids)


def test_failure_pause_and_resume_events_are_persisted():
    box = {}

    def pauser(ctx, task):
        box["engine"].pause(task.run_id)  # a user pauses while this task is executing
        return ok(task)

    def broken(ctx, task):
        raise ValueError("always")

    engine, store = make_engine({"supervisor": supervisor(P("a", "pauser"), P("b", "worker", deps=["a"])),
                                 "pauser": pauser, "worker": broken})
    box["engine"] = engine
    rid = start(engine, max_retries=0)
    assert run(engine.run(rid)).status == RunStatus.PAUSED  # stops at the next safe point
    engine.resume(rid)
    run(engine.run(rid))
    types = [e["type"] for e in store.list_events(rid)]
    assert {"pause_requested", "run_paused", "resume_requested", "run_resumed", "task_failed"} <= set(types)
    assert types.index("run_paused") < types.index("run_resumed") < types.index("task_failed")


# ------------------------------------------------------------------ the API surface
def test_task_detail_returns_executions_and_logs_and_is_isolated(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c, "alice@example.org")
        rid = c.post(f"/projects/{project(c, h)}/runs", json={}, headers=h).json()["id"]
        import time
        for _ in range(600):
            if c.get(f"/runs/{rid}").json()["state"]["status"] == "completed":
                break
            time.sleep(0.1)
        tasks = c.get(f"/runs/{rid}/tasks").json()
        detail = c.get(f"/runs/{rid}/tasks/{tasks[0]['id']}").json()
        assert [e["attempt"] for e in detail["executions"]] == [1] and detail["executions"][0]["status"] == "succeeded"
        assert {"task_started", "task_succeeded"} <= {e["type"] for e in detail["logs"]}
        assert all(e["task_id"] == tasks[0]["id"] for e in detail["logs"])
        assert len(c.get(f"/runs/{rid}/executions").json()) >= len(tasks)
        c.post("/auth/logout", headers=h)
        other = sign_in(c, "mallory@example.org")
        assert c.get(f"/runs/{rid}/executions", headers=other).status_code == 404
        assert c.get(f"/runs/{rid}/tasks/{tasks[0]['id']}", headers=other).status_code == 404
