"""Durable job queue and worker: atomic claiming, leases and heartbeats, backoff, restart recovery, graceful shutdown, lifecycle
phases, and the API workflow with a separate worker. Fake engines are used only where a test needs to control timing or failure;
the end-to-end tests run the real agent engine."""
import asyncio
import threading
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings
from api.db import utcnow
from api.jobs import JobQueue, Worker, lifecycle
from api.models import ResearchJob, Run
from tests.test_api_experiments import QUESTION, make_index, project, sign_in


def settings_for(tmp_path, **kw):
    base = dict(database_url=f"sqlite:///{(tmp_path / 'app.db').as_posix()}", engine_db=tmp_path / "engine.db",
                data_dir=tmp_path / "data", corpus_isolate=False, embedded_worker=False, job_poll_s=0.05, job_lease_s=1.5,
                job_backoff_s=0.2, job_max_attempts=3)
    base.update(kw)
    return Settings(**base)


@pytest.fixture()
def app(tmp_path):
    return create_app(settings_for(tmp_path), index_provider=make_index)


@pytest.fixture()
def client(app):
    with TestClient(app, base_url="http://testserver") as c:
        yield c


def make_run(client, start=True, h=None, email="alice@example.org"):
    h = h or sign_in(client, email)
    pid = project(client, h)
    rid = client.post(f"/projects/{pid}/runs", json={"start": start}, headers=h).json()["id"]
    return h, pid, rid


def engine_id(app, rid):
    with app.state.sessionmaker() as db:
        return db.get(Run, rid).engine_run_id


def job_row(app, rid):
    with app.state.sessionmaker() as db:
        j = db.query(ResearchJob).filter_by(run_id=rid).one_or_none()
        return None if j is None else {c.name: getattr(j, c.name) for c in j.__table__.columns}


def set_job(app, rid, **fields):
    with app.state.sessionmaker() as db:
        j = db.query(ResearchJob).filter_by(run_id=rid).one()
        for k, v in fields.items():
            setattr(j, k, v)
        db.commit()


def new_worker(app, **kw):
    return Worker(app.state.settings, app.state.runtime, app.state.runtime.jobs, app.state.sessionmaker, **kw)


def wait_for(pred, timeout=60, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("condition not reached in time")


def lifecycle_of(client, rid):
    return client.get(f"/runs/{rid}").json()["state"]["lifecycle"]


class FakeEngine:
    """Stands in for the research engine when a test must control timing or failure."""
    def __init__(self, rt, eid, behaviour, log):
        self.rt, self.eid, self.behaviour, self.log = rt, eid, behaviour, log

    async def run(self, eid):
        self.log.append(eid)
        await self.behaviour(self.rt, eid)


def fake_runtime(app, behaviour):
    log = []
    rt = app.state.runtime
    rt.engine = lambda project_id: FakeEngine(rt, None, behaviour, log)  # noqa: E731
    return log


async def completes(rt, eid):
    await asyncio.sleep(0.05)
    rt.store.update_run(eid, status="completed")


# ------------------------------------------------------------------ claiming
def test_a_queued_job_is_claimed_by_exactly_one_of_many_racing_workers(client, app):
    _, _, rid = make_run(client, start=True)
    q = app.state.runtime.jobs
    winners, start = [], threading.Barrier(8)

    def race(i):
        start.wait()
        job = q.claim(f"worker-{i}")
        if job:
            winners.append((i, job["id"]))

    threads = [threading.Thread(target=race, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(winners) == 1
    row = job_row(app, rid)
    assert row["status"] == "running" and row["attempts"] == 1 and row["worker_id"] == f"worker-{winners[0][0]}"
    assert q.claim("late") is None  # nothing left to claim


def test_many_jobs_are_each_executed_exactly_once_by_two_workers(client, app):
    h = sign_in(client)
    pid = project(client, h)
    rids = [client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"] for _ in range(4)]
    app.state.settings.max_active_runs_per_user = 10
    for r in rids:
        assert client.post(f"/runs/{r}/start", headers=h).status_code == 200
    log = fake_runtime(app, completes)
    w1, w2 = new_worker(app), new_worker(app)
    w1.start(), w2.start()
    try:
        wait_for(lambda: all(job_row(app, r)["status"] == "done" for r in rids))
    finally:
        w1.stop(), w2.stop()
    assert sorted(log) == sorted(engine_id(app, r) for r in rids)  # no run executed twice, none skipped
    assert {job_row(app, r)["attempts"] for r in rids} == {1}
    assert {job_row(app, r)["worker_id"] for r in rids} <= {w1.worker_id, w2.worker_id}


# ------------------------------------------------------------------ leases, backoff, recovery
def test_heartbeat_extends_the_lease_and_only_the_owner_may_renew(client, app):
    make_run(client)
    q = app.state.runtime.jobs
    job = q.claim("w1")
    before = job_row(app, job["run_id"])["lease_expires_at"]
    time.sleep(0.05)
    assert q.heartbeat(job["id"], "w1") and job_row(app, job["run_id"])["lease_expires_at"] > before
    assert not q.heartbeat(job["id"], "someone-else")


def test_requeue_uses_exponential_backoff_then_fails_the_job_and_the_run(client, app):
    _, _, rid = make_run(client)
    q = app.state.runtime.jobs
    base = app.state.settings.job_backoff_s
    for attempt in (1, 2):
        job = q.claim("w1")
        assert job["attempts"] == attempt
        assert q.requeue(job["id"], "w1", "boom") == "queued"
        row = job_row(app, rid)
        delay = (row["available_at"] - utcnow()).total_seconds()
        assert base * 2 ** (attempt - 1) - 0.15 < delay <= base * 2 ** (attempt - 1) + 0.05  # 0.2s, then 0.4s
        assert q.claim("w2") is None  # not claimable while backing off
        set_job(app, rid, available_at=utcnow())
    job = q.claim("w1")
    assert q.requeue(job["id"], "w1", "boom") == "failed"
    app.state.runtime.jobs._fail_engine_run(job["id"])
    row = job_row(app, rid)
    assert row["status"] == "failed" and "gave up after 3 attempt" in row["error"]
    assert client.get(f"/runs/{rid}").json()["state"]["lifecycle"] == "failed"
    types = [e["type"] for e in app.state.runtime.store.list_events(engine_id(app, rid))]
    assert types.count("job_requeued") == 2 and "job_failed" in types


def test_a_worker_that_dies_mid_task_loses_its_lease_and_the_run_finishes_elsewhere_without_repeating_work(client, app):
    h, pid, rid = make_run(client, start=True)
    q, store, eid = app.state.runtime.jobs, app.state.runtime.store, engine_id(app, rid)
    job = q.claim("dead-worker")  # claimed, then the process vanished: it never heartbeats again
    first = store.list_tasks(eid)[0]  # the crash happened while this task was running
    store.update_task(first["id"], status="running")
    set_job(app, rid, lease_expires_at=utcnow() - timedelta(seconds=1))
    survivor = new_worker(app)
    survivor.start()
    try:
        wait_for(lambda: job_row(app, rid)["status"] == "done")
    finally:
        survivor.stop()
    row = job_row(app, rid)
    assert row["attempts"] == 2 and row["worker_id"] == survivor.worker_id
    state = client.get(f"/runs/{rid}").json()["state"]
    assert state["status"] == "completed" and state["lifecycle"] == "completed"
    types = [e["type"] for e in store.list_events(eid)]
    assert "job_requeued" in types and "task_recovered" in types and types[-1] in ("job_finished", "run_finished")
    done_tasks = [t for t in store.list_tasks(eid) if t["status"] == "succeeded"]
    assert len(done_tasks) >= 8 and len({t["id"] for t in done_tasks}) == len(done_tasks)


def test_an_unexpected_error_is_retried_then_the_run_fails_visibly(client, app):
    _, _, rid = make_run(client)

    async def explodes(rt, eid):
        raise RuntimeError("engine blew up")

    log = fake_runtime(app, explodes)
    w = new_worker(app)
    w.start()
    try:
        wait_for(lambda: job_row(app, rid)["status"] == "failed", timeout=30)
    finally:
        w.stop()
    row = job_row(app, rid)
    assert len(log) == 3 == row["attempts"] and "engine blew up" in row["error"]
    state = client.get(f"/runs/{rid}").json()["state"]
    assert state["status"] == "failed" and state["stop_reason"] == "worker_lost" and state["lifecycle"] == "failed"


def test_a_project_without_documents_fails_the_job_at_once_without_retrying(tmp_path):
    a = create_app(settings_for(tmp_path), index_provider=make_index)
    with TestClient(a, base_url="http://testserver") as c:
        _, _, rid = make_run(c)
        calls = []

        def no_corpus(project_id):
            from api.runtime import NoCorpus
            calls.append(1)
            raise NoCorpus("this project has no searchable documents")

        a.state.runtime.engine = no_corpus
        w = new_worker(a)
        w.start()
        try:
            wait_for(lambda: job_row(a, rid)["status"] == "failed")
        finally:
            w.stop()
        assert len(calls) == 1 and job_row(a, rid)["attempts"] == 1
        assert c.get(f"/runs/{rid}").json()["state"]["lifecycle"] == "failed"


# ------------------------------------------------------------------ graceful shutdown
def test_stopping_a_worker_hands_its_job_back_without_using_an_attempt(client, app):
    _, _, rid = make_run(client)
    started = threading.Event()

    async def hangs(rt, eid):
        started.set()
        await asyncio.sleep(60)

    fake_runtime(app, hangs)
    w1 = new_worker(app)
    w1.start()
    assert started.wait(10)
    w1.stop()
    row = job_row(app, rid)
    assert row["status"] == "queued" and row["attempts"] == 0 and "shutdown" in row["error"] and row["worker_id"] == ""
    fake_runtime(app, completes)  # a replacement worker picks it up
    w2 = new_worker(app)
    w2.start()
    try:
        wait_for(lambda: job_row(app, rid)["status"] == "done")
    finally:
        w2.stop()


def test_a_worker_that_lost_its_lease_stops_and_does_not_touch_the_new_owners_job(client, app):
    _, _, rid = make_run(client)
    started = threading.Event()

    async def hangs(rt, eid):
        started.set()
        await asyncio.sleep(60)

    fake_runtime(app, hangs)
    w = new_worker(app)
    w.start()
    assert started.wait(10)
    set_job(app, rid, worker_id="new-owner")  # another worker took over (its lease expired here)
    wait_for(lambda: w.running == 0, timeout=15)  # our run was abandoned at the next heartbeat
    w.stop()
    row = job_row(app, rid)
    assert row["worker_id"] == "new-owner" and row["status"] == "running"


# ------------------------------------------------------------------ lifecycle phases
def task(agent, status):
    return {"agent_type": agent, "status": status}


def test_lifecycle_phases_are_derived_from_persisted_state():
    job = {"status": "running"}
    run = {"status": "running", "control": "run"}
    assert lifecycle(None, {"status": "pending", "control": "run"}, []) == "created"
    assert lifecycle({"status": "queued"}, {"status": "pending", "control": "run"}, []) == "queued"
    assert lifecycle(job, run, [task("supervisor", "running")]) == "planning"
    assert lifecycle(job, run, [task("supervisor", "succeeded"), task("discovery", "running")]) == "running"
    assert lifecycle(job, run, [task("supervisor", "succeeded"), task("verifier", "running")]) == "reviewing"
    assert lifecycle(job, run, [task("supervisor", "succeeded"), task("verifier", "succeeded"), task("synthesis", "pending")]) == "reviewing"
    assert lifecycle(job, {"status": "running", "control": "pause"}, [task("supervisor", "succeeded")]) == "paused"
    assert lifecycle({"status": "paused"}, {"status": "paused", "control": "run"}, []) == "paused"
    assert lifecycle(job, {"status": "completed", "control": "run"}, []) == "completed"
    assert lifecycle(job, {"status": "stopped_limit", "control": "run"}, []) == "failed"
    assert lifecycle(job, {"status": "cancelled", "control": "cancel"}, []) == "cancelled"
    assert lifecycle({"status": "failed"}, {"status": "running", "control": "run"}, []) == "failed"


# ------------------------------------------------------------------ the API with a separate worker
def test_run_stays_queued_until_a_worker_claims_it_then_completes(client, app):
    h, pid, rid = make_run(client, start=True)  # embedded worker is off: nothing executes in the API process
    time.sleep(0.5)
    st = client.get(f"/runs/{rid}").json()["state"]
    assert st["lifecycle"] == "queued" and st["active"] and st["job"]["status"] == "queued" and st["status"] == "pending"
    assert client.post(f"/runs/{rid}/start", headers=h).status_code == 409  # already queued
    w = new_worker(app)
    w.start()
    try:
        wait_for(lambda: lifecycle_of(client, rid) == "completed")
    finally:
        w.stop()
    st = client.get(f"/runs/{rid}").json()["state"]
    assert st["job"]["status"] == "done" and st["job"]["attempts"] == 1 and not st["active"]
    types = [e["type"] for e in client.get(f"/runs/{rid}/events").json()]
    assert types.index("job_queued") < types.index("job_claimed") < types.index("run_started") < types.index("run_finished")
    assert client.post(f"/runs/{rid}/reports", headers=h).status_code == 201  # the report comes from what the worker persisted


def test_cancel_pause_and_resume_work_through_the_queue(client, app):
    h, pid, queued = make_run(client)
    assert client.post(f"/runs/{queued}/cancel", headers=h).json()["state"]["lifecycle"] == "cancelled"
    assert job_row(app, queued)["status"] == "cancelled"
    w = new_worker(app)
    w.start()
    try:
        time.sleep(0.5)
        assert job_row(app, queued)["status"] == "cancelled" and job_row(app, queued)["attempts"] == 0  # never executed

        _, _, other = make_run(client, h=h)
        assert client.post(f"/runs/{other}/pause", headers=h).status_code == 200  # paused before a worker gets to it
        wait_for(lambda: lifecycle_of(client, other) == "paused" and job_row(app, other)["status"] == "paused")
        assert not client.get(f"/runs/{other}").json()["state"]["active"]  # holds no lease while paused
        assert client.post(f"/runs/{other}/resume", headers=h).status_code == 200
        wait_for(lambda: lifecycle_of(client, other) == "completed")
    finally:
        w.stop()


def test_recover_endpoint_revives_a_run_whose_job_failed(client, app):
    h, _, rid = make_run(client)
    q = app.state.runtime.jobs
    job = q.claim("w")
    set_job(app, rid, attempts=3)
    q.requeue(job["id"], "w", "worker keeps dying")
    assert job_row(app, rid)["status"] == "failed"
    # the engine run itself never finished, so an operator can bring it back
    assert client.post(f"/runs/{rid}/recover", headers=h).status_code == 200
    assert job_row(app, rid)["status"] == "queued" and job_row(app, rid)["attempts"] == 0
    w = new_worker(app)
    w.start()
    try:
        wait_for(lambda: lifecycle_of(client, rid) == "completed")
    finally:
        w.stop()


def test_per_user_active_run_quota_counts_queued_jobs(client, app):
    h = sign_in(client)
    pid = project(client, h)
    app.state.settings.max_active_runs_per_user = 2
    codes = [client.post(f"/projects/{pid}/runs", json={}, headers=h).status_code for _ in range(3)]
    assert codes == [201, 201, 429]


# ------------------------------------------------------------------ acceptance: the whole workflow, then a restart
def test_end_to_end_workflow_survives_an_application_restart(tmp_path):
    s = settings_for(tmp_path, embedded_worker=True)
    app1 = create_app(s, index_provider=make_index)
    with TestClient(app1, base_url="http://testserver") as c1:
        h = sign_in(c1)  # 1-2 register (signs the user in)
        pid = c1.post("/projects", json={"title": "Perovskites", "question": QUESTION}, headers=h).json()["id"]  # 3-4
        rid = c1.post(f"/projects/{pid}/runs", json={}, headers=h).json()["id"]  # 5 persistent run, queued for a worker
        seen = set()
        def poll():
            seen.add(lifecycle_of(c1, rid))
            return "completed" in seen
        wait_for(poll)  # 6-14 worker executes the graph
        assert "completed" in seen
        tasks = c1.get(f"/runs/{rid}/tasks").json()
        assert {"supervisor", "discovery", "critic", "literature", "gap", "hypothesis", "verifier", "synthesis"} <= {t["agent_type"] for t in tasks}
        assert all(t["status"] == "succeeded" for t in tasks)
        with c1.stream("GET", f"/runs/{rid}/stream") as r:  # 15 live execution updates (replayed from the durable log)
            body = "".join(r.iter_text())
        assert "event: job_claimed" in body and "event: run_finished" in body and "event: end" in body
        rep = c1.post(f"/runs/{rid}/reports", headers=h).json()  # 13-14 report generated from persisted outputs and saved
        assert rep["content"]["status"] in ("complete", "insufficient_evidence")
        findings_before = rep["content"]["findings"]
    # 17: a brand-new application over the same files
    app2 = create_app(s, index_provider=make_index)
    with TestClient(app2, base_url="http://testserver") as c2:
        assert c2.post("/auth/login", json={"email": "alice@example.org", "password": "correct horse battery 42"}).status_code == 200
        assert c2.get(f"/runs/{rid}").json()["state"]["lifecycle"] == "completed"
        again = c2.get(f"/reports/{rep['id']}").json()  # 16 the user retrieves the completed report
        assert again["content"]["findings"] == findings_before and again["content_intact"]
        assert len(c2.get(f"/runs/{rid}/tasks").json()) == len(tasks)


def test_a_job_queued_before_a_restart_runs_after_it(tmp_path):
    s = settings_for(tmp_path)  # no worker at all in the first process
    app1 = create_app(s, index_provider=make_index)
    with TestClient(app1, base_url="http://testserver") as c1:
        h, pid, rid = make_run(c1)
        assert lifecycle_of(c1, rid) == "queued"
    s2 = settings_for(tmp_path, embedded_worker=True)
    app2 = create_app(s2, index_provider=make_index)
    with TestClient(app2, base_url="http://testserver") as c2:  # lifespan starts the embedded worker
        assert c2.post("/auth/login", json={"email": "alice@example.org", "password": "correct horse battery 42"}).status_code == 200
        wait_for(lambda: lifecycle_of(c2, rid) == "completed")
