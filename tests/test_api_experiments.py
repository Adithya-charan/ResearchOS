"""Approvals and sandboxed experiments over HTTP: approval gates (experiments and agent tasks), execution only after a human
decision bound to the exact specification, results, artifacts, reproduction and review."""
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.app import create_app
from api.config import Settings
from api.models import Experiment, Run
from research.bm25.index import BM25Index, Passage
from research.experiments import DockerSandbox, SubprocessSandbox
from research.experiments.examples import example_spec

PW = "correct horse battery 42"
QUESTION = "How much efficiency did the cells retained?"
TEXTS = [
    "Encapsulated cells retained ninety percent of their initial efficiency after two thousand hours of damp heat testing.",
    "We tested cells retained efficiency under humidity and found that cells retained seventy percent after one thousand hours.",
    "Unencapsulated cells retained efficiency poorly; cells lost half of their efficiency within one hundred hours of ageing.",
    "Cells retained eighty percent of efficiency in the third study which measured efficiency after five hundred hours.",
    "Granite quarries supply building stone across the northern valleys and harbours of the region.",
    "Medieval monasteries copied manuscripts by candlelight during the long winter months of the year.",
]


def make_index(_pid):
    idx = BM25Index()
    idx.add_passages([Passage(passage_id=f"d{i}:p001", document_id=f"d{i}", text=t, page_number=1, section="3. Results",
                              quality="ok") for i, t in enumerate(TEXTS)])
    return idx


def make_app(tmp_path, weak=True, **kw):
    s = Settings(database_url="sqlite:///:memory:", engine_db=":memory:", data_dir=tmp_path / "data", corpus_isolate=False, **kw)
    return create_app(s, index_provider=make_index, sandbox_factory=(lambda: SubprocessSandbox()) if weak else None)


@pytest.fixture()
def app(tmp_path):
    return make_app(tmp_path)


@pytest.fixture()
def client(app):
    with TestClient(app, base_url="http://testserver") as c:
        yield c


def sign_in(c, email="alice@example.org"):
    r = c.post("/auth/register", json={"email": email, "password": PW})
    assert r.status_code == 201, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def project(c, h, title="P"):
    return c.post("/projects", json={"title": title, "question": QUESTION}, headers=h).json()["id"]


def new_exp(c, h, pid, spec=None):
    r = c.post(f"/projects/{pid}/experiments", json={"spec": spec or example_spec().model_dump()}, headers=h)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def approve(c, h, eid, comment="code read, safe"):
    assert c.post(f"/experiments/{eid}/request-approval", headers=h).status_code == 200
    return c.post(f"/approvals/experiment/{eid}/approve", json={"comment": comment}, headers=h)


def wait(c, eid, h, until, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        e = c.get(f"/experiments/{eid}").json()
        if until(e):
            return e
        time.sleep(0.1)
    raise AssertionError(f"timed out; last status {e['status']}")


def finished(e):
    return e["status"] in ("succeeded", "failed", "timed_out", "error")


# ================================================================== the full, reviewed lifecycle
def test_experiment_lifecycle_approval_run_artifacts_reproduction_review(client):
    h = sign_in(client)
    pid = project(client, h)
    eid = new_exp(client, h, pid)
    e = client.get(f"/experiments/{eid}").json()
    assert e["status"] == "draft" and e["spec"]["code"] and e["manifest"] is None

    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 409  # nothing runs before approval
    assert client.post(f"/experiments/{eid}/request-approval", headers=h).json()["status"] == "pending_approval"
    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 409  # still waiting for a human

    pending = client.get("/approvals").json()
    assert [(a["kind"], a["id"]) for a in pending] == [("experiment", eid)]
    assert pending[0]["code_sha256"] == e["spec_sha256"] and pending[0]["limits"]["timeout_s"] == 30

    decision = client.post(f"/approvals/experiment/{eid}/approve", json={"comment": "read the code"}, headers=h)
    assert decision.status_code == 200 and decision.json()["experiment_status"] == "approved"
    assert client.get("/approvals").json() == []
    assert client.post(f"/approvals/experiment/{eid}/approve", headers=h).status_code == 409  # decided once

    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 202
    done = wait(client, eid, h, finished)
    assert done["status"] == "succeeded", done
    m = done["manifest"]
    assert m["approved_by"] == "alice@example.org" and m["metrics"]["samples"] == 20000
    assert done["integrity"] == {"ok": True, "problems": []}
    assert done["approval_record"]["comment"] == "read the code"
    assert any("NON-isolating" in line for line in done["limitations"])  # the weak sandbox is never hidden
    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 409  # one execution per approval

    logs = client.get(f"/experiments/{eid}/logs").json()
    assert "RESULT" in logs["stdout"] and logs["stderr"] == ""
    arts = client.get(f"/experiments/{eid}/artifacts").json()
    assert [a["path"] for a in arts] == ["estimate.txt"]
    dl = client.get(f"/experiments/{eid}/artifacts/estimate.txt")
    assert dl.status_code == 200 and float(dl.text) == pytest.approx(m["metrics"]["pi_estimate"], abs=1e-6)
    assert "attachment" in dl.headers["content-disposition"] and dl.headers["x-content-type-options"] == "nosniff"

    assert client.post(f"/experiments/{eid}/reproduce", headers=h).status_code == 202
    again = wait(client, eid, h, lambda x: "reproduced" in (x["reproduction"] or {}))
    assert again["reproduction"]["reproduced"] is True

    rv = client.post(f"/experiments/{eid}/review", json={"decision": "accepted", "notes": "metrics plausible"}, headers=h)
    assert rv.status_code == 200 and rv.json()["review"]["status"] == "accepted" and rv.json()["review"]["reproduced"] is True
    assert client.get(f"/projects/{pid}/experiments").json()[0]["status"] == "succeeded"


def test_rejection_needs_a_reason_and_the_spec_can_be_revised_and_resubmitted(client):
    h = sign_in(client)
    eid = new_exp(client, h, project(client, h))
    client.post(f"/experiments/{eid}/request-approval", headers=h)
    assert client.post(f"/approvals/experiment/{eid}/reject", json={}, headers=h).status_code == 422
    r = client.post(f"/approvals/experiment/{eid}/reject", json={"comment": "too many samples"}, headers=h)
    assert r.json()["experiment_status"] == "rejected"
    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 409

    spec = example_spec(samples=100).model_dump()
    revised = client.put(f"/experiments/{eid}/spec", json={"spec": spec}, headers=h).json()
    assert revised["status"] == "draft" and revised["approval_record"] == {}
    assert approve(client, h, eid).status_code == 200
    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 202
    assert wait(client, eid, h, finished)["manifest"]["metrics"]["samples"] == 100


def test_editing_voids_a_pending_request_and_approved_specs_are_frozen(client, app):
    h = sign_in(client)
    eid = new_exp(client, h, project(client, h))
    client.post(f"/experiments/{eid}/request-approval", headers=h)
    edited = client.put(f"/experiments/{eid}/spec", json={"spec": example_spec(samples=50).model_dump()}, headers=h).json()
    assert edited["status"] == "draft"
    assert client.post(f"/approvals/experiment/{eid}/approve", headers=h).status_code == 409  # nothing pending any more

    assert approve(client, h, eid).status_code == 200
    assert client.put(f"/experiments/{eid}/spec", json={"spec": example_spec().model_dump()}, headers=h).status_code == 409

    # even a direct database edit after approval cannot smuggle different code into the sandbox
    with app.state.sessionmaker() as db:
        e = db.get(Experiment, eid)
        e.spec = {**e.spec, "code": "print('RESULT {\"evil\": 1}')"}
        db.commit()
    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 409
    assert not (app.state.experiments.run_dir(eid)).exists()


# ================================================================== validation and sandbox policy
@pytest.mark.parametrize("patch", [{"timeout_s": 100000}, {"files": {"../x": "a"}}, {"memory_mb": 1}, {"surprise": 1}, {"code": ""}])
def test_invalid_specs_are_rejected(client, patch):
    h = sign_in(client)
    pid = project(client, h)
    spec = {**example_spec().model_dump(), **patch}
    r = client.post(f"/projects/{pid}/experiments", json={"spec": spec}, headers=h)
    assert r.status_code == 422 and "invalid experiment specification" in r.json()["detail"]


def test_code_the_sandbox_refuses_is_reported_and_leaves_no_run(client, app):
    h = sign_in(client)
    spec = {**example_spec().model_dump(), "code": "import socket\nprint('hi')"}
    eid = new_exp(client, h, project(client, h), spec)
    approve(client, h, eid)
    assert client.post(f"/experiments/{eid}/execute", headers=h).status_code == 202
    e = wait(client, eid, h, finished)
    assert e["status"] == "error" and "refused" in e["error"] and e["manifest"] is None
    assert not app.state.experiments.run_dir(eid).exists()


def test_no_isolated_sandbox_means_no_execution(tmp_path):
    if DockerSandbox().available:
        pytest.skip("Docker is available on this machine")
    app = make_app(tmp_path, weak=False)  # default factory: Docker only, weak fallback not enabled
    with TestClient(app, base_url="http://testserver") as c:
        h = sign_in(c)
        eid = new_exp(c, h, project(c, h))
        approve(c, h, eid)
        r = c.post(f"/experiments/{eid}/execute", headers=h)
        assert r.status_code == 503 and "no isolated sandbox" in r.json()["detail"]
        assert c.get(f"/experiments/{eid}").json()["status"] == "approved"  # still approved, not run


def test_model_generated_code_never_runs_without_a_container(client):
    h = sign_in(client)
    spec = {**example_spec().model_dump(), "generated_by_model": True}
    eid = new_exp(client, h, project(client, h), spec)
    approve(client, h, eid)
    r = client.post(f"/experiments/{eid}/execute", headers=h)
    assert r.status_code == 409 and "container" in r.json()["detail"]


def test_quotas(tmp_path):
    app = make_app(tmp_path, max_experiments_per_project=2, max_running_experiments_per_user=1)
    with TestClient(app, base_url="http://testserver") as c:
        h = sign_in(c)
        pid = project(c, h)
        a, b = new_exp(c, h, pid), new_exp(c, h, pid)
        assert c.post(f"/projects/{pid}/experiments", json={"spec": example_spec().model_dump()}, headers=h).status_code == 429
        approve(c, h, a)
        approve(c, h, b)
        with app.state.sessionmaker() as db:
            db.get(Experiment, a).status = "running"  # another run is executing
            db.commit()
        assert c.post(f"/experiments/{b}/execute", headers=h).status_code == 429


# ================================================================== artifacts are untrusted
def test_artifact_downloads_are_confined_hash_checked_and_never_rendered(client, app):
    h = sign_in(client)
    code = ('import json\nopen("out/page.html", "w").write("<script>alert(1)</script>")\n'
            'open("out/data.txt", "w").write("ok")\nprint("RESULT " + json.dumps({"n": 1}))\n')
    eid = new_exp(client, h, project(client, h), {**example_spec().model_dump(), "code": code})
    approve(client, h, eid)
    client.post(f"/experiments/{eid}/execute", headers=h)
    assert wait(client, eid, h, finished)["status"] == "succeeded"

    html = client.get(f"/experiments/{eid}/artifacts/page.html")
    assert html.headers["content-type"].startswith("application/octet-stream") and "attachment" in html.headers["content-disposition"]
    assert html.headers["content-security-policy"] == "sandbox"
    for bad in ("../manifest.json", "..%2Fmanifest.json", "../../spec.json", "nope.txt", "%2e%2e/main.py", "main.py"):
        assert client.get(f"/experiments/{eid}/artifacts/{bad}").status_code == 404, bad

    (app.state.experiments.run_dir(eid) / "out" / "data.txt").write_text("tampered")
    assert client.get(f"/experiments/{eid}/artifacts/data.txt").status_code == 409
    assert client.get(f"/experiments/{eid}").json()["integrity"]["ok"] is False


# ================================================================== authorization
def test_users_cannot_see_decide_run_or_download_each_others_experiments(client, app):
    ha = sign_in(client)
    pa = project(client, ha)
    eid = new_exp(client, ha, pa)
    approve(client, ha, eid)
    client.post(f"/experiments/{eid}/execute", headers=ha)
    wait(client, eid, ha, finished)

    bob = TestClient(app, base_url="http://testserver")
    hb = sign_in(bob, "bob@example.org")
    pb = project(bob, hb, "B")
    assert bob.get("/approvals", params={"status": "all"}).json() == []
    for method, url in (("get", f"/experiments/{eid}"), ("get", f"/experiments/{eid}/logs"), ("get", f"/experiments/{eid}/artifacts"),
                        ("get", f"/experiments/{eid}/artifacts/estimate.txt"), ("get", f"/projects/{pa}/experiments")):
        assert getattr(bob, method)(url).status_code == 404, url
    for url in (f"/experiments/{eid}/request-approval", f"/experiments/{eid}/execute", f"/experiments/{eid}/reproduce",
                f"/approvals/experiment/{eid}/approve", f"/approvals/experiment/{eid}/reject"):
        assert bob.post(url, headers=hb, json={"comment": "x"}).status_code == 404, url
    assert bob.post(f"/experiments/{eid}/review", headers=hb, json={"decision": "accepted"}).status_code == 404
    assert bob.put(f"/experiments/{eid}/spec", headers=hb, json={"spec": example_spec().model_dump()}).status_code == 404
    assert bob.post(f"/projects/{pa}/experiments", headers=hb, json={"spec": example_spec().model_dump()}).status_code == 404
    assert bob.get(f"/projects/{pb}/experiments").json() == []
    assert client.get(f"/experiments/{eid}").json()["review"]["status"] == "pending"


def test_decisions_need_csrf_and_a_session(client, app):
    h = sign_in(client)
    eid = new_exp(client, h, project(client, h))
    client.post(f"/experiments/{eid}/request-approval", headers=h)
    assert client.post(f"/approvals/experiment/{eid}/approve").status_code == 403
    assert TestClient(app, base_url="http://testserver").get("/approvals").status_code == 401
    assert client.get(f"/experiments/{eid}").json()["status"] == "pending_approval"


def test_review_requires_a_finished_experiment(client):
    h = sign_in(client)
    eid = new_exp(client, h, project(client, h))
    assert client.post(f"/experiments/{eid}/review", json={"decision": "accepted"}, headers=h).status_code == 409
    assert client.post(f"/experiments/{eid}/review", json={"decision": "maybe"}, headers=h).status_code == 422
    assert client.post(f"/experiments/{eid}/reproduce", headers=h).status_code == 409


# ================================================================== agent-task approvals through the real scheduler
def _gated_run(client, app, h):
    pid = project(client, h)
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
    with app.state.sessionmaker() as db:
        engine_run = db.get(Run, rid).engine_run_id
    store = app.state.runtime.store
    store.update_task(store.list_tasks(engine_run)[0]["id"], requires_approval=1)  # the scheduler must now ask a human
    return rid


def wait_run(c, rid, h, pred, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        st = c.get(f"/runs/{rid}", headers=h).json()["state"]
        if pred(st):
            return st
        time.sleep(0.1)
    raise AssertionError(f"run state never matched: {st}")


def test_a_run_pauses_for_task_approval_and_continues_after_it(client, app):
    h = sign_in(client)
    rid = _gated_run(client, app, h)
    client.post(f"/runs/{rid}/start", headers=h)
    wait_run(client, rid, h, lambda s: s["status"] == "paused" and not s["active"])
    pending = client.get("/approvals").json()
    assert len(pending) == 1 and pending[0]["kind"] == "task" and pending[0]["run_id"] == rid and pending[0]["agent_type"] == "supervisor"
    out = client.post(f"/approvals/task/{pending[0]['id']}/approve", json={"comment": "go ahead"}, headers=h)
    assert out.status_code == 200 and out.json()["decision"] == "approved"
    assert wait_run(client, rid, h, lambda s: s["status"] == "completed" and not s["active"])["status"] == "completed"
    done = client.get("/approvals", params={"status": "all"}).json()
    assert done[0]["decision"] == "approved" and done[0]["reviewer"] == "alice@example.org" and done[0]["comment"] == "go ahead"
    assert client.post(f"/approvals/task/{pending[0]['id']}/approve", headers=h).status_code == 409


def test_rejecting_a_task_approval_records_the_reason_and_stops_that_work(client, app):
    h = sign_in(client)
    rid = _gated_run(client, app, h)
    client.post(f"/runs/{rid}/start", headers=h)
    wait_run(client, rid, h, lambda s: s["status"] == "paused" and not s["active"])
    aid = client.get("/approvals").json()[0]["id"]
    assert client.post(f"/approvals/task/{aid}/reject", json={}, headers=h).status_code == 422
    assert client.post(f"/approvals/task/{aid}/reject", json={"comment": "not now"}, headers=h).status_code == 200
    st = wait_run(client, rid, h, lambda s: s["status"] in ("completed", "failed", "cancelled") and not s["active"])
    assert st["task_counts"].get("succeeded", 0) == 0  # the rejected task never ran
    assert client.get("/approvals", params={"status": "all"}).json()[0]["comment"] == "not now"


def test_task_approvals_are_private_to_the_runs_owner(client, app):
    ha = sign_in(client)
    rid = _gated_run(client, app, ha)
    client.post(f"/runs/{rid}/start", headers=ha)
    wait_run(client, rid, ha, lambda s: s["status"] == "paused" and not s["active"])
    aid = client.get("/approvals").json()[0]["id"]
    bob = TestClient(app, base_url="http://testserver")
    hb = sign_in(bob, "bob@example.org")
    assert bob.get("/approvals", params={"status": "all"}).json() == []
    assert bob.post(f"/approvals/task/{aid}/approve", headers=hb, json={}).status_code == 404
    assert bob.post(f"/approvals/task/{aid}/reject", headers=hb, json={"comment": "x"}).status_code == 404
    assert bob.post("/approvals/bogus/x/approve", headers=hb, json={}).status_code == 404
    assert client.get(f"/runs/{rid}", headers=ha).json()["state"]["status"] == "paused"  # untouched
