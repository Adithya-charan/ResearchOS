"""The HTTP API: accounts, sessions, CSRF, per-user isolation, projects, and research runs over the real agent engine."""
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.app import create_app
from api.config import Settings
from api.db import utcnow
from api.models import AuditLog, Run, SessionRow, User
from api.runtime import NoCorpus
from research.bm25.index import BM25Index, Passage

PW = "correct horse battery 42"
TEXTS = [
    "Encapsulated cells retained ninety percent of their initial efficiency after two thousand hours of damp heat testing.",
    "We tested cells retained efficiency under humidity and found that cells retained seventy percent after one thousand hours.",
    "Unencapsulated cells retained efficiency poorly; cells lost half of their efficiency within one hundred hours of ageing.",
    "Cells retained eighty percent of efficiency in the third study which measured efficiency after five hundred hours.",
    "Granite quarries supply building stone across the northern valleys and harbours of the region.",
    "Medieval monasteries copied manuscripts by candlelight during the long winter months of the year.",
    "Migrating geese follow river valleys southward every autumn before the first frost arrives.",
]
QUESTION = "How much efficiency did the cells retained?"


def make_index(_project_id: str) -> BM25Index:
    idx = BM25Index()
    idx.add_passages([Passage(passage_id=f"d{i}:p001", document_id=f"d{i}", text=t, page_number=1, section="3. Results",
                              quality="ok") for i, t in enumerate(TEXTS)])
    return idx


def settings(**kw) -> Settings:
    return Settings(database_url="sqlite:///:memory:", engine_db=":memory:", login_attempts=5, **kw)


@pytest.fixture()
def app():
    return create_app(settings(), index_provider=make_index)


@pytest.fixture()
def client(app):
    with TestClient(app, base_url="http://testserver") as c:
        yield c


def other_client(app):
    return TestClient(app, base_url="http://testserver")  # separate cookie jar = a different browser


def register(c, email="alice@example.org", password=PW):
    r = c.post("/auth/register", json={"email": email, "password": password})
    return r


def signed_in(c, email="alice@example.org"):
    r = register(c, email)
    assert r.status_code == 201, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def hold_running(app, run_id):
    """Make a run look like it is executing: a running job whose lease is held by a (fake) live worker."""
    from datetime import timedelta
    from api.db import utcnow
    from api.models import ResearchJob
    with app.state.sessionmaker() as db:
        r = db.get(Run, run_id)
        db.add(ResearchJob(run_id=run_id, engine_run_id=r.engine_run_id, owner_id=r.owner_id, status="running", worker_id="test-holder",
                           attempts=1, lease_expires_at=utcnow() + timedelta(hours=1)))
        db.commit()


def release_job(app, run_id):
    from api.models import ResearchJob
    with app.state.sessionmaker() as db:
        db.query(ResearchJob).filter(ResearchJob.run_id == run_id).delete()
        db.commit()


def wait_done(c, run_id, h, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        st = c.get(f"/runs/{run_id}", headers=h).json()["state"]
        if st["status"] in ("completed", "failed", "cancelled") and not st["active"]:
            return st
        time.sleep(0.1)
    raise AssertionError("run did not finish")


# ================================================================== authentication
def test_register_login_logout_and_me(client):
    r = register(client)
    assert r.status_code == 201 and r.json()["user"]["email"] == "alice@example.org"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie
    assert client.get("/auth/me").json()["user"]["email"] == "alice@example.org"

    h = {"X-CSRF-Token": client.get("/auth/me").json()["csrf_token"]}
    assert client.post("/auth/logout", headers=h).status_code == 204
    assert client.get("/auth/me").status_code == 401

    login = client.post("/auth/login", json={"email": "ALICE@example.org", "password": PW})  # email is case-insensitive
    assert login.status_code == 200 and client.get("/auth/me").status_code == 200


def test_passwords_are_argon2id_and_session_tokens_are_stored_hashed(client, app):
    register(client)
    with app.state.sessionmaker() as db:
        user = db.scalar(select(User))
        assert user.password_hash.startswith("$argon2id$") and PW not in user.password_hash
        token = client.cookies.get("rs_session")
        sessions = db.scalars(select(SessionRow)).all()
        assert len(sessions) == 1 and sessions[0].id != token and len(sessions[0].id) == 64


@pytest.mark.parametrize("password,why", [("short1!", "short"), ("passwordpassword1", "common"), ("alice-example-1234", "email name"),
                                          ("aaaaaaaaaaaaaaaaaaaa", "few distinct"), ("onlyletterslongenough", "no digit")])
def test_weak_passwords_are_rejected(client, password, why):
    r = register(client, password=password)
    assert r.status_code == 422, why
    assert PW not in r.text


def test_invalid_email_and_unknown_fields_are_rejected(client):
    assert register(client, email="not-an-email").status_code == 422
    assert client.post("/auth/register", json={"email": "a@b.org", "password": PW, "is_admin": True}).status_code == 422


def test_duplicate_email_conflicts(client, app):
    assert register(client).status_code == 201
    assert register(other_client(app)).status_code == 409


def test_wrong_password_and_unknown_user_are_indistinguishable(client, app):
    register(client)
    c2 = other_client(app)
    a = c2.post("/auth/login", json={"email": "alice@example.org", "password": "wrong password 99"})
    b = c2.post("/auth/login", json={"email": "nobody@example.org", "password": "wrong password 99"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


def test_repeated_failed_logins_are_rate_limited(client, app):
    register(client)
    c2 = other_client(app)
    codes = [c2.post("/auth/login", json={"email": "alice@example.org", "password": f"bad password {i}"}).status_code for i in range(7)]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    assert c2.post("/auth/login", json={"email": "alice@example.org", "password": PW}).status_code == 429  # even the right one


def test_expired_session_is_refused_and_removed(client, app):
    register(client)
    with app.state.sessionmaker() as db:
        s = db.scalar(select(SessionRow))
        s.expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
    assert client.get("/auth/me").status_code == 401
    with app.state.sessionmaker() as db:
        assert db.scalar(select(SessionRow)) is None


def test_protected_routes_need_a_session(client):
    for method, url in (("get", "/projects"), ("post", "/projects"), ("get", "/runs/run_x"), ("get", "/auth/me")):
        assert getattr(client, method)(url).status_code == 401


# ================================================================== CSRF and origin
def test_state_changes_need_the_csrf_token(client):
    h = signed_in(client)
    body = {"title": "T", "question": "What is this?"}
    assert client.post("/projects", json=body).status_code == 403
    assert client.post("/projects", json=body, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert client.post("/projects", json=body, headers=h).status_code == 201
    assert client.get("/projects").status_code == 200  # reads need no token


def test_a_csrf_token_from_another_session_is_rejected(client, app):
    signed_in(client)
    other = other_client(app)
    h_other = signed_in(other, "bob@example.org")
    assert client.post("/projects", json={"title": "T", "question": "What is this?"}, headers=h_other).status_code == 403


def test_cross_site_origin_is_refused(client):
    h = signed_in(client)
    r = client.post("/projects", json={"title": "T", "question": "What is this?"}, headers={**h, "Origin": "http://evil.example"})
    assert r.status_code == 403
    ok = client.post("/projects", json={"title": "T", "question": "What is this?"}, headers={**h, "Origin": "http://localhost:3000"})
    assert ok.status_code == 201


def test_unhandled_errors_return_a_safe_body(app):
    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internals /etc/passwd")

    with TestClient(app, base_url="http://testserver", raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 500 and "secret" not in r.text and r.json()["request_id"]
    assert r.headers["x-content-type-options"] == "nosniff"


# ================================================================== projects and isolation
def test_project_crud_and_validation(client):
    h = signed_in(client)
    r = client.post("/projects", json={"title": "Perovskites", "question": "How stable are perovskite cells?"}, headers=h)
    pid = r.json()["id"]
    assert client.get(f"/projects/{pid}").json()["title"] == "Perovskites"
    assert client.patch(f"/projects/{pid}", json={"title": "Renamed"}, headers=h).json()["title"] == "Renamed"
    assert [p["id"] for p in client.get("/projects").json()] == [pid]
    assert client.post("/projects", json={"title": "", "question": "ok?"}, headers=h).status_code == 422
    assert client.post("/projects", json={"title": "x" * 500, "question": "ok?"}, headers=h).status_code == 422
    assert client.delete(f"/projects/{pid}", headers=h).status_code == 204
    assert client.get(f"/projects/{pid}").status_code == 404


def test_project_quota(app):
    a = create_app(settings(max_projects_per_user=2), index_provider=make_index)
    with TestClient(a, base_url="http://testserver") as c:
        h = signed_in(c)
        codes = [c.post("/projects", json={"title": f"P{i}", "question": "What is this?"}, headers=h).status_code for i in range(3)]
    assert codes == [201, 201, 429]


def test_users_cannot_see_or_change_each_others_projects_or_runs(client, app):
    ha = signed_in(client)
    pid = client.post("/projects", json={"title": "A's", "question": QUESTION}, headers=ha).json()["id"]
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=ha).json()["id"]

    bob = other_client(app)
    hb = signed_in(bob, "bob@example.org")
    assert bob.get("/projects").json() == []
    assert bob.get(f"/projects/{pid}").status_code == 404
    assert bob.patch(f"/projects/{pid}", json={"title": "pwned"}, headers=hb).status_code == 404
    assert bob.delete(f"/projects/{pid}", headers=hb).status_code == 404
    assert bob.post(f"/projects/{pid}/runs", json={}, headers=hb).status_code == 404
    assert bob.get(f"/projects/{pid}/runs").status_code == 404
    for url in (f"/runs/{rid}", f"/runs/{rid}/tasks", f"/runs/{rid}/events", f"/runs/{rid}/report"):
        assert bob.get(url).status_code == 404, url
    for action in ("start", "pause", "resume", "cancel", "recover"):
        assert bob.post(f"/runs/{rid}/{action}", headers=hb).status_code == 404, action
    assert client.get(f"/projects/{pid}").json()["title"] == "A's"  # untouched


def test_a_task_id_from_another_run_is_not_readable_through_my_run(client, app):
    ha = signed_in(client)
    pa = client.post("/projects", json={"title": "A", "question": QUESTION}, headers=ha).json()["id"]
    ra = client.post(f"/projects/{pa}/runs", json={"start": False}, headers=ha).json()["id"]
    task_id = client.get(f"/runs/{ra}/tasks").json()[0]["id"]

    bob = other_client(app)
    hb = signed_in(bob, "bob@example.org")
    pb = bob.post("/projects", json={"title": "B", "question": QUESTION}, headers=hb).json()["id"]
    rb = bob.post(f"/projects/{pb}/runs", json={"start": False}, headers=hb).json()["id"]
    assert bob.get(f"/runs/{rb}/tasks/{task_id}").status_code == 404
    assert client.get(f"/runs/{ra}/tasks/{task_id}").status_code == 200


# ================================================================== research runs over the real engine
def test_a_run_executes_through_the_scheduler_and_produces_a_linked_report(client):
    h = signed_in(client)
    pid = client.post("/projects", json={"title": "Perovskites", "question": QUESTION}, headers=h).json()["id"]
    run = client.post(f"/projects/{pid}/runs", json={}, headers=h)
    assert run.status_code == 201
    rid = run.json()["id"]
    st = wait_done(client, rid, h)
    assert st["status"] == "completed" and st["task_counts"]["succeeded"] >= 8

    tasks = client.get(f"/runs/{rid}/tasks").json()
    assert {"supervisor", "discovery", "critic", "literature", "gap", "hypothesis", "verifier", "synthesis"} <= {t["agent_type"] for t in tasks}
    assert all(t["status"] == "succeeded" and t["depends_on"] is not None for t in tasks)
    full = client.get(f"/runs/{rid}/tasks/{tasks[0]['id']}").json()
    assert "output" in full and "input" in full

    events = client.get(f"/runs/{rid}/events").json()
    assert {"run_created", "task_succeeded", "run_finished"} <= {e["type"] for e in events}
    tail = client.get(f"/runs/{rid}/events", params={"after": events[3]["id"]}).json()
    assert tail and tail[0]["id"] > events[3]["id"]

    report = client.get(f"/runs/{rid}/report").json()
    assert report["human_review"]["status"] == "pending" and report["limitations"]
    findings = [f for fs in report["findings"].values() for f in fs]
    assert findings and all(f["citation"]["passage_id"].startswith(f["citation"]["document_id"]) for f in findings)
    assert client.get(f"/projects/{pid}/runs").json()[0]["id"] == rid


def test_run_can_be_created_unstarted_then_cancelled(client):
    h = signed_in(client)
    pid = client.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
    assert client.get(f"/runs/{rid}").json()["state"]["status"] == "pending"
    assert client.post(f"/runs/{rid}/cancel", headers=h).json()["state"]["status"] == "cancelled"
    assert client.post(f"/runs/{rid}/start", headers=h).status_code == 409
    assert client.post(f"/runs/{rid}/cancel", headers=h).status_code == 409  # already finished
    assert client.get(f"/runs/{rid}/report").status_code == 404  # no report is fabricated


def test_pause_then_resume_continues_the_same_run(client):
    h = signed_in(client)
    pid = client.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
    assert client.post(f"/runs/{rid}/pause", headers=h).json()["state"]["control"] == "pause"
    assert client.post(f"/runs/{rid}/resume", headers=h).status_code == 200
    st = wait_done(client, rid, h)
    assert st["status"] == "completed"


def test_recover_requeues_interrupted_tasks_and_finishes_the_run(client, app):
    h = signed_in(client)
    pid = client.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
    with app.state.sessionmaker() as db:
        engine_run = db.get(Run, rid).engine_run_id
    store = app.state.runtime.store
    first = store.list_tasks(engine_run)[0]
    store.update_task(first["id"], status="running")  # simulate a worker that died mid-task
    assert client.post(f"/runs/{rid}/recover", headers=h).status_code == 200
    assert wait_done(client, rid, h)["status"] == "completed"
    assert any(e["type"] == "task_recovered" for e in client.get(f"/runs/{rid}/events").json())


def test_requested_limits_cannot_exceed_server_ceilings(app):
    a = create_app(settings(run_max_tasks=12, run_max_runtime_s=30.0), index_provider=make_index)
    with TestClient(a, base_url="http://testserver") as c:
        h = signed_in(c)
        pid = c.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
        r = c.post(f"/projects/{pid}/runs", json={"start": False, "limits": {"max_tasks": 9999, "max_runtime_s": 99999}}, headers=h)
        with a.state.sessionmaker() as db:
            limits = db.scalar(select(Run)).limits
        assert r.status_code == 201 and limits["max_tasks"] == 12 and limits["max_runtime_s"] == 30.0
        assert c.post(f"/projects/{pid}/runs", json={"limits": {"bogus": 1}}, headers=h).status_code == 422


def test_active_run_quota(app):
    a = create_app(settings(max_active_runs_per_user=1), index_provider=make_index)
    release = []

    class Slow:
        pass

    with TestClient(a, base_url="http://testserver") as c:
        h = signed_in(c)
        pid = c.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
        r1 = c.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
        r2 = c.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
        rt = a.state.runtime
        with a.state.sessionmaker() as db:
            e1 = db.get(Run, r1).engine_run_id
        hold_running(a, r1)  # r1 counts as executing (a job held by a live worker)
        assert c.post(f"/runs/{r2}/start", headers=h).status_code == 429
        release_job(a, r1)
        assert c.post(f"/runs/{r2}/start", headers=h).status_code == 200
        wait_done(c, r2, h)


def test_run_without_a_corpus_is_refused_not_faked(tmp_path):
    a = create_app(settings(default_corpus_dir=tmp_path / "nothing"))  # no index_provider: uses the (missing) corpus dir
    with TestClient(a, base_url="http://testserver") as c:
        h = signed_in(c)
        pid = c.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
        r = c.post(f"/projects/{pid}/runs", json={}, headers=h)
        assert r.status_code == 409 and "no searchable documents" in r.json()["detail"]
        assert c.get(f"/projects/{pid}/runs").json() == []


def test_deleting_a_project_with_an_active_run_is_refused(client, app):
    h = signed_in(client)
    pid = client.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
    rid = client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
    hold_running(app, rid)
    assert client.delete(f"/projects/{pid}", headers=h).status_code == 409
    release_job(app, rid)
    assert client.delete(f"/projects/{pid}", headers=h).status_code == 204


# ================================================================== audit trail
def test_actions_are_audited_without_secrets(client, app):
    h = signed_in(client)
    pid = client.post("/projects", json={"title": "P", "question": QUESTION}, headers=h).json()["id"]
    client.post(f"/projects/{pid}/runs", json={"start": False}, headers=h)
    other_client(app).post("/auth/login", json={"email": "alice@example.org", "password": "definitely wrong 77"})
    with app.state.sessionmaker() as db:
        rows = db.scalars(select(AuditLog)).all()
        actions = {r.action for r in rows}
        assert {"register", "project_create", "run_create", "login_failed"} <= actions
        blob = " ".join(str(r.detail) + r.target for r in rows)
        assert PW not in blob and "definitely wrong" not in blob
