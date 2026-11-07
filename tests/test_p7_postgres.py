"""P7: PostgreSQL integration tests. Every test talks to a real PostgreSQL server over TCP (see tests/pg_support.py) - nothing here
mocks the database or substitutes SQLite. Each test gets its own schemas or databases."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from agent_engine.state import dburl, migrate as engine_migrate
from agent_engine.state.migrate import SchemaMismatch
from agent_engine.state.store import Store, StoreUnavailable
from agent_engine.state.transitions import InvalidTransition
from api import migrate as api_migrate
from api.config import Settings
from api.db import Base, make_engine, make_sessionmaker, utcnow
from api.jobs import JobQueue
from api.models import (Claim, ClaimEvidence, ClaimVerification, NoveltyAssessmentRow, Project, Report, ResearchJob, Run, User)
from tests import pg_support as pg

pytestmark = [pytest.mark.postgres,
              pytest.mark.skipif(not pg.available(), reason="needs psycopg and a PostgreSQL server (pip install pgserver psycopg[binary])")]
ROOT = Path(__file__).resolve().parents[1]


def run_py(code: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code), *args], capture_output=True, text=True, timeout=timeout, env=env)


@pytest.fixture()
def engine_url():
    url = pg.new_schema_url(prefix="engine")
    yield url
    pg.drop_schema(url)


@pytest.fixture()
def api_url():
    url = pg.new_schema_url(prefix="api")
    yield url
    pg.drop_schema(url)


def table_names(url: str) -> set[str]:
    base, schema = dburl.split_schema(url)
    with pg.connect(base) as c:
        rows = c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=%s", (schema or "public",)).fetchall()
    return {r[0] for r in rows}


# ================================================================================================ the server itself
def test_the_server_is_real_postgresql():
    version = pg.server_version()
    assert int(version.split(".")[0]) >= 14
    with pg.connect() as c:
        assert "PostgreSQL" in c.execute("SELECT version()").fetchone()[0]


# ================================================================================================ URL handling
def test_schema_parameter_round_trip_and_validation():
    url = "postgresql://u:secret@h:5432/db?sslmode=require&schema=engine"
    base, schema = dburl.split_schema(url)
    assert schema == "engine" and "schema" not in base and "sslmode=require" in base
    assert dburl.split_schema(dburl.with_schema(base, "api"))[1] == "api"
    assert dburl.with_schema(url, None) == base
    assert dburl.sqlalchemy_url(url)[0].startswith("postgresql+psycopg://")
    assert "secret" not in dburl.redact(url)
    with pytest.raises(ValueError):
        dburl.with_schema(base, 'x"; DROP SCHEMA public; --')
    assert dburl.is_postgres("postgres://x/y") and not dburl.is_postgres("sqlite:///x.db") and not dburl.is_postgres(":memory:")


def test_settings_defaults_use_two_schemas_in_one_database(monkeypatch):
    for k in ("RESEARCHOS_DATABASE_SCHEMA", "RESEARCHOS_ENGINE_DATABASE_URL", "RESEARCHOS_ENGINE_DATABASE_SCHEMA", "RESEARCHOS_ENGINE_DB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RESEARCHOS_DATABASE_URL", "postgresql://u:p@db:5432/researchos")
    s = Settings.from_env()
    assert dburl.split_schema(s.database_url)[1] == "api"
    assert dburl.split_schema(str(s.engine_db))[1] == "engine"
    assert dburl.split_schema(s.database_url)[0] == dburl.split_schema(str(s.engine_db))[0]
    monkeypatch.setenv("RESEARCHOS_ENGINE_DATABASE_URL", "postgresql://u:p@other:5432/engine_db")
    s = Settings.from_env()
    assert "other" in str(s.engine_db)
    monkeypatch.setenv("RESEARCHOS_ENGINE_DATABASE_URL", "postgresql://u:p@db:5432/researchos")
    monkeypatch.setenv("RESEARCHOS_ENGINE_DATABASE_SCHEMA", "api")
    with pytest.raises(ValueError):  # the two layers define tables with the same names: they may not share a namespace
        Settings.from_env()


def test_sqlite_remains_the_default(monkeypatch):
    monkeypatch.delenv("RESEARCHOS_DATABASE_URL", raising=False)
    assert Settings.from_env().database_url.startswith("sqlite:///")


# ================================================================================================ engine migrations
def test_engine_fresh_install_creates_schema_at_head_and_is_idempotent(engine_url):
    assert engine_migrate.upgrade(engine_url) == engine_migrate.head_revision()
    assert engine_migrate.upgrade(engine_url) == engine_migrate.head_revision()
    assert {"projects", "runs", "tasks", "events", "approvals", "memory", "task_executions", "alembic_version"} <= table_names(engine_url)
    assert engine_migrate.current_revision(engine_url) == engine_migrate.head_revision()


def test_api_fresh_install_matches_the_models_exactly(api_url):
    eng = make_engine(api_url)
    try:
        assert api_migrate.upgrade(eng) == api_migrate.head_revision()
        assert api_migrate.upgrade(eng) == api_migrate.head_revision()
        assert api_migrate.schema_diff(eng) == []  # the migrated PostgreSQL schema and api/models.py agree
        assert set(Base.metadata.tables) <= table_names(api_url)
    finally:
        eng.dispose()


def test_api_and_engine_share_one_database_in_separate_schemas():
    db = pg.new_database()
    api, eng_url = dburl.with_schema(db, "api"), dburl.with_schema(db, "engine")
    e = make_engine(api)
    try:
        api_migrate.upgrade(e)
        store = Store(eng_url)
        store.create_project("t", "q")
        assert "runs" in table_names(api) and "runs" in table_names(eng_url)  # same table name, two namespaces
        assert "claims" in table_names(api) and "claims" not in table_names(eng_url)
        assert "tasks" in table_names(eng_url) and "tasks" not in table_names(api)
        with e.connect() as c:
            assert c.execute(select(func.count()).select_from(Project)).scalar() == 0  # the engine's project is not the API's
        store.close()
    finally:
        e.dispose()


def test_engine_refuses_an_unversioned_schema_and_leaves_it_untouched(engine_url):
    base, schema = dburl.split_schema(engine_url)
    with pg.connect(base) as c:
        c.execute(f'CREATE SCHEMA "{schema}"')
        c.execute(f'CREATE TABLE "{schema}".runs (id TEXT PRIMARY KEY, precious TEXT)')
        c.execute(f"INSERT INTO \"{schema}\".runs VALUES ('r1', 'keep me')")
    with pytest.raises(SchemaMismatch):
        engine_migrate.upgrade(engine_url)
    with pytest.raises(SchemaMismatch):
        Store(engine_url)
    with pg.connect(base) as c:
        assert c.execute(f'SELECT precious FROM "{schema}".runs').fetchall() == [("keep me",)]
        assert "alembic_version" not in table_names(engine_url)


def test_api_refuses_an_unversioned_foreign_schema_and_leaves_it_untouched(api_url):
    base, schema = dburl.split_schema(api_url)
    with pg.connect(base) as c:
        c.execute(f'CREATE SCHEMA "{schema}"')
        c.execute(f'CREATE TABLE "{schema}".users (id TEXT PRIMARY KEY, email TEXT, secret_notes TEXT)')
        c.execute(f"INSERT INTO \"{schema}\".users VALUES ('u1', 'a@b.c', 'keep me')")
    eng = make_engine(api_url)
    try:
        with pytest.raises(api_migrate.SchemaMismatch):
            api_migrate.upgrade(eng)
    finally:
        eng.dispose()
    with pg.connect(base) as c:
        assert c.execute(f'SELECT secret_notes FROM "{schema}".users').fetchall() == [("keep me",)]
        assert "alembic_version" not in table_names(api_url) and "claims" not in table_names(api_url)


def test_databases_from_newer_code_are_refused_not_downgraded(engine_url, api_url):
    engine_migrate.upgrade(engine_url)
    store = Store(engine_url)
    pid = store.create_project("t", "q")
    store.close()
    base, schema = dburl.split_schema(engine_url)
    with pg.connect(base) as c:
        c.execute(f'UPDATE "{schema}".alembic_version SET version_num=\'9999\'')
    with pytest.raises(SchemaMismatch):
        engine_migrate.upgrade(engine_url)
    with pg.connect(base) as c:  # nothing was dropped or reset
        assert c.execute(f'SELECT COUNT(*) FROM "{schema}".projects WHERE id=%s', (pid,)).fetchone()[0] == 1
        assert c.execute(f'SELECT version_num FROM "{schema}".alembic_version').fetchone()[0] == "9999"
    eng = make_engine(api_url)
    try:
        api_migrate.upgrade(eng)
        abase, aschema = dburl.split_schema(api_url)
        with pg.connect(abase) as c:
            c.execute(f'UPDATE "{aschema}".alembic_version SET version_num=\'9999\'')
        with pytest.raises(api_migrate.SchemaMismatch):
            api_migrate.upgrade(eng)
    finally:
        eng.dispose()


def test_a_failing_migration_rolls_back_completely(engine_url, tmp_path, monkeypatch):
    """PostgreSQL DDL is transactional: if revision 2 fails, revision 1's tables must not be left behind half-applied."""
    tree = tmp_path / "migrations"
    (tree / "versions").mkdir(parents=True)
    src = engine_migrate.MIGRATIONS
    (tree / "env.py").write_text((src / "env.py").read_text(encoding="utf-8"), encoding="utf-8")
    (tree / "script.py.mako").write_text((src / "script.py.mako").read_text(encoding="utf-8"), encoding="utf-8")
    (tree / "versions" / "0001_ok.py").write_text(textwrap.dedent('''
        from alembic import op
        revision, down_revision, branch_labels, depends_on = "0001", None, None, None
        def upgrade(): op.execute("CREATE TABLE half_applied (id INTEGER)")
        def downgrade(): op.execute("DROP TABLE half_applied")
    '''), encoding="utf-8")
    (tree / "versions" / "0002_boom.py").write_text(textwrap.dedent('''
        from alembic import op
        revision, down_revision, branch_labels, depends_on = "0002", "0001", None, None
        def upgrade(): op.execute("SELECT 1/0")
        def downgrade(): pass
    '''), encoding="utf-8")
    monkeypatch.setattr(engine_migrate, "MIGRATIONS", tree)
    with pytest.raises(Exception, match="division by zero"):
        engine_migrate.upgrade(engine_url)
    assert "half_applied" not in table_names(engine_url) and "alembic_version" not in table_names(engine_url)


_START_ENGINE = """
    import sys
    from agent_engine.state.store import Store
    s = Store(sys.argv[1]); print("ok", s.create_project("p", "q")[:4]); s.close()
"""
_START_API = """
    import sys
    from api.db import make_engine
    from api import migrate
    e = make_engine(sys.argv[1]); print("ok", migrate.upgrade(e)); e.dispose()
"""


def test_concurrent_startup_of_the_engine_database_from_many_processes(engine_url):
    procs = [subprocess.Popen([sys.executable, "-c", textwrap.dedent(_START_ENGINE), engine_url], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)}) for _ in range(5)]
    results = [p.communicate(timeout=180) + (p.returncode,) for p in procs]
    assert all(rc == 0 and out.startswith("ok") for out, err, rc in results), results
    assert engine_migrate.current_revision(engine_url) == engine_migrate.head_revision()
    base, schema = dburl.split_schema(engine_url)
    with pg.connect(base) as c:
        assert c.execute(f'SELECT COUNT(*) FROM "{schema}".alembic_version').fetchone()[0] == 1
        assert c.execute(f'SELECT COUNT(*) FROM "{schema}".projects').fetchone()[0] == 5  # every process really used the schema


def test_concurrent_startup_of_the_api_database_from_many_processes(api_url):
    procs = [subprocess.Popen([sys.executable, "-c", textwrap.dedent(_START_API), api_url], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)}) for _ in range(5)]
    results = [p.communicate(timeout=180) + (p.returncode,) for p in procs]
    assert all(rc == 0 and out.startswith("ok") for out, err, rc in results), results
    eng = make_engine(api_url)
    try:
        assert api_migrate.current_revision(eng) == api_migrate.head_revision() and api_migrate.schema_diff(eng) == []
    finally:
        eng.dispose()


def test_concurrent_startup_from_threads_in_one_process(engine_url):
    errors, stores = [], []

    def start():
        try:
            stores.append(Store(engine_url))
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(6)]
    [t.start() for t in threads]
    [t.join(120) for t in threads]
    assert not errors and len(stores) == 6
    [s.close() for s in stores]


# ================================================================================================ engine store semantics
def test_store_round_trip_and_restart_persistence(engine_url):
    s = Store(engine_url)
    pid = s.create_project("Title", "Question?")
    rid = s.create_run(pid, "Question?", {"max_tasks": 9}, {"workflow": "full"})
    s.add_tasks(rid, [{"id": "t1", "agent_type": "discovery", "title": "find", "input": {"q": "ünïcode ✓"}, "depends_on": []},
                      {"id": "t2", "agent_type": "critic", "title": "judge", "depends_on": ["t1"]}])
    s.update_task("t1", status="running")
    s.update_task("t1", status="succeeded", output={"hits": [1, 2, 3]})
    s.add_event(rid, "note", {"x": 1}, task_id="t1")
    s.memory_put(pid, "scope/a", {"v": 1}, rid, "t1")
    aid = s.add_approval(rid, "t2", "run experiment", {"hash": "abc"})
    s.close()

    s = Store(engine_url)  # a new process would see exactly this
    assert s.get_project(pid)["title"] == "Title"
    assert s.get_run(rid)["limits"] == {"max_tasks": 9} and s.get_run(rid)["config"] == {"workflow": "full"}
    tasks = s.list_tasks(rid)
    assert [t["id"] for t in tasks] == ["t1", "t2"] and tasks[0]["input"] == {"q": "ünïcode ✓"} and tasks[0]["output"] == {"hits": [1, 2, 3]}
    assert tasks[1]["depends_on"] == ["t1"] and s.task_counts(rid) == {"succeeded": 1, "pending": 1}
    assert [e["type"] for e in s.list_events(rid)] == ["note"]
    assert s.memory_get(pid, "scope/a")["value"] == {"v": 1} and s.get_approval(aid)["details"] == {"hash": "abc"}
    s.close()


def test_invalid_transitions_and_unknown_ids_are_refused(engine_url):
    s = Store(engine_url)
    pid = s.create_project("t", "q")
    rid = s.create_run(pid, "q", {})
    s.add_tasks(rid, [{"id": "t1", "agent_type": "discovery", "title": "x"}])
    s.update_task("t1", status="running")
    s.update_task("t1", status="succeeded")
    with pytest.raises(InvalidTransition):
        s.update_task("t1", status="running")  # a finished task cannot be resurrected
    with pytest.raises(KeyError):
        s.update_task("nope", status="running")
    with pytest.raises(KeyError):
        s.update_run("nope", status="running")
    import psycopg
    with pytest.raises(psycopg.errors.IntegrityError):
        s.create_run("proj_missing", "q", {})  # foreign keys are enforced by the server
    assert s.get_task("t1")["status"] == "succeeded"  # and the connection is still usable afterwards
    s.close()


def test_a_failed_transaction_rolls_back_every_write(engine_url):
    s = Store(engine_url)
    pid = s.create_project("t", "q")
    rid = s.create_run(pid, "q", {})
    with pytest.raises(RuntimeError, match="boom"):
        with s.transaction():
            s.add_tasks(rid, [{"id": "t1", "agent_type": "discovery", "title": "x"}])
            s.add_event(rid, "inside", {})
            s.memory_put(pid, "k", {}, rid, None)
            raise RuntimeError("boom")
    assert s.list_tasks(rid) == [] and s.list_events(rid) == [] and s.memory_get(pid, "k") is None
    s.add_tasks(rid, [{"id": "t1", "agent_type": "discovery", "title": "x"}])  # the id is free again
    assert [t["id"] for t in s.list_tasks(rid)] == ["t1"]
    s.close()


def test_a_database_error_inside_a_transaction_rolls_back_earlier_statements(engine_url):
    s = Store(engine_url)
    pid = s.create_project("t", "q")
    rid = s.create_run(pid, "q", {})
    import psycopg
    with pytest.raises(psycopg.errors.UniqueViolation):
        s.add_tasks(rid, [{"id": "dup", "agent_type": "a", "title": "1"}, {"id": "dup", "agent_type": "a", "title": "2"}])
    assert s.list_tasks(rid) == []  # the first insert of the batch did not survive the second one failing
    s.close()


def test_duplicate_task_ids_are_prevented_by_the_database_under_concurrency(engine_url):
    s0 = Store(engine_url)
    rid = s0.create_run(s0.create_project("t", "q"), "q", {})
    wins, errors = [], []

    def add():
        s = Store(engine_url)
        try:
            s.add_tasks(rid, [{"id": "same", "agent_type": "a", "title": "x"}])
            wins.append(1)
        except Exception as exc:
            errors.append(type(exc).__name__)
        finally:
            s.close()

    threads = [threading.Thread(target=add) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert len(wins) == 1 and len(errors) == 7 and len(s0.list_tasks(rid)) == 1
    s0.close()


def _race(engine_url, n, work):
    """Run `work(store, i)` in `n` threads, each with its own connection, released together. Returns (results, errors)."""
    gate, results, errors = threading.Barrier(n), [], []

    def go(i):
        s = Store(engine_url)
        try:
            gate.wait(30)
            results.append(work(s, i))
        except Exception as exc:
            errors.append(exc)
        finally:
            s.close()

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join(120) for t in threads]
    return results, errors


def test_concurrent_task_transition_has_exactly_one_winner(engine_url):
    s0 = Store(engine_url)
    rid = s0.create_run(s0.create_project("t", "q"), "q", {})
    s0.add_tasks(rid, [{"id": "t1", "agent_type": "a", "title": "x"}])

    def claim(s, i):
        s.update_task("t1", status="running", started_at=f"w{i}")
        return i

    # pending -> running is legal once; a second writer sees 'running' and 'running' -> 'running' is a no-op transition, so use a
    # transition that is only legal from pending: pending -> skipped, racing against pending -> running.
    def mover(s, i):
        s.update_task("t1", status="running" if i % 2 == 0 else "skipped")
        return s.get_task("t1")["status"]

    results, errors = _race(engine_url, 8, mover)
    final = s0.get_task("t1")["status"]
    assert final in ("running", "skipped")
    # exactly one transition out of 'pending' was applied; every writer that lost saw a state it could not leave (or repeated the winner)
    assert all(isinstance(e, InvalidTransition) for e in errors)
    assert len(results) + len(errors) == 8 and results.count(final) == len(results)
    s0.close()


def test_concurrent_run_creation_gives_unique_consecutive_run_numbers(engine_url):
    s0 = Store(engine_url)
    pid = s0.create_project("t", "q")
    results, errors = _race(engine_url, 8, lambda s, i: s.create_run(pid, "q", {}))
    assert not errors and len(set(results)) == 8
    numbers = sorted(s0.get_run(r)["run_number"] for r in results)
    assert numbers == list(range(1, 9))  # no duplicates, no gaps: the project row lock serialises the count-then-insert
    s0.close()


def test_token_budget_reservation_never_exceeds_the_limit_under_concurrency(engine_url):
    s0 = Store(engine_url)
    rid = s0.create_run(s0.create_project("t", "q"), "q", {})
    results, errors = _race(engine_url, 12, lambda s, i: s.reserve_model_tokens(rid, 10, 55))
    assert not errors and results.count(True) == 5 and results.count(False) == 7
    assert s0.get_run(rid)["model_tokens_used"] == 50
    s0.close()


def test_an_approval_is_decided_exactly_once_under_concurrency(engine_url):
    s0 = Store(engine_url)
    rid = s0.create_run(s0.create_project("t", "q"), "q", {})
    aid = s0.add_approval(rid, "t1", "act", {})
    results, errors = _race(engine_url, 6, lambda s, i: s.decide_approval(aid, "approved" if i % 2 else "rejected", f"u{i}") or i)
    assert len(results) == 1 and len(errors) == 5 and all(isinstance(e, ValueError) for e in errors)
    assert s0.get_approval(aid)["reviewer_id"] == f"u{results[0]}"
    s0.close()


def test_events_become_visible_in_id_order_for_a_polling_reader(engine_url):
    """Sequence values are allocated before commit. Without the per-run lock a reader polling `id > last` can skip an event whose
    id was allocated first but committed last. The reader below polls continuously while writers race; it must see every event."""
    s0 = Store(engine_url)
    rid = s0.create_run(s0.create_project("t", "q"), "q", {})
    total, seen, stop = 6 * 40, [], threading.Event()

    def reader():
        r, last = Store(engine_url), 0
        while not stop.is_set() or len(seen) < total:
            for e in r.list_events(rid, after_id=last):
                seen.append(e["id"])
                last = e["id"]
            if len(seen) >= total:
                break
            time.sleep(0.001)
        r.close()

    rt = threading.Thread(target=reader)
    rt.start()
    _, errors = _race(engine_url, 6, lambda s, i: [s.add_event(rid, "e", {"w": i, "n": n}) for n in range(40)])
    stop.set()
    rt.join(60)
    all_ids = [e["id"] for e in s0.list_events(rid, limit=10_000)]
    assert not errors and len(all_ids) == total
    assert seen == sorted(seen) and sorted(seen) == all_ids  # nothing skipped, nothing out of order
    s0.close()


def test_connection_loss_is_survived_and_writes_are_never_silently_repeated(engine_url):
    s = Store(engine_url)
    pid = s.create_project("t", "q")
    rid = s.create_run(pid, "q", {})

    def kill_connection():
        with pg.connect() as c:
            c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE application_name='researchos-engine' "
                      "AND pid <> pg_backend_pid()")
        time.sleep(0.2)

    kill_connection()
    assert s.get_run(rid)["id"] == rid  # a read is retried on a fresh connection
    kill_connection()
    with pytest.raises(StoreUnavailable):  # a write might already have been applied, so it is reported, not repeated
        s.add_event(rid, "maybe", {})
    n_after_error = len(s.list_events(rid))  # reconnects by itself
    s.add_event(rid, "again", {})
    assert len(s.list_events(rid)) == n_after_error + 1
    s.close()


def test_connections_are_released_on_close_and_dispose(engine_url, api_url):
    def conns():
        with pg.connect() as c:
            return c.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE application_name LIKE 'researchos-%%' "
                             "AND application_name NOT LIKE '%%lock%%'").fetchone()[0]

    before = conns()
    stores = [Store(engine_url) for _ in range(4)]
    for st in stores:
        st.create_project("t", "q")
    assert conns() >= before + 4
    [st.close() for st in stores]
    eng = make_engine(api_url, pool_size=3, max_overflow=0)
    api_migrate.upgrade(eng)
    sm = make_sessionmaker(eng)
    for _ in range(10):  # more checkouts than the pool holds: sessions must return their connection
        with sm() as db:
            db.execute(text("SELECT 1"))
    assert eng.pool.checkedout() == 0
    eng.dispose()
    time.sleep(0.3)
    assert conns() <= before


# ================================================================================================ job queue
def seed(sm, n_runs=1):
    """A user, a project and `n_runs` runs; returns the run ids."""
    with sm() as db:
        u = User(email=f"u{time.monotonic_ns()}@example.org", password_hash="x")
        db.add(u)
        db.flush()
        p = Project(owner_id=u.id, title="p", question="q?")
        db.add(p)
        db.flush()
        runs = [Run(project_id=p.id, owner_id=u.id, engine_run_id=f"eng_{time.monotonic_ns()}_{i}", question="q?", workflow="full")
                for i in range(n_runs)]
        db.add_all(runs)
        db.commit()
        return [r.id for r in runs], u.id


@pytest.fixture()
def api(api_url):
    eng = make_engine(api_url)
    api_migrate.upgrade(eng)
    yield api_url, eng, make_sessionmaker(eng)
    eng.dispose()


def queue_settings(**kw):
    base = dict(job_lease_s=1.0, job_backoff_s=0.05, job_max_attempts=3)
    base.update(kw)
    return Settings(**base)


def load_runs(sm, ids):
    with sm() as db:
        return [db.get(Run, i) for i in ids]


def test_concurrent_workers_never_claim_the_same_job(api):
    url, _, sm = api
    ids, _ = seed(sm, 24)
    q = JobQueue(sm, queue_settings())
    for r in load_runs(sm, ids):
        assert q.enqueue(r)
    claimed, errors, gate = [], [], threading.Barrier(6)

    def worker(n):  # each worker has its own engine, pool and connections, like a separate process
        eng = make_engine(url)
        try:
            wq = JobQueue(make_sessionmaker(eng), queue_settings())
            gate.wait(30)
            while True:
                job = wq.claim(f"worker-{n}")
                if job is None:
                    return
                claimed.append((job["id"], f"worker-{n}"))
        except Exception as exc:
            errors.append(exc)
        finally:
            eng.dispose()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join(120) for t in threads]
    assert not errors
    job_ids = [j for j, _ in claimed]
    assert len(job_ids) == len(set(job_ids)) == 24  # every job claimed exactly once
    assert len({w for _, w in claimed}) > 1  # and the work really was spread over several workers
    with sm() as db:
        rows = db.scalars(select(ResearchJob)).all()
        assert all(j.status == "running" and j.attempts == 1 and j.lease_expires_at is not None for j in rows)


def test_concurrent_enqueue_of_one_run_creates_one_job(api):
    url, _, sm = api
    ids, _ = seed(sm)
    run = load_runs(sm, ids)[0]
    results, gate = [], threading.Barrier(8)

    def enq():
        eng = make_engine(url)
        try:
            q = JobQueue(make_sessionmaker(eng), queue_settings())
            gate.wait(30)
            results.append(q.enqueue(run))
        finally:
            eng.dispose()

    threads = [threading.Thread(target=enq) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert results.count(True) == 1 and results.count(False) == 7
    with sm() as db:
        assert db.scalar(select(func.count()).select_from(ResearchJob)) == 1


def test_lease_heartbeat_expiry_retry_and_give_up(api):
    _, _, sm = api
    ids, _ = seed(sm)
    q = JobQueue(sm, queue_settings(job_lease_s=0.4, job_max_attempts=2))
    q.enqueue(load_runs(sm, ids)[0])
    job = q.claim("w1")
    assert job and job["attempts"] == 1 and q.claim("w2") is None  # nothing else to claim while it is running
    assert q.heartbeat(job["id"], "w1") and not q.heartbeat(job["id"], "w2")  # only the owner can renew
    assert q.reap_expired() == 0  # lease still valid
    time.sleep(0.6)
    assert q.reap_expired() == 1  # lease expired: back to the queue
    row = q.get(job["engine_run_id"])
    assert row["status"] == "queued" and row["worker_id"] == "" and "heartbeating" in row["error"]
    assert not q.heartbeat(job["id"], "w1")  # the dead worker has lost it
    assert q.finish(job["id"], "w1", "done") is False  # and cannot complete it either
    time.sleep(0.2)  # retry backoff
    again = q.claim("w2")
    assert again and again["id"] == job["id"] and again["attempts"] == 2
    time.sleep(0.6)
    assert q.reap_expired() == 1
    assert q.get(job["engine_run_id"])["status"] == "failed"  # attempts used up: visible failure, not an endless loop


def test_a_renewed_lease_is_not_taken_away_by_a_concurrent_reaper(api):
    _, _, sm = api
    ids, _ = seed(sm)
    q = JobQueue(sm, queue_settings(job_lease_s=0.3))
    q.enqueue(load_runs(sm, ids)[0])
    job = q.claim("w1")
    time.sleep(0.45)  # lease has expired as far as a stale list of candidates is concerned...
    assert q.heartbeat(job["id"], "w1")  # ...but the worker renews it before the reaper acts on its stale list
    assert q.requeue(job["id"], "w1", "stale reaper", only_if_expired=True) == "running"
    assert q.get(job["engine_run_id"])["status"] == "running"


def test_competing_reapers_and_a_finishing_worker_leave_one_consistent_state(api):
    url, _, sm = api
    ids, _ = seed(sm)
    q = JobQueue(sm, queue_settings(job_lease_s=0.2, job_max_attempts=5))
    q.enqueue(load_runs(sm, ids)[0])
    job = q.claim("w1")
    time.sleep(0.4)
    outcomes, gate = [], threading.Barrier(5)

    def reap():
        eng = make_engine(url)
        try:
            rq = JobQueue(make_sessionmaker(eng), queue_settings(job_lease_s=0.2, job_max_attempts=5))
            gate.wait(30)
            outcomes.append(("reap", rq.reap_expired()))
        finally:
            eng.dispose()

    def finish():
        eng = make_engine(url)
        try:
            gate.wait(30)
            outcomes.append(("finish", JobQueue(make_sessionmaker(eng), queue_settings()).finish(job["id"], "w1", "done")))
        finally:
            eng.dispose()

    threads = [threading.Thread(target=reap) for _ in range(4)] + [threading.Thread(target=finish)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    final = q.get(job["engine_run_id"])
    reaps = sum(n for k, n in outcomes if k == "reap")
    finished = [v for k, v in outcomes if k == "finish"][0]
    # either the worker finished first (job done, no reaper touched it) or exactly one reaper requeued it (and the finish was refused)
    assert (final["status"] == "done" and finished and reaps == 0) or (final["status"] == "queued" and not finished and reaps == 1)


def test_worker_crash_recovery_after_a_killed_process(api):
    url, _, sm = api
    ids, _ = seed(sm)
    q = JobQueue(sm, queue_settings(job_lease_s=1.0))
    q.enqueue(load_runs(sm, ids)[0])
    crashing = """
        import sys, time
        from api.config import Settings
        from api.db import make_engine, make_sessionmaker
        from api.jobs import JobQueue
        eng = make_engine(sys.argv[1])
        q = JobQueue(make_sessionmaker(eng), Settings(job_lease_s=1.0, job_backoff_s=0.05, job_max_attempts=3))
        job = q.claim("doomed-worker")
        print("claimed", job["id"], flush=True)
        time.sleep(600)
    """
    proc = subprocess.Popen([sys.executable, "-c", textwrap.dedent(crashing), url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT)})
    try:
        assert proc.stdout.readline().startswith("claimed")
        proc.kill()  # no cleanup, no goodbye: the worker simply disappears
        proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert q.get(load_runs(sm, ids)[0].engine_run_id)["status"] == "running"  # as far as the database knows it is still running
    time.sleep(1.3)  # the lease runs out
    assert q.reap_expired() == 1
    survivor = q.claim("survivor")
    time.sleep(0.1)
    survivor = survivor or q.claim("survivor")
    assert survivor and survivor["worker_id"] == "survivor" and survivor["attempts"] == 2
    assert q.finish(survivor["id"], "survivor", "done") and q.get(survivor["engine_run_id"])["status"] == "done"


def test_engine_run_is_recovered_after_a_crash_without_repeating_finished_work(engine_url):
    """What the engine does for a run whose worker died: finished tasks stay finished, the interrupted one is re-queued, the open
    execution record is closed as abandoned, and the history shows both attempts."""
    from tests.conftest_agents import make_engine as make_test_engine, ok

    store = Store(engine_url)
    runs = {"count": 0}

    def worker(ctx, task):
        runs["count"] += 1
        return ok(task)

    engine, _ = make_test_engine({"worker": worker}, store=store)
    pid = store.create_project("t", "q")
    rid = store.create_run(pid, "q", {})
    store.add_tasks(rid, [{"id": "done1", "agent_type": "worker", "title": "finished"},
                          {"id": "mid", "agent_type": "worker", "title": "interrupted"}])
    store.update_task("done1", status="running")
    store.update_task("done1", status="succeeded", output={"ok": True})
    store.update_run(rid, status="running")
    store.update_task("mid", status="running")
    store.begin_execution(rid, "mid", "worker", 1)  # opened by the worker that then died
    store.close()

    store2 = Store(engine_url)  # a different process, later
    engine2, _ = make_test_engine({"worker": worker}, store=store2)
    engine2.recover(rid)
    assert store2.get_task("mid")["status"] == "pending" and store2.get_task("done1")["status"] == "succeeded"
    assert [e["status"] for e in store2.list_executions(rid, "mid")] == ["abandoned"]
    assert store2.get_task("done1")["output"] == {"ok": True}  # untouched
    store2.close()


# ================================================================================================ persistence of the P6 records
def test_claims_evidence_verifications_novelty_and_reports_persist_across_a_restart(api):
    url, eng, sm = api
    ids, uid = seed(sm)
    with sm() as db:
        run = db.get(Run, ids[0])
        pid = run.project_id
        claim = Claim(project_id=pid, run_id=run.id, owner_id=uid, text="Cells retained 91 percent capacity", claim_type="result",
                      source="agent", status="SUPPORTED", strength={"level": "moderate"}, explanation=["one direct passage"])
        db.add(claim)
        db.flush()
        db.add(ClaimEvidence(claim_id=claim.id, role="supporting", directness="direct", document_id="doc1", passage_id="doc1:p001",
                             page_number=None, quote="retained 91 percent", retrieval_method="bm25", retrieval_score=3.25, origin="verification"))
        db.add(ClaimVerification(claim_id=claim.id, status="SUPPORTED", strength={"level": "moderate"}, explanation=["x"], round=1,
                                 verifier="deterministic-lexical-v1"))
        db.add(Report(run_id=run.id, project_id=pid, owner_id=uid, content={"structured": {"sections": [1, 2]}}, editable={"notes": "ü"},
                      content_sha256="abc"))
        db.commit()
        cid = claim.id
    eng.dispose()

    eng2 = make_engine(url)  # "restart": a brand-new engine and pool
    sm2 = make_sessionmaker(eng2)
    try:
        with sm2() as db:
            c = db.get(Claim, cid)
            assert c.status == "SUPPORTED" and c.strength == {"level": "moderate"}
            ev = db.scalars(select(ClaimEvidence).where(ClaimEvidence.claim_id == cid)).all()
            assert len(ev) == 1 and ev[0].page_number is None and ev[0].retrieval_score == 3.25 and ev[0].quote == "retained 91 percent"
            assert db.scalar(select(func.count()).select_from(ClaimVerification)) == 1
            rep = db.scalar(select(Report))
            assert rep.content == {"structured": {"sections": [1, 2]}} and rep.editable == {"notes": "ü"}
            db.delete(db.get(Project, c.project_id))  # deleting the project removes everything that hangs off it
            db.commit()
        with sm2() as db:
            for model in (Claim, ClaimEvidence, ClaimVerification, Report, Run):
                assert db.scalar(select(func.count()).select_from(model)) == 0, model.__name__
    finally:
        eng2.dispose()


def test_foreign_keys_and_unique_constraints_are_enforced_by_postgresql(api):
    _, _, sm = api
    ids, uid = seed(sm)
    with sm() as db:
        db.add(Claim(project_id="prj_missing", owner_id=uid, text="x", claim_type="result", source="user"))
        with pytest.raises(IntegrityError):
            db.commit()
    with sm() as db:
        db.add(User(email=db.get(User, uid).email, password_hash="y"))  # duplicate e-mail
        with pytest.raises(IntegrityError):
            db.commit()


def test_concurrent_api_writes_do_not_lose_updates(api):
    url, _, sm = api
    ids, uid = seed(sm)
    with sm() as db:
        pid = db.get(Run, ids[0]).project_id
    gate, errors = threading.Barrier(8), []

    def add_claims(n):
        eng = make_engine(url)
        try:
            maker = make_sessionmaker(eng)
            gate.wait(30)
            for i in range(10):
                with maker() as db:
                    db.add(Claim(project_id=pid, owner_id=uid, text=f"claim {n}-{i}", claim_type="result", source="user"))
                    db.commit()
        except Exception as exc:
            errors.append(exc)
        finally:
            eng.dispose()

    threads = [threading.Thread(target=add_claims, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join(120) for t in threads]
    assert not errors
    with sm() as db:
        assert db.scalar(select(func.count()).select_from(Claim)) == 80
        assert db.scalar(select(func.count(func.distinct(Claim.id))).select_from(Claim)) == 80


# ================================================================================================ end to end through the HTTP API
def make_app(api_url, engine_url, tmp_path):
    from api.app import create_app
    from tests.test_api_experiments import make_index
    s = Settings(database_url=api_url, engine_db=engine_url, data_dir=tmp_path / "data", corpus_isolate=False, embedded_worker=False,
                 job_poll_s=0.05, job_lease_s=2.0, login_attempts=50)
    return create_app(s, index_provider=make_index)


def test_http_api_on_postgresql_restart_persistence_and_cross_project_isolation(api_url, engine_url, tmp_path):
    from fastapi.testclient import TestClient
    from tests.test_api import PW
    from tests.test_api_experiments import project, sign_in

    app = make_app(api_url, engine_url, tmp_path)
    with TestClient(app, base_url="http://testserver") as alice, TestClient(app, base_url="http://testserver") as mallory:
        ha = sign_in(alice, "alice@example.org")
        pid = project(alice, ha)
        rid = alice.post(f"/projects/{pid}/runs", json={"start": False}, headers=ha).json()["id"]
        claim = alice.post(f"/projects/{pid}/claims", json={"text": "Cells retained 91 percent of capacity", "claim_type": "result"}, headers=ha)
        assert claim.status_code in (200, 201), claim.text
        cid = claim.json()["id"]
        hm = sign_in(mallory, "mallory@example.org")
        assert mallory.get(f"/projects/{pid}/claims", headers=hm).status_code == 404  # another user's project does not exist for her
        assert mallory.get(f"/projects/{pid}/claims/{cid}", headers=hm).status_code == 404
        assert mallory.post(f"/projects/{pid}/claims/{cid}/verify", headers=hm).status_code == 404
        assert mallory.get(f"/runs/{rid}", headers=hm).status_code == 404
    app.state.db_engine.dispose()
    app.state.runtime.store.close()

    app2 = make_app(api_url, engine_url, tmp_path)  # a restarted API process on the same databases
    with TestClient(app2, base_url="http://testserver") as alice2:
        login = alice2.post("/auth/login", json={"email": "alice@example.org", "password": PW})
        assert login.status_code == 200, login.text
        listed = alice2.get(f"/projects/{pid}/claims").json()
        items = listed["claims"]
        assert [c["id"] for c in items] == [cid]
        assert alice2.get(f"/runs/{rid}").status_code == 200
    app2.state.db_engine.dispose()
    app2.state.runtime.store.close()


# ================================================================================================ SQLite -> PostgreSQL copy
def make_sqlite_installation(tmp_path):
    """A real SQLite installation made by the application itself: users, a project, a run, a claim, engine rows."""
    from fastapi.testclient import TestClient
    from api.app import create_app
    from tests.test_api_experiments import make_index, project, sign_in

    api_db, engine_db = tmp_path / "app.db", tmp_path / "engine.db"
    s = Settings(database_url=f"sqlite:///{api_db.as_posix()}", engine_db=engine_db, data_dir=tmp_path / "data", corpus_isolate=False,
                 embedded_worker=False, login_attempts=50)
    app = create_app(s, index_provider=make_index)
    with TestClient(app, base_url="http://testserver") as c:
        h = sign_in(c, "alice@example.org")
        pid = project(c, h)
        rid = c.post(f"/projects/{pid}/runs", json={"start": False}, headers=h).json()["id"]
        c.post(f"/projects/{pid}/claims", json={"text": "Cells retained 91 percent of capacity", "claim_type": "result"}, headers=h)
        c.post(f"/projects/{pid}/claims", json={"text": "Second claim about electrolytes", "claim_type": "method"}, headers=h)
    store = app.state.runtime.store
    erid = app.state.sessionmaker().get(Run, rid).engine_run_id
    store.add_event(erid, "note", {"n": 1})
    store.add_event(erid, "note", {"n": 2})
    store.memory_put("proj_x", "k", {"v": 1}, erid, None)
    app.state.db_engine.dispose()
    store.close()
    return api_db, engine_db


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts_sqlite(api_db, engine_db):
    a = sqlite3.connect(api_db)
    e = sqlite3.connect(engine_db)
    api = {t: a.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in Base.metadata.tables}
    eng = {t: e.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
           ("projects", "runs", "tasks", "task_executions", "events", "approvals", "memory")}
    a.close(); e.close()
    return api, eng


def test_existing_sqlite_installation_copies_to_postgresql_without_loss(tmp_path):
    from api import dbcopy
    api_db, engine_db = make_sqlite_installation(tmp_path)
    before_api, before_engine = sha(api_db), sha(engine_db)
    want_api, want_engine = counts_sqlite(api_db, engine_db)
    assert want_api["users"] == 1 and want_api["claims"] == 2 and want_engine["events"] >= 2

    db = pg.new_database()
    api_target, engine_target = dburl.with_schema(db, "api"), dburl.with_schema(db, "engine")
    got_api = dbcopy.copy_api(str(api_db), api_target)
    got_engine = dbcopy.copy_engine(str(engine_db), engine_target)
    assert got_api == want_api and got_engine == want_engine

    # the data is really in PostgreSQL, with values intact
    eng = make_engine(api_target)
    try:
        with eng.connect() as c:
            assert c.execute(text("SELECT email FROM users")).scalars().all() == ["alice@example.org"]
            assert sorted(c.execute(text("SELECT text FROM claims")).scalars().all()) == [
                "Cells retained 91 percent of capacity", "Second claim about electrolytes"]
            assert api_migrate.schema_diff(eng) == [] and api_migrate.current_revision(eng) == api_migrate.head_revision()
    finally:
        eng.dispose()
    store = Store(engine_target)
    src = Store(str(engine_db))
    for run in [r["run_id"] for r in src._all("SELECT DISTINCT run_id FROM events")]:
        assert store.list_events(run) == src.list_events(run)  # every event, in order, byte for byte
    erid = store._one("SELECT run_id FROM events LIMIT 1")["run_id"]
    assert {"n": 1} in [e["payload"] for e in store.list_events(erid)]
    src.close()
    # identity sequences were advanced past the copied ids: a new event gets a fresh id instead of colliding
    new_id = store.add_event(erid, "after-copy", {})
    assert new_id > max(e["id"] for e in store.list_events(erid)[:-1])
    store.close()
    assert sha(api_db) == before_api and sha(engine_db) == before_engine  # the source was only read


def test_copy_refuses_a_non_empty_target_and_changes_nothing(tmp_path):
    from api import dbcopy
    api_db, engine_db = make_sqlite_installation(tmp_path)
    db = pg.new_database()
    api_target, engine_target = dburl.with_schema(db, "api"), dburl.with_schema(db, "engine")
    dbcopy.copy_api(str(api_db), api_target)
    dbcopy.copy_engine(str(engine_db), engine_target)
    eng = make_engine(api_target)
    with eng.connect() as c:
        before = c.execute(text("SELECT COUNT(*) FROM claims")).scalar()
    eng.dispose()
    with pytest.raises(dbcopy.CopyRefused, match="not empty"):
        dbcopy.copy_api(str(api_db), api_target)  # a second copy would duplicate or merge: refused
    with pytest.raises(dbcopy.CopyRefused, match="not empty"):
        dbcopy.copy_engine(str(engine_db), engine_target)
    eng = make_engine(api_target)
    with eng.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM claims")).scalar() == before
    eng.dispose()


def test_a_failed_copy_leaves_the_target_empty(tmp_path, monkeypatch):
    from api import dbcopy
    api_db, engine_db = make_sqlite_installation(tmp_path)
    db = pg.new_database()
    target = dburl.with_schema(db, "engine")
    real = dbcopy.ENGINE_TABLES
    monkeypatch.setattr(dbcopy, "ENGINE_TABLES", [*real, "no_such_table"])  # fails after every real table was copied in the transaction
    with pytest.raises(Exception):
        dbcopy.copy_engine(str(engine_db), target)
    monkeypatch.setattr(dbcopy, "ENGINE_TABLES", real)
    store = Store(target)
    assert store._one("SELECT COUNT(*) FROM projects")[0] == 0 and store._one("SELECT COUNT(*) FROM events")[0] == 0
    store.close()


def test_copy_refuses_a_source_that_is_not_at_the_current_migration(tmp_path):
    from api import dbcopy
    legacy = tmp_path / "old.db"
    c = sqlite3.connect(legacy)
    c.execute("CREATE TABLE users (id TEXT)")
    c.commit(); c.close()
    with pytest.raises(Exception):
        dbcopy.copy_api(str(legacy), dburl.with_schema(pg.new_database(), "api"))
    assert sqlite3.connect(legacy).execute("SELECT name FROM sqlite_master").fetchall() == [("users",)]  # never altered


def test_the_copy_script_runs_from_the_command_line(tmp_path):
    api_db, engine_db = make_sqlite_installation(tmp_path)
    db = pg.new_database()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "migrate_database.py"), "--to", db, "--api-db", str(api_db),
                        "--engine-db", str(engine_db)], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=180)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["copied_rows"]["api"]["claims"] == 2
    again = subprocess.run([sys.executable, str(ROOT / "scripts" / "migrate_database.py"), "--to", db, "--api-db", str(api_db)],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=180)
    assert again.returncode == 2 and "refused" in again.stderr
