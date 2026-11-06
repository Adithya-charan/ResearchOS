"""The agent engine's own database is created and upgraded by Alembic migrations (a tree separate from the API's)."""
import sqlite3
import threading

import pytest

from agent_engine.state import migrate
from agent_engine.state.migrate import SchemaMismatch
from agent_engine.state.store import Store

# What the engine created for itself before migrations existed (copied verbatim from the old Store.SCHEMA, deliberately not imported).
LEGACY = [
    "CREATE TABLE schema_info (version INTEGER NOT NULL)",
    "INSERT INTO schema_info VALUES (1)",
    "CREATE TABLE projects (id TEXT PRIMARY KEY, title TEXT NOT NULL, question TEXT NOT NULL, created_at TEXT NOT NULL)",
    """CREATE TABLE runs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), run_number INTEGER NOT NULL,
    question TEXT NOT NULL, status TEXT NOT NULL, control TEXT NOT NULL DEFAULT 'run', limits_json TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, stop_reason TEXT,
    model_tokens_used INTEGER NOT NULL DEFAULT 0, elapsed_s REAL NOT NULL DEFAULT 0, summary_json TEXT)""",
    """CREATE TABLE tasks (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, run_id TEXT NOT NULL REFERENCES runs(id),
    parent_task_id TEXT, agent_type TEXT NOT NULL, title TEXT NOT NULL, task_input TEXT NOT NULL DEFAULT '{}',
    depends_on TEXT NOT NULL DEFAULT '[]', task_output TEXT, status TEXT NOT NULL, iteration INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0, not_before REAL NOT NULL DEFAULT 0, error TEXT,
    requires_approval INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT)""",
    "CREATE INDEX idx_tasks_run ON tasks(run_id, seq)",
    """CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, task_id TEXT, event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)""",
    "CREATE INDEX idx_events_run ON events(run_id, id)",
    """CREATE TABLE approvals (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL, requested_action TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}', decision TEXT, reviewer_id TEXT, notes TEXT, created_at TEXT NOT NULL, decided_at TEXT)""",
    """CREATE TABLE memory (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, key TEXT NOT NULL,
    value_json TEXT NOT NULL, run_id TEXT, task_id TEXT, created_at TEXT NOT NULL)""",
    "CREATE INDEX idx_memory_key ON memory(project_id, key, id)",
]


def legacy_db(path):
    c = sqlite3.connect(path)
    for s in LEGACY:
        c.execute(s)
    c.execute("INSERT INTO projects VALUES ('prj_old', 'Old', 'Q?', '2026-01-01T00:00:00.000+00:00')")
    c.commit()
    c.close()


def dump(path):
    c = sqlite3.connect(path)
    try:
        return migrate.structure(c)
    finally:
        c.close()


def test_a_new_file_database_is_created_by_migrations(tmp_path):
    db = tmp_path / "engine.db"
    Store(db).close()
    assert migrate.current_revision(str(db)) == migrate.head_revision() == "0001"
    assert {"projects", "runs", "tasks", "events", "approvals", "memory", "alembic_version"} <= \
        {r[0] for r in sqlite3.connect(db).execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_reopening_is_a_no_op_and_keeps_data(tmp_path):
    db = tmp_path / "engine.db"
    s = Store(db)
    pid = s.create_project("t", "q")
    s.close()
    s2 = Store(db)
    assert s2.get_project(pid)["question"] == "q"
    s2.close()
    assert migrate.current_revision(str(db)) == "0001"


def test_in_memory_and_file_stores_get_the_same_schema(tmp_path):
    db = tmp_path / "engine.db"
    Store(db).close()
    mem = Store(":memory:")
    try:
        assert migrate.structure(mem._conn) == dump(db)
    finally:
        mem.close()


def test_a_database_made_before_migrations_is_adopted_with_its_data(tmp_path):
    db = tmp_path / "engine.db"
    legacy_db(db)
    assert migrate.current_revision(str(db)) is None
    s = Store(db)
    assert s.get_project("prj_old")["title"] == "Old"
    s.close()
    assert migrate.current_revision(str(db)) == "0001"
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM schema_info").fetchone()[0] == 1  # not duplicated


def test_a_legacy_database_that_differs_is_refused_and_left_alone(tmp_path):
    db = tmp_path / "engine.db"
    legacy_db(db)
    c = sqlite3.connect(db)
    c.execute("ALTER TABLE runs ADD COLUMN surprise TEXT")
    c.commit()
    c.close()
    with pytest.raises(SchemaMismatch, match="does not match"):
        Store(db)
    assert migrate.current_revision(str(db)) is None
    assert "surprise" in [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(runs)")]


def test_a_database_newer_than_this_code_is_refused(tmp_path):
    db = tmp_path / "engine.db"
    Store(db).close()
    c = sqlite3.connect(db)
    c.execute("UPDATE alembic_version SET version_num='9999'")
    c.commit()
    c.close()
    with pytest.raises(SchemaMismatch, match="does not know"):
        Store(db)


def test_downgrade_then_upgrade_round_trips(tmp_path):
    from alembic import command
    from sqlalchemy import create_engine
    db = tmp_path / "engine.db"
    Store(db).close()
    before = dump(db)
    eng = create_engine(f"sqlite:///{db}")
    with eng.begin() as conn:
        command.downgrade(migrate._config(conn), "base")
    assert dump(db) == {}
    assert migrate.upgrade(str(db)) == "0001"
    assert dump(db) == before
    eng.dispose()


def test_two_processes_starting_together_both_succeed(tmp_path):
    db = tmp_path / "engine.db"
    errors, stores = [], []

    def open_store():
        try:
            stores.append(Store(db))
        except Exception as e:  # pragma: no cover - failure path is the assertion below
            errors.append(e)

    threads = [threading.Thread(target=open_store) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(stores) == 6
    assert migrate.current_revision(str(db)) == "0001"
    [s.close() for s in stores]


def test_separate_processes_starting_together_both_succeed(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    db = tmp_path / "engine.db"
    code = f"from agent_engine.state.store import Store; Store(r'{db}').close()"
    procs = [subprocess.Popen([sys.executable, "-c", code], cwd=root, stderr=subprocess.PIPE) for _ in range(4)]
    results = [(p.wait(timeout=120), p.stderr.read().decode()[-300:]) for p in procs]
    assert all(rc == 0 for rc, _ in results), results
    assert migrate.current_revision(str(db)) == "0001"
