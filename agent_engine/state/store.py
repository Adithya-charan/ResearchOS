"""Durable run state in SQLite (WAL mode).

The schema mirrors the blueprint's tables (research_projects, research_runs, agent_tasks, agent_events,
approvals) plus a provenance-tracked memory table. It is deliberately behind this one class so the API phase can
swap in PostgreSQL without touching the scheduler. Every mutation that must be atomic goes through `transaction()`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, question TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    run_number INTEGER NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    control TEXT NOT NULL DEFAULT 'run',          -- run | pause | cancel (requested by a user/API)
    limits_json TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
    stop_reason TEXT,
    model_tokens_used INTEGER NOT NULL DEFAULT 0,
    elapsed_s REAL NOT NULL DEFAULT 0,
    summary_json TEXT);
CREATE TABLE IF NOT EXISTS tasks (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES runs(id),
    parent_task_id TEXT,
    agent_type TEXT NOT NULL,
    title TEXT NOT NULL,
    task_input TEXT NOT NULL DEFAULT '{}',
    depends_on TEXT NOT NULL DEFAULT '[]',
    task_output TEXT,
    status TEXT NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    not_before REAL NOT NULL DEFAULT 0,
    error TEXT,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id, seq);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL, task_id TEXT, event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL,
    requested_action TEXT NOT NULL, details_json TEXT NOT NULL DEFAULT '{}',
    decision TEXT, reviewer_id TEXT, notes TEXT, created_at TEXT NOT NULL, decided_at TEXT);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL, key TEXT NOT NULL, value_json TEXT NOT NULL,
    run_id TEXT, task_id TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(project_id, key, id);
"""

_RUN_COLUMNS = {
    "status", "control", "started_at", "completed_at", "stop_reason", "model_tokens_used", "elapsed_s",
    "summary_json", "config_json",
}
_TASK_COLUMNS = {
    "status", "retry_count", "not_before", "error", "requires_approval", "started_at", "completed_at",
    "task_output", "task_input", "depends_on",
}
_TASK_ALIASES = {"output": "task_output", "input": "task_input"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            if self._conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0] == 0:
                self._conn.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """All-or-nothing block. Re-entrant: nested use joins the outer transaction."""
        with self._lock:
            if self._depth:
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
            finally:
                self._depth = 0

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _run(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    # ------------------------------------------------------------------ projects / runs
    def create_project(self, title: str, question: str) -> str:
        pid = new_id("proj")
        self._run("INSERT INTO projects VALUES (?,?,?,?)", (pid, title, question, utcnow()))
        return pid

    def get_project(self, project_id: str) -> dict | None:
        row = self._one("SELECT * FROM projects WHERE id=?", (project_id,))
        return dict(row) if row else None

    def create_run(self, project_id: str, question: str, limits: dict, config: dict | None = None) -> str:
        rid = new_id("run")
        with self.transaction():
            n = self._one("SELECT COUNT(*) FROM runs WHERE project_id=?", (project_id,))[0]
            self._run(
                "INSERT INTO runs (id, project_id, run_number, question, status, limits_json, config_json, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (rid, project_id, n + 1, question, "pending", _dumps(limits), _dumps(config or {}), utcnow()),
            )
        return rid

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["limits"] = json.loads(d.pop("limits_json"))
        d["config"] = json.loads(d.pop("config_json"))
        s = d.pop("summary_json")
        d["summary"] = json.loads(s) if s else None
        return d

    def get_run(self, run_id: str) -> dict:
        row = self._one("SELECT * FROM runs WHERE id=?", (run_id,))
        if row is None:
            raise KeyError(f"unknown run {run_id}")
        return self._run_row(row)

    def update_run(self, run_id: str, **fields: Any) -> None:
        if "summary" in fields:
            fields["summary_json"] = _dumps(fields.pop("summary"))
        bad = set(fields) - _RUN_COLUMNS
        if bad:
            raise ValueError(f"cannot update run columns {sorted(bad)}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self._run(f"UPDATE runs SET {sets} WHERE id=?", (*fields.values(), run_id))

    def add_model_tokens(self, run_id: str, n: int) -> int:
        """Adjust the run's token counter by `n` (may be negative when a reservation is refunded)."""
        with self.transaction():
            self._run("UPDATE runs SET model_tokens_used = model_tokens_used + ? WHERE id=?", (n, run_id))
            return self._one("SELECT model_tokens_used FROM runs WHERE id=?", (run_id,))[0]

    def reserve_model_tokens(self, run_id: str, n: int, limit: int) -> bool:
        """Atomically claim `n` tokens of the run's budget; False (and no change) if it would exceed `limit`."""
        with self.transaction():
            used = self._one("SELECT model_tokens_used FROM runs WHERE id=?", (run_id,))[0]
            if used + n > limit:
                return False
            self._run("UPDATE runs SET model_tokens_used = model_tokens_used + ? WHERE id=?", (n, run_id))
            return True

    # ------------------------------------------------------------------ tasks
    @staticmethod
    def _task_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["input"] = json.loads(d.pop("task_input"))
        d["depends_on"] = json.loads(d["depends_on"])
        out = d.pop("task_output")
        d["output"] = json.loads(out) if out else None
        d["requires_approval"] = bool(d["requires_approval"])
        return d

    def add_tasks(self, run_id: str, tasks: list[dict]) -> None:
        with self.transaction():
            for t in tasks:
                self._run(
                    "INSERT INTO tasks (id, run_id, parent_task_id, agent_type, title, task_input, depends_on, status,"
                    " iteration, requires_approval, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        t["id"], run_id, t.get("parent_task_id"), t["agent_type"], t["title"],
                        _dumps(t.get("input", {})), _dumps(t.get("depends_on", [])), "pending",
                        t.get("iteration", 0), int(t.get("requires_approval", False)), utcnow(),
                    ),
                )

    def get_task(self, task_id: str) -> dict:
        row = self._one("SELECT * FROM tasks WHERE id=?", (task_id,))
        if row is None:
            raise KeyError(f"unknown task {task_id}")
        return self._task_row(row)

    def list_tasks(self, run_id: str) -> list[dict]:
        return [self._task_row(r) for r in self._all("SELECT * FROM tasks WHERE run_id=? ORDER BY seq", (run_id,))]

    def update_task(self, task_id: str, **fields: Any) -> None:
        cols: dict[str, Any] = {}
        for k, v in fields.items():
            k = _TASK_ALIASES.get(k, k)
            if k in ("task_output", "task_input", "depends_on") and v is not None:
                v = _dumps(v)
            cols[k] = v
        bad = set(cols) - _TASK_COLUMNS
        if bad:
            raise ValueError(f"cannot update task columns {sorted(bad)}")
        sets = ", ".join(f"{k}=?" for k in cols)
        self._run(f"UPDATE tasks SET {sets} WHERE id=?", (*cols.values(), task_id))

    def task_counts(self, run_id: str) -> dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) n FROM tasks WHERE run_id=? GROUP BY status", (run_id,))
        return {r["status"]: r["n"] for r in rows}

    # ------------------------------------------------------------------ events
    def add_event(self, run_id: str, event_type: str, payload: dict | None = None, task_id: str | None = None) -> int:
        cur = self._run(
            "INSERT INTO events (run_id, task_id, event_type, payload_json, created_at) VALUES (?,?,?,?,?)",
            (run_id, task_id, event_type, _dumps(payload or {}), utcnow()),
        )
        return int(cur.lastrowid)

    def list_events(self, run_id: str, after_id: int = 0, limit: int = 1000) -> list[dict]:
        rows = self._all(
            "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT ?", (run_id, after_id, limit)
        )
        return [
            {"id": r["id"], "run_id": r["run_id"], "task_id": r["task_id"], "type": r["event_type"],
             "payload": json.loads(r["payload_json"]), "created_at": r["created_at"]}
            for r in rows
        ]

    # ------------------------------------------------------------------ approvals
    def add_approval(self, run_id: str, task_id: str, requested_action: str, details: dict) -> str:
        aid = new_id("appr")
        self._run(
            "INSERT INTO approvals (id, run_id, task_id, requested_action, details_json, created_at) VALUES (?,?,?,?,?,?)",
            (aid, run_id, task_id, requested_action, _dumps(details), utcnow()),
        )
        return aid

    @staticmethod
    def _approval_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["details"] = json.loads(d.pop("details_json"))
        return d

    def get_approval(self, approval_id: str) -> dict:
        row = self._one("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row is None:
            raise KeyError(f"unknown approval {approval_id}")
        return self._approval_row(row)

    def approval_for_task(self, task_id: str) -> dict | None:
        row = self._one("SELECT * FROM approvals WHERE task_id=? ORDER BY created_at DESC LIMIT 1", (task_id,))
        return self._approval_row(row) if row else None

    def list_approvals(self, run_id: str, pending_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM approvals WHERE run_id=?" + (" AND decision IS NULL" if pending_only else "")
        return [self._approval_row(r) for r in self._all(sql + " ORDER BY created_at", (run_id,))]

    def decide_approval(self, approval_id: str, decision: str, reviewer_id: str, notes: str = "") -> None:
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        with self.transaction():
            row = self._one("SELECT decision FROM approvals WHERE id=?", (approval_id,))
            if row is None:
                raise KeyError(f"unknown approval {approval_id}")
            if row["decision"] is not None:
                raise ValueError(f"approval {approval_id} already decided ({row['decision']})")
            self._run(
                "UPDATE approvals SET decision=?, reviewer_id=?, notes=?, decided_at=? WHERE id=?",
                (decision, reviewer_id, notes, utcnow(), approval_id),
            )

    # ------------------------------------------------------------------ memory (append-only, with provenance)
    def memory_put(self, project_id: str, key: str, value: Any, run_id: str | None, task_id: str | None) -> None:
        self._run(
            "INSERT INTO memory (project_id, key, value_json, run_id, task_id, created_at) VALUES (?,?,?,?,?,?)",
            (project_id, key, _dumps(value), run_id, task_id, utcnow()),
        )

    def memory_get(self, project_id: str, key: str) -> dict | None:
        row = self._one(
            "SELECT * FROM memory WHERE project_id=? AND key=? ORDER BY id DESC LIMIT 1", (project_id, key)
        )
        return self._memory_row(row) if row else None

    @staticmethod
    def _memory_row(row: sqlite3.Row) -> dict:
        return {"key": row["key"], "value": json.loads(row["value_json"]), "run_id": row["run_id"],
                "task_id": row["task_id"], "created_at": row["created_at"]}

    def memory_list(self, project_id: str, prefix: str = "") -> list[dict]:
        rows = self._all(
            # substr compare, not LIKE: the prefix is literal (no wildcards) and case-sensitive
            "SELECT m.* FROM memory m JOIN (SELECT key, MAX(id) mid FROM memory WHERE project_id=? "
            "AND substr(key, 1, ?) = ? GROUP BY key) l ON m.id = l.mid ORDER BY m.key",
            (project_id, len(prefix), prefix),
        )
        return [self._memory_row(r) for r in rows]
