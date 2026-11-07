"""Durable run state in SQLite (WAL mode) or PostgreSQL.

The schema mirrors the blueprint's tables (research_projects, research_runs, agent_tasks, agent_events,
approvals) plus a provenance-tracked memory table. Both backends sit behind this one class, so the scheduler and the agents do
not know which one they run on. `Store("path.db")` / `Store(":memory:")` use SQLite; `Store("postgresql://...")` uses PostgreSQL
(see `agent_engine/state/dburl.py` for the optional `schema` parameter and `docs/DATABASE.md`).

Every mutation that must be atomic goes through `transaction()`. SQLite serialises writers with `BEGIN IMMEDIATE`; PostgreSQL
uses ordinary READ COMMITTED transactions plus row locks (`SELECT ... FOR UPDATE`) wherever a value is read and then changed, and a
per-run advisory lock so a run's events become visible in id order.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_engine.state import dburl, migrate
from agent_engine.state.transitions import check_run, check_task

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


class _Row(dict):
    """A PostgreSQL row that also answers `row[0]`, like `sqlite3.Row`."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def _row_factory(cursor):
    names = [c.name for c in cursor.description] if cursor.description else []

    def make(values):
        return _Row(zip(names, values))
    return make


class StoreUnavailable(RuntimeError):
    """The PostgreSQL server could not be reached (a write is not retried because it may already have been applied)."""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.postgres = dburl.is_postgres(self.path)
        self._lock = threading.RLock()
        self._depth = 0
        if self.postgres:
            self._init_postgres()
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            if self.path != ":memory:":
                for attempt in range(12):  # switching the journal mode needs a lock busy_timeout does not always cover
                    try:
                        self._conn.execute("PRAGMA journal_mode = WAL")
                        break
                    except sqlite3.OperationalError as exc:
                        if attempt == 11 or "locked" not in str(exc).lower():
                            raise
                        time.sleep(0.05 * (attempt + 1))
        if self.path == ":memory:":
            migrate.apply_in_memory(self._conn)
        else:
            migrate.upgrade(self.path)  # Alembic: creates, upgrades or adopts the schema; refuses unknown ones

    # ------------------------------------------------------------------ PostgreSQL connection
    def _init_postgres(self) -> None:
        import psycopg  # imported here so SQLite-only installations do not need the driver
        self._psycopg = psycopg
        self._dsn, self._schema = dburl.psycopg_dsn(self.path)
        self._conn = None
        self._connect()
        migrate.upgrade(self.path)  # Alembic under an advisory lock: creates, upgrades or refuses; never drops

    def _connect(self) -> None:
        opts = dburl.connect_options(self._schema, "researchos-engine")
        try:
            self._conn = self._psycopg.connect(self._dsn, autocommit=True, row_factory=_row_factory, **opts)
        except self._psycopg.OperationalError as exc:
            raise StoreUnavailable(f"cannot connect to {dburl.redact(self.path)}: {exc}") from exc

    def _healthy(self) -> bool:
        return self._conn is not None and not self._conn.closed and not self._conn.broken

    def _exec(self, sql: str, params: tuple = ()):
        """Run one statement on the PostgreSQL connection. A connection already known to be dead is replaced first. A statement that
        fails because the connection died mid-flight is retried only when it is a read outside a transaction: a write may have been
        applied before the connection dropped, so repeating it blindly could duplicate it."""
        sql = sql.replace("?", "%s")
        if not self._depth and not self._healthy():
            self._connect()
        try:
            return self._conn.execute(sql, params)
        except (self._psycopg.OperationalError, self._psycopg.InterfaceError) as exc:
            if self._depth == 0 and sql.lstrip()[:6].upper() == "SELECT" and not self._healthy():
                self._connect()
                return self._conn.execute(sql, params)
            raise StoreUnavailable(f"database connection lost: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """All-or-nothing block. Re-entrant: nested use joins the outer transaction."""
        with self._lock:
            if self._depth:
                yield
                return
            if self.postgres:
                if not self._healthy():
                    self._connect()
                try:
                    self._conn.execute("BEGIN")
                except (self._psycopg.OperationalError, self._psycopg.InterfaceError) as exc:
                    raise StoreUnavailable(f"database connection lost: {exc}") from exc  # nothing was sent, nothing was applied
            else:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except Exception:  # the connection may be gone: the server rolls back an abandoned transaction by itself
                    pass
                raise
            else:
                try:
                    self._conn.execute("COMMIT")
                except Exception as exc:
                    if self.postgres and isinstance(exc, (self._psycopg.OperationalError, self._psycopg.InterfaceError)):
                        # The server may or may not have committed before the connection dropped: say so instead of guessing.
                        raise StoreUnavailable(f"database connection lost while committing (outcome unknown): {exc}") from exc
                    raise
            finally:
                self._depth = 0

    @property
    def _for_update(self) -> str:
        """Row lock for read-then-write sequences. SQLite needs none (BEGIN IMMEDIATE already serialises writers)."""
        return " FOR UPDATE" if self.postgres else ""

    def _all(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            return (self._exec(sql, params) if self.postgres else self._conn.execute(sql, params)).fetchall()

    def _one(self, sql: str, params: tuple = ()):
        with self._lock:
            return (self._exec(sql, params) if self.postgres else self._conn.execute(sql, params)).fetchone()

    def _run(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._exec(sql, params) if self.postgres else self._conn.execute(sql, params)

    def _insert_id(self, sql: str, params: tuple) -> int:
        """Insert one row of a table with an integer `id` and return that id."""
        with self._lock:
            if self.postgres:
                return int(self._exec(sql + " RETURNING id", params).fetchone()[0])
            return int(self._conn.execute(sql, params).lastrowid)

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
            if self.postgres:  # two processes creating runs for one project must not both read the same count
                self._one("SELECT id FROM projects WHERE id=? FOR UPDATE", (project_id,))
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
        with self.transaction():  # the transition is checked against the row we hold, so two writers cannot both pass the check
            if "status" in fields:
                row = self._one("SELECT status FROM runs WHERE id=?" + self._for_update, (run_id,))
                if row is None:
                    raise KeyError(f"unknown run {run_id}")
                check_run(row["status"], fields["status"])
            self._run(f"UPDATE runs SET {sets} WHERE id=?", (*fields.values(), run_id))

    def add_model_tokens(self, run_id: str, n: int) -> int:
        """Adjust the run's token counter by `n` (may be negative when a reservation is refunded)."""
        with self.transaction():
            self._run("UPDATE runs SET model_tokens_used = model_tokens_used + ? WHERE id=?", (n, run_id))
            return self._one("SELECT model_tokens_used FROM runs WHERE id=?", (run_id,))[0]

    def reserve_model_tokens(self, run_id: str, n: int, limit: int) -> bool:
        """Atomically claim `n` tokens of the run's budget; False (and no change) if it would exceed `limit`."""
        with self.transaction():
            used = self._one("SELECT model_tokens_used FROM runs WHERE id=?" + self._for_update, (run_id,))[0]
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
        with self.transaction():  # check and write on the same locked row: a task cannot be moved twice from the same state
            if "status" in cols:
                row = self._one("SELECT status FROM tasks WHERE id=?" + self._for_update, (task_id,))
                if row is None:
                    raise KeyError(f"unknown task {task_id}")
                check_task(row["status"], cols["status"])
            self._run(f"UPDATE tasks SET {sets} WHERE id=?", (*cols.values(), task_id))

    def task_counts(self, run_id: str) -> dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) n FROM tasks WHERE run_id=? GROUP BY status", (run_id,))
        return {r["status"]: r["n"] for r in rows}

    # ------------------------------------------------------------------ execution records
    def begin_execution(self, run_id: str, task_id: str, agent_type: str, attempt: int) -> int:
        return self._insert_id("INSERT INTO task_executions (run_id, task_id, agent_type, attempt, started_at) VALUES (?,?,?,?,?)",
                               (run_id, task_id, agent_type, attempt, utcnow()))

    def finish_execution(self, exec_id: int, status: str, duration_s: float, cpu_process_s: float, error: str | None) -> None:
        self._run("UPDATE task_executions SET status=?, finished_at=?, duration_s=?, cpu_process_s=?, error=? "
                  "WHERE id=? AND finished_at IS NULL", (status, utcnow(), duration_s, cpu_process_s, error, exec_id))

    def abandon_open_executions(self, run_id: str) -> int:
        """Executions still open when the run is recovered belong to a worker that died: close them honestly."""
        cur = self._run("UPDATE task_executions SET status='abandoned', finished_at=?, error='worker lost before this attempt finished' "
                        "WHERE run_id=? AND finished_at IS NULL", (utcnow(), run_id))
        return cur.rowcount

    def list_executions(self, run_id: str, task_id: str | None = None) -> list[dict]:
        if task_id:
            rows = self._all("SELECT * FROM task_executions WHERE run_id=? AND task_id=? ORDER BY id", (run_id, task_id))
        else:
            rows = self._all("SELECT * FROM task_executions WHERE run_id=? ORDER BY id", (run_id,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ events
    def add_event(self, run_id: str, event_type: str, payload: dict | None = None, task_id: str | None = None) -> int:
        sql = "INSERT INTO events (run_id, task_id, event_type, payload_json, created_at) VALUES (?,?,?,?,?)"
        params = (run_id, task_id, event_type, _dumps(payload or {}), utcnow())
        if not self.postgres:
            return self._insert_id(sql, params)
        # Sequence values are handed out before commit, so two writers could commit out of id order and a reader polling
        # `id > last_seen` would skip the earlier one. Holding a per-run lock until commit makes a run's events commit in id order.
        with self.transaction():
            self._run("SELECT pg_advisory_xact_lock(hashtext(?))", (f"events:{run_id}",))
            return self._insert_id(sql, params)

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
            row = self._one("SELECT decision FROM approvals WHERE id=?" + self._for_update, (approval_id,))
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
