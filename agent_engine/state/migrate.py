"""Schema management for the engine's own database. Same policy as the API's: apply migrations, adopt a pre-migration database
only if its tables match the frozen baseline exactly, and refuse (never alter) anything else or a database newer than this code."""
from __future__ import annotations

import importlib.util
import sqlite3
import threading
import time
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

MIGRATIONS = Path(__file__).resolve().parent / "migrations"
_LOCK = threading.Lock()  # Alembic script loading is not thread-safe; processes are serialised by SQLite itself (plus the retry below)


class SchemaMismatch(RuntimeError):
    pass


def _scripts() -> ScriptDirectory:
    return ScriptDirectory(str(MIGRATIONS))


def head_revision() -> str:
    return _scripts().get_current_head()


def _chain() -> list:
    """Revision scripts from the baseline to head, in order."""
    return list(reversed(list(_scripts().walk_revisions())))


def _module(script):
    spec = importlib.util.spec_from_file_location(f"engine_migration_{script.revision}", script.path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config(connection) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.attributes["connection"] = connection
    return cfg


def current_revision(path: str) -> str | None:
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()


def structure(conn: sqlite3.Connection) -> dict:
    """Tables -> columns and indexes, ignoring Alembic's bookkeeping table (used to compare databases)."""
    out = {}
    names = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                         "AND name != 'alembic_version' ORDER BY name").fetchall()
    for (t,) in names:
        cols = [tuple(r)[1:] for r in conn.execute(f"PRAGMA table_info({t})")]
        idx = sorted(tuple(r)[1:3] for r in conn.execute(f"PRAGMA index_list({t})")
                     if not str(tuple(r)[1]).startswith("sqlite_autoindex"))
        out[t] = {"columns": cols, "indexes": idx}
    return out


def apply_in_memory(conn: sqlite3.Connection) -> None:
    """Bring a brand-new in-memory store to head using the same SQL the migrations run (no Alembic involved)."""
    for script in _chain():
        for statement in _module(script).UP:
            conn.execute(statement)


def _baseline_structure() -> dict:
    scratch = sqlite3.connect(":memory:")
    try:
        for statement in _module(_chain()[0]).UP:
            scratch.execute(statement)
        return structure(scratch)
    finally:
        scratch.close()


def upgrade(path: str) -> str:
    """Bring the database file at `path` to head; returns the resulting revision. Safe to call from several threads/processes."""
    with _LOCK:
        for attempt in range(12):
            try:
                return _upgrade(path)
            except SchemaMismatch:
                raise
            except Exception as exc:
                if attempt == 11 or "locked" not in str(exc).lower():
                    raise
                time.sleep(0.1 * (attempt + 1))
    raise AssertionError("unreachable")


def _upgrade(path: str) -> str:
    probe = sqlite3.connect(path)
    try:
        has_tables = probe.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()[0] > 0
        versioned = probe.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='alembic_version'").fetchone()[0] > 0
        adopt = False
        if has_tables and not versioned:
            if structure(probe) != _baseline_structure():
                raise SchemaMismatch("the engine database has tables but no migration history and does not match the expected "
                                     "schema; back it up and migrate it by hand")
            adopt = True
    finally:
        probe.close()
    engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 30})
    try:
        try:
            with engine.begin() as conn:
                if adopt:
                    command.stamp(_config(conn), "0001")
                command.upgrade(_config(conn), "head")
        except Exception as exc:
            # Two processes starting at once can race on the version table; if the other one finished, we are done.
            if current_revision(path) == head_revision():
                return head_revision()
            if "locate revision" in str(exc):
                raise SchemaMismatch("the engine database is at a revision this version does not know "
                                     "(written by newer code?)") from exc
            raise
    finally:
        engine.dispose()
    return current_revision(path) or ""
