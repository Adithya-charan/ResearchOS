"""Copy an existing SQLite installation (API database + engine database) into PostgreSQL.

Guarantees, in this order:
* the source is only read, never altered; it must already be at the current migration head (start the application on it once, or run
  `alembic upgrade head`, so the copy never has to guess what an old schema meant);
* the target is brought to head by the normal migrations first, and the copy refuses to run unless every target table is empty -
  it never merges into, overwrites or deletes existing data;
* each database is copied in ONE PostgreSQL transaction and committed only after every table's row count has been compared with the
  source, so a failure leaves the target exactly as it was (empty);
* identity sequences are advanced past the copied ids so new rows do not collide with copied ones."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import func, insert, select, text

from agent_engine.state import dburl
from agent_engine.state import migrate as engine_migrate
from api import migrate as api_migrate
from api.db import Base, make_engine
from api import models  # noqa: F401  (registers the tables)

ENGINE_TABLES = ["projects", "runs", "tasks", "task_executions", "events", "approvals", "memory"]  # parents before children
BATCH = 500


class CopyRefused(RuntimeError):
    pass


def _sqlite_path(location: str) -> str:
    return location.split("///", 1)[1] if location.startswith("sqlite:///") else location


def _reset_sequences(conn, tables: list[str]) -> None:
    for t in tables:
        seq = conn.execute(text("SELECT pg_get_serial_sequence(:t, 'id')"), {"t": t}).scalar()
        if seq:
            conn.execute(text(f"SELECT setval(:s, COALESCE((SELECT MAX(id) FROM {t}), 1), (SELECT MAX(id) FROM {t}) IS NOT NULL)"),
                         {"s": seq})


def copy_api(source_url: str, target_url: str) -> dict[str, int]:
    src = make_engine(source_url if source_url.startswith("sqlite") else f"sqlite:///{source_url}")
    try:
        if api_migrate.current_revision(src) != api_migrate.head_revision():
            raise CopyRefused("the source API database is not at the current migration head; start the application on it (or run "
                              "`alembic upgrade head`) before copying")
        dst = make_engine(target_url)
        try:
            api_migrate.upgrade(dst)
            tables = list(Base.metadata.sorted_tables)  # foreign-key order
            with dst.begin() as out, src.connect() as inp:
                for t in tables:
                    if out.execute(select(func.count()).select_from(t)).scalar():
                        raise CopyRefused(f"target table {t.name} is not empty; refusing to merge into existing data")
                counts: dict[str, int] = {}
                for t in tables:
                    rows = [dict(r._mapping) for r in inp.execute(select(t))]
                    for i in range(0, len(rows), BATCH):
                        out.execute(insert(t), rows[i:i + BATCH])
                    counts[t.name] = len(rows)
                for t in tables:
                    got = out.execute(select(func.count()).select_from(t)).scalar()
                    if got != counts[t.name]:
                        raise CopyRefused(f"row count mismatch in {t.name}: source {counts[t.name]}, target {got}")  # rolls back
                _reset_sequences(out, [t.name for t in tables if "id" in t.c and t.c.id.autoincrement is not False
                                       and t.c.id.type.python_type is int])
            return counts
        finally:
            dst.dispose()
    finally:
        src.dispose()


def copy_engine(source_path: str, target_url: str) -> dict[str, int]:
    import psycopg
    path = _sqlite_path(source_path)
    if not Path(path).is_file():
        raise CopyRefused(f"source engine database {path} does not exist")
    if engine_migrate.current_revision(path) != engine_migrate.head_revision():
        raise CopyRefused("the source engine database is not at the current migration head; open it with this version first")
    engine_migrate.upgrade(target_url)
    dsn, schema = dburl.psycopg_dsn(target_url)
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    out = psycopg.connect(dsn, **dburl.connect_options(schema, "researchos-dbcopy"))  # not autocommit: one transaction
    try:
        for t in ENGINE_TABLES:
            if out.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]:
                raise CopyRefused(f"target table {t} is not empty; refusing to merge into existing data")
        counts: dict[str, int] = {}
        for t in ENGINE_TABLES:
            rows = src.execute(f"SELECT * FROM {t}").fetchall()
            if rows:
                cols = rows[0].keys()
                sql = f"INSERT INTO {t} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"
                with out.cursor() as cur:
                    cur.executemany(sql, [tuple(r) for r in rows])
            counts[t] = len(rows)
        for t in ENGINE_TABLES:
            got = out.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            if got != counts[t]:
                raise CopyRefused(f"row count mismatch in {t}: source {counts[t]}, target {got}")
        for t, col in (("tasks", "seq"), ("events", "id"), ("memory", "id"), ("task_executions", "id")):
            seq = out.execute("SELECT pg_get_serial_sequence(%s, %s)", (t, col)).fetchone()[0]
            out.execute(f"SELECT setval(%s, COALESCE((SELECT MAX({col}) FROM {t}), 1), (SELECT MAX({col}) FROM {t}) IS NOT NULL)", (seq,))
        out.commit()
        return counts
    except BaseException:
        out.rollback()
        raise
    finally:
        out.close()
        src.close()
