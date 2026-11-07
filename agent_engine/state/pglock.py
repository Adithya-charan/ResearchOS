"""Cross-process serialisation of schema changes on PostgreSQL.

Two API processes and a worker may all start at once on a fresh database. Alembic is not safe to run twice concurrently (both would
try to create the same tables), so each process takes a session-level advisory lock first and the loser finds the schema already at
head. The lock lives on its own connection: it is released when that connection closes, even if this process dies mid-migration."""
from __future__ import annotations

import time
from contextlib import contextmanager

from agent_engine.state import dburl


@contextmanager
def migration_lock(target: str, name: str, timeout_s: float = 120.0):
    """Hold the advisory lock `name` (scoped to the schema in `target`) for the duration of the block."""
    import psycopg
    dsn, schema = dburl.psycopg_dsn(target)
    key = f"researchos-migrate:{name}:{schema or 'public'}"
    conn = psycopg.connect(dsn, autocommit=True, **dburl.connect_options(None, "researchos-migrate-lock"))
    try:
        deadline = time.monotonic() + timeout_s
        while True:  # try-lock in a loop so a stuck migration elsewhere produces an error instead of an endless wait
            got = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (key,)).fetchone()[0]
            if got:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"another process has held the {name} migration lock for more than {timeout_s:.0f}s")
            time.sleep(0.1)
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (key,))
    finally:
        conn.close()
