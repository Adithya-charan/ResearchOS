"""A real PostgreSQL server for the integration tests.

The server is the unmodified PostgreSQL 16 build shipped by the `pgserver` package (pip-installable, no Docker, no admin rights). It
listens on a local TCP port and is used exactly like any other PostgreSQL: the application connects with psycopg over TCP, so the code
paths that run in production (connection pooling, advisory locks, row locks, transactional DDL) are the ones that run in the tests.

Set RESEARCHOS_TEST_POSTGRES_URL=postgresql://user:pw@host:port/db to use another PostgreSQL server (for example the compose service)
instead of starting one. Every test gets its own schemas, so tests never see each other's data."""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import uuid
from pathlib import Path

_state: dict = {}


def available() -> bool:
    try:
        import psycopg  # noqa: F401
        if os.environ.get("RESEARCHOS_TEST_POSTGRES_URL"):
            return True
        import pgserver  # noqa: F401
        return True
    except ImportError:
        return False


def base_url() -> str:
    """postgresql:// URL of the server's maintenance database (started on first use)."""
    if "url" in _state:
        return _state["url"]
    external = os.environ.get("RESEARCHOS_TEST_POSTGRES_URL")
    if external:
        _state["url"] = external
        return external
    import pgserver
    data = Path(tempfile.mkdtemp(prefix="researchos-pg-"))
    server = pgserver.get_server(data, cleanup_mode="stop")
    _state["server"] = server
    _state["url"] = server.get_uri()
    atexit.register(shutil.rmtree, data, True)
    return _state["url"]


def connect(url: str | None = None, autocommit: bool = True):
    import psycopg
    from agent_engine.state import dburl
    dsn, _ = dburl.psycopg_dsn(url or base_url())
    return psycopg.connect(dsn, autocommit=autocommit)


def server_version() -> str:
    with connect() as c:
        return c.execute("SHOW server_version").fetchone()[0]


def new_database() -> str:
    """A brand-new empty database on the server; returns its URL."""
    from urllib.parse import urlsplit, urlunsplit
    name = f"t_{uuid.uuid4().hex[:12]}"
    with connect() as c:
        c.execute(f'CREATE DATABASE "{name}"')
    parts = urlsplit(base_url())
    return urlunsplit(parts._replace(path=f"/{name}"))


def new_schema_url(db_url: str | None = None, prefix: str = "s") -> str:
    """URL of a fresh schema name (created lazily by the application's migration, exactly as in a real installation)."""
    from agent_engine.state import dburl
    return dburl.with_schema(db_url or base_url(), f"{prefix}_{uuid.uuid4().hex[:12]}")


def drop_schema(url: str) -> None:
    from agent_engine.state import dburl
    base, schema = dburl.split_schema(url)
    if schema:
        with connect(base) as c:
            c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
