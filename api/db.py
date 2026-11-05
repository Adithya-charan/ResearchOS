from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Naive UTC (SQLite drops tzinfo, so everything is stored and compared as naive UTC)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def make_engine(url: str) -> Engine:
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
        else:
            Path(url.split("///", 1)[-1]).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _fk(dbapi_conn, _):  # SQLite ignores foreign keys unless asked
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return eng


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
