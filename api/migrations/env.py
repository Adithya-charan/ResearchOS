"""Alembic environment. The application passes a live connection (config.attributes["connection"]); the CLI falls back to
RESEARCHOS_DATABASE_URL. SQLite needs batch mode for ALTER TABLE, so it is enabled for every dialect (it is a no-op elsewhere)."""
from alembic import context
from sqlalchemy import create_engine

from api import models  # noqa: F401  (registers every table on Base.metadata)
from api.config import Settings
from api.db import Base

config = context.config
target_metadata = Base.metadata


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    from api.db import make_engine
    url = config.get_main_option("sqlalchemy.url") or Settings.from_env().database_url
    engine = make_engine(url)  # same URL handling as the application (PostgreSQL schema, pool settings)
    with engine.connect() as conn:
        schema = getattr(engine, "_researchos_schema", None)
        if schema:
            from sqlalchemy import text
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            conn.commit()
        _run(conn)
        conn.commit()


if context.is_offline_mode():
    raise RuntimeError("offline (SQL script) migrations are not supported; run against a database")
run_migrations_online()
