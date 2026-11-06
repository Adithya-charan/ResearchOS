"""Schema management. The API applies Alembic migrations on startup (file databases) and refuses to guess when an existing
database was not created by migrations and does not match the current models."""
from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from api.db import Base

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


class SchemaMismatch(RuntimeError):
    pass


def _config(connection) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.attributes["connection"] = connection
    return cfg


def head_revision() -> str:
    return ScriptDirectory(str(MIGRATIONS)).get_current_head()


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def schema_diff(engine: Engine) -> list:
    """Differences between the live database and the models (empty means they agree)."""
    with engine.connect() as conn:
        return compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)


def upgrade(engine: Engine, revision: str = "head") -> str:
    """Bring `engine` to `revision`. A database that already has our tables but no migration history (created before migrations
    existed) is adopted only if it matches the models exactly; otherwise this raises instead of altering unknown data."""
    tables = set(inspect(engine).get_table_names())
    if tables and "alembic_version" not in tables and tables & set(Base.metadata.tables):
        diff = schema_diff(engine)
        # Whole tables that later migrations add are fine (the upgrade creates them); anything else is an unknown schema.
        other = [d for d in diff if not (isinstance(d, tuple) and d[0] == "add_table")]
        if other:
            raise SchemaMismatch("the database has application tables but no migration history and its schema differs from the "
                                 f"models ({len(other)} difference(s), first: {other[0]!r}); back it up and migrate it by hand")
        # Already matches the models exactly -> it is at head; missing only later tables -> it is at the baseline.
        with engine.begin() as conn:
            command.stamp(_config(conn), "head" if not diff else _baseline())
    with engine.begin() as conn:
        command.upgrade(_config(conn), revision)
    return current_revision(engine) or ""


def _baseline() -> str:
    scripts = ScriptDirectory(str(MIGRATIONS))
    return next(s.revision for s in scripts.walk_revisions() if s.down_revision is None)


def downgrade(engine: Engine, revision: str) -> None:
    with engine.begin() as conn:
        command.downgrade(_config(conn), revision)
