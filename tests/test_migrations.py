"""Schema migrations: the migration history must always reproduce the models exactly, run both ways, adopt databases created
before migrations existed (only when they match), and be applied by the app at startup."""
import pytest
from sqlalchemy import create_engine, inspect, text

from api import migrate
from api.app import create_app
from api.config import Settings
from api.db import Base


def file_engine(tmp_path, name="app.db"):
    return create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


def test_migrations_reproduce_the_models_exactly(tmp_path):
    e = file_engine(tmp_path)
    assert migrate.upgrade(e) == migrate.head_revision()
    assert set(inspect(e).get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
    # If this fails, a model changed without a new migration: run `alembic revision --autogenerate -m "..."`.
    assert migrate.schema_diff(e) == []


def test_upgrade_is_idempotent_and_downgrade_round_trips(tmp_path):
    e = file_engine(tmp_path)
    migrate.upgrade(e)
    assert migrate.upgrade(e) == migrate.head_revision()
    migrate.downgrade(e, "base")
    assert set(inspect(e).get_table_names()) <= {"alembic_version"}
    assert migrate.upgrade(e) == migrate.head_revision()


def test_a_database_made_before_migrations_is_adopted_with_its_data_intact(tmp_path):
    e = file_engine(tmp_path)
    Base.metadata.create_all(e)  # how every database was created before this increment
    with e.begin() as c:
        c.execute(text("INSERT INTO users (id, email, password_hash, display_name, created_at, is_active) "
                       "VALUES ('usr_old', 'old@example.org', 'h', '', '2026-01-01 00:00:00', 1)"))
    assert migrate.current_revision(e) is None
    assert migrate.upgrade(e) == migrate.head_revision()
    with e.connect() as c:
        assert c.execute(text("SELECT email FROM users")).scalar() == "old@example.org"


def test_a_database_that_differs_from_the_models_is_refused_not_altered(tmp_path):
    e = file_engine(tmp_path)
    Base.metadata.create_all(e)
    with e.begin() as c:
        c.execute(text("ALTER TABLE users ADD COLUMN surprise TEXT"))
    with pytest.raises(migrate.SchemaMismatch, match="differs from the models"):
        migrate.upgrade(e)
    assert migrate.current_revision(e) is None  # nothing was stamped
    with e.connect() as c:
        assert "surprise" in [r[1] for r in c.execute(text("PRAGMA table_info(users)"))]


def test_app_migrates_a_file_database_at_startup(tmp_path):
    s = Settings(database_url=f"sqlite:///{(tmp_path / 'new' / 'app.db').as_posix()}", engine_db=":memory:", data_dir=tmp_path / "d",
                 corpus_isolate=False)
    app = create_app(s)
    assert migrate.current_revision(app.state.db_engine) == migrate.head_revision()


def test_app_refuses_to_start_on_an_outdated_schema_when_auto_migrate_is_off(tmp_path):
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    base = dict(database_url=url, engine_db=":memory:", data_dir=tmp_path / "d", corpus_isolate=False)
    with pytest.raises(migrate.SchemaMismatch, match="alembic upgrade head"):
        create_app(Settings(**base, auto_migrate=False))
    migrate.upgrade(file_engine(tmp_path))
    create_app(Settings(**base, auto_migrate=False))  # at head: starts normally
