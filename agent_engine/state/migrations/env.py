"""Alembic environment for the agent engine's own database (separate from the API's). The caller passes a live connection."""
from alembic import context

config = context.config


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        raise RuntimeError("engine migrations are applied by agent_engine.state.migrate, which supplies the connection")
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("offline (SQL script) migrations are not supported")
run_migrations_online()
