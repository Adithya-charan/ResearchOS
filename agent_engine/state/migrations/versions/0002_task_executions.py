"""task execution records (one row per attempt)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-06
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

UP: list[str] = [
    """CREATE TABLE IF NOT EXISTS task_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    task_id TEXT NOT NULL REFERENCES tasks(id),
    agent_type TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL, finished_at TEXT,
    duration_s REAL,
    cpu_process_s REAL,
    error TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_exec_task ON task_executions(task_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_exec_run ON task_executions(run_id, id)",
]
DOWN: list[str] = ["DROP TABLE IF EXISTS task_executions"]


def upgrade() -> None:
    for statement in UP:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWN:
        op.execute(statement)
