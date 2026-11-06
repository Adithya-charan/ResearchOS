"""Allowed status transitions for runs, tasks and queue jobs. The store enforces the run and task tables on every status write, so a
bug (or a stale worker) cannot move a finished run back to running or resurrect a cancelled task."""
from __future__ import annotations


class InvalidTransition(ValueError):
    pass


RUN: dict[str, set[str]] = {
    "pending": {"running", "cancelled", "failed"},
    "running": {"paused", "completed", "failed", "cancelled", "stopped_limit"},
    "paused": {"running", "cancelled", "failed"},
    "completed": set(), "failed": set(), "cancelled": set(), "stopped_limit": set(),
}

TASK: dict[str, set[str]] = {
    "pending": {"running", "skipped", "cancelled", "awaiting_approval", "failed"},
    "running": {"succeeded", "failed", "pending", "cancelled"},  # pending = retry or crash recovery
    "awaiting_approval": {"pending", "rejected", "cancelled", "skipped"},
    "succeeded": set(), "failed": set(), "skipped": set(), "cancelled": set(), "rejected": set(),
}

# Queue jobs (api/models.ResearchJob): queued -> running -> done|failed|cancelled|paused; running -> queued on retry/hand-back.
JOB: dict[str, set[str]] = {
    "queued": {"running", "cancelled", "paused", "failed"},
    "running": {"done", "failed", "cancelled", "paused", "queued"},
    "paused": {"queued", "cancelled"},
    "done": set(), "failed": {"queued"}, "cancelled": set(),  # failed -> queued only through an operator's explicit recover
}


def check(kind: str, table: dict[str, set[str]], old: str, new: str) -> None:
    if old == new:
        return
    if old not in table:
        raise InvalidTransition(f"{kind} is in unknown status '{old}'")
    if new not in table[old]:
        raise InvalidTransition(f"{kind} cannot go from '{old}' to '{new}'")


def check_run(old: str, new: str) -> None:
    check("run", RUN, old, new)


def check_task(old: str, new: str) -> None:
    check("task", TASK, old, new)


def check_job(old: str, new: str) -> None:
    check("job", JOB, old, new)
