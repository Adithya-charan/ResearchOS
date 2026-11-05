"""Project-scoped memory with provenance. Writes are append-only: history is never overwritten."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_engine.state.store import Store


class MemoryManager:
    def __init__(self, store: Store):
        self.store = store

    def bind(self, project_id: str, run_id: str, task_id: str, emit: Callable[[str, dict], None] | None = None) -> BoundMemory:
        return BoundMemory(self.store, project_id, run_id, task_id, emit)


class BoundMemory:
    """What one task sees: reads the project's latest values, stamps writes with (run, task)."""

    def __init__(self, store: Store, project_id: str, run_id: str, task_id: str, emit: Callable[[str, dict], None] | None):
        self._store, self._project, self._run, self._task, self._emit = store, project_id, run_id, task_id, emit

    def put(self, key: str, value: Any) -> None:
        self._store.memory_put(self._project, key, value, self._run, self._task)
        if self._emit:
            self._emit("memory_write", {"key": key})

    def get(self, key: str, default: Any = None) -> Any:
        row = self._store.memory_get(self._project, key)
        return row["value"] if row else default

    def list(self, prefix: str = "") -> dict[str, Any]:
        return {r["key"]: r["value"] for r in self._store.memory_list(self._project, prefix)}
