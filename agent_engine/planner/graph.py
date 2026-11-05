"""Task dependency graph: validation, ordering, readiness."""
from __future__ import annotations

from collections import deque

from agent_engine.contracts import DEAD_TASK, TaskStatus
from agent_engine.errors import GraphError


class TaskGraph:
    """Operates on task rows (dicts with id, depends_on, status, not_before)."""

    def __init__(self, tasks: list[dict]):
        self.tasks = {t["id"]: t for t in tasks}
        if len(self.tasks) != len(tasks):
            raise GraphError("duplicate task ids")
        self.validate()

    def validate(self) -> None:
        for t in self.tasks.values():
            for dep in t["depends_on"]:
                if dep not in self.tasks:
                    raise GraphError(f"task {t['id']} depends on unknown task {dep}")
                if dep == t["id"]:
                    raise GraphError(f"task {t['id']} depends on itself")
        self.topological_order()  # raises on cycles

    def topological_order(self) -> list[str]:
        indegree = {tid: len(set(t["depends_on"])) for tid, t in self.tasks.items()}
        children: dict[str, list[str]] = {tid: [] for tid in self.tasks}
        for tid, t in self.tasks.items():
            for dep in set(t["depends_on"]):
                children[dep].append(tid)
        queue = deque(sorted(tid for tid, n in indegree.items() if n == 0))
        order: list[str] = []
        while queue:
            tid = queue.popleft()
            order.append(tid)
            for child in sorted(children[tid]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.tasks):
            raise GraphError("dependency cycle detected")
        return order

    def ready(self, now: float) -> list[dict]:
        """Pending tasks whose dependencies have all succeeded and whose retry back-off has elapsed."""
        out = []
        for t in self.tasks.values():
            if t["status"] != TaskStatus.PENDING.value or t["not_before"] > now:
                continue
            if all(self.tasks[d]["status"] == TaskStatus.SUCCEEDED.value for d in t["depends_on"]):
                out.append(t)
        return out

    def doomed(self) -> list[tuple[dict, str]]:
        """Pending tasks that can never run because a dependency failed / was rejected / skipped / cancelled."""
        out = []
        dead = {d.value for d in DEAD_TASK}
        for t in self.tasks.values():
            if t["status"] != TaskStatus.PENDING.value:
                continue
            for dep in t["depends_on"]:
                if self.tasks[dep]["status"] in dead:
                    out.append((t, dep))
                    break
        return out
