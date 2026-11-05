"""Event bus. The database is the bus: every event is persisted first, so a UI can replay history and then follow
live, from any process. Server-Sent-Events formatting is provided for the API layer."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from agent_engine.contracts import TERMINAL_RUN, RunStatus
from agent_engine.state.store import Store


class EventBus:
    def __init__(self, store: Store):
        self.store = store

    def publish(self, run_id: str, event_type: str, payload: dict | None = None, task_id: str | None = None) -> int:
        return self.store.add_event(run_id, event_type, payload, task_id)

    async def stream(
        self,
        run_id: str,
        after_id: int = 0,
        poll_interval: float = 0.05,
        until: frozenset[RunStatus] = frozenset(TERMINAL_RUN | {RunStatus.PAUSED}),
    ) -> AsyncIterator[dict]:
        """Yield every event after `after_id` in order; finish once the run reaches a status in `until`
        and the backlog is drained. Safe to call before the run starts."""
        last = after_id
        while True:
            status = RunStatus(self.store.get_run(run_id)["status"])  # read BEFORE draining so no event is missed
            batch = self.store.list_events(run_id, last)
            for ev in batch:
                last = ev["id"]
                yield ev
            if not batch:
                if status in until:
                    return
                await asyncio.sleep(poll_interval)


def to_sse(event: dict) -> str:
    """Format one event as a Server-Sent-Events frame (id lets browsers resume with Last-Event-ID)."""
    data = json.dumps({"task_id": event["task_id"], "payload": event["payload"], "created_at": event["created_at"]})
    return f"id: {event['id']}\nevent: {event['type']}\ndata: {data}\n\n"
