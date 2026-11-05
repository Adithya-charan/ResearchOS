"""Server-Sent Events plumbing. Streams are built on polling the same durable state the REST endpoints read (engine event log,
training metrics file, experiment status): there is no separate message bus, so a missed event can always be re-read by
reconnecting with `Last-Event-ID`. Every stream is bounded: per-user and global caps, a maximum lifetime, a heartbeat, and a
periodic re-check that the viewer's session is still valid."""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import anyio
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse


HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff"}
_NAME = re.compile(r"[^a-z0-9_.-]")


@dataclass
class Item:
    event: str
    data: object
    id: int | None = None


def sse_format(item: Item) -> str:
    """One SSE frame. Event names are restricted to a safe alphabet and data is always one line of JSON, so nothing a user or
    an experiment printed can add fields or frames to the stream."""
    name = _NAME.sub("_", item.event.lower())[:40] or "message"
    lines = []
    if item.id is not None:
        lines.append(f"id: {int(item.id)}")
    lines.append(f"event: {name}")
    lines.append("data: " + json.dumps(item.data, ensure_ascii=False, default=str).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    return "\n".join(lines) + "\n\n"


class StreamGate:
    """Caps concurrent streams per user and in total (each one holds a connection and polls state)."""

    def __init__(self, per_user: int, total: int, max_age_s: float):
        self.per_user, self.total, self.max_age = per_user, total, max_age_s
        self._slots: dict[int, tuple[str, float]] = {}
        self._n = 0
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:  # a slot whose generator never ran is reclaimed after the maximum stream lifetime
        for k in [k for k, (_, t) in self._slots.items() if now - t > self.max_age + 30]:
            del self._slots[k]

    def acquire(self, user_id: str) -> int:
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            if sum(1 for u, _ in self._slots.values() if u == user_id) >= self.per_user:
                raise HTTPException(429, f"at most {self.per_user} live streams per user; close one first")
            if len(self._slots) >= self.total:
                raise HTTPException(429, "the server is at its live-stream capacity; try again shortly")
            self._n += 1
            self._slots[self._n] = (user_id, now)
            return self._n

    def release(self, slot: int) -> None:
        with self._lock:
            self._slots.pop(slot, None)

    def active(self, user_id: str | None = None) -> int:
        with self._lock:
            return sum(1 for u, _ in self._slots.values() if user_id in (None, u))


Poll = Callable[[], tuple[list[Item], bool]]  # -> (new items, finished). Called in a worker thread; must drain before finishing.


async def event_stream(poll: Poll, *, is_disconnected: Callable[[], Awaitable[bool]], still_valid: Callable[[], bool],
                       interval: float, heartbeat: float, max_seconds: float, reauth_every: float,
                       on_close: Callable[[], None] = lambda: None) -> AsyncIterator[str]:
    start = last_sent = last_auth = time.monotonic()
    try:
        yield "retry: 3000\n\n"
        while True:
            if await is_disconnected():
                return
            items, finished = await anyio.to_thread.run_sync(poll)
            for it in items:
                yield sse_format(it)
            if items:
                last_sent = time.monotonic()
            if finished:
                yield sse_format(Item("end", {"reason": "finished"}))
                return
            now = time.monotonic()
            if now - start > max_seconds:
                yield sse_format(Item("end", {"reason": "max_duration", "hint": "reconnect with Last-Event-ID to continue"}))
                return
            if now - last_auth > reauth_every:
                last_auth = now
                if not await anyio.to_thread.run_sync(still_valid):
                    yield sse_format(Item("end", {"reason": "session_ended"}))
                    return
            if now - last_sent > heartbeat:
                last_sent = now
                yield ": keepalive\n\n"
            await asyncio.sleep(interval)
    finally:
        on_close()


def session_checker(request: Request, session_id: str) -> Callable[[], bool]:
    from api.db import utcnow
    from api.models import SessionRow
    maker = request.app.state.sessionmaker

    def check() -> bool:
        with maker() as db:
            row = db.get(SessionRow, session_id)
            return row is not None and row.expires_at > utcnow()
    return check


def start_stream(request: Request, auth, poll: Poll) -> StreamingResponse:
    """Authorisation has already happened (the route looked up the object for this user). This registers the stream slot and
    returns the response; the slot is released when the generator ends or the client disconnects."""
    s = request.app.state.settings
    gate: StreamGate = request.app.state.streams
    slot = gate.acquire(auth.user.id)
    body = event_stream(poll, is_disconnected=request.is_disconnected, still_valid=session_checker(request, auth.session.id),
                        interval=s.sse_interval_s, heartbeat=s.sse_heartbeat_s, max_seconds=s.sse_max_seconds,
                        reauth_every=s.sse_reauth_s, on_close=lambda: gate.release(slot))
    return StreamingResponse(body, media_type="text/event-stream", headers=HEADERS)


def resume_point(request: Request, after: int) -> int:
    """`Last-Event-ID` (sent automatically by EventSource on reconnect) wins over the `after` query parameter."""
    raw = request.headers.get("last-event-id", "")
    return int(raw) if raw.isdigit() else after
