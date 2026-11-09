"""Durable research-job queue and worker.

* `JobQueue` is the only code that changes a job row. Claiming is an atomic conditional UPDATE (portable to SQLite and PostgreSQL), so
  two workers - threads or separate processes - can never execute the same run.
* A running job holds a lease that its worker renews with a heartbeat. If the worker dies, the lease expires and the job is
  re-queued with exponential backoff (up to `job_max_attempts` claims, then it fails visibly). The engine then re-queues the tasks the
  dead worker left mid-flight (`engine.recover`), so finished work is never repeated.
* `Worker` claims jobs and runs the agent engine for them; `lifecycle()` derives the user-facing phase from persisted state."""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
import uuid
from datetime import timedelta

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from agent_engine.contracts import RunStatus
from api.db import utcnow
from api.models import ResearchJob, Run
from api.runtime import NoCorpus

log = logging.getLogger("researchos.jobs")
ACTIVE = ("queued", "running")
BACKOFF_CAP_S = 60.0
ENGINE_TERMINAL = {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value, RunStatus.STOPPED_LIMIT.value}


def new_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


class JobQueue:
    def __init__(self, sessionmaker, settings, store=None):
        self.sessions, self.settings, self.store = sessionmaker, settings, store

    def _event(self, job: ResearchJob | dict, kind: str, **payload) -> None:
        if self.store is not None:
            eid = job["engine_run_id"] if isinstance(job, dict) else job.engine_run_id
            try:
                self.store.add_event(eid, kind, payload)
            except Exception:  # an event must never break queue bookkeeping
                log.exception("could not record %s", kind)

    # ------------------------------------------------------------------ producers (API side)
    def enqueue(self, run: Run, force: bool = False) -> bool:
        """Queue (or re-queue) a run. False when it is already queued/running, or finished. `force` also revives a failed job
        whose engine run never finished (operator recovery)."""
        with self.sessions() as db:
            job = db.scalar(select(ResearchJob).where(ResearchJob.run_id == run.id))
            if job is None:
                job = ResearchJob(run_id=run.id, engine_run_id=run.engine_run_id, owner_id=run.owner_id,
                                  max_attempts=self.settings.job_max_attempts)
                db.add(job)
                try:
                    db.commit()
                except IntegrityError:  # another process queued this run between our SELECT and INSERT (run_id is unique)
                    db.rollback()
                    return False
                self._event(job, "job_queued", job_id=job.id)
                return True
            revivable = job.status == "paused" or (force and job.status == "failed")
            if not revivable:
                return False
            if force:
                job.attempts = 0
            job.status, job.available_at, job.error, job.finished_at = "queued", utcnow(), "", None
            job.worker_id, job.lease_expires_at = "", None
            db.commit()
            self._event(job, "job_queued", job_id=job.id, requeued=True)
            return True

    def get(self, engine_run_id: str) -> dict | None:
        with self.sessions() as db:
            j = db.scalar(select(ResearchJob).where(ResearchJob.engine_run_id == engine_run_id))
            return None if j is None else _view(j)

    def is_active(self, engine_run_id: str) -> bool:
        with self.sessions() as db:
            return db.scalar(select(ResearchJob.id).where(ResearchJob.engine_run_id == engine_run_id,
                                                          ResearchJob.status.in_(ACTIVE))) is not None

    def active_count(self, owner_id: str) -> int:
        with self.sessions() as db:
            return db.scalar(select(func.count()).select_from(ResearchJob).where(ResearchJob.owner_id == owner_id,
                                                                                 ResearchJob.status.in_(ACTIVE))) or 0

    def cancel_if_idle(self, engine_run_id: str) -> None:
        """A cancelled run that no worker holds is closed here; a running one is closed by its worker when the engine stops."""
        with self.sessions() as db:
            db.execute(update(ResearchJob).where(ResearchJob.engine_run_id == engine_run_id, ResearchJob.status.in_(("queued", "paused")))
                       .values(status="cancelled", finished_at=utcnow(), worker_id="", lease_expires_at=None))
            db.commit()

    # ------------------------------------------------------------------ consumers (worker side)
    def claim(self, worker_id: str) -> dict | None:
        now = utcnow()
        with self.sessions() as db:
            for _ in range(5):  # another worker may win the race for a candidate: look at the next one
                pick = (select(ResearchJob.id).where(ResearchJob.status == "queued", ResearchJob.available_at <= now)
                        .order_by(ResearchJob.available_at, ResearchJob.enqueued_at).limit(1))
                if db.get_bind().dialect.name == "postgresql":
                    # Lock the candidate row and skip rows other workers have locked, so concurrent workers each get a different job
                    # instead of queueing behind one another. (SQLAlchemy ignores FOR UPDATE on SQLite.)
                    pick = pick.with_for_update(skip_locked=True)
                cand = db.scalar(pick)
                if cand is None:
                    return None
                won = db.execute(update(ResearchJob).where(ResearchJob.id == cand, ResearchJob.status == "queued")
                                 .values(status="running", worker_id=worker_id, attempts=ResearchJob.attempts + 1, started_at=now,
                                         heartbeat_at=now, lease_expires_at=now + timedelta(seconds=self.settings.job_lease_s)))
                db.commit()
                if won.rowcount == 1:
                    job = db.get(ResearchJob, cand)
                    db.refresh(job)
                    self._event(job, "job_claimed", worker=worker_id, attempt=job.attempts)
                    return _view(job)
        return None

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        """Renew the lease. False means this worker no longer owns the job and must stop."""
        now = utcnow()
        with self.sessions() as db:
            r = db.execute(update(ResearchJob).where(ResearchJob.id == job_id, ResearchJob.worker_id == worker_id,
                                                     ResearchJob.status == "running")
                           .values(heartbeat_at=now, lease_expires_at=now + timedelta(seconds=self.settings.job_lease_s)))
            db.commit()
            return r.rowcount == 1

    def finish(self, job_id: str, worker_id: str, status: str, error: str = "") -> bool:
        with self.sessions() as db:
            r = db.execute(update(ResearchJob).where(ResearchJob.id == job_id, ResearchJob.worker_id == worker_id,
                                                     ResearchJob.status == "running")
                           .values(status=status, error=error[:2000], finished_at=utcnow() if status != "paused" else None,
                                   lease_expires_at=None))
            db.commit()
            if r.rowcount == 1:
                job = db.get(ResearchJob, job_id)
                self._event(job, "job_finished", status=status, error=error[:300])
            return r.rowcount == 1

    def requeue(self, job_id: str, worker_id: str, reason: str, penalty: bool = True, only_if_expired: bool = False) -> str:
        """Give a job back (shutdown, lost lease, crash). Returns the new status: queued, or failed once attempts are used up."""
        return self._requeue(job_id, worker_id, reason, penalty, only_if_expired)[0]

    def _requeue(self, job_id: str, worker_id: str, reason: str, penalty: bool, only_if_expired: bool) -> tuple[str, bool]:
        """As `requeue`, also reporting whether *this call* changed the job (a racing caller that finds it already given back did not)."""
        with self.sessions() as db:
            # Lock the row for the whole decision: a heartbeat (a single UPDATE) then waits instead of renewing a lease we are
            # about to take away, and two reapers cannot both requeue the same job.
            job = db.scalar(select(ResearchJob).where(ResearchJob.id == job_id).with_for_update()
                            .execution_options(populate_existing=True))
            if job is None or job.worker_id != worker_id or job.status != "running":
                db.rollback()
                return (job.status if job else "gone"), False
            if only_if_expired and not (job.lease_expires_at is not None and job.lease_expires_at < utcnow()):
                db.rollback()  # the worker renewed its lease after we looked: it is alive
                return "running", False
            if not penalty:
                job.attempts = max(0, job.attempts - 1)
            if job.attempts >= job.max_attempts:
                job.status, job.error, job.finished_at = "failed", f"gave up after {job.attempts} attempt(s): {reason}"[:2000], utcnow()
            else:
                delay = min(BACKOFF_CAP_S, self.settings.job_backoff_s * 2 ** max(0, job.attempts - 1)) if penalty else 0.0
                job.status, job.error = "queued", reason[:2000]
                job.available_at = utcnow() + timedelta(seconds=delay)
            job.worker_id, job.lease_expires_at = "", None
            db.commit()
            self._event(job, "job_requeued" if job.status == "queued" else "job_failed", reason=reason[:300], attempts=job.attempts)
            return job.status, True

    def reap_expired(self) -> int:
        """Re-queue (or fail) jobs whose worker stopped heartbeating."""
        now = utcnow()
        n = 0
        with self.sessions() as db:
            stale = db.scalars(select(ResearchJob).where(ResearchJob.status == "running", ResearchJob.lease_expires_at < now)).all()
            ids = [(j.id, j.worker_id) for j in stale]
        for jid, wid in ids:
            status, acted = self._requeue(jid, wid, "worker stopped heartbeating (lease expired)", True, True)
            if not acted:
                continue  # the worker renewed its lease, or a competing reaper (or the worker's own finish) got there first
            if status == "failed":
                self._fail_engine_run(jid)
            n += 1
        return n

    def _fail_engine_run(self, job_id: str) -> None:
        if self.store is None:
            return
        with self.sessions() as db:
            job = db.get(ResearchJob, job_id)
            if job is None:
                return
            eid, err = job.engine_run_id, job.error
        run = self.store.get_run(eid)
        if run["status"] not in ENGINE_TERMINAL:
            self.store.update_run(eid, status="failed", stop_reason="worker_lost")
            self.store.add_event(eid, "run_finished", {"status": "failed", "stop_reason": "worker_lost", "detail": err[:300]})


def _view(j: ResearchJob) -> dict:
    return {"id": j.id, "run_id": j.run_id, "engine_run_id": j.engine_run_id, "owner_id": j.owner_id, "status": j.status,
            "worker_id": j.worker_id, "attempts": j.attempts, "max_attempts": j.max_attempts, "error": j.error,
            "available_at": j.available_at, "lease_expires_at": j.lease_expires_at, "enqueued_at": j.enqueued_at,
            "started_at": j.started_at, "finished_at": j.finished_at}


# ------------------------------------------------------------------ user-facing phase
def lifecycle(job: dict | None, run: dict, tasks: list[dict]) -> str:
    """created | queued | planning | running | reviewing | paused | completed | failed | cancelled - derived from persisted state only."""
    status = run["status"]
    if status == RunStatus.COMPLETED.value:
        return "completed"
    if status == RunStatus.CANCELLED.value:
        return "cancelled"
    if status in (RunStatus.FAILED.value, RunStatus.STOPPED_LIMIT.value):
        return "failed"
    if job is None:
        return "created"
    if job["status"] == "failed":
        return "failed"
    if job["status"] == "cancelled":
        return "cancelled"
    if job["status"] == "queued":
        return "queued"
    if job["status"] == "paused" or status == RunStatus.PAUSED.value or run.get("control") == "pause":
        return "paused"
    by = {}
    for t in tasks:
        by.setdefault(t["agent_type"], []).append(t["status"])
    if not any(s == "succeeded" for s in by.get("supervisor", [])):
        return "planning"
    if any(s in ("running", "succeeded") for s in by.get("verifier", []) + by.get("synthesis", [])):
        return "reviewing"
    return "running"


# ------------------------------------------------------------------ worker
class Worker:
    """Claims queued research jobs and runs the agent engine for them (up to `worker_concurrency` at once)."""

    def __init__(self, settings, runtime, queue: JobQueue, sessionmaker, worker_id: str | None = None):
        self.settings, self.runtime, self.queue, self.sessions = settings, runtime, queue, sessionmaker
        self.worker_id = worker_id or new_worker_id()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._runs: dict[str, threading.Thread] = {}
        self._cancel_hooks: dict[str, callable] = {}
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self.loop, name=f"worker-{self.worker_id}", daemon=True)
            self._thread.start()

    def ensure_started(self) -> None:
        """Embedded mode only: the API process hosts the worker thread (started on first use, so apps created and never served stay idle)."""
        if self.settings.embedded_worker and not self._stop.is_set():
            self.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Graceful shutdown: stop claiming, interrupt running jobs, hand them back (no attempt penalty) so another worker resumes them."""
        self._stop.set()
        for hook in list(self._cancel_hooks.values()):
            hook()
        deadline = time.time() + timeout
        for t in [self._thread, *list(self._runs.values())]:
            if t is not None and t.is_alive():
                t.join(max(0.1, deadline - time.time()))

    @property
    def running(self) -> int:
        return sum(1 for t in self._runs.values() if t.is_alive())

    def loop(self) -> None:
        last_reap = 0.0
        while not self._stop.is_set():
            try:
                if time.monotonic() - last_reap > max(1.0, self.settings.job_lease_s / 3):
                    self.queue.reap_expired()
                    last_reap = time.monotonic()
                self._runs = {k: t for k, t in self._runs.items() if t.is_alive()}
                claimed = False
                if len(self._runs) < self.settings.worker_concurrency:
                    job = self.queue.claim(self.worker_id)
                    if job is not None:
                        claimed = True
                        t = threading.Thread(target=self._execute, args=(job,), name=f"job-{job['id']}", daemon=True)
                        self._runs[job["id"]] = t
                        t.start()
                if not claimed:
                    self._stop.wait(self.settings.job_poll_s)
            except Exception:
                log.exception("worker loop error")
                self._stop.wait(1.0)

    # -- one job -------------------------------------------------------
    def _execute(self, job: dict) -> None:
        jid, eid = job["id"], job["engine_run_id"]
        interrupted = {"reason": None}
        loop_box: dict = {}
        beat_stop = threading.Event()

        def abandon(reason: str) -> None:
            interrupted["reason"] = reason
            loop, task = loop_box.get("loop"), loop_box.get("task")
            if loop is not None and task is not None:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:  # the run already finished and its loop closed: nothing left to interrupt
                    pass

        def beat() -> None:
            while not beat_stop.wait(max(0.05, self.settings.job_lease_s / 3)):
                if not self.queue.heartbeat(jid, self.worker_id):
                    abandon("lease_lost")
                    return

        self._cancel_hooks[jid] = lambda: abandon("shutdown")
        threading.Thread(target=beat, name=f"beat-{jid}", daemon=True).start()
        try:
            with self.sessions() as db:
                run = db.get(Run, job["run_id"])
                project_id = run.project_id if run else None
            if project_id is None:
                self.queue.finish(jid, self.worker_id, "failed", "the run no longer exists")
                return
            engine = self.runtime.engine(project_id)  # raises NoCorpus if the project has no searchable documents

            async def main() -> None:
                loop_box["loop"], loop_box["task"] = asyncio.get_running_loop(), asyncio.current_task()
                await engine.run(eid)

            asyncio.run(main())
            self._finalise(job)
        except asyncio.CancelledError:
            self._handed_back(job, interrupted["reason"] or "interrupted")
        except NoCorpus as e:  # deterministic: retrying cannot help
            self._fail_permanently(job, str(e))
        except Exception as e:
            log.exception("job %s crashed", jid)
            self._after_crash(job, f"{type(e).__name__}: {e}")
        finally:
            beat_stop.set()
            self._cancel_hooks.pop(jid, None)

    def _handed_back(self, job: dict, reason: str) -> None:
        if reason == "lease_lost":  # someone else owns it now; do not touch the row
            return
        self.queue.requeue(job["id"], self.worker_id, f"worker {reason}", penalty=False)

    def _after_crash(self, job: dict, error: str) -> None:
        status = self.queue.requeue(job["id"], self.worker_id, error)
        if status == "failed":
            self.queue._fail_engine_run(job["id"])

    def _fail_permanently(self, job: dict, error: str) -> None:
        self.queue.finish(job["id"], self.worker_id, "failed", error)
        self.queue._fail_engine_run(job["id"])

    def _finalise(self, job: dict) -> None:
        run = self.runtime.store.get_run(job["engine_run_id"])
        st = run["status"]
        if st == RunStatus.COMPLETED.value:
            self.queue.finish(job["id"], self.worker_id, "done")
        elif st == RunStatus.CANCELLED.value:
            self.queue.finish(job["id"], self.worker_id, "cancelled")
        elif st in (RunStatus.FAILED.value, RunStatus.STOPPED_LIMIT.value):
            self.queue.finish(job["id"], self.worker_id, "failed", run.get("stop_reason") or st)
        else:  # paused by the user or waiting for approval: resumable, holds no lease
            self.queue.finish(job["id"], self.worker_id, "paused")
