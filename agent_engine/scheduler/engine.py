"""The scheduler: drives a run's task graph to completion with concurrency, retries, timeouts, policy checks,
approval gates, stop controls, pause/resume/cancel and crash recovery. All state lives in the Store, so a run
survives a process restart: call `run()` again and it carries on from what was persisted."""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent_engine.agents.base import AgentContext
from agent_engine.agents.specs import spec_for
from agent_engine.contracts import (
    TERMINAL_RUN,
    AgentOutput,
    RunLimits,
    RunResult,
    RunStatus,
    TaskSpec,
    TaskStatus,
)
from agent_engine.errors import NonRetryableError, OutputValidationError, RunStateError
from agent_engine.memory.manager import MemoryManager
from agent_engine.messaging.events import EventBus
from agent_engine.model_gateway import ModelGateway
from agent_engine.planner.graph import TaskGraph
from agent_engine.planner.planner import ROOT_AGENT, expand, initial_plan
from agent_engine.policies.policy import PolicyEngine
from agent_engine.registry.registry import AgentRegistry
from agent_engine.scheduler.retry import RetryPolicy
from agent_engine.scheduler.summary import build_summary
from agent_engine.state.store import Store, new_id, utcnow
from agent_engine.tools.registry import ToolRegistry


@dataclass
class _RunCtl:
    """Per-call run state shared with task coroutines."""

    run_id: str
    project_id: str
    question: str
    limits: RunLimits
    elapsed_base: float
    t0: float
    clock: Any
    abandon: bool = False  # simulate/handle process death: leave the database exactly as it is
    cancel_reason: str | None = None
    runtime_cut: bool = False  # a task was stopped because the run's runtime budget ran out
    execs: dict = field(default_factory=dict)  # task id -> open execution record

    def elapsed(self) -> float:
        return self.elapsed_base + (self.clock() - self.t0)


def _task_payload(row: dict) -> dict:
    return {"agent_type": row["agent_type"], "title": row["title"], "depends_on": row["depends_on"],
            "iteration": row["iteration"], "parent_task_id": row["parent_task_id"]}


class ResearchEngine:
    def __init__(
        self,
        store: Store,
        registry: AgentRegistry,
        tools: ToolRegistry,
        policy: PolicyEngine,
        *,
        model: ModelGateway | None = None,
        resolver: Any = None,  # object with .check(EvidenceReference) -> str | None
        poll_interval: float = 0.05,
        clock=time.monotonic,
        wall=time.time,
    ):
        self.store, self.registry, self.tools, self.policy = store, registry, tools, policy
        self.model, self.resolver = model, resolver
        self.poll_interval, self.clock, self.wall = poll_interval, clock, wall
        self.memory = MemoryManager(store)
        self.bus = EventBus(store)

    # ------------------------------------------------------------------ setup
    def create_project(self, title: str, question: str) -> str:
        return self.store.create_project(title, question)

    def create_run(self, project_id: str, question: str, limits: RunLimits | None = None) -> str:
        if ROOT_AGENT not in self.registry:
            raise ValueError(f"a '{ROOT_AGENT}' agent must be registered to start a run")
        limits = limits or RunLimits()
        run_id = self.store.create_run(project_id, question, limits.model_dump())
        plan = initial_plan(question)
        self.store.add_tasks(run_id, plan)
        self._event(run_id, "run_created", {"question": question, "limits": limits.model_dump()})
        for t in plan:
            self._event(run_id, "task_created", _task_payload(t), t["id"])
        return run_id

    def _event(self, run_id: str, event_type: str, payload: dict | None = None, task_id: str | None = None) -> None:
        self.bus.publish(run_id, event_type, payload, task_id)

    def events(self, run_id: str, after_id: int = 0, **kw) -> AsyncIterator[dict]:
        return self.bus.stream(run_id, after_id, **kw)

    # ------------------------------------------------------------------ controls (safe to call from another process)
    def _set_control(self, run_id: str, control: str, event: str) -> None:
        run = self.store.get_run(run_id)
        if RunStatus(run["status"]) in TERMINAL_RUN:
            raise RunStateError(f"run {run_id} is already {run['status']}")
        self.store.update_run(run_id, control=control)
        self._event(run_id, event, {})

    def pause(self, run_id: str) -> None:
        self._set_control(run_id, "pause", "pause_requested")

    def resume(self, run_id: str) -> None:
        """Clear a pause request. Call `run()` again to continue."""
        self._set_control(run_id, "run", "resume_requested")

    def cancel(self, run_id: str) -> None:
        self._set_control(run_id, "cancel", "cancel_requested")
        if RunStatus(self.store.get_run(run_id)["status"]) in (RunStatus.PENDING, RunStatus.PAUSED):
            run = self.store.get_run(run_id)  # nothing is executing, so finalise right away
            rc = _RunCtl(run_id, run["project_id"], run["question"], RunLimits(**run["limits"]),
                         run["elapsed_s"], self.clock(), self.clock)
            self._close_open_tasks(run_id, RunStatus.CANCELLED, "cancel_requested")
            self._finish(run_id, rc, RunStatus.CANCELLED, "cancel_requested")

    def decide_approval(self, approval_id: str, approved: bool, reviewer_id: str, notes: str = "") -> None:
        approval = self.store.get_approval(approval_id)
        self.store.decide_approval(approval_id, "approved" if approved else "rejected", reviewer_id, notes)
        self._event(approval["run_id"], "approval_decided",
                    {"approval_id": approval_id, "decision": "approved" if approved else "rejected", "reviewer": reviewer_id},
                    approval["task_id"])
        self._sync_approvals(approval["run_id"])

    def result(self, run_id: str) -> RunResult:
        run = self.store.get_run(run_id)
        return RunResult(run_id=run_id, status=RunStatus(run["status"]), stop_reason=run["stop_reason"],
                         task_counts=self.store.task_counts(run_id), summary=run["summary"] or {})

    # ------------------------------------------------------------------ recovery
    def recover(self, run_id: str) -> list[str]:
        """Tasks left 'running' by a dead process are re-queued. The interrupted attempt counts against the
        retry budget so a task that crashes the worker every time cannot loop forever."""
        run = self.store.get_run(run_id)
        policy = RetryPolicy(RunLimits(**run["limits"]).max_retries, 0.0)
        recovered = []
        self.store.abandon_open_executions(run_id)
        for t in self.store.list_tasks(run_id):
            if t["status"] != TaskStatus.RUNNING.value:
                continue
            if policy.can_retry(t["retry_count"]):
                self.store.update_task(t["id"], status="pending", retry_count=t["retry_count"] + 1, not_before=0,
                                       error="interrupted (worker stopped mid-task)")
            else:
                self.store.update_task(t["id"], status="failed", completed_at=utcnow(),
                                       error="interrupted repeatedly; retry budget exhausted")
            self._event(run_id, "task_recovered", {"attempts_used": t["retry_count"] + 1}, t["id"])
            recovered.append(t["id"])
        if recovered:
            self._event(run_id, "run_recovered", {"tasks": recovered})
        return recovered

    # ------------------------------------------------------------------ main loop
    async def run(self, run_id: str) -> RunResult:
        run = self.store.get_run(run_id)
        if RunStatus(run["status"]) in TERMINAL_RUN or run["control"] == "pause":
            return self.result(run_id)
        limits = RunLimits(**run["limits"])
        self.recover(run_id)
        first = run["started_at"] is None
        self.store.update_run(run_id, status="running", started_at=run["started_at"] or utcnow(), stop_reason=None)
        self._event(run_id, "run_started" if first else "run_resumed", {})
        rc = _RunCtl(run_id, run["project_id"], run["question"], limits, run["elapsed_s"], self.clock(), self.clock)
        running: dict[str, asyncio.Task] = {}
        outcome: tuple[RunStatus, str | None] | None = None
        paused_reason: str | None = None

        try:
            while True:
                self._reap(running)
                control = self.store.get_run(run_id)["control"]
                if rc.runtime_cut or rc.elapsed() >= limits.max_runtime_s:  # runtime_cut: timers may fire a tick early
                    outcome = (RunStatus.STOPPED_LIMIT, "max_runtime_s")
                    break
                if control == "cancel":
                    outcome = (RunStatus.CANCELLED, "cancel_requested")
                    break
                self._sync_approvals(run_id)
                self._skip_doomed(run_id)
                graph = TaskGraph(self.store.list_tasks(run_id))
                now = self.wall()
                if control != "pause":
                    ready = sorted(graph.ready(now), key=lambda t: t["iteration"])  # stable: creation order inside a round
                    for t in ready:
                        if t["requires_approval"]:
                            self._request_approval(run_id, t)
                        elif len(running) < limits.max_concurrency:
                            self._launch(rc, t, running)
                if running:
                    await asyncio.wait(list(running.values()), timeout=self.poll_interval, return_when=asyncio.FIRST_COMPLETED)
                    continue
                tasks = self.store.list_tasks(run_id)
                pending = [t for t in tasks if t["status"] == "pending"]
                awaiting = [t for t in tasks if t["status"] == "awaiting_approval"]
                if control == "pause":
                    paused_reason = "user_pause"
                    break
                waits = [t["not_before"] - now for t in pending if t["not_before"] > now]
                if waits:  # a retry is cooling down
                    await asyncio.sleep(min(min(waits), self.poll_interval * 4))
                    continue
                if awaiting:
                    paused_reason = "awaiting_approval"
                    break
                if pending:  # nothing can ever satisfy these dependencies; do not spin
                    for t in pending:
                        self.store.update_task(t["id"], status="skipped", completed_at=utcnow(), error="unsatisfiable dependencies")
                        self._event(run_id, "task_skipped", {"reason": "unsatisfiable"}, t["id"])
                    continue
                break  # every task is terminal
        except BaseException:
            rc.abandon = True  # a crash must leave the database untouched; recovery handles it next start
            for t in running.values():
                t.cancel()
            raise

        if outcome is not None:
            status, reason = outcome
            await self._cancel_running(rc, running, reason or "stopped")
            self._close_open_tasks(run_id, status, reason)
            self._finish(run_id, rc, status, reason)
        elif paused_reason is not None:
            with self.store.transaction():  # status and event become visible together to other processes
                self.store.update_run(run_id, status="paused", elapsed_s=rc.elapsed(), stop_reason=paused_reason)
                self._event(run_id, "run_paused", {"reason": paused_reason})
        else:
            failed = any(t["status"] == "failed" for t in self.store.list_tasks(run_id))
            self._finish(run_id, rc, RunStatus.FAILED if failed else RunStatus.COMPLETED, "task_failed" if failed else None)
        return self.result(run_id)

    # ------------------------------------------------------------------ helpers: lifecycle
    def _reap(self, running: dict[str, asyncio.Task]) -> None:
        for tid, at in list(running.items()):
            if at.done():
                del running[tid]
                if not at.cancelled() and at.exception() is not None:
                    raise at.exception()  # only BaseExceptions get here: task code handles every Exception itself

    async def _cancel_running(self, rc: _RunCtl, running: dict[str, asyncio.Task], reason: str) -> None:
        rc.cancel_reason = reason
        for t in running.values():
            t.cancel()
        await asyncio.gather(*running.values(), return_exceptions=True)
        running.clear()

    def _close_open_tasks(self, run_id: str, status: RunStatus, reason: str | None) -> None:
        """Tasks that never got to run when the run stopped early."""
        final = TaskStatus.CANCELLED if status == RunStatus.CANCELLED else TaskStatus.SKIPPED
        for t in self.store.list_tasks(run_id):
            if t["status"] in ("pending", "awaiting_approval", "running"):
                self.store.update_task(t["id"], status=final.value, completed_at=utcnow(), error=f"run stopped: {reason}")
                self._event(run_id, f"task_{final.value}", {"reason": f"run stopped: {reason}"}, t["id"])

    def _finish(self, run_id: str, rc: _RunCtl, status: RunStatus, reason: str | None) -> None:
        with self.store.transaction():  # a watcher that sees the terminal status is guaranteed to see run_finished too
            summary = build_summary(self.store, run_id)
            self.store.update_run(run_id, status=status.value, completed_at=utcnow(), elapsed_s=rc.elapsed(),
                                  stop_reason=reason, summary=summary)
            if status == RunStatus.STOPPED_LIMIT:
                self._event(run_id, "limit_reached", {"limit": reason})
            self._event(run_id, "run_finished", {"status": status.value, "stop_reason": reason,
                                                 "task_counts": self.store.task_counts(run_id)})

    # ------------------------------------------------------------------ helpers: approvals and graph hygiene
    def _request_approval(self, run_id: str, task: dict) -> None:
        if self.store.approval_for_task(task["id"]) is None:
            details = {"agent_type": task["agent_type"], "title": task["title"], "input": task["input"]}
            aid = self.store.add_approval(run_id, task["id"], f"run {task['agent_type']} task: {task['title']}", details)
            self._event(run_id, "approval_requested", {"approval_id": aid, "agent_type": task["agent_type"]}, task["id"])
        self.store.update_task(task["id"], status="awaiting_approval")

    def _sync_approvals(self, run_id: str) -> None:
        """Apply human decisions (made here or by another process) to the tasks waiting on them."""
        for t in self.store.list_tasks(run_id):
            if t["status"] != TaskStatus.AWAITING_APPROVAL.value:
                continue
            appr = self.store.approval_for_task(t["id"])
            if appr is None or appr["decision"] is None:
                continue
            if appr["decision"] == "approved":
                self.store.update_task(t["id"], status="pending", requires_approval=0)
                self._event(run_id, "task_approved", {"approval_id": appr["id"]}, t["id"])
            else:
                self.store.update_task(t["id"], status="rejected", completed_at=utcnow(), error="rejected by reviewer")
                self._event(run_id, "task_rejected", {"approval_id": appr["id"]}, t["id"])

    def _skip_doomed(self, run_id: str) -> None:
        while True:
            doomed = TaskGraph(self.store.list_tasks(run_id)).doomed()
            if not doomed:
                return
            for t, dep in doomed:
                self.store.update_task(t["id"], status="skipped", completed_at=utcnow(),
                                       error=f"upstream task {dep} did not succeed")
                self._event(run_id, "task_skipped", {"reason": "upstream", "upstream": dep}, t["id"])

    # ------------------------------------------------------------------ helpers: executing one task
    def _launch(self, rc: _RunCtl, row: dict, running: dict[str, asyncio.Task]) -> None:
        attempt = row["retry_count"] + 1
        self.store.update_task(row["id"], status="running", started_at=utcnow())
        if row["agent_type"] == "supervisor" and row["iteration"] == 0 and attempt == 1:
            self._event(rc.run_id, "planning_started", {"question": rc.question[:200]}, row["id"])
        self._event(rc.run_id, "agent_started", {"agent_type": row["agent_type"], "attempt": attempt}, row["id"])
        rc.execs[row["id"]] = self.store.begin_execution(rc.run_id, row["id"], row["agent_type"], attempt)
        self._event(rc.run_id, "task_started", {"agent_type": row["agent_type"], "attempt": attempt,
                                                "iteration": row["iteration"]}, row["id"])
        running[row["id"]] = asyncio.create_task(self._run_task(rc, row, attempt))

    def _build_spec(self, row: dict, run_id: str, attempt: int) -> TaskSpec:
        deps: dict[str, AgentOutput] = {}
        for dep_id in row["depends_on"]:
            dep = self.store.get_task(dep_id)
            if dep["status"] == "succeeded" and dep["output"]:
                deps[dep_id] = AgentOutput.model_validate(dep["output"])
        return TaskSpec(id=row["id"], run_id=run_id, agent_type=row["agent_type"], title=row["title"], input=row["input"],
                        depends_on=row["depends_on"], iteration=row["iteration"], parent_task_id=row["parent_task_id"],
                        attempt=attempt, dependency_outputs=deps)

    def _validate_output(self, raw: Any, spec: TaskSpec) -> AgentOutput:
        try:
            out = raw if isinstance(raw, AgentOutput) else AgentOutput.model_validate(raw)
            out = AgentOutput.model_validate(out.model_dump())  # re-check: the object may have been mutated after creation
            out.model_dump_json()
        except (ValidationError, TypeError, ValueError) as e:
            raise OutputValidationError(f"output failed schema validation: {str(e)[:300]}") from e
        if out.task_id != spec.id or out.agent_type != spec.agent_type:
            raise OutputValidationError(
                f"output identifies as task {out.task_id}/{out.agent_type}, expected {spec.id}/{spec.agent_type}")
        if self.resolver is not None:
            for ref in out.evidence:
                problem = self.resolver.check(ref)
                if problem:
                    raise OutputValidationError(f"unresolvable evidence: {problem}")
        return out

    async def _run_task(self, rc: _RunCtl, row: dict, attempt: int) -> None:
        run_id, limits, task_id = rc.run_id, rc.limits, row["id"]

        def emit(event_type: str, payload: dict | None = None) -> None:
            self._event(run_id, event_type, payload, task_id)

        spec = spec_for(row["agent_type"])
        task_cap = min(limits.task_timeout_s, spec.timeout_s) if spec and spec.timeout_s else limits.task_timeout_s
        timeout = max(0.001, min(task_cap, limits.max_runtime_s - rc.elapsed()))
        t0, cpu0 = time.monotonic(), time.process_time()
        try:
            await self._execute_task(rc, row, attempt, timeout, task_cap, emit)
        finally:
            self._close_execution(rc, task_id, time.monotonic() - t0, time.process_time() - cpu0)

    def _close_execution(self, rc: _RunCtl, task_id: str, duration: float, cpu: float) -> None:
        exec_id = rc.execs.pop(task_id, None)
        if exec_id is None or rc.abandon:  # abandoned runs are closed by recover(), which knows the worker is gone
            return
        row = self.store.get_task(task_id)
        outcome = {"succeeded": "succeeded", "pending": "retry", "failed": "failed", "cancelled": "cancelled"}.get(row["status"], row["status"])
        self.store.finish_execution(exec_id, outcome, round(duration, 4), round(cpu, 4), row["error"] if outcome != "succeeded" else None)

    async def _execute_task(self, rc: _RunCtl, row: dict, attempt: int, timeout: float, task_cap: float, emit) -> None:
        run_id, limits, task_id = rc.run_id, rc.limits, row["id"]
        try:
            if row["agent_type"] not in self.registry:
                raise NonRetryableError(f"no agent registered for type '{row['agent_type']}'")
            agent = self.registry.get(row["agent_type"])
            spec = self._build_spec(row, run_id, attempt)
            ctx = AgentContext(
                run_id=run_id, project_id=rc.project_id, question=rc.question, limits=limits,
                tools=self.tools.bind(row["agent_type"], self.policy, emit),
                memory=self.memory.bind(rc.project_id, run_id, task_id, emit),
                model=self.model.handle(run_id, limits.max_model_tokens, emit) if self.model and self.model.available else None,
                _emit=emit,
            )
            raw = await asyncio.wait_for(agent.run(ctx, spec), timeout)
            out = self._validate_output(raw, spec)
        except asyncio.CancelledError:
            if not rc.abandon:
                self.store.update_task(task_id, status="cancelled", completed_at=utcnow(), error=rc.cancel_reason or "cancelled")
                emit("task_cancelled", {"reason": rc.cancel_reason})
            raise
        except asyncio.TimeoutError:
            if timeout < task_cap:  # cut short by the run's own runtime limit, not the task's fault
                rc.runtime_cut = True
                self.store.update_task(task_id, status="cancelled", completed_at=utcnow(), error="run runtime limit")
                emit("task_cancelled", {"reason": "max_runtime_s"})
            else:
                emit("task_timeout", {"attempt": attempt, "timeout_s": timeout})
                self._on_failure(rc, row, attempt, f"timeout after {timeout:g}s", retryable=True)
        except NonRetryableError as e:
            self._on_failure(rc, row, attempt, f"{type(e).__name__}: {e}", retryable=False)
        except Exception as e:  # agent bug, bad output, tool error: bounded retry
            self._on_failure(rc, row, attempt, f"{type(e).__name__}: {str(e)[:400]}", retryable=True)
        else:
            self._on_success(rc, row, out)

    def _on_failure(self, rc: _RunCtl, row: dict, attempt: int, error: str, retryable: bool) -> None:
        spec = spec_for(row["agent_type"])
        ceiling = rc.limits.max_retries if spec is None or spec.max_retries is None else min(rc.limits.max_retries, spec.max_retries)
        policy = RetryPolicy(ceiling, rc.limits.retry_backoff_s)
        if retryable and policy.can_retry(row["retry_count"]):
            delay = policy.delay_s(row["retry_count"])
            self.store.update_task(row["id"], status="pending", retry_count=row["retry_count"] + 1,
                                   not_before=self.wall() + delay, error=error)
            self._event(rc.run_id, "task_retry", {"attempt": attempt, "next_attempt": attempt + 1, "delay_s": delay,
                                                  "error": error}, row["id"])
        else:
            self.store.update_task(row["id"], status="failed", completed_at=utcnow(), error=error)
            self._event(rc.run_id, "task_failed", {"attempt": attempt, "error": error, "retryable": retryable}, row["id"])
            self._maybe_replan(rc, row, error)

    REPLANNABLE = frozenset({"discovery"})

    def _maybe_replan(self, rc: _RunCtl, row: dict, error: str) -> None:
        """A search task failed for good: ask the Supervisor (once per allowed replan, within max_tasks) for an alternative plan."""
        if row["agent_type"] not in self.REPLANNABLE or ROOT_AGENT not in self.registry:
            return
        tasks = self.store.list_tasks(rc.run_id)
        used = sum(1 for t in tasks if t["agent_type"] == ROOT_AGENT and t["input"].get("replan_for"))
        if used >= rc.limits.max_replans or len(tasks) + 1 > rc.limits.max_tasks:
            return
        dependents = [t for t in tasks if row["id"] in t["depends_on"] and t["agent_type"] == "critic"]
        critic = dependents[0] if dependents else None
        survivors = [d for d in (critic["depends_on"] if critic else []) if d != row["id"]
                     and self.store.get_task(d)["status"] == "succeeded"]
        new = {"id": new_id("task"), "agent_type": ROOT_AGENT, "title": f"Replan after failed {row['agent_type']} task",
               "input": {"question": rc.question, "replan_for": row["id"], "failed_input": row["input"], "error": error[:300],
                         "critic_input": critic["input"] if critic else None, "critic_depends_on": survivors},
               "depends_on": [], "iteration": row["iteration"], "parent_task_id": row["id"], "requires_approval": 0}
        self.store.add_tasks(rc.run_id, [new])
        self._event(rc.run_id, "replan_requested", {"failed_task": row["id"], "agent_type": row["agent_type"], "error": error[:300]}, new["id"])
        self._event(rc.run_id, "task_created", _task_payload(new), new["id"])

    def _on_success(self, rc: _RunCtl, row: dict, out: AgentOutput) -> None:
        run_id = rc.run_id
        with self.store.transaction():  # the output and the tasks it spawns are persisted together or not at all
            self.store.update_task(row["id"], status="succeeded", output=out.model_dump(mode="json"),
                                   completed_at=utcnow(), error=None)
            new_rows, rejections = expand(row, out.proposed_tasks, self.store.list_tasks(run_id), rc.limits,
                                          self.policy, set(self.registry.types()))
            self.store.add_tasks(run_id, new_rows)
        self._event(run_id, "task_succeeded", {"status": out.status, "summary": out.summary[:300],
                                               "findings": len(out.findings), "evidence": len(out.evidence),
                                               "proposed": len(out.proposed_tasks)}, row["id"])
        for r in rejections:
            self._event(run_id, "proposal_rejected", {"key": r.key, "agent_type": r.agent_type, "reason": r.reason}, row["id"])
        for t in new_rows:
            self._event(run_id, "task_created", _task_payload(t), t["id"])
