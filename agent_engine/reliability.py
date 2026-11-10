"""Run reliability summary (P11.6): what actually happened to each agent, read from the persisted task rows, execution records and events.

The scheduler already bounds retries and timeouts, validates every output against the AgentOutput contract, re-queues work lost to a crash,
skips tasks whose upstream failed, and persists each task's output as it finishes. This module only *reports* that, in one place, so a
degraded run is visible instead of being something a reader has to reconstruct from raw events:

  - per agent: tasks, outcomes, attempts, retries, timeouts, abandoned (crashed) attempts, time spent
  - every failure with its error, and every skipped task traced to the failed task that caused it (the root cause, not the nearest parent)
  - partial results: outputs that succeeded and are persisted even though the run did not complete
  - a `degraded` flag and a plain-language note; a report is only ever built from a completed synthesis, so a degraded run has no report,
    and a completed run that needed retries says so.

Pure reading: it never changes a run."""
from __future__ import annotations

from collections import defaultdict

from agent_engine.state.store import Store

TERMINAL_BAD = ("failed", "skipped", "cancelled", "rejected")
NOTE_OK = "Every task succeeded."
NOTE_RETRIED = "The run completed, but some attempts failed and were retried; the retries are listed under agents."
NOTE_DEGRADED = ("The run did not complete. Succeeded tasks keep their persisted output (partial results). No report is built from an "
                 "incomplete run, so no missing agent can silently leave a gap in a report.")
NOTE_ACTIVE = "The run has not finished; this is a snapshot."


def _root_causes(tasks: dict[str, dict], task_id: str, seen: set[str] | None = None) -> list[str]:
    """Failed (or rejected/cancelled) ancestors that explain why `task_id` did not run, nearest-first, without duplicates."""
    seen = seen if seen is not None else set()
    out: list[str] = []
    for dep in tasks[task_id]["depends_on"]:
        if dep in seen or dep not in tasks:
            continue
        seen.add(dep)
        if tasks[dep]["status"] in ("failed", "rejected", "cancelled"):
            out.append(dep)
        elif tasks[dep]["status"] == "skipped":
            out += _root_causes(tasks, dep, seen)
    return list(dict.fromkeys(out))


def run_reliability(store: Store, run_id: str) -> dict:
    run = store.get_run(run_id)
    tasks = {t["id"]: t for t in store.list_tasks(run_id)}
    execs = store.list_executions(run_id)
    events = store.list_events(run_id)
    timeouts: dict[str, int] = defaultdict(int)
    for e in events:
        if e["type"] == "task_timeout" and e["task_id"]:
            timeouts[e["task_id"]] += 1
    by_task: dict[str, list[dict]] = defaultdict(list)
    for x in execs:
        by_task[x["task_id"]].append(x)

    agents: dict[str, dict] = {}
    for t in tasks.values():
        a = agents.setdefault(t["agent_type"], {"tasks": 0, "succeeded": 0, "failed": 0, "skipped": 0, "cancelled": 0, "rejected": 0,
                                                "attempts": 0, "retries": 0, "timeouts": 0, "abandoned_attempts": 0, "duration_s": 0.0})
        a["tasks"] += 1
        if t["status"] in a:
            a[t["status"]] += 1
        xs = by_task.get(t["id"], [])
        a["attempts"] += len(xs)
        a["retries"] += t["retry_count"]
        a["timeouts"] += timeouts.get(t["id"], 0)
        a["abandoned_attempts"] += sum(x["status"] == "abandoned" for x in xs)
        a["duration_s"] = round(a["duration_s"] + sum(x["duration_s"] or 0.0 for x in xs), 4)

    failures = [{"task_id": t["id"], "agent_type": t["agent_type"], "title": t["title"], "status": t["status"], "error": t["error"],
                 "attempts": len(by_task.get(t["id"], [])), "retries_used": t["retry_count"], "timed_out": timeouts.get(t["id"], 0) > 0,
                 "attempt_errors": [x["error"] for x in by_task.get(t["id"], []) if x["error"]]}
                for t in tasks.values() if t["status"] in ("failed", "cancelled", "rejected")]
    skipped = []
    for t in tasks.values():
        if t["status"] == "skipped":
            roots = _root_causes(tasks, t["id"])
            skipped.append({"task_id": t["id"], "agent_type": t["agent_type"], "title": t["title"], "reason": t["error"],
                            "caused_by": [{"task_id": r, "agent_type": tasks[r]["agent_type"], "status": tasks[r]["status"]} for r in roots]})
    partial = [{"task_id": t["id"], "agent_type": t["agent_type"], "title": t["title"]}
               for t in tasks.values() if t["status"] == "succeeded" and t["output"]]
    retried = any(t["retry_count"] for t in tasks.values())
    finished = run["status"] in ("completed", "failed", "cancelled")
    degraded = any(t["status"] in TERMINAL_BAD for t in tasks.values()) or run["status"] in ("failed", "cancelled")
    report_possible = any(t["agent_type"] == "synthesis" and t["status"] == "succeeded" and t["output"] for t in tasks.values())
    return {
        "run_id": run_id, "run_status": run["status"], "finished": finished, "degraded": degraded, "report_available": report_possible,
        "tasks": len(tasks), "agents": agents, "failures": failures, "skipped": skipped,
        "partial_results": partial if degraded else [],
        "retried_tasks": [{"task_id": t["id"], "agent_type": t["agent_type"], "retries_used": t["retry_count"]}
                          for t in tasks.values() if t["retry_count"]],
        "note": NOTE_ACTIVE if not finished else NOTE_DEGRADED if degraded else NOTE_RETRIED if retried else NOTE_OK,
        "guarantees": ["retries are bounded per task and per run limits", "each attempt has a timeout", "every output is validated against the "
                       "AgentOutput contract before it is stored", "a failed task skips only its dependents", "attempts lost to a crash are "
                       "re-queued and count against the retry budget", "reports are built only from a completed synthesis task"],
    }
