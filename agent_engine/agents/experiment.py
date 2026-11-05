"""Experiment agent: runs a human-approved experiment spec in a sandbox. The engine's approval gate holds every
'experiment' task until a named reviewer approves it; this agent then refuses to run without finding that approval."""
from __future__ import annotations

import asyncio
from pathlib import Path

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.contracts import AgentOutput, TaskSpec
from agent_engine.errors import NonRetryableError
from agent_engine.state.store import Store
from research.experiments import (ExperimentSpec, NotApproved, QuotaExceeded, SandboxRefused, run_experiment)
from research.experiments.sandbox import Sandbox


class ExperimentAgent(Agent):
    agent_type = "experiment"
    description = "Runs an approved experiment spec in a sandbox and records a reproducibility manifest."

    def __init__(self, sandbox: Sandbox, root: Path, store: Store):
        self.sandbox, self.root, self.store = sandbox, Path(root), store

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        approval = self.store.approval_for_task(task.id)
        if not approval or approval["decision"] != "approved":
            raise NonRetryableError("experiment has no recorded human approval")
        try:
            spec = ExperimentSpec.model_validate(task.input.get("spec") or {})
        except ValueError as e:
            raise NonRetryableError(f"invalid experiment spec: {e}") from e
        try:
            run_dir, m = await asyncio.to_thread(run_experiment, spec, self.sandbox, self.root,
                                                 approved_by=approval["reviewer_id"] or "")
        except (SandboxRefused, QuotaExceeded, NotApproved) as e:
            raise NonRetryableError(str(e)) from e
        ok = m["status"] == "succeeded"
        uncertainties = ["An experiment result is evidence only about this code, data and seed; a human must review the "
                         "manifest and judge whether it tests the hypothesis."]
        if not m["isolation"]["network_blocked"]:
            uncertainties.append("This run used the process fallback: " + " ".join(m["isolation"]["notes"]))
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok" if ok else "partial",
            summary=f"Experiment '{spec.name}' {m['status']} in {m['result']['duration_s']}s; "
                    f"{len(m['metrics'])} metric(s), {len(m['artifacts'])} artifact(s). Review: pending.",
            findings=[f"{k} = {v}" for k, v in list(m["metrics"].items())[:10]], uncertainties=uncertainties + m["warnings"],
            data={"experiment": {"id": m["id"], "run_dir": str(run_dir), "status": m["status"], "metrics": m["metrics"],
                                 "hypothesis_id": m["hypothesis_id"], "manifest_sha256": m["manifest_sha256"],
                                 "isolation": m["isolation"]["level"], "review": "pending"}},
        )
