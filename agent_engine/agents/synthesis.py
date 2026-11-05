"""Synthesis: assembles the final report from the other agents' persisted outputs. Extractive only - see reporting.builder."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.agents.literature import critic_assessment
from agent_engine.contracts import AgentOutput, EvidenceReference, TaskSpec
from agent_engine.reporting.builder import build_report


class SynthesisAgent(Agent):
    agent_type = "synthesis"
    description = "Assembles the report: scope, closest documents, verified findings, unsupported claims, gaps, hypotheses, uncertainty."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        critic_assessment(task)  # raises if the evidence assessment is missing
        by_type = {o.agent_type: o for o in task.dependency_outputs.values()}
        report = build_report(str(task.input.get("question") or ctx.question), by_type)
        refs = [EvidenceReference(**f["citation"]) for fs in report["findings"].values() for f in fs]
        n_find = sum(len(v) for v in report["findings"].values())
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type,
            status="ok" if report["status"] == "complete" else "insufficient_evidence",
            summary=f"Report {report['status']}: {n_find} verified finding(s), {len(report['unsupported_claims'])} unsupported, "
                    f"{len(report['candidate_gaps'])} candidate gap(s), {len(report['hypotheses'])} hypothesis proposal(s).",
            findings=[report["novelty_statement"]], evidence=refs, uncertainties=report["uncertainties"][:12],
            data={"report": report},
        )
