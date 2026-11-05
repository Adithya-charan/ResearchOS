"""Hypothesis agent: turns candidate gaps into falsifiable, clearly-labelled hypotheses. The statements are templates filled
with quoted text, so they are only as good as the quotes: they are starting points for a human, never findings."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.contracts import AgentOutput, EvidenceReference, TaskSpec

MAX_HYPOTHESES = 5


class HypothesisAgent(Agent):
    agent_type = "hypothesis"
    description = "Proposes falsifiable hypotheses (statement, assumptions, falsification criteria) from candidate gaps."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        gap_out = next(o for o in task.dependency_outputs.values() if o.agent_type == "gap")
        lit = next(o for o in task.dependency_outputs.values() if o.agent_type == "literature")
        titles = {d["document_id"]: d["title"] for d in lit.data["documents"]}
        results = {}
        for s in lit.data["statements"]:
            if s["kind"] == "result":
                results.setdefault(s["document_id"], s)  # statements are ordered best-first within a document

        hyps: list[dict] = []
        for g in gap_out.data["gaps"]:
            if len(hyps) >= MAX_HYPOTHESES:
                break
            if g["kind"] == "stated_limitation" and g["evidence"]:
                ref = g["evidence"][0]
                doc, title = ref["document_id"], titles.get(ref["document_id"], ref["document_id"])
                quote = g["quote"]
                outcome = results.get(doc)
                tail = f" (reported outcome: \"{outcome['text']}\")" if outcome else ""
                hyps.append({
                    "statement": f"Addressing the limitation stated in {title} (\"{quote}\") would measurably change the outcome "
                                 f"that document reports{tail}.",
                    "assumptions": ["The stated limitation is a material factor, not a minor caveat.",
                                    f"The conditions described in {title} can be reproduced."],
                    "falsification_criteria": "Refuted if a controlled comparison that removes the stated limitation, with "
                                              "everything else fixed and enough repeats to estimate run-to-run variation, shows "
                                              "no difference in the reported outcome beyond that variation.",
                    "gap_id": g["id"], "evidence": g["evidence"]})
            elif g["kind"] == "unaddressed_term":
                word = g["term"]
                hyps.append({
                    "statement": f"The relationship examined in the accepted evidence still holds when '{word}' is varied.",
                    "assumptions": [f"'{word}' can be varied independently of the other factors in the question.",
                                    "Results in the accepted documents generalise beyond their reported conditions."],
                    "falsification_criteria": f"Refuted if varying '{word}' over at least three levels, other factors fixed, "
                                              "produces a systematic change (or none where the evidence implies one) larger "
                                              "than the measurement noise.",
                    "gap_id": g["id"], "evidence": []})
        for i, h in enumerate(hyps, 1):
            h.update({"id": f"h{i}", "status": "proposed",
                      "label": "Hypothesis - an untested proposal, not a finding. No experiment has been run."})
        refs = [EvidenceReference(**r) for h in hyps for r in h["evidence"]]
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok" if hyps else "insufficient_evidence",
            summary=f"{len(hyps)} falsifiable hypothesis proposal(s) drafted from candidate gaps.",
            findings=[h["statement"][:200] for h in hyps], evidence=refs,
            uncertainties=["Hypotheses are templated from gaps and quotations; they are not facts and have not been tested. "
                           "Testing one needs a human-approved experiment design run in the sandbox (scripts/run_experiment.py)."],
            data={"hypotheses": hyps},
        )
