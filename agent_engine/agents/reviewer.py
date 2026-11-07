"""Critical reviewer: a sceptical pass over the finished analysis. It challenges assumptions, flags statements that rest on no
cited passage, lists possible contradictions between documents and checks each experiment plan for methodological holes.

It reports findings with severities and never "approves": a clean review means these specific checks found nothing, not that
the research is correct. Contradiction detection is lexical (shared terms + opposite direction words) and is labelled possible."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.agents.evidence import merged_verification
from agent_engine.agents.textutil import polarity
from agent_engine.contracts import AgentOutput, TaskSpec
from agent_engine.errors import NonRetryableError
from research.bm25.analysis import analyze

MIN_SHARED = 3
SEVERITIES = ("blocking", "major", "minor")


class ReviewerAgent(Agent):
    agent_type = "reviewer"
    description = "Challenges assumptions, flags unsupported statements, possible contradictions and methodology problems."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        by = {o.agent_type: o for o in task.dependency_outputs.values()}
        if "critic" not in by or "hypothesis" not in by:
            raise NonRetryableError("reviewer needs the critic and hypothesis outputs as dependencies")
        findings: list[dict] = []

        def flag(severity: str, category: str, target: str, message: str) -> None:
            findings.append({"id": f"r{len(findings) + 1:03d}", "severity": severity, "category": category, "target": target,
                             "message": message})

        assessment = by["critic"].data["assessment"]
        if not assessment.get("sufficient"):
            flag("blocking", "evidence_quality", "critic", "The evidence check found the accepted evidence insufficient: "
                 + "; ".join(assessment.get("reasons", [])) + ".")
        elif assessment.get("documents", 0) == 1:
            flag("major", "evidence_quality", "critic", "All accepted evidence comes from a single document; nothing corroborates it.")

        for h in by["hypothesis"].data["hypotheses"]:
            for a in h["assumptions"]:
                flag("minor", "assumption", h["id"], f"Untested assumption: {a}")
            if not h["evidence"]:
                flag("major", "unsupported_statement", h["id"],
                     "This hypothesis is not anchored to any cited passage; it is derived from a question term alone.")

        lit = by.get("literature")
        if lit:
            results = [s for s in lit.data["statements"] if s["kind"] == "result"]
            for i, a in enumerate(results):
                for b in results[i + 1:]:
                    if a["document_id"] == b["document_id"]:
                        continue
                    shared = sorted(set(analyze(a["text"])) & set(analyze(b["text"])))
                    if len(shared) >= MIN_SHARED and polarity(a["text"]) * polarity(b["text"]) == -1:
                        flag("major", "possible_contradiction", f"{a['id']} vs {b['id']}",
                             f"Possible contradiction (shared terms: {', '.join(shared)}): \"{a['text']}\" vs \"{b['text']}\". "
                             "Word overlap with opposite direction words only; a human must decide.")

        merged = merged_verification(task.dependency_outputs.values())
        for c in merged["claims"]:
            if c["claim_type"] in ("gap", "hypothesis"):
                continue
            quote = next((e["quote"] for e in c["evidence"] if e["role"] == "contradicting"), "")
            if c["status"] == "CONTRADICTED":
                flag("major", "contradicted_claim", c["key"], f"This claim is contradicted by a stored passage: \"{quote}\" "
                     "(lexical check; a human must read both).")
            elif c["conflicting"]:
                flag("major", "contradictory_evidence", c["key"], "Stored passages both support and contradict this claim"
                     + (f" (contradicting: \"{quote}\")." if quote else "."))
            elif c["status"] == "INSUFFICIENT_EVIDENCE":
                flag("major", "unsupported_statement", c["key"], "No stored passage addresses this claim; it should not be relied on.")
            elif c["status"] == "UNVERIFIED":
                flag("major", "unverifiable_claim", c["key"], "This claim could not be verified: " + "; ".join(c["explanation"][:2]) + ".")
            elif c["status"] == "PARTIALLY_SUPPORTED":
                flag("minor", "partial_support", c["key"], "Only indirect evidence supports this claim (" + "; ".join(c["explanation"][:1]) + ").")
        for n in merged["novelty"]:
            if n["label"] == "Similar prior work identified":
                flag("major", "novelty", n["claim_key"], "Similar prior work was identified in the searched corpus ("
                     + ", ".join(d.get("title") or d["document_id"] for d in n["similar_existing_research"][:3])
                     + "); the proposal may not be new.")
            elif n["label"] == "Insufficient literature coverage":
                flag("minor", "novelty", n["claim_key"], "The corpus is too small to judge novelty; this label says nothing about the idea.")
        if merged["needs_followup"]:
            flag("major", "evidence_follow_up", "supervisor", f"{len(merged['needs_followup'])} claim(s) remain not fully supported after "
                 f"{max(merged['rounds'] - 1, 0)} follow-up round(s).")
        for p in (by["experiment_design"].data["plans"] if "experiment_design" in by else []):
            if not p["metrics"]:
                flag("major", "methodology", p["id"], "No evaluation metric is named in the accepted evidence; one must be chosen before running.")
            if not p["baselines"]:
                flag("major", "methodology", p["id"], "No baseline or control is identified; a comparison needs one.")
            if not p["datasets"]:
                flag("minor", "methodology", p["id"], "No dataset or sample is named in the accepted evidence.")
            if p["reproducibility"]["repeats_min"] < 3:
                flag("major", "methodology", p["id"], "Fewer than three repeats cannot estimate run-to-run variation.")
            if p["dependent_variable"]["name"].startswith("to be defined"):
                flag("major", "methodology", p["id"], "The dependent variable is undefined.")
        if "experiment_design" not in by:
            flag("minor", "methodology", "workflow", "No experiment plan was available to review.")

        counts = {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITIES}
        blocking = counts["blocking"] > 0
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="partial" if blocking else "ok",
            summary=f"Review: {counts['blocking']} blocking, {counts['major']} major, {counts['minor']} minor finding(s).",
            findings=[f"[{f['severity']}] {f['message'][:160]}" for f in findings if f["severity"] != "minor"][:10],
            uncertainties=["This review applies a fixed set of checks. No findings would not mean the research is sound; "
                           "possible contradictions come from word overlap and may be false alarms."],
            data={"review": {"findings": findings, "counts": counts, "blocking": blocking,
                             "disclaimer": "Automated checklist review - not a peer review and not an approval."}},
        )
