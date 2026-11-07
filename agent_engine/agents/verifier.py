"""Verifier: deterministic checks on every claim the workflow is about to report.

Extracted claims (quoted sentences) must be traceable: the passage exists on the cited page of the cited document, the quote
and every number in it are in that stored passage, the page text was extracted reliably and - when a document library is
attached - the passage still matches the original file. Counts are recomputed. Hypotheses and gaps are never "verified".
Whether a passage truly SUPPORTS a claim, and whether a tension is a real contradiction, stay with a human reviewer."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.agents.evidence import build_novelty, build_verification
from agent_engine.agents.literature import critic_assessment
from agent_engine.agents.textutil import digit_tokens, polarity, squash
from agent_engine.contracts import AgentOutput, EvidenceReference, TaskSpec
from research.bm25.analysis import analyze

MIN_SHARED_FOR_TENSION = 3


def _dep(task: TaskSpec, agent_type: str):
    return next((o for o in task.dependency_outputs.values() if o.agent_type == agent_type), None)


class VerifierAgent(Agent):
    agent_type = "verifier"
    description = "Checks citations, quotes, numbers and counts for every claim; labels unsupported claims; flags possible tensions."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        assessment, _ = critic_assessment(task)
        lit, gap, hyp = _dep(task, "literature"), _dep(task, "gap"), _dep(task, "hypothesis")
        source_checks = ctx.tools.has("corpus.verify_passage")
        claims: list[dict] = []

        def add(**kw) -> dict:
            claim = {"id": f"c{len(claims) + 1:03d}", "reasons": [], "checks": {}, **kw}
            claims.append(claim)
            return claim

        for s in (lit.data["statements"] if lit else []):
            p = await ctx.tools.call("corpus.get_passage", passage_id=s["passage_id"])
            checks: dict[str, bool | None] = {
                "passage_exists": bool(p.found and p.document_id == s["document_id"] and p.page_number == s["page_number"]),
                "quote_in_passage": bool(p.found and squash(s["text"]) in squash(p.text)),
                "numbers_in_passage": bool(p.found and all(n in digit_tokens(p.text) for n in digit_tokens(s["text"]))),
                "extraction_reliable": bool(p.found and p.quality == "ok"),
                "source_file": None,
            }
            if source_checks and checks["passage_exists"]:
                v = await ctx.tools.call("corpus.verify_passage", passage_id=s["passage_id"])
                checks["source_file"] = v.ok
                if not v.ok:
                    checks["source_message"] = v.message  # type: ignore[assignment]
            claim = add(type="extracted", kind=s["kind"], text=s["text"], document_id=s["document_id"],
                        page_number=s["page_number"], passage_id=s["passage_id"], section=s["section"],
                        source_id=s["id"], checks=checks)
            hard = [k for k in ("passage_exists", "quote_in_passage", "numbers_in_passage", "source_file") if checks[k] is False]
            if hard:
                claim["status"] = "unsupported"
                claim["reasons"] = [("the passage no longer matches the original file: " + str(checks.get("source_message", "")))
                                    if k == "source_file" else f"check failed: {k.replace('_', ' ')}" for k in hard]
            elif not checks["extraction_reliable"]:
                claim["status"], claim["reasons"] = "needs_review", ["the page text was extracted unreliably"]
            else:
                claim["status"] = "verified"

        # numerical verification: the counts the Critic reported are recomputed from the accepted passages
        acc = assessment["accepted"]
        recomputed = {"passages": len(acc), "documents": len({a["document_id"] for a in acc})}
        reported = {"passages": len(assessment["accepted_evidence"]), "documents": assessment["documents"]}
        c = add(type="count", kind="aggregate", document_id=None, page_number=None, passage_id=None, section="",
                text=f"{recomputed['passages']} accepted passage(s) from {recomputed['documents']} document(s) match the question terms.",
                source_id="critic", checks={"counts_recomputed": recomputed == reported})
        c["status"] = "verified" if recomputed == reported else "unsupported"
        if recomputed != reported:
            c["reasons"] = [f"reported {reported} but recomputed {recomputed}"]

        for h in (hyp.data["hypotheses"] if hyp else []):
            add(type="hypothesis", kind="hypothesis", text=h["statement"], document_id=None, page_number=None, passage_id=None,
                section="", source_id=h["id"], status="hypothesis",
                reasons=["a hypothesis is a proposal; it cannot be verified by reading documents"])
        for g in (gap.data["gaps"] if gap else []):
            add(type="gap", kind=g["kind"], text=g["statement"], document_id=None, page_number=None, passage_id=None,
                section="", source_id=g["id"], status="candidate_gap",
                reasons=["a gap is a property of the searched corpus, not an established fact"])

        tensions = self._tensions([c for c in claims if c["type"] == "extracted" and c["kind"] == "result" and c["status"] == "verified"])
        for t in tensions:
            for cid in (t["a"], t["b"]):
                cl = next(c for c in claims if c["id"] == cid)
                cl["status"] = "needs_review"
                cl["reasons"].append(f"possible tension with {t['b'] if cid == t['a'] else t['a']} (opposite direction on shared terms)")

        counts: dict[str, int] = {}
        for c in claims:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        verified = [c for c in claims if c["status"] == "verified" and c["passage_id"]]
        uncertainties = ["'verified' means the quoted sentence is in the cited stored passage"
                         + (" and the passage still matches the original file" if source_checks else " (original files were not re-checked)")
                         + "; it does not mean the sentence is true or that it answers the question. A human must judge support."]
        if counts.get("unsupported"):
            uncertainties.append(f"{counts['unsupported']} claim(s) failed verification and are labelled unsupported.")
        if tensions:
            uncertainties.append(f"{len(tensions)} possible tension(s) between documents were flagged by word overlap and "
                                 "direction words; they may not be real contradictions.")
        round_no = int(task.input.get("round", 1))
        verification = await build_verification(ctx, task, claims, round_no)
        novelty = await build_novelty(ctx, task) if round_no == 1 else []
        vc = verification["counts"]
        if verification["needs_followup"]:
            uncertainties.append(f"{len(verification['needs_followup'])} claim(s) are not fully supported by stored passages "
                                 "(partially supported, insufficient evidence or contradicted); more retrieval may change that.")
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="partial" if counts.get("unsupported") else "ok",
            summary=f"Checked {len(claims)} claim(s): " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + ". "
                    f"Evidence verification (round {round_no}): " + (", ".join(f"{n} {k}" for k, n in sorted(vc.items())) or "no claims") + ".",
            findings=[f"{c['id']} {c['status']}: {c['text'][:120]}" for c in claims if c["status"] in ("unsupported", "needs_review")][:10],
            evidence=[EvidenceReference(document_id=c["document_id"], page_number=c["page_number"], passage_id=c["passage_id"])
                      for c in verified],
            uncertainties=uncertainties, data={"claims": claims, "tensions": tensions, "counts": counts,
                                               "source_files_checked": source_checks, "verification": verification,
                                               "novelty": novelty, "round": round_no},
        )

    @staticmethod
    def _tensions(results: list[dict]) -> list[dict]:
        out = []
        for i, a in enumerate(results):
            for b in results[i + 1:]:
                if a["document_id"] == b["document_id"]:
                    continue
                shared = sorted(set(analyze(a["text"])) & set(analyze(b["text"])))
                if len(shared) >= MIN_SHARED_FOR_TENSION and polarity(a["text"]) * polarity(b["text"]) == -1:
                    out.append({"a": a["id"], "b": b["id"], "shared_terms": shared})
        return out
