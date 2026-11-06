"""Critic: challenges the evidence. It never trusts Discovery's claims - it re-fetches every candidate passage and
recomputes coverage itself, then checks sufficiency and diversity against the Supervisor's stop criteria."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.contracts import AgentOutput, EvidenceReference, ProposedTask, TaskSpec
from agent_engine.errors import NonRetryableError
from research.bm25.analysis import analyze, key_terms

MAX_VERIFY = 60  # candidate passages re-checked per review
DEEP_K = 40  # passages retrieved by the follow-up search


class CriticAgent(Agent):
    agent_type = "critic"
    description = "Independently verifies candidate passages, checks evidence sufficiency/diversity, requests follow-up search."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        question = str(task.input.get("question") or ctx.question)
        terms = [str(t) for t in task.input.get("terms") or key_terms(question, limit=12)]
        if not terms:
            raise NonRetryableError("critic has no terms to check evidence against")
        min_accepted = int(task.input.get("min_accepted", 3))
        min_docs = int(task.input.get("min_documents", 3))
        min_cov = float(task.input.get("min_coverage", 0.5))

        discovery = [o for o in task.dependency_outputs.values() if o.agent_type == "discovery"]
        candidates: dict[str, dict] = {}
        for o in discovery:
            for h in o.data.get("hits", []):
                if h["passage_id"] not in candidates or h["score"] > candidates[h["passage_id"]]["score"]:
                    candidates[h["passage_id"]] = h
        searches_done = {(o.data.get("query"), o.data.get("k")) for o in discovery}
        corpus = next((o.data["corpus"] for o in discovery if "corpus" in o.data), None)

        accepted: list[dict] = []  # solid evidence: enough term coverage AND cleanly extracted page text
        low_quality: list[dict] = []  # matches from pages with unreliable extraction: reported, never counted as evidence
        rejected: list[dict] = []
        for pid, hit in sorted(candidates.items(), key=lambda kv: (-kv[1]["score"], kv[0]))[:MAX_VERIFY]:
            p = await ctx.tools.call("corpus.get_passage", passage_id=pid)
            if not p.found:
                rejected.append({"passage_id": pid, "reason": "not_found_in_index"})
                continue
            if p.document_id != hit["document_id"]:
                rejected.append({"passage_id": pid, "reason": "document_mismatch"})
                continue
            present = set(analyze(p.text))
            covered = [t for t in terms if t in present]
            record = {"passage_id": pid, "document_id": p.document_id, "page_number": p.page_number,
                      "section": p.section, "coverage": round(len(covered) / len(terms), 3), "score": hit["score"],
                      "covered_terms": covered, "quality": p.quality}
            if record["coverage"] >= min_cov:
                (accepted if p.quality == "ok" else low_quality).append(record)
            else:
                rejected.append({**record, "reason": "low_term_coverage"})

        accepted_docs = sorted({a["document_id"] for a in accepted})
        covered_terms = {t for a in accepted for t in a["covered_terms"]}
        uncovered = [t for t in terms if t not in covered_terms]
        reasons = []
        if len(accepted) < min_accepted:
            reasons.append(f"only {len(accepted)} accepted passage(s); need {min_accepted}")
        if len(accepted_docs) < min_docs:
            reasons.append(f"evidence from only {len(accepted_docs)} document(s); need {min_docs}")
        sufficient = not reasons

        proposals: list[ProposedTask] = []
        # Follow-up = look deeper before concluding "insufficient": the full question again, many more candidates.
        # (Facet queries were already run; a passage needs >= min_coverage of ALL terms to be accepted, so a search
        # for only the uncovered terms could not produce acceptable evidence.) Uses surface words, never stems.
        surface = task.input.get("surface") or {}
        followup_query = " ".join(surface.get(t, t) for t in terms)
        if (not sufficient and task.iteration < ctx.limits.max_iterations
                and (followup_query, DEEP_K) not in searches_done):
            proposals = [
                ProposedTask(key="followup_search", agent_type="discovery", followup=True,
                             title="Deeper search before concluding evidence is insufficient",
                             input={"query": followup_query, "label": "deeper search", "k": DEEP_K}),
                ProposedTask(key="followup_critic", agent_type="critic", followup=True,
                             title="Re-check evidence after follow-up search",
                             depends_on=[*task.depends_on, "followup_search"], input=dict(task.input)),
            ]

        if not proposals and task.input.get("workflow"):
            # Final review reached: hand the accepted evidence to the research workflow (analysis -> report).
            base = {"question": question, "terms": terms, "surface": surface}
            me = task.id
            proposals = [
                ProposedTask(key="literature", agent_type="literature", title="Analyse accepted documents",
                             depends_on=[me], input=base),
                ProposedTask(key="gap", agent_type="gap", title="Identify candidate gaps",
                             depends_on=[me, "literature"], input=base),
                ProposedTask(key="hypothesis", agent_type="hypothesis", title="Draft falsifiable hypotheses",
                             depends_on=["gap", "literature"], input=base),
                ProposedTask(key="experiment_design", agent_type="experiment_design", title="Design experiment plans",
                             depends_on=["hypothesis", "literature", "gap"], input=base),
                ProposedTask(key="reviewer", agent_type="reviewer", title="Critically review the analysis and plans",
                             depends_on=[me, "literature", "gap", "hypothesis", "experiment_design"], input=base),
                ProposedTask(key="verifier", agent_type="verifier", title="Verify claims and citations",
                             depends_on=[me, "literature", "gap", "hypothesis"], input=base),
                ProposedTask(key="synthesis", agent_type="synthesis", title="Assemble the research report",
                             depends_on=[me, "literature", "gap", "hypothesis", "experiment_design", "reviewer", "verifier"],
                             input=base),
            ]
        accepted_refs = [EvidenceReference(document_id=a["document_id"], page_number=a["page_number"], passage_id=a["passage_id"])
                         for a in accepted]
        uncertainties = [
            f"Checked {len(candidates)} candidate passage(s) lexically ({len(accepted)} accepted). Term overlap does not "
            "show that a passage supports a specific claim; human review is required."
        ]
        if low_quality:
            spots = ", ".join(f"{m['document_id']} p.{m['page_number']}" if m["page_number"] else m["document_id"]
                              for m in low_quality[:5])
            more = f" and {len(low_quality) - 5} more" if len(low_quality) > 5 else ""
            uncertainties.append(
                f"{len(low_quality)} matching passage(s) come from pages whose text extraction looks unreliable "
                f"({spots}{more}); they were not counted as evidence - inspect the source document.")
        if not accepted and not low_quality:
            where = f" ({corpus['documents']} documents, {corpus['passages']} passages)" if corpus else ""
            uncertainties.append(f"No close match was identified in the indexed corpus searched for this run{where}. "
                                 "This is not proof that the idea has never been published.")
        elif len(accepted_docs) == 1:
            uncertainties.append("All accepted evidence comes from a single document.")
        if uncovered and accepted:
            uncertainties.append(f"Question terms with no accepted evidence: {', '.join(uncovered)}.")
        if reasons and not any(p.followup for p in proposals):
            uncertainties.append("Evidence is insufficient and no further search round is available: " + "; ".join(reasons) + ".")

        assessment = {
            "sufficient": sufficient, "reasons": reasons, "documents": len(accepted_docs), "uncovered_terms": uncovered,
            "accepted": accepted, "low_quality_matches": low_quality[:20], "rejected": rejected[:20],
            "accepted_evidence": [r.model_dump() for r in accepted_refs], "followup_proposed": any(p.followup for p in proposals),
        }
        ctx.memory.put(f"assessment/{task.id}", {k: assessment[k] for k in ("sufficient", "documents", "uncovered_terms")})
        out = AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok" if sufficient else "insufficient_evidence",
            summary=(f"Evidence is sufficient: {len(accepted)} passage(s) from {len(accepted_docs)} document(s)."
                     if sufficient else "Evidence is insufficient: " + "; ".join(reasons) + "."),
            findings=[f"Accepted {len(accepted)} of {len(candidates)} candidate passage(s) from {len(accepted_docs)} document(s)."]
                     + [f"Rejected {r['passage_id']}: {r['reason']}" for r in rejected[:5]],
            evidence=accepted_refs, uncertainties=uncertainties, proposed_tasks=proposals,
            data={"assessment": assessment, **({"corpus": corpus} if corpus else {})},
        )
        if ctx.model is not None:
            note = await ctx.model_note(f"Title: Review of {question}\n", max_new_tokens=40)
            if note is not None:
                out.model_notes.append(note)
            else:
                out.uncertainties.append("Model note skipped (token budget exhausted or model unavailable).")
        return out
