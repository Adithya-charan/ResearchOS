"""Supervisor: scopes the question, plans delegated work, sets the stop criteria the Critic will enforce."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.contracts import AgentOutput, ProposedTask, TaskSpec
from agent_engine.errors import NonRetryableError
from research.bm25.analysis import analyze, phrase_chunks, term_surface_forms

MAX_FACETS = 3  # sub-queries besides the full-question query
MAX_TERMS = 12


class SupervisorAgent(Agent):
    agent_type = "supervisor"
    description = "Scopes the question into search facets, plans Discovery and Critic tasks, and sets stop criteria."

    def __init__(self, discovery_k: int = 8, workflow: bool = False):
        self.discovery_k = discovery_k  # passages each Discovery task retrieves in the first round
        self.workflow = workflow  # True = after the evidence check, run analysis -> gaps -> hypotheses -> verification -> report

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        if task.input.get("replan_for"):
            return self._replan(ctx, task)
        question = str(task.input.get("question") or ctx.question).strip()
        if not question:
            raise NonRetryableError("empty research question")
        pairs = term_surface_forms(question, limit=MAX_TERMS)
        terms = [s for s, _ in pairs]  # stems: for matching evidence
        surface = dict(pairs)  # stem -> original word: for building queries
        if not terms:
            raise NonRetryableError("the question contains no searchable terms")

        # Facets: the full question, plus up to MAX_FACETS distinctive noun-phrase-like chunks (split at stopwords).
        queries: list[tuple[str, str]] = [("full question", " ".join(surface.values()))]
        seen = {frozenset(terms)}
        chunks = [c for c in phrase_chunks(question)]
        picked = sorted(range(len(chunks)), key=lambda i: (-len(chunks[i]), i))[:MAX_FACETS]
        for i in sorted(picked):
            facet_terms = frozenset(analyze(" ".join(chunks[i])))
            if facet_terms and facet_terms not in seen:
                seen.add(facet_terms)
                queries.append((" ".join(chunks[i]), " ".join(chunks[i])))

        stop_criteria = {"min_accepted": 3, "min_documents": 3, "min_coverage": 0.5, "max_followup_rounds": ctx.limits.max_iterations}
        proposals = [
            ProposedTask(key=f"disc{i}", agent_type="discovery", title=f"Search: {label}",
                         input={"query": q, "label": label, "k": self.discovery_k})
            for i, (label, q) in enumerate(queries)
        ]
        proposals.append(ProposedTask(
            key="critic", agent_type="critic", title="Check evidence sufficiency and diversity",
            depends_on=[p.key for p in proposals],
            input={"question": question, "terms": terms, "surface": surface, **stop_criteria,
                   **({"workflow": True} if self.workflow else {})},
        ))

        ctx.memory.put("scope", {"question": question, "terms": terms, "queries": [q for _, q in queries]})
        out = AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok",
            summary=f"Scoped the question into {len(queries)} search facet(s) and planned {len(queries)} discovery task(s) "
                    "followed by one critic review.",
            findings=[f"Key terms: {', '.join(terms)}"] + [f"Facet '{label}': {q}" for label, q in queries],
            uncertainties=["Scoping is lexical (stopword splitting); it does not understand the meaning of the question."],
            proposed_tasks=proposals,
            data={"terms": terms, "queries": [q for _, q in queries], "stop_criteria": stop_criteria},
        )
        if ctx.model is not None:
            note = await ctx.model_note(f"Title: {question}\n", max_new_tokens=40)
            if note is not None:
                out.model_notes.append(note)
            else:
                out.uncertainties.append("Model note skipped (token budget exhausted or model unavailable).")
        return out

    def _replan(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        """A discovery task failed permanently. Try once with a broader query (the last term dropped) and re-run the
        evidence review over what did succeed. If there is nothing new to try, say so instead of repeating the failed task."""
        failed = task.input.get("failed_input") or {}
        old_query = str(failed.get("query", "")).strip()
        words = old_query.split()
        new_query = " ".join(words[:-1]) if len(words) > 1 else ""
        critic_input = task.input.get("critic_input")
        if not new_query or new_query == old_query or critic_input is None:
            return AgentOutput(
                task_id=task.id, agent_type=self.agent_type, status="insufficient_evidence",
                summary="No alternative plan is available for the failed search; nothing was added.",
                uncertainties=[f"The failed search could not be broadened any further (error: {task.input.get('error', 'unknown')})."],
                data={"replan": {"for": task.input["replan_for"], "action": "none"}})
        proposals = [
            ProposedTask(key="replan_search", agent_type="discovery", title=f"Replanned search: {new_query}",
                         input={"query": new_query, "label": "replanned search", "k": int(failed.get("k", self.discovery_k))}),
            ProposedTask(key="replan_critic", agent_type="critic", title="Re-check evidence after replanned search",
                         depends_on=[*task.input.get("critic_depends_on", []), "replan_search"], input=dict(critic_input)),
        ]
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok",
            summary=f"Replanned after a failed search: retrying with the broader query '{new_query}' and re-running the evidence review.",
            findings=[f"Failed query: {old_query}", f"Replacement query: {new_query}"], proposed_tasks=proposals,
            uncertainties=["A broader query (one term fewer) may return less precise passages than the failed one would have."],
            data={"replan": {"for": task.input["replan_for"], "action": "broaden_query", "query": new_query}},
        )
