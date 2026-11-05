"""Gap Analyst: lists *candidate* gaps in the indexed material - things the accepted evidence does not cover, limitations the
documents state about themselves, and findings that rest on a single source. A gap here is a gap in this corpus, never a
claim about the whole literature."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.agents.literature import critic_assessment
from agent_engine.contracts import AgentOutput, EvidenceReference, TaskSpec

MAX_UNADDRESSED, MAX_LIMITATIONS, MAX_SINGLE = 3, 4, 2


class GapAnalystAgent(Agent):
    agent_type = "gap"
    description = "Identifies candidate gaps in the indexed corpus: uncovered question terms, stated limitations, single-source findings."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        assessment, _ = critic_assessment(task)
        lit = next(o for o in task.dependency_outputs.values() if o.agent_type == "literature")
        surface = task.input.get("surface") or {}
        n_docs = assessment["documents"]
        titles = {d["document_id"]: d["title"] for d in lit.data["documents"]}
        gaps: list[dict] = []

        def add(kind: str, statement: str, refs: list[dict], **extra: str) -> None:
            gaps.append({"id": f"g{len(gaps) + 1}", "kind": kind, "statement": statement, "evidence": refs, **extra})

        # with no accepted evidence there is nothing to compare against: report that, not a list of 'gaps'
        for term in (assessment["uncovered_terms"][:MAX_UNADDRESSED] if n_docs else []):
            word = surface.get(term, term)
            add("unaddressed_term",
                f"None of the {n_docs} accepted document(s) addresses '{word}' together with the rest of the question. "
                "This describes the indexed corpus searched, not the literature as a whole.", [], term=word)

        docs_seen: set[str] = set()
        for s in sorted((s for s in lit.data["statements"] if s["kind"] == "limitation"),
                        key=lambda s: (s["document_id"] in docs_seen, s["id"])):  # spread across documents first
            if sum(g["kind"] == "stated_limitation" for g in gaps) >= MAX_LIMITATIONS:
                break
            docs_seen.add(s["document_id"])
            add("stated_limitation",
                f"{titles.get(s['document_id'], s['document_id'])} states a limitation that later work could address: "
                f"\"{s['text']}\"",
                [{"document_id": s["document_id"], "page_number": s["page_number"], "passage_id": s["passage_id"]}],
                quote=s["text"])

        term_docs: dict[str, dict[str, list[dict]]] = {}
        for a in assessment["accepted"]:
            for t in a["covered_terms"]:
                term_docs.setdefault(t, {}).setdefault(a["document_id"], []).append(a)
        if n_docs >= 2:
            single = [(t, d) for t, d in sorted(term_docs.items()) if len(d) == 1]
            for t, d in single[:MAX_SINGLE]:
                (doc_id, accs), = d.items()
                add("single_source", f"Evidence for '{surface.get(t, t)}' comes from a single document "
                                     f"({titles.get(doc_id, doc_id)}); it has not been corroborated within the indexed corpus.",
                    [{"document_id": doc_id, "page_number": accs[0]["page_number"], "passage_id": accs[0]["passage_id"]}])

        refs = [EvidenceReference(**r) for g in gaps for r in g["evidence"]]
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok" if gaps else "insufficient_evidence",
            summary=f"{len(gaps)} candidate gap(s) identified in the indexed corpus.",
            findings=[g["statement"][:200] for g in gaps], evidence=refs,
            uncertainties=([] if n_docs else ["No accepted evidence, so no gaps were derived: with nothing found, gap analysis "
                                              "would only restate the question."]) + ["Gaps are candidates derived from word overlap and quoted limitations; absence in this corpus is "
                           "not absence in the literature, and a human must judge whether a gap is real or worthwhile."],
            data={"gaps": gaps},
        )
