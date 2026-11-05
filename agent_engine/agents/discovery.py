"""Discovery: finds candidate passages in the indexed corpus for one query."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.contracts import AgentOutput, EvidenceReference, TaskSpec
from agent_engine.errors import NonRetryableError


class DiscoveryAgent(Agent):
    agent_type = "discovery"
    description = "Searches the indexed corpus (BM25) for passages relevant to one query and reports them as evidence."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        query = str(task.input.get("query", "")).strip()
        if not query:
            raise NonRetryableError("discovery task has no query")
        k = int(task.input.get("k", 8))
        res = await ctx.tools.call("corpus.search", query=query, k=k)

        hits = res.hits
        corpus = {"documents": res.corpus_documents, "passages": res.corpus_passages}
        uncertainties = [
            f"Searched {corpus['documents']} documents / {corpus['passages']} passages with lexical BM25 only; "
            "relevant text that uses different words would not be found."
        ]
        if not hits:
            uncertainties.append("No passage matched any query term.")
        low = [h for h in hits if h.quality != "ok"]
        if low:
            uncertainties.append(f"{len(low)} of {len(hits)} returned passage(s) come from pages whose text extraction looks "
                                 "unreliable; check the source document before relying on them.")
        ctx.memory.put(f"discovery/{task.id}", {"query": query, "passage_ids": [h.passage_id for h in hits]})
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok" if hits else "insufficient_evidence",
            summary=f"Query '{query[:120]}' returned {len(hits)} passage(s) from {len({h.document_id for h in hits})} document(s).",
            # snippets are untrusted document text, already sanitised and length-capped by the tool
            findings=[f"{h.document_id} [{h.section or 'body'}]{'' if h.quality == 'ok' else ' (unreliable extraction)'} "
                      f"score {h.score:.2f}: {h.snippet}" for h in hits[:5]],
            evidence=[EvidenceReference(document_id=h.document_id, page_number=h.page_number, passage_id=h.passage_id) for h in hits],
            uncertainties=uncertainties,
            data={
                "query": query, "k": k, "label": task.input.get("label"), "query_terms": res.query_terms, "corpus": corpus,
                "hits": [{"passage_id": h.passage_id, "document_id": h.document_id, "page_number": h.page_number,
                          "section": h.section, "score": h.score, "quality": h.quality} for h in hits],
            },
        )
