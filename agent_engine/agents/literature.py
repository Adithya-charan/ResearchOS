"""Literature Analyst: reads the documents the Critic accepted and extracts, per document, the methods, results and stated
limitations - as verbatim sentences with their page and passage. Extractive only: nothing is paraphrased or generated."""
from __future__ import annotations

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.agents.textutil import classify, has_number, sentences
from agent_engine.contracts import AgentOutput, EvidenceReference, TaskSpec
from agent_engine.errors import NonRetryableError
from research.bm25.analysis import analyze

MAX_DOCS = 8
PER_KIND = {"method": 3, "result": 3, "limitation": 2}


def critic_assessment(task: TaskSpec) -> tuple[dict, AgentOutput]:
    critic = [o for o in task.dependency_outputs.values() if o.agent_type == "critic"]
    if not critic:
        raise NonRetryableError(f"'{task.agent_type}' needs the critic's assessment as a dependency")
    return critic[-1].data["assessment"], critic[-1]


class LiteratureAgent(Agent):
    agent_type = "literature"
    description = "Extracts methods, results and stated limitations (verbatim sentences with page citations) from accepted documents."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        assessment, _ = critic_assessment(task)
        terms = [str(t) for t in task.input.get("terms", [])]
        accepted_by_doc: dict[str, dict] = {}
        for a in assessment["accepted"]:  # solid evidence only: unreliable pages never feed the analysis
            best = accepted_by_doc.setdefault(a["document_id"], {"score": 0.0, "passages": set()})
            best["score"] = max(best["score"], a["score"])
            best["passages"].add(a["passage_id"])
        ranked = sorted(accepted_by_doc, key=lambda d: (-accepted_by_doc[d]["score"], d))[:MAX_DOCS]

        documents, statements, evidence = [], [], []
        for doc_id in ranked:
            res = await ctx.tools.call("corpus.document_passages", document_id=doc_id)
            if not res.found:
                continue
            candidates: dict[str, list[tuple[float, int, dict]]] = {k: [] for k in PER_KIND}
            read = skipped = 0
            for order, p in enumerate(res.passages):
                if p.quality != "ok":
                    skipped += 1
                    continue
                read += 1
                for sent in sentences(p.text):
                    kind = classify(sent, p.section)
                    if kind is None:
                        continue
                    overlap = len(set(analyze(sent)) & set(terms)) / max(1, len(terms))
                    relevant = overlap > 0 or p.passage_id in accepted_by_doc[doc_id]["passages"]
                    if kind != "limitation" and not relevant:
                        continue
                    score = overlap + (0.1 if has_number(sent) else 0.0) + (0.05 if p.passage_id in accepted_by_doc[doc_id]["passages"] else 0)
                    candidates[kind].append((-score, order, {
                        "kind": kind, "text": sent, "document_id": doc_id, "passage_id": p.passage_id,
                        "page_number": p.page_number, "section": p.section}))
            entry = {"document_id": doc_id, "title": res.info.get("title", doc_id), "year": res.info.get("year"),
                     "doi": res.info.get("doi"), "license": res.info.get("license"), "passages_read": read,
                     "unreliable_passages_skipped": skipped, "statement_ids": []}
            for kind, cap in PER_KIND.items():
                seen: set[str] = set()
                for _, _, st in sorted(candidates[kind], key=lambda c: c[:2]):
                    if st["text"] in seen:
                        continue
                    if len(seen) >= cap:
                        break
                    seen.add(st["text"])
                    st["id"] = f"s{len(statements) + 1:03d}"
                    statements.append(st)
                    entry["statement_ids"].append(st["id"])
                    evidence.append(EvidenceReference(document_id=doc_id, page_number=st["page_number"], passage_id=st["passage_id"]))
            documents.append(entry)

        counts = {k: sum(1 for s in statements if s["kind"] == k) for k in PER_KIND}
        uncertainties = ["Statements are picked by section titles and cue words (heuristics): they are quotations, but whether "
                         "each is a document's main method, result or limitation needs human judgement."]
        if not documents:
            uncertainties.append("No accepted documents to analyse: the evidence check found nothing usable.")
        if any(d["unreliable_passages_skipped"] for d in documents):
            uncertainties.append("Passages from pages with unreliable text extraction were not analysed.")
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok" if statements else "insufficient_evidence",
            summary=f"Analysed {len(documents)} document(s): {counts['method']} method, {counts['result']} result and "
                    f"{counts['limitation']} limitation statement(s) extracted.",
            findings=[f"{d['title']}: {len(d['statement_ids'])} statement(s)" for d in documents],
            evidence=evidence, uncertainties=uncertainties,
            data={"documents": documents, "statements": statements, "terms": terms},
        )
