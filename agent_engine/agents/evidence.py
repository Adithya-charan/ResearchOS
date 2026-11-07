"""Claim extraction, evidence matching and novelty assessment for the Verifier. Pure orchestration around research.evidence: passages
are fetched through the policy-checked corpus tools (so every lookup is an auditable tool_call event) and the decisions are made by
the deterministic engine in research/evidence. Nothing here writes prose of its own."""
from __future__ import annotations

from agent_engine.agents.base import AgentContext
from agent_engine.contracts import TaskSpec
from research.bm25.analysis import term_surface_forms
from research.evidence import Candidate, assess_novelty, verify_claim
from research.evidence.types import UNCHECKABLE_TYPES

VERIFIER_VERSION = "deterministic-lexical-v1"
MAX_CLAIMS = 24      # claims searched per verification pass (the rest are listed as UNVERIFIED with the reason)
SEARCH_K = 5         # corroboration passages retrieved per claim
NOVELTY_K = 8
MAX_NOVELTY = 6      # gap / hypothesis statements assessed per pass
FOLLOWUP_LABEL = "evidence follow-up"
NEEDS_FOLLOWUP = ("PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONTRADICTED")


class _Passages:
    """Fetches each passage once per pass, via the corpus tools."""

    def __init__(self, ctx: AgentContext):
        self.ctx, self.cache = ctx, {}

    async def get(self, passage_id: str):
        if passage_id not in self.cache:
            self.cache[passage_id] = await self.ctx.tools.call("corpus.get_passage", passage_id=passage_id)
        return self.cache[passage_id]

    async def candidate(self, passage_id: str, method: str, score: float | None, document_id: str | None = None) -> Candidate:
        p = await self.get(passage_id)
        if not p.found:
            return Candidate(passage_id=passage_id, document_id=document_id or "", resolved=False, retrieval_method=method,
                             retrieval_score=score)
        return Candidate(passage_id=p.passage_id, document_id=p.document_id, text=p.text, page_number=p.page_number,
                         quality=p.quality, resolved=True, retrieval_method=method, retrieval_score=score)


def _hit_candidate(h, method: str) -> Candidate:
    """A search hit already carries the stored passage text, so it needs no second lookup."""
    return Candidate(passage_id=h.passage_id, document_id=h.document_id, text=h.text, page_number=h.page_number, quality=h.quality,
                     resolved=True, retrieval_method=method, retrieval_score=h.score)


def _dep_ids(task: TaskSpec) -> dict[str, str]:
    return {o.agent_type: tid for tid, o in task.dependency_outputs.items()}


def claim_sources(task: TaskSpec) -> list[dict]:
    """The claims the workflow made, as structured records with their originating agent and task."""
    ids, by = _dep_ids(task), {o.agent_type: o for o in task.dependency_outputs.values()}
    out: list[dict] = []
    lit, hyp, gap = by.get("literature"), by.get("hypothesis"), by.get("gap")
    for s in (lit.data["statements"] if lit else []):
        out.append({"key": f"literature:{s['id']}", "text": s["text"], "claim_type": s["kind"], "origin_agent": "literature",
                    "origin_task_id": ids.get("literature"), "source_id": s["id"],
                    "source": {"document_id": s["document_id"], "passage_id": s["passage_id"], "page_number": s["page_number"]}})
    for g in (gap.data["gaps"] if gap else []):
        out.append({"key": f"gap:{g['id']}", "text": g["statement"], "claim_type": "gap", "origin_agent": "gap",
                    "origin_task_id": ids.get("gap"), "source_id": g["id"], "source": None, "refs": g["evidence"]})
    for h in (hyp.data["hypotheses"] if hyp else []):
        out.append({"key": f"hypothesis:{h['id']}", "text": h["statement"], "claim_type": "hypothesis", "origin_agent": "hypothesis",
                    "origin_task_id": ids.get("hypothesis"), "source_id": h["id"], "source": None, "refs": h["evidence"]})
    return out


def _followup_hits(task: TaskSpec) -> list[dict]:
    hits: list[dict] = []
    for o in task.dependency_outputs.values():
        if o.agent_type == "discovery" and o.data.get("label") == FOLLOWUP_LABEL:
            hits += [{**h, "query": o.data.get("query")} for h in o.data.get("hits", [])]
    return hits


async def build_verification(ctx: AgentContext, task: TaskSpec, old_claims: list[dict], round_no: int) -> dict:
    """Verify every checkable claim against its source passage plus corroborating/contradicting passages found by search."""
    can_search = ctx.tools.has("corpus.evidence_search")
    passages = _Passages(ctx)
    only = set(task.input.get("claim_keys") or [])
    old = {c["source_id"]: c for c in old_claims if c.get("source_id")}
    follow = _followup_hits(task)
    records: list[dict] = []
    sources = [c for c in claim_sources(task) if not only or c["key"] in only]
    capped = {c["key"] for c in [c for c in sources if c["claim_type"] not in UNCHECKABLE_TYPES][MAX_CLAIMS:]}
    for c in sources:
        cands: list[Candidate] = []
        searched = False
        if c["claim_type"] in UNCHECKABLE_TYPES:
            for ref in c.get("refs", []):
                if ref.get("passage_id"):
                    cands.append(await passages.candidate(ref["passage_id"], "source_passage", None, ref["document_id"]))
        elif c["key"] not in capped:
            src = c["source"]
            cand = await passages.candidate(src["passage_id"], "source_passage", None, src["document_id"])
            if old.get(c["source_id"], {}).get("status") == "unsupported":
                cand.resolved = False  # the earlier checks found the passage no longer matches its original file / quote
            cands.append(cand)
            if can_search:
                res = await ctx.tools.call("corpus.evidence_search", query=c["text"][:480], k=SEARCH_K)
                searched = True
                for h in res.hits:
                    if h.passage_id != src["passage_id"]:
                        cands.append(_hit_candidate(h, "bm25_search"))
            seen = {x.passage_id for x in cands}
            for h in follow:
                if h["passage_id"] not in seen:
                    seen.add(h["passage_id"])
                    cands.append(await passages.candidate(h["passage_id"], "followup_search", h["score"], h["document_id"]))
            searched = searched or bool(follow)
        v = verify_claim(c["text"], c["claim_type"], cands, searched=searched)
        if c["key"] in capped:
            v["explanation"].append(f"not checked: more than {MAX_CLAIMS} claims in this pass")
        rec = {k: c[k] for k in ("key", "text", "claim_type", "origin_agent", "origin_task_id", "source_id", "source")}
        rec.update(v, round=round_no, verifier=VERIFIER_VERSION)
        records.append(rec)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    needs = [{"key": r["key"], "text": r["text"], "status": r["status"]} for r in records
             if r["status"] in NEEDS_FOLLOWUP and r["claim_type"] not in UNCHECKABLE_TYPES]
    return {"round": round_no, "verifier": VERIFIER_VERSION, "claims": records, "counts": counts, "needs_followup": needs,
            "scope_note": "Statuses are lexical evidence-strength labels, not proofs; see docs/EVIDENCE_VERIFICATION.md."}


async def build_novelty(ctx: AgentContext, task: TaskSpec) -> list[dict]:
    """A bounded novelty assessment for each gap and hypothesis, excluding the documents the statement was derived from."""
    if not ctx.tools.has("corpus.evidence_search"):
        return []
    passages = _Passages(ctx)
    lit = next((o for o in task.dependency_outputs.values() if o.agent_type == "literature"), None)
    info = {d["document_id"]: {"title": d.get("title"), "year": d.get("year"), "doi": d.get("doi")}
            for d in (lit.data["documents"] if lit else [])}
    out: list[dict] = []
    for c in [c for c in claim_sources(task) if c["claim_type"] in UNCHECKABLE_TYPES][:MAX_NOVELTY]:
        source_docs = {r["document_id"] for r in c.get("refs", [])}
        query = " ".join(s for _, s in term_surface_forms(c["text"], limit=12)) or c["text"][:200]
        res = await ctx.tools.call("corpus.evidence_search", query=query[:480], k=NOVELTY_K)
        cands = [_hit_candidate(h, "bm25_search") for h in res.hits if h.document_id not in source_docs]
        a = assess_novelty(c["text"], passages=cands, corpus={"documents": res.corpus_documents, "passages": res.corpus_passages},
                           queries=[query], doc_info=info, statement_kind=c["claim_type"])
        a.update(claim_key=c["key"], excluded_source_documents=sorted(source_docs), origin_agent=c["origin_agent"],
                 origin_task_id=c["origin_task_id"])
        out.append(a)
    return out


def merged_verification(outputs) -> dict:
    """Combine the verification passes of one run: the latest pass that looked at a claim decides its status; every pass is kept
    in `history`. `outputs` are AgentOutputs of any agent types (non-verifier outputs are ignored)."""
    passes = sorted((o for o in outputs if o.agent_type == "verifier" and "verification" in o.data),
                    key=lambda o: o.data.get("round", 1))
    claims: dict[str, dict] = {}
    history: list[dict] = []
    novelty: list[dict] = []
    for o in passes:
        v = o.data["verification"]
        for c in v["claims"]:
            if c["key"] in claims:
                c = {**c, "previous_status": claims[c["key"]]["status"]}
            claims[c["key"]] = c
        history.append({"round": v["round"], "claims_checked": len(v["claims"]), "counts": v["counts"],
                        "needs_followup": len(v["needs_followup"])})
        novelty = novelty or o.data.get("novelty", [])
    recs = list(claims.values())
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    needs = [{"key": r["key"], "text": r["text"], "status": r["status"]} for r in recs
             if r["status"] in NEEDS_FOLLOWUP and r["claim_type"] not in UNCHECKABLE_TYPES]
    return {"claims": recs, "counts": counts, "needs_followup": needs, "history": history, "rounds": len(passes),
            "novelty": novelty}
