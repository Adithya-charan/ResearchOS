"""Claim persistence and verification for the API. The decisions are made by research.evidence (deterministic, lexical); this module
only gathers real passages from the project's own library, stores results, and refuses anything it cannot resolve - a caller can
attach a passage id but never supply a quote, page or document: those always come from the stored passage."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from agent_engine.reporting import report_from_store
from api.db import utcnow
from api.models import Claim, ClaimEvidence, ClaimVerification, NoveltyAssessmentRow
from api.runtime import NoCorpus
from research.bm25.analysis import term_surface_forms
from research.evidence import Candidate, assess_novelty, verify_claim

VERIFIER = "deterministic-lexical-v1"
SEARCH_K = 5
NOVELTY_K = 8
MAX_CLAIMS_PER_PROJECT = 2000
MAX_EVIDENCE_PER_CLAIM = 50


class Unresolvable(Exception):
    """A passage id that does not exist in this project's library (or no longer matches its original file)."""


class _Library:
    """The project's searchable passages, plus the optional re-check against the stored original files."""

    def __init__(self, rt, project_id: str):
        self.rt, self.project_id = rt, project_id
        self.index = rt.index_for(project_id)  # raises NoCorpus when the project has no documents

    def candidate(self, passage_id: str, method: str, score: float | None = None) -> Candidate:
        p = self.index.get(passage_id)
        if p is None:
            return Candidate(passage_id=passage_id, document_id="", resolved=False, retrieval_method=method, retrieval_score=score)
        ok = True
        mgr = self.rt.corpora
        if mgr is not None and self.rt._index_provider is None and mgr.exists(self.project_id):
            with mgr.lock(self.project_id):
                ok = mgr.get(self.project_id).verify_passage(passage_id).ok  # the stored original must still match
        return Candidate(passage_id=p.passage_id, document_id=p.document_id, text=p.text, page_number=p.page_number,
                         quality=p.quality, resolved=ok, retrieval_method=method, retrieval_score=score)

    def doc_info(self, doc_ids: set[str]) -> dict[str, dict]:
        mgr, out = self.rt.corpora, {}
        if mgr is None or self.rt._index_provider is not None or not mgr.exists(self.project_id):
            return out
        with mgr.lock(self.project_id):
            corpus = mgr.get(self.project_id)
            for d in doc_ids:
                rec = corpus.get_document(d)
                if rec is not None:
                    out[d] = {"title": rec.title, "year": rec.year, "doi": rec.doi}
        return out


def library(rt, project_id: str) -> _Library:
    return _Library(rt, project_id)


# ------------------------------------------------------------------ serialisation
def evidence_view(e: ClaimEvidence) -> dict:
    return {"id": e.id, "role": e.role, "directness": e.directness, "document_id": e.document_id, "passage_id": e.passage_id,
            "page_number": e.page_number, "quote": e.quote, "retrieval_method": e.retrieval_method,
            "retrieval_score": e.retrieval_score, "coverage": e.coverage, "citation_valid": bool(e.citation_valid),
            "reason": e.reason, "origin": e.origin}


def claim_view(c: Claim, detail: bool = False) -> dict:
    view = {"id": c.id, "project_id": c.project_id, "run_id": c.run_id, "claim_key": c.claim_key, "text": c.text,
            "claim_type": c.claim_type, "origin_agent": c.origin_agent, "origin_task_id": c.origin_task_id, "source": c.source,
            "status": c.status, "strength": c.strength, "created_at": c.created_at, "updated_at": c.updated_at,
            "evidence_counts": {r: sum(1 for e in c.evidence if e.role == r) for r in ("supporting", "contradicting", "contextual")}}
    if detail:
        view.update(explanation=c.explanation, evidence=[evidence_view(e) for e in c.evidence],
                    verifications=len(c.verifications),
                    scope_note="Statuses are lexical evidence-strength labels, not proofs; see docs/EVIDENCE_VERIFICATION.md.")
    return view


def verification_view(v: ClaimVerification) -> dict:
    return {"id": v.id, "claim_id": v.claim_id, "round": v.round, "status": v.status, "strength": v.strength,
            "explanation": v.explanation, "evidence": v.evidence, "verifier": v.verifier, "requested_by": v.requested_by,
            "created_at": v.created_at}


def novelty_view(n: NoveltyAssessmentRow) -> dict:
    return {"id": n.id, "project_id": n.project_id, "run_id": n.run_id, "claim_id": n.claim_id, "statement": n.statement,
            "label": n.label, "created_at": n.created_at, "assessment": n.content}


# ------------------------------------------------------------------ claims and evidence
def create_claim(db: Session, *, project_id: str, owner_id: str, text: str, claim_type: str, run_id: str | None = None,
                 key: str | None = None, origin_agent: str = "user", origin_task_id: str | None = None, source: str = "user") -> Claim:
    c = Claim(project_id=project_id, owner_id=owner_id, run_id=run_id, claim_key=key, text=text.strip(), claim_type=claim_type,
              origin_agent=origin_agent, origin_task_id=origin_task_id, source=source, status="UNVERIFIED",
              strength={"level": "none", "note": "not verified yet"}, explanation=["not verified yet"])
    db.add(c)
    db.flush()
    return c


def attach_evidence(db: Session, lib: _Library, claim: Claim, passage_id: str) -> ClaimEvidence:
    """Associate a stored passage with a claim. The passage must resolve in this project's library; page, quote and document are
    read from it. The role stays 'contextual' until a verification decides whether it supports or contradicts the claim."""
    if len(claim.evidence) >= MAX_EVIDENCE_PER_CLAIM:
        raise Unresolvable(f"a claim can have at most {MAX_EVIDENCE_PER_CLAIM} evidence items")
    cand = lib.candidate(passage_id, "attached")
    if cand.resolved is not True:
        raise Unresolvable(f"passage '{passage_id}' does not exist in this project's library")
    if any(e.passage_id == passage_id for e in claim.evidence):
        return next(e for e in claim.evidence if e.passage_id == passage_id)
    e = ClaimEvidence(claim_id=claim.id, role="contextual", document_id=cand.document_id, passage_id=cand.passage_id,
                      page_number=cand.page_number, quote=" ".join(cand.text.split())[:400], retrieval_method="attached",
                      reason="attached by a person; not yet verified", origin="user")
    db.add(e)
    claim.evidence.append(e)
    db.flush()
    return e


def _store_verification(db: Session, claim: Claim, v: dict, requested_by: str, round_no: int, replace_origin: str = "verification") -> ClaimVerification:
    by_passage = {e["passage_id"]: e for e in [*v["evidence"], *v["invalid_citations"], *v["unreliable_passages"]]}
    for row in list(claim.evidence):
        if row.origin == replace_origin:
            claim.evidence.remove(row)
        elif row.passage_id in by_passage:  # a person-attached item: keep the row, refresh what the verifier decided about it
            r = by_passage[row.passage_id]
            row.role, row.directness = r.get("role", "contextual"), r.get("directness")
            row.coverage, row.reason = r.get("coverage"), r.get("reason", "")
            row.citation_valid = int(r.get("citation_valid", True))
            row.quote = r.get("quote", row.quote)
            row.retrieval_method, row.retrieval_score = r.get("retrieval_method", row.retrieval_method), r.get("retrieval_score")
        else:
            row.role, row.directness, row.reason = "contextual", None, "shares too little with the claim to count as evidence"
    attached = {row.passage_id for row in claim.evidence}
    for r in [*v["evidence"], *v["invalid_citations"], *v["unreliable_passages"]]:
        if r["passage_id"] in attached or r["retrieval_method"] == "attached":
            continue
        claim.evidence.append(ClaimEvidence(
            claim_id=claim.id, role=r.get("role", "contextual"), directness=r.get("directness"), document_id=r["document_id"],
            passage_id=r["passage_id"], page_number=r["page_number"], quote=r.get("quote", ""),
            retrieval_method=r["retrieval_method"], retrieval_score=r["retrieval_score"], coverage=r.get("coverage"),
            citation_valid=int(r.get("citation_valid", True)), reason=r.get("reason", ""), origin="verification"))
        attached.add(r["passage_id"])
    claim.status, claim.strength, claim.explanation = v["status"], v["strength"], v["explanation"]
    ver = ClaimVerification(claim_id=claim.id, round=round_no, status=v["status"], strength=v["strength"],
                            explanation=v["explanation"], evidence=[*v["evidence"], *v["invalid_citations"], *v["unreliable_passages"]],
                            verifier=VERIFIER, requested_by=requested_by)
    claim.verifications.append(ver)
    claim.updated_at = utcnow()
    db.flush()
    return ver


def verify(db: Session, lib: _Library, claim: Claim, requested_by: str) -> ClaimVerification:
    """Verify a claim against (a) the passages attached to it and (b) passages retrieved by searching its own wording."""
    cands: list[Candidate] = [lib.candidate(e.passage_id, "attached") for e in claim.evidence if e.origin == "user"]
    seen = {c.passage_id for c in cands}
    searching = claim.claim_type not in ("gap", "hypothesis")
    if searching:
        for h in lib.index.search(claim.text[:480], k=SEARCH_K):
            if h.passage.passage_id not in seen:
                seen.add(h.passage.passage_id)
                cands.append(lib.candidate(h.passage.passage_id, "bm25_search", round(h.score, 6)))
    v = verify_claim(claim.text, claim.claim_type, cands, searched=searching)
    return _store_verification(db, claim, v, requested_by, round_no=len(claim.verifications) + 1)


# ------------------------------------------------------------------ novelty
def assess(db: Session, lib: _Library, *, project_id: str, owner_id: str, statement: str, claim_id: str | None = None,
           run_id: str | None = None, exclude_documents: set[str] | None = None) -> NoveltyAssessmentRow:
    exclude_documents = exclude_documents or set()
    query = " ".join(s for _, s in term_surface_forms(statement, limit=12)) or statement[:200]
    hits = lib.index.search(query[:480], k=NOVELTY_K)
    cands = [lib.candidate(h.passage.passage_id, "bm25_search", round(h.score, 6)) for h in hits
             if h.passage.document_id not in exclude_documents]
    info = lib.doc_info({c.document_id for c in cands})
    content = assess_novelty(statement, passages=cands, corpus={"documents": lib.index.num_documents, "passages": lib.index.num_passages},
                             queries=[query], doc_info=info)
    content["excluded_source_documents"] = sorted(exclude_documents)
    row = NoveltyAssessmentRow(project_id=project_id, owner_id=owner_id, run_id=run_id, claim_id=claim_id, statement=statement,
                               label=content["label"], content=content)
    db.add(row)
    db.flush()
    return row


# ------------------------------------------------------------------ claims from a finished run
def persist_run_claims(db: Session, rt, run, owner_id: str) -> dict:
    """Store the claims, evidence, verification history and novelty assessments the run's agents produced. Idempotent: running it
    again updates the same rows. Returns counts; zeros when the run did not reach the report."""
    report = report_from_store(rt.store, run.engine_run_id)
    if report is None or "verification" not in report:
        return {"claims": 0, "created": 0, "updated": 0, "novelty": 0}
    existing = {c.claim_key: c for c in db.scalars(select(Claim).where(Claim.run_id == run.id)).all()}
    created = updated = 0
    for rec in report["verification"]["claims"]:
        c = existing.get(rec["key"])
        if c is None:
            c = create_claim(db, project_id=run.project_id, owner_id=owner_id, run_id=run.id, text=rec["text"],
                             claim_type=rec["claim_type"], key=rec["key"], origin_agent=rec["origin_agent"],
                             origin_task_id=rec["origin_task_id"], source="agent")
            existing[rec["key"]] = c
            created += 1
        else:
            updated += 1
        have = {v.round for v in c.verifications}
        if rec["round"] not in have or rec["status"] != c.status:
            _store_verification(db, c, rec, "system", rec["round"])
    novelty = 0
    for n in report["verification"]["novelty"]:
        c = existing.get(n["claim_key"])
        for old in db.scalars(select(NoveltyAssessmentRow).where(NoveltyAssessmentRow.run_id == run.id,
                                                                  NoveltyAssessmentRow.claim_id == (c.id if c else None))).all():
            db.delete(old)
        db.add(NoveltyAssessmentRow(project_id=run.project_id, owner_id=owner_id, run_id=run.id, claim_id=c.id if c else None,
                                    statement=n["statement"], label=n["label"], content=n))
        novelty += 1
    db.commit()
    return {"claims": len(report["verification"]["claims"]), "created": created, "updated": updated, "novelty": novelty}
