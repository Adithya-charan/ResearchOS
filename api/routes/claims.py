"""Claims, evidence association, verification, verification history and novelty assessment. Everything is scoped to a project the
signed-in user owns; another user's project or claim looks like a missing one (404)."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from api import claims_service as svc
from api.deps import DB, CurrentAuth, audit
from api.models import Claim, ClaimVerification, NoveltyAssessmentRow
from api.routes.projects import owned_project
from api.routes.runs import owned_run
from api.runtime import NoCorpus
from research.evidence import CLAIM_TYPES, STATUSES

router = APIRouter(tags=["claims"])
ClaimType = Literal[CLAIM_TYPES]  # type: ignore[valid-type]


class EvidenceIn(BaseModel):
    passage_id: str = Field(min_length=1, max_length=80)


class ClaimIn(BaseModel):
    text: str = Field(min_length=8, max_length=2000)
    claim_type: ClaimType = "finding"
    run_id: str | None = Field(default=None, max_length=32)
    evidence: list[EvidenceIn] = Field(default_factory=list, max_length=20)


class NoveltyIn(BaseModel):
    statement: str = Field(min_length=10, max_length=2000)
    claim_id: str | None = Field(default=None, max_length=32)
    exclude_document_ids: list[str] = Field(default_factory=list, max_length=50)


def _claim(db, project_id: str, claim_id: str, user_id: str) -> Claim:
    c = db.get(Claim, claim_id)
    if c is None or c.project_id != project_id or c.owner_id != user_id:
        raise HTTPException(404, "claim not found")
    return c


def _library(request: Request, project_id: str):
    try:
        return svc.library(request.app.state.runtime, project_id)
    except NoCorpus as e:
        raise HTTPException(409, str(e))


@router.post("/projects/{project_id}/claims", status_code=201)
def create_claim(project_id: str, body: ClaimIn, request: Request, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    if db.scalar(select(func.count()).select_from(Claim).where(Claim.project_id == project_id)) >= svc.MAX_CLAIMS_PER_PROJECT:
        raise HTTPException(429, f"claim quota reached ({svc.MAX_CLAIMS_PER_PROJECT})")
    run_id = None
    if body.run_id:
        run = owned_run(db, body.run_id, auth.user.id)
        if run.project_id != project_id:
            raise HTTPException(404, "run not found")
        run_id = run.id
    lib = _library(request, project_id) if body.evidence else None
    c = svc.create_claim(db, project_id=project_id, owner_id=auth.user.id, text=body.text, claim_type=body.claim_type, run_id=run_id)
    try:
        for e in body.evidence:
            svc.attach_evidence(db, lib, c, e.passage_id)
    except svc.Unresolvable as e:
        db.rollback()  # nothing is stored when any cited passage is unresolvable: no half-cited claim
        raise HTTPException(422, str(e))
    db.commit()
    audit(db, request, "claim_create", auth.user.id, c.id, project=project_id)
    return svc.claim_view(c, detail=True)


@router.get("/projects/{project_id}/claims")
def list_claims(project_id: str, auth: CurrentAuth, db: DB, run_id: str | None = None, status: str | None = Query(default=None),
                claim_type: str | None = None, source: Literal["agent", "user"] | None = None,
                limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0)):
    owned_project(db, project_id, auth.user.id)
    if status is not None and status not in STATUSES:
        raise HTTPException(422, f"status must be one of {', '.join(STATUSES)}")
    q = select(Claim).where(Claim.project_id == project_id, Claim.owner_id == auth.user.id)
    for col, val in ((Claim.run_id, run_id), (Claim.status, status), (Claim.claim_type, claim_type), (Claim.source, source)):
        if val is not None:
            q = q.where(col == val)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(Claim.created_at, Claim.id).limit(limit).offset(offset)).all()
    return {"total": total, "claims": [svc.claim_view(c) for c in rows]}


@router.get("/projects/{project_id}/claims/{claim_id}")
def read_claim(project_id: str, claim_id: str, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    return svc.claim_view(_claim(db, project_id, claim_id, auth.user.id), detail=True)


@router.post("/projects/{project_id}/claims/{claim_id}/evidence", status_code=201)
def add_evidence(project_id: str, claim_id: str, body: EvidenceIn, request: Request, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    c = _claim(db, project_id, claim_id, auth.user.id)
    lib = _library(request, project_id)
    try:
        e = svc.attach_evidence(db, lib, c, body.passage_id)
    except svc.Unresolvable as err:
        db.rollback()
        raise HTTPException(422, str(err))
    db.commit()
    audit(db, request, "claim_evidence_add", auth.user.id, c.id, passage=body.passage_id)
    return svc.evidence_view(e)


@router.delete("/projects/{project_id}/claims/{claim_id}/evidence/{evidence_id}", status_code=204)
def remove_evidence(project_id: str, claim_id: str, evidence_id: int, request: Request, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    c = _claim(db, project_id, claim_id, auth.user.id)
    row = next((e for e in c.evidence if e.id == evidence_id and e.origin == "user"), None)
    if row is None:
        raise HTTPException(404, "evidence not found (only evidence you attached can be removed)")
    c.evidence.remove(row)
    db.commit()
    audit(db, request, "claim_evidence_remove", auth.user.id, c.id, evidence=evidence_id)


@router.post("/projects/{project_id}/claims/{claim_id}/verify")
def verify_claim(project_id: str, claim_id: str, request: Request, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    c = _claim(db, project_id, claim_id, auth.user.id)
    lib = _library(request, project_id)
    svc.verify(db, lib, c, auth.user.id)
    db.commit()
    audit(db, request, "claim_verify", auth.user.id, c.id, status=c.status)
    return svc.claim_view(c, detail=True)


@router.get("/projects/{project_id}/claims/{claim_id}/verifications")
def claim_verifications(project_id: str, claim_id: str, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    c = _claim(db, project_id, claim_id, auth.user.id)
    return [svc.verification_view(v) for v in c.verifications]


@router.get("/projects/{project_id}/verifications")
def project_verifications(project_id: str, auth: CurrentAuth, db: DB, limit: int = Query(default=100, ge=1, le=500)):
    """Verification history across the project's claims, newest first."""
    owned_project(db, project_id, auth.user.id)
    rows = db.scalars(select(ClaimVerification).join(Claim, Claim.id == ClaimVerification.claim_id)
                      .where(Claim.project_id == project_id, Claim.owner_id == auth.user.id)
                      .order_by(ClaimVerification.id.desc()).limit(limit)).all()
    return [svc.verification_view(v) for v in rows]


@router.post("/runs/{run_id}/claims/extract")
def extract_run_claims(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    """Persist the claims the run's agents produced (with their evidence, verification history and novelty assessments)."""
    run = owned_run(db, run_id, auth.user.id)
    rt = request.app.state.runtime
    if rt.is_active(run.engine_run_id):
        raise HTTPException(409, "the run is still executing")
    result = svc.persist_run_claims(db, rt, run, auth.user.id)
    if not result["claims"]:
        raise HTTPException(409, "this run produced no claims: it did not reach verification")
    audit(db, request, "claims_extract", auth.user.id, run.id, **result)
    return result


@router.post("/projects/{project_id}/novelty", status_code=201)
def assess_novelty(project_id: str, body: NoveltyIn, request: Request, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    claim = _claim(db, project_id, body.claim_id, auth.user.id) if body.claim_id else None
    lib = _library(request, project_id)
    row = svc.assess(db, lib, project_id=project_id, owner_id=auth.user.id, statement=body.statement,
                     claim_id=claim.id if claim else None, run_id=claim.run_id if claim else None,
                     exclude_documents=set(body.exclude_document_ids))
    db.commit()
    audit(db, request, "novelty_assess", auth.user.id, row.id, label=row.label)
    return svc.novelty_view(row)


@router.get("/projects/{project_id}/novelty")
def list_novelty(project_id: str, auth: CurrentAuth, db: DB, claim_id: str | None = None):
    owned_project(db, project_id, auth.user.id)
    q = select(NoveltyAssessmentRow).where(NoveltyAssessmentRow.project_id == project_id, NoveltyAssessmentRow.owner_id == auth.user.id)
    if claim_id:
        q = q.where(NoveltyAssessmentRow.claim_id == claim_id)
    return [svc.novelty_view(n) for n in db.scalars(q.order_by(NoveltyAssessmentRow.created_at.desc())).all()]


@router.get("/projects/{project_id}/novelty/{novelty_id}")
def read_novelty(project_id: str, novelty_id: str, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    n = db.get(NoveltyAssessmentRow, novelty_id)
    if n is None or n.project_id != project_id or n.owner_id != auth.user.id:
        raise HTTPException(404, "novelty assessment not found")
    return svc.novelty_view(n)
