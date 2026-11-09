from __future__ import annotations

import copy
import hashlib
import json

from fastapi import Depends, APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select

from agent_engine.reporting import render_structured_markdown, report_from_store
from api import claims_service
from api.db import utcnow
from api.ratelimit import rate_limit
from api.deps import DB, CurrentAuth, audit
from api.models import Report, ReportVersion
from api.report_export import display_title, slug, to_markdown, to_pdf
from api.routes.projects import owned_project
from api.routes.runs import owned_run
from api.runtime import NoCorpus
from api.schemas import ReportPatch

router = APIRouter(tags=["reports"])
MAX_ANNOTATION = 1000
MAX_ANNOTATIONS = 200
EMPTY_EDITABLE = {"title": "", "sections": [], "annotations": {}, "review": {"status": "pending", "comment": "", "reviewer": None, "at": None}}


def content_hash(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def owned_report(db, report_id: str, user_id: str) -> Report:
    r = db.get(Report, report_id)
    if r is None or r.owner_id != user_id:
        raise HTTPException(404, "report not found")
    return r


def referenceable_ids(content: dict) -> set[str]:
    ids = {f["claim_id"] for fs in content["findings"].values() for f in fs}
    ids |= {n["claim_id"] for n in content["needs_review"]} | {u["claim_id"] for u in content["unsupported_claims"]}
    ids |= {g["id"] for g in content["candidate_gaps"]} | {h["id"] for h in content["hypotheses"]}
    return ids


def meta(r: Report) -> dict:
    return {"run_id": r.run_id, "generated_at": r.created_at.isoformat(timespec="seconds"), "content_sha256": r.content_sha256,
            "version": r.version, "edits": r.version - 1}


def _intact(r: Report) -> bool:
    return content_hash(r.content) == r.content_sha256


def _view(r: Report) -> dict:
    return {"id": r.id, "project_id": r.project_id, "run_id": r.run_id, "title": display_title(r.content, r.editable),
            "version": r.version, "created_at": r.created_at, "updated_at": r.updated_at, "meta": meta(r),
            "content_intact": _intact(r), "content": r.content, "editable": r.editable,
            "notice": "'content' is the immutable generated analysis; only 'editable' can be changed, and it is always shown "
                      "separately from the generated findings."}


def _summary(r: Report) -> dict:
    return {"id": r.id, "run_id": r.run_id, "title": display_title(r.content, r.editable), "status": r.content["status"],
            "version": r.version, "review": (r.editable.get("review") or {}).get("status", "pending"), "updated_at": r.updated_at}


@router.post("/runs/{run_id}/reports", status_code=201, dependencies=[Depends(rate_limit("report"))])
def generate_report(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    run = owned_run(db, run_id, auth.user.id)
    existing = db.scalar(select(Report).where(Report.run_id == run.id))
    if existing is not None:
        return JSONResponse(_view_json(existing), status_code=200)
    rt = request.app.state.runtime
    if rt.is_active(run.engine_run_id):
        raise HTTPException(409, "the run is still executing")
    generated = report_from_store(rt.store, run.engine_run_id)
    if generated is None:
        raise HTTPException(409, "this run has no report: it did not reach synthesis")
    content = copy.deepcopy(generated)
    try:  # embed the full source passage for every citation, so the report stands on its own
        index = rt.index_for(run.project_id)
    except NoCorpus:
        index = None
    for group in [*content["findings"].values(), content["needs_review"]]:
        for item in group:
            c = item.get("citation")
            if c:
                p = index.get(c["passage_id"]) if index is not None else None
                c["source_text"] = " ".join(p.text.split()) if p else None
    for sec in content.get("structured", {}).get("sections", []):  # every structured citation carries the stored passage text
        for block in sec["blocks"]:
            for c in block["citations"]:
                p = index.get(c["passage_id"]) if index is not None and c.get("passage_id") else None
                c["source_text"] = " ".join(p.text.split()) if p else None
                c["resolvable"] = p is not None
    content["generated_from"] = {"run_id": run.id, "question": run.question, "workflow": run.workflow}
    r = Report(project_id=run.project_id, owner_id=auth.user.id, run_id=run.id, content=content,
               content_sha256=content_hash(content), editable=copy.deepcopy(EMPTY_EDITABLE))
    db.add(r)
    db.flush()
    db.add(ReportVersion(report_id=r.id, version=1, editor_id=auth.user.id, summary="generated", editable=r.editable))
    db.commit()
    audit(db, request, "report_generate", auth.user.id, r.id, run=run.id)
    try:  # the report's claims become queryable records (idempotent); a failure here must not lose the report
        claims_service.persist_run_claims(db, rt, run, auth.user.id)
    except Exception:  # pragma: no cover - defensive
        db.rollback()
    return JSONResponse(_view_json(r), status_code=201)


def _view_json(r: Report) -> dict:
    return jsonable_encoder(_view(r))  # datetimes -> ISO strings


@router.get("/projects/{project_id}/reports")
def list_reports(project_id: str, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    rows = db.scalars(select(Report).where(Report.project_id == project_id).order_by(Report.created_at.desc())).all()
    return [_summary(r) for r in rows]


@router.get("/reports/{report_id}")
def read_report(report_id: str, auth: CurrentAuth, db: DB):
    return _view(owned_report(db, report_id, auth.user.id))


def _structured(r: Report) -> dict:
    s = r.content.get("structured")
    if s is None:
        raise HTTPException(404, "this report was generated before the structured format existed")
    return s


@router.get("/reports/{report_id}/structured")
def read_structured(report_id: str, auth: CurrentAuth, db: DB):
    """The 18-section view of the immutable generated content (every block cites a stored passage or is labelled UNVERIFIED)."""
    r = owned_report(db, report_id, auth.user.id)
    return {"report_id": r.id, "version": r.version, "content_sha256": r.content_sha256, "structured": _structured(r),
            "verification": r.content.get("verification", {}).get("history", [])}


@router.get("/reports/{report_id}/structured.md")
def read_structured_markdown(report_id: str, auth: CurrentAuth, db: DB):
    r = owned_report(db, report_id, auth.user.id)
    _export_guard(r)
    return _download(render_structured_markdown(_structured(r)), "text/markdown; charset=utf-8", r, "md")


@router.patch("/reports/{report_id}")
def update_report(report_id: str, body: ReportPatch, request: Request, auth: CurrentAuth, db: DB):
    r = owned_report(db, report_id, auth.user.id)
    if body.base_version != r.version:
        raise HTTPException(409, f"the report is at version {r.version}; reload it before editing (you edited version {body.base_version})")
    new = copy.deepcopy(r.editable)
    if body.title is not None:
        new["title"] = body.title.strip()
    if body.sections is not None:
        new["sections"] = [{"heading": s.heading.strip(), "body": s.body} for s in body.sections]
    if body.annotations is not None:
        valid = referenceable_ids(r.content)
        if len(body.annotations) > MAX_ANNOTATIONS:
            raise HTTPException(422, f"at most {MAX_ANNOTATIONS} annotations")
        for cid, text in body.annotations.items():
            if cid not in valid:
                raise HTTPException(422, f"annotation refers to unknown claim, gap or hypothesis '{cid}'")
            if len(text) > MAX_ANNOTATION:
                raise HTTPException(422, f"annotation on {cid} is longer than {MAX_ANNOTATION} characters")
        new["annotations"] = {k: v for k, v in body.annotations.items() if v.strip()}
    if body.review is not None:
        new["review"] = {"status": body.review.status, "comment": body.review.comment,
                         "reviewer": auth.user.email if body.review.status != "pending" else None,
                         "at": utcnow().isoformat(timespec="seconds") if body.review.status != "pending" else None}
    if new == r.editable:
        return _view(r)  # nothing changed: no new version
    r.editable = new
    r.version += 1
    db.add(ReportVersion(report_id=r.id, version=r.version, editor_id=auth.user.id, summary=body.summary, editable=new))
    db.commit()
    audit(db, request, "report_update", auth.user.id, r.id, version=r.version)
    return _view(r)


@router.get("/reports/{report_id}/versions")
def versions(report_id: str, auth: CurrentAuth, db: DB):
    r = owned_report(db, report_id, auth.user.id)
    return [{"version": v.version, "editor_id": v.editor_id, "summary": v.summary, "at": v.at, "editable": v.editable}
            for v in r.versions]


@router.delete("/reports/{report_id}", status_code=204)
def delete_report(report_id: str, request: Request, auth: CurrentAuth, db: DB):
    r = owned_report(db, report_id, auth.user.id)
    db.delete(r)
    db.commit()
    audit(db, request, "report_delete", auth.user.id, report_id)


def _export_guard(r: Report) -> None:
    if not _intact(r):
        raise HTTPException(409, "the stored report no longer matches its content hash; it will not be exported")


def _download(data: bytes | str, media: str, r: Report, ext: str) -> Response:
    return Response(data, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{slug(display_title(r.content, r.editable))}.{ext}"',
        "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"})


@router.get("/reports/{report_id}/export.md", dependencies=[Depends(rate_limit("report"))])
def export_markdown(report_id: str, auth: CurrentAuth, db: DB):
    r = owned_report(db, report_id, auth.user.id)
    _export_guard(r)
    return _download(to_markdown(r.content, r.editable, meta(r)), "text/markdown; charset=utf-8", r, "md")


@router.get("/reports/{report_id}/export.pdf", dependencies=[Depends(rate_limit("report"))])
def export_pdf(report_id: str, auth: CurrentAuth, db: DB):
    r = owned_report(db, report_id, auth.user.id)
    _export_guard(r)
    return _download(to_pdf(r.content, r.editable, meta(r)), "application/pdf", r, "pdf")


@router.get("/reports/{report_id}/export.json", dependencies=[Depends(rate_limit("report"))])
def export_json(report_id: str, auth: CurrentAuth, db: DB):
    r = owned_report(db, report_id, auth.user.id)
    _export_guard(r)
    body = json.dumps({"meta": meta(r), "content": r.content, "editable": r.editable}, indent=2, ensure_ascii=False)
    return _download(body, "application/json", r, "json")
