"""Read-only aggregates for the web dashboard and activity page. Everything is scoped to the signed-in user and computed from stored
records; nothing is estimated or invented. (Added in P9: the UI would otherwise have to fetch every run of every project.)"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

from fastapi import APIRouter, Query, Request
from sqlalchemy import func, select

from api.db import utcnow
from api.deps import DB, CurrentAuth
from api.models import AuditLog, Claim, Project, Report, Run
from api.routes.approvals import _experiment_items, _task_items

router = APIRouter(tags=["overview"])
MAX_RUNS_SCANNED = 200
DAYS = 30
CLAIM_STATUSES = ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "UNVERIFIED")


@router.get("/overview")
def overview(request: Request, auth: CurrentAuth, db: DB):
    uid, rt = auth.user.id, request.app.state.runtime
    projects = db.scalars(select(Project).where(Project.owner_id == uid).order_by(Project.updated_at.desc())).all()
    total_runs = db.scalar(select(func.count()).select_from(Run).where(Run.owner_id == uid)) or 0
    runs = db.scalars(select(Run).where(Run.owner_id == uid).order_by(Run.created_at.desc()).limit(MAX_RUNS_SCANNED)).all()
    status_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    recent, active = [], 0
    for i, r in enumerate(runs):
        st = rt.status(r.engine_run_id)
        active += 1 if st["active"] else 0
        status_counts[st["status"]] += 1
        for k, v in (st.get("task_counts") or {}).items():
            task_counts[k] += v
        if i < 8:
            recent.append({"id": r.id, "project_id": r.project_id, "question": r.question, "created_at": r.created_at,
                           "status": st["status"], "active": st["active"]})
    since = utcnow() - timedelta(days=DAYS - 1)
    per_day: Counter[str] = Counter(r.created_at.date().isoformat() for r in runs if r.created_at >= since)
    series = [{"date": (since + timedelta(days=d)).date().isoformat(),
               "runs": per_day.get((since + timedelta(days=d)).date().isoformat(), 0)} for d in range(DAYS)]
    mgr = request.app.state.corpora
    docs = 0
    for p in projects:
        if mgr.exists(p.id):
            with mgr.lock(p.id):
                docs += mgr.get(p.id).stats().get("documents", 0)
    claim_rows = db.execute(select(Claim.status, func.count()).where(Claim.owner_id == uid).group_by(Claim.status)).all()
    claims = {s: 0 for s in CLAIM_STATUSES} | {s: n for s, n in claim_rows}
    approvals = len(_task_items(request, db, uid, True)) + len(_experiment_items(db, uid, True))
    reports = db.scalar(select(func.count()).select_from(Report).where(Report.owner_id == uid)) or 0
    return {
        "projects": len(projects), "runs": total_runs, "runs_scanned": len(runs),
        "active_runs": active,
        "completed_runs": status_counts.get("completed", 0), "pending_approvals": approvals, "documents": docs,
        "claims": claims, "claims_total": sum(claims.values()), "reports": reports,
        "run_status": dict(status_counts), "task_status": dict(task_counts), "runs_per_day": series,
        "recent_runs": recent,
        "recent_projects": [{"id": p.id, "title": p.title, "question": p.question, "updated_at": p.updated_at} for p in projects[:5]],
        "note": f"Run statuses and task counts cover the latest {MAX_RUNS_SCANNED} runs; the run total is exact.",
    }


@router.get("/audit")
def my_audit(auth: CurrentAuth, db: DB, action: str | None = Query(None, max_length=60), limit: int = Query(50, ge=1, le=200),
             offset: int = Query(0, ge=0)):
    """The caller's own audit entries only (never another user's). Details never contain passwords, tokens or document text."""
    q = select(AuditLog).where(AuditLog.user_id == auth.user.id)
    if action:
        q = q.where(AuditLog.action == action)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(AuditLog.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "entries": [{"id": a.id, "at": a.at, "action": a.action, "target": a.target, "detail": a.detail} for a in rows],
            "actions": [a for (a,) in db.execute(select(AuditLog.action).where(AuditLog.user_id == auth.user.id).distinct().order_by(AuditLog.action))]}
