"""Human approval gates: experiments awaiting execution approval and agent-engine tasks held by the scheduler."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select

from api.db import utcnow
from api.deps import DB, CurrentAuth, audit
from api.models import Experiment, Run
from api.routes.runs import _launch
from api.schemas import ApprovalDecision

router = APIRouter(prefix="/approvals", tags=["approvals"])
MAX_RUNS_SCANNED = 100


def _task_items(request: Request, db, user_id: str, pending_only: bool) -> list[dict]:
    store = request.app.state.runtime.store
    runs = db.scalars(select(Run).where(Run.owner_id == user_id).order_by(Run.created_at.desc()).limit(MAX_RUNS_SCANNED)).all()
    items = []
    for r in runs:
        for a in store.list_approvals(r.engine_run_id, pending_only=pending_only):
            task = store.get_task(a["task_id"])
            items.append({"kind": "task", "id": a["id"], "run_id": r.id, "project_id": r.project_id,
                          "title": a["requested_action"], "agent_type": task["agent_type"], "task_id": task["id"],
                          "task_input": task.get("input"), "decision": a["decision"], "reviewer": a["reviewer_id"],
                          "comment": a["notes"], "requested_at": a["created_at"], "decided_at": a["decided_at"]})
    return items


def _experiment_items(db, user_id: str, pending_only: bool) -> list[dict]:
    q = select(Experiment).where(Experiment.owner_id == user_id, Experiment.status != "draft")
    items = []
    for e in db.scalars(q.order_by(Experiment.created_at.desc())).all():
        a = e.approval or {}
        if not a.get("requested_at") or (pending_only and a.get("decision")):
            continue
        items.append({"kind": "experiment", "id": e.id, "project_id": e.project_id,
                      "title": f"run experiment '{e.name}' in a sandbox", "spec_sha256": a.get("spec_sha256"),
                      "decision": a.get("decision"), "reviewer": a.get("reviewer_id"), "comment": a.get("comment"),
                      "requested_at": a.get("requested_at"), "decided_at": a.get("decided_at"),
                      "review_hint": "Read the code (GET /experiments/{id}) before approving; approval is bound to its hash.",
                      "code_sha256": e.spec_sha256, "generated_by_model": bool(e.spec.get("generated_by_model")),
                      "limits": {k: e.spec.get(k) for k in ("timeout_s", "memory_mb", "cpus", "max_output_bytes")}})
    return items


@router.get("")
def list_approvals(request: Request, auth: CurrentAuth, db: DB, status: str = Query("pending", pattern="^(pending|all)$")):
    pending = status == "pending"
    items = _task_items(request, db, auth.user.id, pending) + _experiment_items(db, auth.user.id, pending)
    return sorted(items, key=lambda i: str(i["requested_at"]), reverse=True)


def _decide(request: Request, db, auth, kind: str, ident: str, approve: bool, body: ApprovalDecision) -> dict:
    if not approve and not body.comment.strip():
        raise HTTPException(422, "a comment is required when rejecting")
    verdict = "approved" if approve else "rejected"
    if kind == "experiment":
        e = db.get(Experiment, ident)
        if e is None or e.owner_id != auth.user.id:
            raise HTTPException(404, "approval not found")
        a = dict(e.approval or {})
        if e.status != "pending_approval" or a.get("decision"):
            raise HTTPException(409, "this experiment is not awaiting a decision")
        if a.get("spec_sha256") != e.spec_sha256:
            raise HTTPException(409, "the specification changed since approval was requested; request approval again")
        e.approval = {**a, "decision": verdict, "reviewer_id": auth.user.email, "comment": body.comment,
                      "decided_at": utcnow().isoformat(timespec="seconds")}
        e.status = "approved" if approve else "rejected"
        db.commit()
        audit(db, request, f"experiment_{verdict}", auth.user.id, e.id, spec_sha256=e.spec_sha256)
        return {"kind": kind, "id": e.id, "decision": verdict, "experiment_status": e.status}
    if kind == "task":
        store = request.app.state.runtime.store
        try:
            ap = store.get_approval(ident)
        except KeyError:
            raise HTTPException(404, "approval not found") from None
        run = db.scalar(select(Run).where(Run.engine_run_id == ap["run_id"], Run.owner_id == auth.user.id))
        if run is None:
            raise HTTPException(404, "approval not found")  # another user's approval looks like a missing one
        if ap["decision"] is not None:
            raise HTTPException(409, "this approval was already decided")
        request.app.state.runtime.controller().decide_approval(ident, approve, auth.user.email, body.comment)
        audit(db, request, f"task_{verdict}", auth.user.id, ident, run=run.id)
        _launch(request, db, auth, run)  # the run was paused waiting for this decision: continue it
        return {"kind": kind, "id": ident, "decision": verdict, "run_id": run.id}
    raise HTTPException(404, "unknown approval kind")


@router.post("/{kind}/{ident}/approve")
def approve(kind: str, ident: str, request: Request, auth: CurrentAuth, db: DB, body: ApprovalDecision = ApprovalDecision()):
    return _decide(request, db, auth, kind, ident, True, body)


@router.post("/{kind}/{ident}/reject")
def reject(kind: str, ident: str, request: Request, auth: CurrentAuth, db: DB, body: ApprovalDecision = ApprovalDecision()):
    return _decide(request, db, auth, kind, ident, False, body)
