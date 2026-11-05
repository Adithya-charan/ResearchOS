from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlalchemy import func, select

from api.deps import DB, CurrentAuth, audit
from api.db import utcnow
from api.experiments_service import SandboxUnavailable, spec_hash
from api.models import Experiment
from api.routes.projects import owned_project
from api.schemas import ExperimentIn, ExperimentSpecPatch, ReviewIn
from research.experiments import ExperimentSpec, read_review, review_experiment
from research.experiments.runner import sha256_file

router = APIRouter(tags=["experiments"])
LOG_LIMIT = 200_000


def owned_experiment(db, exp_id: str, user_id: str) -> Experiment:
    e = db.get(Experiment, exp_id)
    if e is None or e.owner_id != user_id:
        raise HTTPException(404, "experiment not found")
    return e


def _spec(raw: dict) -> dict:
    try:
        return ExperimentSpec.model_validate(raw).model_dump()
    except ValidationError as e:
        raise HTTPException(422, "invalid experiment specification: " + "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5])) from e


def _summary(e: Experiment) -> dict:
    return {"id": e.id, "project_id": e.project_id, "name": e.name, "status": e.status, "created_at": e.created_at,
            "updated_at": e.updated_at, "approval": (e.approval or {}).get("decision") or "none"}


def _view(request: Request, e: Experiment) -> dict:
    svc = request.app.state.experiments
    manifest = svc.manifest(e.id)
    d = {**_summary(e), "spec": e.spec, "spec_sha256": e.spec_sha256, "approval_record": e.approval, "error": e.error,
         "manifest": manifest, "reproduction": e.reproduction, "review": e.review or {"status": "pending"},
         "limitations": ["A result is evidence about this code, data and seed only; a human must review it.",
                         "'Reproduced' means repeatable on this machine and sandbox, not that the experiment is valid."]}
    if manifest:
        problems = svc.integrity(e)
        d["integrity"] = {"ok": not problems, "problems": problems}
        if not manifest["isolation"]["network_blocked"]:
            d["limitations"].append("This run used the NON-isolating process fallback: " + " ".join(manifest["isolation"]["notes"]))
    return d


@router.post("/projects/{project_id}/experiments", status_code=201)
def create_experiment(project_id: str, body: ExperimentIn, request: Request, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    limit = request.app.state.settings.max_experiments_per_project
    if db.scalar(select(func.count()).select_from(Experiment).where(Experiment.project_id == project_id)) >= limit:
        raise HTTPException(429, f"experiment quota reached ({limit})")
    spec = _spec(body.spec)
    e = Experiment(project_id=project_id, owner_id=auth.user.id, name=spec["name"], spec=spec, spec_sha256=spec_hash(spec))
    db.add(e)
    db.commit()
    audit(db, request, "experiment_create", auth.user.id, e.id)
    return _view(request, e)


@router.get("/projects/{project_id}/experiments")
def list_experiments(project_id: str, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    rows = db.scalars(select(Experiment).where(Experiment.project_id == project_id).order_by(Experiment.created_at.desc())).all()
    return [_summary(e) for e in rows]


@router.get("/experiments/{exp_id}")
def read_experiment(exp_id: str, request: Request, auth: CurrentAuth, db: DB):
    return _view(request, owned_experiment(db, exp_id, auth.user.id))


@router.put("/experiments/{exp_id}/spec")
def replace_spec(exp_id: str, body: ExperimentSpecPatch, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    if e.status not in ("draft", "rejected", "pending_approval"):
        raise HTTPException(409, f"the specification cannot change while the experiment is {e.status}")
    spec = _spec(body.spec)
    e.spec, e.spec_sha256, e.name = spec, spec_hash(spec), spec["name"]
    e.status, e.approval = "draft", {}  # any edit voids a pending or earlier decision
    db.commit()
    audit(db, request, "experiment_edit", auth.user.id, e.id)
    return _view(request, e)


@router.post("/experiments/{exp_id}/request-approval")
def request_approval(exp_id: str, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    if e.status not in ("draft", "rejected"):
        raise HTTPException(409, f"cannot request approval while the experiment is {e.status}")
    e.status = "pending_approval"
    e.approval = {"requested_at": utcnow().isoformat(timespec="seconds"), "spec_sha256": e.spec_sha256, "decision": None}
    db.commit()
    audit(db, request, "experiment_request_approval", auth.user.id, e.id, spec_sha256=e.spec_sha256)
    return _view(request, e)


@router.post("/experiments/{exp_id}/execute", status_code=202)
def execute(exp_id: str, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    s, svc = request.app.state.settings, request.app.state.experiments
    if e.status != "approved":
        raise HTTPException(409, "only an approved experiment can be executed" if e.status != "pending_approval"
                            else "this experiment is waiting for human approval")
    if e.approval.get("spec_sha256") != spec_hash(e.spec):
        raise HTTPException(409, "the specification changed after it was approved; request approval again")
    running = db.scalar(select(func.count()).select_from(Experiment).where(Experiment.owner_id == auth.user.id,
                                                                           Experiment.status == "running"))
    if running >= s.max_running_experiments_per_user:
        raise HTTPException(429, f"running experiment quota reached ({s.max_running_experiments_per_user})")
    try:
        sandbox = svc.sandbox()
    except SandboxUnavailable as err:
        raise HTTPException(503, str(err)) from err
    if e.spec.get("generated_by_model") and not getattr(sandbox, "container", False):
        raise HTTPException(409, "model-generated code may only run in a container sandbox")
    e.status = "running"
    db.commit()
    audit(db, request, "experiment_execute", auth.user.id, e.id, sandbox=sandbox.name)
    svc.submit_execute(e.id, sandbox)
    return _view(request, e)


@router.get("/experiments/{exp_id}/logs")
def logs(exp_id: str, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    d = request.app.state.experiments.run_dir(e.id)
    out = {}
    for name in ("stdout", "stderr"):
        p = d / f"{name}.log"
        out[name] = p.read_text(encoding="utf-8", errors="replace")[:LOG_LIMIT] if p.is_file() else ""
    return out


@router.get("/experiments/{exp_id}/artifacts")
def list_artifacts(exp_id: str, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    m = request.app.state.experiments.manifest(e.id)
    return m["artifacts"] if m else []


@router.get("/experiments/{exp_id}/artifacts/{path:path}")
def download_artifact(exp_id: str, path: str, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    svc = request.app.state.experiments
    m = svc.manifest(e.id)
    listed = {a["path"]: a for a in (m["artifacts"] if m else [])}
    if path not in listed:  # only files the manifest recorded - never an arbitrary path
        raise HTTPException(404, "artifact not found")
    out_dir = (svc.run_dir(e.id) / "out").resolve()
    f = (out_dir / path)
    resolved = f.resolve()
    if f.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(out_dir):
        raise HTTPException(404, "artifact not found")
    if resolved.stat().st_size > request.app.state.settings.max_artifact_download_bytes:
        raise HTTPException(413, "artifact is too large to download through the API")
    if sha256_file(resolved) != listed[path]["sha256"]:
        raise HTTPException(409, "artifact no longer matches its recorded hash; it will not be served")
    ctype = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    if ctype in ("text/html", "image/svg+xml") or "javascript" in ctype:
        ctype = "application/octet-stream"  # experiment output is untrusted: never rendered by the browser
    return FileResponse(resolved, media_type=ctype, filename=Path(path).name,
                        headers={"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"})


@router.post("/experiments/{exp_id}/reproduce", status_code=202)
def reproduce_experiment(exp_id: str, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    svc = request.app.state.experiments
    if e.status != "succeeded":
        raise HTTPException(409, "only a succeeded experiment can be reproduced")
    if (e.reproduction or {}).get("state") == "running":
        raise HTTPException(409, "a reproduction is already running")
    try:
        sandbox = svc.sandbox()
    except SandboxUnavailable as err:
        raise HTTPException(503, str(err)) from err
    e.reproduction = {"state": "running"}
    db.commit()
    audit(db, request, "experiment_reproduce", auth.user.id, e.id)
    svc.submit_reproduce(e.id, sandbox, auth.user.email)
    return _view(request, e)


@router.post("/experiments/{exp_id}/review")
def review(exp_id: str, body: ReviewIn, request: Request, auth: CurrentAuth, db: DB):
    e = owned_experiment(db, exp_id, auth.user.id)
    if e.status not in ("succeeded", "failed", "timed_out"):
        raise HTTPException(409, "only a finished experiment can be reviewed")
    if (e.reproduction or {}).get("state") == "running":
        raise HTTPException(409, "wait for the reproduction to finish")
    d = request.app.state.experiments.run_dir(e.id)
    repro = (d / "reproduction.json")
    if e.reproduction and "reproduced" in e.reproduction and not repro.is_file():
        repro.write_text(__import__("json").dumps(e.reproduction), encoding="utf-8")
    try:
        e.review = review_experiment(d, auth.user.email, body.decision, body.notes)
    except ValueError as err:
        raise HTTPException(409, str(err)) from err
    db.commit()
    audit(db, request, "experiment_review", auth.user.id, e.id, decision=body.decision)
    return _view(request, e)


__all__ = ["router", "read_review"]
