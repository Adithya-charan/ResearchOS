from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func, select

from agent_engine.errors import RunStateError
from agent_engine.reporting import report_from_store
from api.deps import DB, CurrentAuth, audit
from api.models import Project, Run
from api.routes.projects import owned_project
from api.runtime import NoCorpus
from api.schemas import RunIn, RunOut

router = APIRouter(tags=["runs"])


def owned_run(db, run_id: str, user_id: str) -> Run:
    r = db.get(Run, run_id)
    if r is None or r.owner_id != user_id:
        raise HTTPException(404, "run not found")
    return r


def _out(request: Request, r: Run) -> RunOut:
    return RunOut(id=r.id, project_id=r.project_id, question=r.question, workflow=r.workflow, created_at=r.created_at,
                  state=request.app.state.runtime.status(r.engine_run_id))


def _active_count(request: Request, db, user_id: str) -> int:
    rt = request.app.state.runtime
    ids = db.scalars(select(Run.engine_run_id).where(Run.owner_id == user_id)).all()
    return rt.active_count(list(ids))


def _launch(request: Request, db, auth, r: Run) -> bool:
    limit = request.app.state.settings.max_active_runs_per_user
    if _active_count(request, db, auth.user.id) >= limit and not request.app.state.runtime.is_active(r.engine_run_id):
        raise HTTPException(429, f"active run quota reached ({limit}); wait for a run to finish or cancel one")
    try:
        started = request.app.state.runtime.launch(r.engine_run_id, r.project_id)
    except NoCorpus as e:
        raise HTTPException(409, str(e)) from e
    if started:
        audit(db, request, "run_start", auth.user.id, r.id)
    return started


@router.post("/projects/{project_id}/runs", status_code=201, response_model=RunOut)
def create_run(project_id: str, body: RunIn, request: Request, auth: CurrentAuth, db: DB):
    p: Project = owned_project(db, project_id, auth.user.id)
    s, rt = request.app.state.settings, request.app.state.runtime
    if db.scalar(select(func.count()).select_from(Run).where(Run.project_id == p.id)) >= s.max_runs_per_project:
        raise HTTPException(429, f"run quota for this project reached ({s.max_runs_per_project})")
    try:
        limits = rt.clamp(body.limits)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"invalid limits: {e}") from e
    question = body.question or p.question
    try:
        engine_run_id = rt.create(p.id, question, limits)
    except NoCorpus as e:
        raise HTTPException(409, str(e)) from e
    r = Run(project_id=p.id, owner_id=auth.user.id, engine_run_id=engine_run_id, question=question,
            workflow=s.workflow, limits=limits.model_dump())
    db.add(r)
    db.commit()
    audit(db, request, "run_create", auth.user.id, r.id)
    if body.start:
        _launch(request, db, auth, r)
    return _out(request, r)


@router.get("/projects/{project_id}/runs", response_model=list[RunOut])
def list_runs(project_id: str, request: Request, auth: CurrentAuth, db: DB):
    owned_project(db, project_id, auth.user.id)
    rows = db.scalars(select(Run).where(Run.project_id == project_id).order_by(Run.created_at.desc())).all()
    return [_out(request, r) for r in rows]


@router.get("/runs/{run_id}", response_model=RunOut)
def read_run(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    return _out(request, owned_run(db, run_id, auth.user.id))


@router.post("/runs/{run_id}/start", response_model=RunOut)
def start_run(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    r = owned_run(db, run_id, auth.user.id)
    if not _launch(request, db, auth, r):
        raise HTTPException(409, "run is already executing or has finished")
    return _out(request, r)


def _control(request: Request, db, auth, run_id: str, action: str) -> RunOut:
    r = owned_run(db, run_id, auth.user.id)
    eng = request.app.state.runtime.controller()
    try:
        getattr(eng, action)(r.engine_run_id)
    except RunStateError as e:
        raise HTTPException(409, str(e)) from e
    audit(db, request, f"run_{action}", auth.user.id, r.id)
    if action == "resume":
        _launch(request, db, auth, r)
    return _out(request, r)


@router.post("/runs/{run_id}/pause", response_model=RunOut)
def pause_run(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    return _control(request, db, auth, run_id, "pause")


@router.post("/runs/{run_id}/resume", response_model=RunOut)
def resume_run(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    return _control(request, db, auth, run_id, "resume")


@router.post("/runs/{run_id}/cancel", response_model=RunOut)
def cancel_run(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    return _control(request, db, auth, run_id, "cancel")


@router.post("/runs/{run_id}/recover", response_model=RunOut)
def recover_run(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    """After a worker restart: tasks left 'running' by the dead process are re-queued, then the run continues."""
    r = owned_run(db, run_id, auth.user.id)
    rt = request.app.state.runtime
    if rt.is_active(r.engine_run_id):
        raise HTTPException(409, "run is still executing in this process")
    recovered = rt.controller().recover(r.engine_run_id)
    audit(db, request, "run_recover", auth.user.id, r.id, recovered=len(recovered))
    _launch(request, db, auth, r)
    return _out(request, r)


@router.get("/runs/{run_id}/tasks")
def list_tasks(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    r = owned_run(db, run_id, auth.user.id)
    return [_task_view(t) for t in request.app.state.runtime.store.list_tasks(r.engine_run_id)]


@router.get("/runs/{run_id}/tasks/{task_id}")
def read_task(run_id: str, task_id: str, request: Request, auth: CurrentAuth, db: DB):
    r = owned_run(db, run_id, auth.user.id)
    try:
        t = request.app.state.runtime.store.get_task(task_id)
    except KeyError:
        raise HTTPException(404, "task not found") from None
    if t["run_id"] != r.engine_run_id:
        raise HTTPException(404, "task not found")  # a task of another run (or user) looks like a missing one
    return _task_view(t, full=True)


@router.get("/runs/{run_id}/events")
def list_events(run_id: str, request: Request, auth: CurrentAuth, db: DB, after: int = Query(0, ge=0),
                limit: int = Query(200, ge=1, le=1000)):
    r = owned_run(db, run_id, auth.user.id)
    return request.app.state.runtime.store.list_events(r.engine_run_id, after, limit)


@router.get("/runs/{run_id}/report")
def read_report(run_id: str, request: Request, auth: CurrentAuth, db: DB):
    r = owned_run(db, run_id, auth.user.id)
    report = report_from_store(request.app.state.runtime.store, r.engine_run_id)
    if report is None:
        raise HTTPException(404, "no report yet: the run has not reached synthesis")
    return report


_HIDE = {"task_input", "task_output"}


def _task_view(t: dict, full: bool = False) -> dict:
    view = {k: v for k, v in t.items() if k not in _HIDE and k not in ("input", "output")}
    if full:
        view["input"], view["output"] = t.get("input"), t.get("output")
    else:
        out = t.get("output") or {}
        view["summary"] = out.get("summary")
        view["output_status"] = out.get("status")
    return view
