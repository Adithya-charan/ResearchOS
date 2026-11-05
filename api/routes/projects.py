from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import func, select

from api.deps import DB, CurrentAuth, audit
from api.models import Project
from api.schemas import ProjectIn, ProjectOut, ProjectPatch

router = APIRouter(prefix="/projects", tags=["projects"])


def owned_project(db, project_id: str, user_id: str) -> Project:
    """Other users' projects look exactly like missing ones (404), so ids cannot be probed."""
    p = db.get(Project, project_id)
    if p is None or p.owner_id != user_id:
        raise HTTPException(404, "project not found")
    return p


@router.post("", status_code=201, response_model=ProjectOut)
def create_project(body: ProjectIn, request: Request, auth: CurrentAuth, db: DB):
    limit = request.app.state.settings.max_projects_per_user
    if db.scalar(select(func.count()).select_from(Project).where(Project.owner_id == auth.user.id)) >= limit:
        raise HTTPException(429, f"project quota reached ({limit})")
    p = Project(owner_id=auth.user.id, title=body.title, question=body.question, description=body.description)
    db.add(p)
    db.commit()
    audit(db, request, "project_create", auth.user.id, p.id)
    return p


@router.get("", response_model=list[ProjectOut])
def list_projects(auth: CurrentAuth, db: DB):
    return db.scalars(select(Project).where(Project.owner_id == auth.user.id).order_by(Project.created_at.desc())).all()


@router.get("/{project_id}", response_model=ProjectOut)
def read_project(project_id: str, auth: CurrentAuth, db: DB):
    return owned_project(db, project_id, auth.user.id)


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(project_id: str, body: ProjectPatch, request: Request, auth: CurrentAuth, db: DB):
    p = owned_project(db, project_id, auth.user.id)
    for k, v in body.model_dump(exclude_unset=True).items():
        if v is not None:
            setattr(p, k, v)
    db.commit()
    audit(db, request, "project_update", auth.user.id, p.id)
    return p


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request, auth: CurrentAuth, db: DB):
    p = owned_project(db, project_id, auth.user.id)
    rt = request.app.state.runtime
    if any(rt.is_active(r.engine_run_id) for r in p.runs):
        raise HTTPException(409, "cancel the project's active runs before deleting it")
    db.delete(p)
    db.commit()
    request.app.state.corpora.drop(project_id)  # the project's documents go with it
    audit(db, request, "project_delete", auth.user.id, project_id)
