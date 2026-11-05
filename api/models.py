from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db import Base, utcnow


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(80), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    is_active: Mapped[int] = mapped_column(Integer, default=1)


class SessionRow(Base):
    """Server-side session. Only a hash of the cookie token is stored, so a database leak does not yield live sessions."""
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(token)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    user_agent: Mapped[str] = mapped_column(String(200), default="")


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("prj"))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(120))
    question: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    runs: Mapped[list["Run"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Run(Base):
    """An API-level research run; the agent engine's own durable state is addressed through engine_run_id."""
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("run"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    engine_run_id: Mapped[str] = mapped_column(String(64), unique=True)
    question: Mapped[str] = mapped_column(Text)
    workflow: Mapped[str] = mapped_column(String(10))
    limits: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    project: Mapped[Project] = relationship(back_populates="runs")


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(60), index=True)
    target: Mapped[str] = mapped_column(String(64), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)  # never holds passwords, tokens or document text


Index("ix_runs_owner_created", Run.owner_id, Run.created_at)


class Experiment(Base):
    """A sandboxed experiment. The approval is bound to the exact spec hash, so code cannot change after it is approved."""
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("exp"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    spec: Mapped[dict] = mapped_column(JSON)
    spec_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)
    # draft | pending_approval | approved | rejected | running | succeeded | failed | timed_out | error
    approval: Mapped[dict] = mapped_column(JSON, default=dict)  # {requested_at, spec_sha256, decision, reviewer_id, comment, decided_at}
    manifest: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    reproduction: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    review: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Report(Base):
    """A report generated from a finished run. `content` is an immutable snapshot (with source passage text); humans can only
    change `editable` (title, notes, annotations, review status), so an edit can never alter a claim or its evidence."""
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("rep"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), unique=True)
    content: Mapped[dict] = mapped_column(JSON)
    content_sha256: Mapped[str] = mapped_column(String(64))
    editable: Mapped[dict] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    versions: Mapped[list["ReportVersion"]] = relationship(cascade="all, delete-orphan", order_by="ReportVersion.version")


class ReportVersion(Base):
    __tablename__ = "report_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    editor_id: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(String(200), default="")
    editable: Mapped[dict] = mapped_column(JSON)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class TrainingJob(Base):
    """A training run of the from-scratch Transformer, executed in its own OS process (so it can be paused, cancelled or killed
    without touching the API). `config` is the exact effective training configuration that was written for the job."""
    __tablename__ = "training_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("trn"))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    base_config: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)
    # running | pausing | paused | cancelling | cancelled | succeeded | failed | interrupted
    control: Mapped[str] = mapped_column(String(10), default="")  # pending operator request: "" | pause | cancel
    segments: Mapped[int] = mapped_column(Integer, default=0)  # how many times a process was started for this job
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
