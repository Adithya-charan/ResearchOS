from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
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


class ResearchJob(Base):
    """The unit of work a worker claims: one per research run. Claiming is an atomic conditional UPDATE, so two workers (threads or
    processes) can never execute the same run; a worker that stops heartbeating loses its lease and the job is re-queued."""
    __tablename__ = "research_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("job"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), unique=True)
    engine_run_id: Mapped[str] = mapped_column(String(64), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(10), default="queued", index=True)  # queued | running | paused | done | failed | cancelled
    worker_id: Mapped[str] = mapped_column(String(80), default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)  # times a worker claimed it
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    available_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)  # not claimable before this (retry backoff)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    enqueued_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    __table_args__ = (Index("ix_research_jobs_claim", "status", "available_at"),)


class Claim(Base):
    """A research claim. Agent-made claims keep their originating agent and task; user-made claims have source='user'.
    `status` is the latest verification outcome (UNVERIFIED until one has run); each outcome is kept in claim_verifications."""
    __tablename__ = "claims"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("clm"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), nullable=True, index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    claim_key: Mapped[str | None] = mapped_column(String(120), nullable=True)  # the engine key, e.g. literature:s003
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(16))
    origin_agent: Mapped[str] = mapped_column(String(32), default="user")
    origin_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(8), default="agent")  # agent | user
    status: Mapped[str] = mapped_column(String(24), default="UNVERIFIED", index=True)
    strength: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    evidence: Mapped[list["ClaimEvidence"]] = relationship(cascade="all, delete-orphan", order_by="ClaimEvidence.id")
    verifications: Mapped[list["ClaimVerification"]] = relationship(cascade="all, delete-orphan", order_by="ClaimVerification.id")
    __table_args__ = (UniqueConstraint("run_id", "claim_key", name="uq_claims_run_key"),)


class ClaimEvidence(Base):
    """A stored passage associated with a claim. Page, quote and document come from the stored passage, never from the caller.
    origin='user' rows were attached by a person and survive re-verification; origin='verification' rows are rewritten by it."""
    __tablename__ = "claim_evidence"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(14), default="contextual")  # supporting | contradicting | contextual
    directness: Mapped[str | None] = mapped_column(String(8), nullable=True)  # direct | indirect
    document_id: Mapped[str] = mapped_column(String(64))
    passage_id: Mapped[str] = mapped_column(String(80))
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote: Mapped[str] = mapped_column(Text, default="")
    retrieval_method: Mapped[str] = mapped_column(String(20), default="attached")
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_valid: Mapped[int] = mapped_column(Integer, default=1)
    reason: Mapped[str] = mapped_column(Text, default="")
    origin: Mapped[str] = mapped_column(String(14), default="verification")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ClaimVerification(Base):
    __tablename__ = "claim_verifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    round: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(24))
    strength: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[list] = mapped_column(JSON, default=list)  # snapshot of the evidence at that time, incl. invalid citations
    verifier: Mapped[str] = mapped_column(String(40), default="")
    requested_by: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class NoveltyAssessmentRow(Base):
    __tablename__ = "novelty_assessments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("nov"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), nullable=True, index=True)
    claim_id: Mapped[str | None] = mapped_column(ForeignKey("claims.id", ondelete="SET NULL"), nullable=True, index=True)
    statement: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(60))
    content: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
