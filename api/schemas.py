from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RegisterIn(_In):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)
    display_name: str = Field(default="", max_length=80)


class LoginIn(_In):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str


class SessionOut(BaseModel):
    user: UserOut
    csrf_token: str
    expires_at: datetime


class ProjectIn(_In):
    title: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=3, max_length=2000)
    description: str = Field(default="", max_length=5000)


class ProjectPatch(_In):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    question: str | None = Field(default=None, min_length=3, max_length=2000)
    description: str | None = Field(default=None, max_length=5000)


class ProjectOut(BaseModel):
    id: str
    title: str
    question: str
    description: str
    created_at: datetime
    updated_at: datetime


class RunIn(_In):
    question: str | None = Field(default=None, min_length=3, max_length=2000)
    limits: dict[str, Any] | None = None
    start: bool = True


class RunOut(BaseModel):
    id: str
    project_id: str
    question: str
    workflow: str
    created_at: datetime
    state: dict[str, Any]


class ExperimentIn(_In):
    spec: dict[str, Any]


class ExperimentSpecPatch(_In):
    spec: dict[str, Any]


class ApprovalDecision(_In):
    comment: str = Field(default="", max_length=2000)


class ReviewIn(_In):
    decision: str = Field(pattern="^(accepted|rejected)$")
    notes: str = Field(default="", max_length=2000)


class SectionIn(_In):
    heading: str = Field(min_length=1, max_length=120)
    body: str = Field(max_length=10_000)


class ReviewStatusIn(_In):
    status: str = Field(pattern="^(pending|approved|changes_requested)$")
    comment: str = Field(default="", max_length=2000)


class ReportPatch(_In):
    base_version: int = Field(ge=1)
    title: str | None = Field(default=None, max_length=200)
    sections: list[SectionIn] | None = Field(default=None, max_length=20)
    annotations: dict[str, str] | None = None
    review: ReviewStatusIn | None = None
    summary: str = Field(default="", max_length=200)


class TrainingJobIn(_In):
    name: str = Field(min_length=1, max_length=80)
    base_config: str = Field(min_length=1, max_length=64, pattern="^[a-z0-9][a-z0-9_]*$")
    max_steps: int | None = Field(default=None, ge=1, le=100_000)
    learning_rate: float | None = Field(default=None, gt=0, le=0.1)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    grad_accum_steps: int | None = Field(default=None, ge=1, le=16)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    eval_interval: int | None = Field(default=None, ge=1, le=100_000)
    checkpoint_interval: int | None = Field(default=None, ge=1, le=100_000)
    warmup_steps: int | None = Field(default=None, ge=0, le=100_000)
    dataset_id: str | None = Field(default=None, pattern="^dts_[0-9a-f]{16}$")  # a prepared dataset of the caller's own


class GenerateIn(_In):
    prompt: str = Field(min_length=1, max_length=20_000)  # the real ceiling is Settings.infer_max_prompt_chars
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=500)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    checkpoint: str = Field(default="best", pattern="^(best|latest)$")


class DatasetDocIn(_In):
    title: str = Field(default="", max_length=120)
    text: str = Field(min_length=1, max_length=200_000)  # the real ceiling is Settings.dataset_max_doc_chars


class DatasetIn(_In):
    name: str = Field(min_length=1, max_length=80)
    license: str = Field(min_length=1, max_length=32)
    attribution: str = Field(default="", max_length=300)
    rights_confirmed: bool = False  # the uploader states they may train a model on this text
    documents: list[DatasetDocIn] = Field(min_length=1, max_length=200)


class PrepareIn(_In):
    vocab_size: int = Field(default=1024, ge=262, le=4096)
    pretokenizer: str = Field(default="v2", pattern="^(v1|v2)$")
    min_chars: int = Field(default=100, ge=20, le=5000)


class LifecycleIn(_In):
    target: str = Field(min_length=1, max_length=16)
    note: str = Field(default="", max_length=300)


class SelectionIn(_In):
    provider: str = Field(pattern="^(deterministic|custom_transformer)$")
    model_id: str | None = Field(default=None, max_length=48, pattern="^(sys_[a-z0-9_]+|trn_[0-9a-f]+)$")
    mode: str = Field(default="experimental", pattern="^(experimental|production)$")
    allow_experimental: bool = False
