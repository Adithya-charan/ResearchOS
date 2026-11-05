"""Experiment specification. Code is untrusted text: it is hashed, stored, and only ever executed by a sandbox runner."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TIMEOUT_S = 300.0
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    code: str = Field(min_length=1, max_length=200_000)  # contents of main.py
    files: dict[str, str] = Field(default_factory=dict)  # extra input files (text datasets), flat names only
    seed: int = 0
    config: dict[str, Any] = Field(default_factory=dict)  # written to config.json and readable by the code
    packages: list[str] = Field(default_factory=list)  # installed distributions whose versions go in the manifest
    hypothesis_id: str | None = None
    timeout_s: float = Field(default=30.0, gt=0, le=MAX_TIMEOUT_S)
    max_output_bytes: int = Field(default=1_000_000, gt=0, le=20_000_000)
    memory_mb: int = Field(default=512, ge=64, le=8192)
    cpus: float = Field(default=1.0, gt=0, le=8)
    generated_by_model: bool = False  # model-written code may only run in a container sandbox

    @field_validator("files")
    @classmethod
    def _flat_safe_names(cls, v: dict[str, str]) -> dict[str, str]:
        for name in v:
            if not _SAFE_NAME.match(name) or name in {"main.py", "config.json"}:
                raise ValueError(f"unsafe or reserved file name: {name!r}")
        if sum(len(t) for t in v.values()) > 5_000_000:
            raise ValueError("input files too large (5 MB limit)")
        return v
