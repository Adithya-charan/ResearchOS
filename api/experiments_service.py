"""Runs approved experiments in a sandbox on worker threads and writes the outcome back to the database.
The FastAPI process only supervises: the experiment code itself runs in the sandbox (a container, or the explicitly
non-isolating process fallback when the operator allowed it)."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api.config import Settings
from api.db import utcnow
from api.models import Experiment
from research.experiments import (DockerSandbox, ExperimentSpec, NotApproved, QuotaExceeded, SandboxRefused,
                                  SubprocessSandbox, load_manifest, reproduce, run_experiment, verify_manifest)
from research.experiments.sandbox import Sandbox

log = logging.getLogger("researchos.experiments")


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SandboxUnavailable(Exception):
    pass


class ExperimentService:
    def __init__(self, settings: Settings, sessionmaker, sandbox_factory: Callable[[], Sandbox] | None = None):
        self.settings, self.sessions = settings, sessionmaker
        self._factory = sandbox_factory
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="experiment")
        self.root = settings.data_dir / settings.experiments_dir_name

    def sandbox(self) -> Sandbox:
        if self._factory is not None:
            return self._factory()
        docker = DockerSandbox()
        if docker.available:
            return docker
        if self.settings.allow_weak_sandbox:
            return SubprocessSandbox()
        raise SandboxUnavailable("no isolated sandbox is available: install Docker (a weak, non-isolating fallback exists but "
                                 "the operator has not enabled it)")

    def run_dir(self, exp_id: str) -> Path:
        return self.root / exp_id

    def submit_execute(self, exp_id: str, sandbox: Sandbox) -> None:
        self.pool.submit(self._execute, exp_id, sandbox)

    def submit_reproduce(self, exp_id: str, sandbox: Sandbox, approver: str) -> None:
        self.pool.submit(self._reproduce, exp_id, sandbox, approver)

    def _execute(self, exp_id: str, sandbox: Sandbox) -> None:
        with self.sessions() as db:
            exp = db.get(Experiment, exp_id)
            try:
                if spec_hash(exp.spec) != exp.approval.get("spec_sha256"):
                    raise NotApproved("the specification changed after it was approved")
                spec = ExperimentSpec.model_validate(exp.spec)
                run_dir, manifest = run_experiment(spec, sandbox, self.root, approved_by=exp.approval["reviewer_id"],
                                                   run_id=exp_id)
                exp.manifest, exp.status = manifest, manifest["status"]
            except (SandboxRefused, NotApproved, QuotaExceeded, FileExistsError, ValueError) as e:
                exp.status, exp.error = "error", str(e)[:500]
            except Exception:  # never leave an experiment stuck in 'running'
                log.exception("experiment %s crashed", exp_id)
                exp.status, exp.error = "error", "internal error while running the experiment"
            exp.updated_at = utcnow()
            db.commit()

    def _reproduce(self, exp_id: str, sandbox: Sandbox, approver: str) -> None:
        with self.sessions() as db:
            exp = db.get(Experiment, exp_id)
            try:
                exp.reproduction = reproduce(self.run_dir(exp_id), sandbox, approved_by=approver)
            except (SandboxRefused, NotApproved, QuotaExceeded, ValueError) as e:
                exp.reproduction = {"reproduced": False, "reason": str(e)[:300], "differences": []}
            except Exception:
                log.exception("reproduction of %s crashed", exp_id)
                exp.reproduction = {"reproduced": False, "reason": "internal error", "differences": []}
            exp.updated_at = utcnow()
            db.commit()

    def integrity(self, exp: Experiment) -> list[str]:
        d = self.run_dir(exp.id)
        return verify_manifest(d) if (d / "manifest.json").is_file() else ["no stored run"]

    def manifest(self, exp_id: str) -> dict | None:
        d = self.run_dir(exp_id)
        return load_manifest(d) if (d / "manifest.json").is_file() else None

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
