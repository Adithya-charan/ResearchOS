"""Settings come from the environment (RESEARCHOS_*); no secrets live in source. Defaults suit local development."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    database_url: str = f"sqlite:///{(ROOT / 'artifacts' / 'api' / 'researchos.db').as_posix()}"
    engine_db: Path = ROOT / "artifacts" / "api" / "engine.db"  # durable agent-engine state (tasks, events, approvals)
    default_corpus_dir: Path = ROOT / "datasets" / "raw" / "synthetic_v1"  # until per-project document upload exists
    data_dir: Path = ROOT / "artifacts" / "api" / "data"  # per-project document libraries live under here
    demo_corpus: bool = False  # True: projects without documents search default_corpus_dir (synthetic demo text)
    max_upload_bytes: int = 25 * 1024 * 1024
    max_documents_per_project: int = 200
    max_corpus_bytes: int = 500 * 1024 * 1024  # stored originals per project
    corpus_isolate: bool = True  # extract inside an isolated subprocess (leave on outside tests)
    experiments_dir_name: str = "experiments"  # under data_dir
    allow_weak_sandbox: bool = False  # True: permit the NON-isolating process fallback when Docker is unavailable (dev only)
    max_experiments_per_project: int = 100
    max_running_experiments_per_user: int = 1
    max_artifact_download_bytes: int = 20 * 1024 * 1024
    max_corpus_docs: int = 1500
    workflow: str = "full"
    session_ttl_s: int = 12 * 3600
    cookie_name: str = "rs_session"
    cookie_secure: bool = False  # set RESEARCHOS_COOKIE_SECURE=true behind TLS
    allowed_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000", "http://testserver")
    min_password_length: int = 12
    max_projects_per_user: int = 50
    max_active_runs_per_user: int = 2
    max_runs_per_project: int = 200
    login_attempts: int = 5  # failures per (ip, email) per window before 429
    login_window_s: int = 900
    register_attempts: int = 10  # registrations per ip per window
    register_window_s: int = 3600
    run_max_tasks: int = 60  # hard ceilings on user-requested limits
    run_max_runtime_s: float = 900.0
    run_max_iterations: int = 4
    auto_migrate: bool = True  # apply Alembic migrations at startup; False = refuse to start unless already at head
    # --- research job queue and worker
    embedded_worker: bool = True  # run a worker thread inside the API process; set false when `scripts/run_worker.py` runs separately
    worker_concurrency: int = 2  # runs one worker executes at the same time
    job_poll_s: float = 0.25
    job_lease_s: float = 30.0  # a worker must heartbeat within this window or its job is re-queued
    job_max_attempts: int = 3  # claims per run (crashes, lease losses and unexpected errors each use one)
    job_backoff_s: float = 2.0  # re-queue delay = backoff * 2**(attempt-1), capped at 60 s
    # --- models: registry, training jobs, inference
    models_dir: Path = ROOT / "artifacts" / "checkpoints"  # operator-trained checkpoints, read-only to the API (one run per sub-folder)
    tokenizer_dir: Path = ROOT / "artifacts" / "tokenizer"
    train_configs_dir: Path = ROOT / "configs"  # the ONLY configs a training job may start from (operator-controlled)
    training_dir_name: str = "training"  # under data_dir: one folder per job
    training_enabled: bool = False  # training uses the host's CPU/GPU for minutes to hours; the operator must opt in
    max_train_steps: int = 5000
    max_running_training_jobs: int = 1  # whole platform
    max_training_jobs_per_user: int = 20
    train_stop_grace_s: float = 60.0  # after a pause/cancel request the process gets this long to checkpoint, then it is killed
    inference_device: str = "cpu"
    infer_max_prompt_chars: int = 4000
    infer_max_new_tokens: int = 256
    infer_timeout_s: float = 30.0
    infer_per_minute: int = 20
    infer_cache_size: int = 2
    # --- server-sent events
    sse_max_streams_per_user: int = 4
    sse_max_streams_total: int = 16
    sse_max_seconds: float = 600.0  # a stream ends after this long; EventSource reconnects with Last-Event-ID
    sse_interval_s: float = 0.5
    sse_heartbeat_s: float = 15.0
    sse_reauth_s: float = 30.0  # how often a live stream re-checks that the viewer's session is still valid

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.database_url = os.environ.get("RESEARCHOS_DATABASE_URL", s.database_url)
        if "RESEARCHOS_ENGINE_DB" in os.environ:
            s.engine_db = Path(os.environ["RESEARCHOS_ENGINE_DB"])
        if "RESEARCHOS_CORPUS_DIR" in os.environ:
            s.default_corpus_dir = Path(os.environ["RESEARCHOS_CORPUS_DIR"])
        if "RESEARCHOS_DATA_DIR" in os.environ:
            s.data_dir = Path(os.environ["RESEARCHOS_DATA_DIR"])
        s.allow_weak_sandbox = _bool("RESEARCHOS_ALLOW_WEAK_SANDBOX", s.allow_weak_sandbox)
        s.embedded_worker = _bool("RESEARCHOS_EMBEDDED_WORKER", s.embedded_worker)
        s.auto_migrate = _bool("RESEARCHOS_AUTO_MIGRATE", s.auto_migrate)
        s.training_enabled = _bool("RESEARCHOS_TRAINING_ENABLED", s.training_enabled)
        s.inference_device = os.environ.get("RESEARCHOS_INFERENCE_DEVICE", s.inference_device)
        if "RESEARCHOS_MODELS_DIR" in os.environ:
            s.models_dir = Path(os.environ["RESEARCHOS_MODELS_DIR"])
        s.demo_corpus = _bool("RESEARCHOS_DEMO_CORPUS", s.demo_corpus)
        s.cookie_secure = _bool("RESEARCHOS_COOKIE_SECURE", s.cookie_secure)
        if "RESEARCHOS_ALLOWED_ORIGINS" in os.environ:
            s.allowed_origins = tuple(o.strip() for o in os.environ["RESEARCHOS_ALLOWED_ORIGINS"].split(",") if o.strip())
        return s
