"""Settings come from the environment (RESEARCHOS_*); no secrets live in source. Defaults suit local development."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from agent_engine.state import dburl

ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


# Per-user rate limits (count, window seconds) for expensive endpoints; see api/ratelimit.py. RESEARCHOS_RATE_<NAME>="count/window" or "off".
RATE_DEFAULTS: dict[str, tuple[int, int]] = {
    "upload": (120, 600), "run_create": (20, 3600), "search": (120, 60), "verify": (60, 600), "report": (30, 600),
    "experiment": (10, 3600), "evaluate": (10, 3600), "reprocess": (30, 600),
}


# Settings an operator may override with an integer environment variable.
_INT_ENV = {
    "max_upload_bytes": "RESEARCHOS_MAX_UPLOAD_BYTES", "max_documents_per_project": "RESEARCHOS_MAX_DOCUMENTS_PER_PROJECT",
    "max_corpus_bytes": "RESEARCHOS_MAX_CORPUS_BYTES", "worker_concurrency": "RESEARCHOS_WORKER_CONCURRENCY",
    "session_ttl_s": "RESEARCHOS_SESSION_TTL_S", "min_password_length": "RESEARCHOS_MIN_PASSWORD_LENGTH",
    "login_attempts": "RESEARCHOS_LOGIN_ATTEMPTS", "login_window_s": "RESEARCHOS_LOGIN_WINDOW_S",
    "register_attempts": "RESEARCHOS_REGISTER_ATTEMPTS", "max_projects_per_user": "RESEARCHOS_MAX_PROJECTS_PER_USER",
    "max_active_runs_per_user": "RESEARCHOS_MAX_ACTIVE_RUNS_PER_USER", "sse_max_streams_per_user": "RESEARCHOS_SSE_MAX_STREAMS_PER_USER",
    "sse_max_streams_total": "RESEARCHOS_SSE_MAX_STREAMS_TOTAL",
}


def parse_rate(text: str) -> tuple[int, int] | None:
    t = text.strip().lower()
    if t in {"off", "none", "disabled"}:
        return None
    count, _, window = t.partition("/")
    out = (int(count), int(window))
    if out[0] < 1 or out[1] < 1:
        raise ValueError(f"invalid rate limit {text!r}: use '<count>/<window seconds>' or 'off'")
    return out


def rates_from_env(env=None) -> dict[str, tuple[int, int] | None]:
    env = os.environ if env is None else env
    out: dict[str, tuple[int, int] | None] = dict(RATE_DEFAULTS)
    for name in RATE_DEFAULTS:
        raw = env.get(f"RESEARCHOS_RATE_{name.upper()}")
        if raw:
            out[name] = parse_rate(raw)
    return out


@dataclass
class Settings:
    database_url: str = f"sqlite:///{(ROOT / 'artifacts' / 'api' / 'researchos.db').as_posix()}"
    engine_db: Path | str = ROOT / "artifacts" / "api" / "engine.db"  # durable agent-engine state: a SQLite file, or a postgresql:// URL
    db_pool_size: int = 10  # PostgreSQL only: connections kept per process
    db_max_overflow: int = 20  # PostgreSQL only: extra connections allowed under load
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
    # --- model lab: user datasets, evaluation, privileged actions
    datasets_dir_name: str = "datasets"  # under data_dir: one folder per user dataset
    dataset_max_docs: int = 200
    dataset_max_doc_chars: int = 200_000
    dataset_max_total_chars: int = 2_000_000
    dataset_max_per_user: int = 10
    dataset_max_vocab: int = 4096
    eval_max_windows: int = 100
    admin_emails: tuple[str, ...] = ()  # operators who may deploy models and change system models' lifecycle (RESEARCHOS_ADMIN_EMAILS)
    # --- server-sent events
    sse_max_streams_per_user: int = 4
    sse_max_streams_total: int = 16
    sse_max_seconds: float = 600.0  # a stream ends after this long; EventSource reconnects with Last-Event-ID
    sse_interval_s: float = 0.5
    sse_heartbeat_s: float = 15.0
    rate_limits: dict = field(default_factory=lambda: dict(RATE_DEFAULTS))  # per-user limits, see RATE_DEFAULTS
    enable_docs: bool = True  # serve /docs and /openapi.json (Swagger UI); switch off in production (RESEARCHOS_ENABLE_DOCS=false)
    hsts: bool = False  # send Strict-Transport-Security (only when the site is served over HTTPS; implied by cookie_secure)
    sse_reauth_s: float = 30.0  # how often a live stream re-checks that the viewer's session is still valid

    def warnings(self) -> list[str]:
        """Risky-but-legal combinations, logged at startup. A wildcard origin is refused outright by `from_env`."""
        out = []
        if any(o.startswith("https://") for o in self.allowed_origins) and not self.cookie_secure:
            out.append("an https origin is allowed but RESEARCHOS_COOKIE_SECURE is false: the session cookie would also travel over http")
        if self.cookie_secure and not self.hsts:
            out.append("cookies are Secure but HSTS is off (RESEARCHOS_HSTS=true is recommended behind TLS)")
        if self.allow_weak_sandbox:
            out.append("RESEARCHOS_ALLOW_WEAK_SANDBOX is on: experiments run WITHOUT isolation (development only)")
        if self.min_password_length < 12:
            out.append("minimum password length is below 12")
        return out

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.database_url = os.environ.get("RESEARCHOS_DATABASE_URL", s.database_url)
        if dburl.is_postgres(s.database_url):
            # One server and one database, two schemas: the API and the engine both have tables called `runs` and `projects`, so they
            # must not share a namespace. `?schema=` in a URL (or the two variables below) overrides the defaults.
            api_schema = os.environ.get("RESEARCHOS_DATABASE_SCHEMA") or dburl.split_schema(s.database_url)[1] or "api"
            s.database_url = dburl.with_schema(s.database_url, api_schema)
            engine_url = os.environ.get("RESEARCHOS_ENGINE_DATABASE_URL")
            if engine_url:
                if not dburl.is_postgres(engine_url):
                    raise ValueError("RESEARCHOS_ENGINE_DATABASE_URL must be a postgresql:// URL when RESEARCHOS_DATABASE_URL is one")
                engine_schema = os.environ.get("RESEARCHOS_ENGINE_DATABASE_SCHEMA") or dburl.split_schema(engine_url)[1]
            else:
                engine_url, engine_schema = s.database_url, os.environ.get("RESEARCHOS_ENGINE_DATABASE_SCHEMA") or "engine"
            s.engine_db = dburl.with_schema(engine_url, engine_schema)
            if s.engine_db == s.database_url:
                raise ValueError("the API and the engine need separate schemas or databases (their table names collide)")
        elif "RESEARCHOS_ENGINE_DB" in os.environ:
            s.engine_db = Path(os.environ["RESEARCHOS_ENGINE_DB"])
        s.db_pool_size = int(os.environ.get("RESEARCHOS_DB_POOL_SIZE", s.db_pool_size))
        s.db_max_overflow = int(os.environ.get("RESEARCHOS_DB_MAX_OVERFLOW", s.db_max_overflow))
        if "RESEARCHOS_CORPUS_DIR" in os.environ:
            s.default_corpus_dir = Path(os.environ["RESEARCHOS_CORPUS_DIR"])
        if "RESEARCHOS_DATA_DIR" in os.environ:
            s.data_dir = Path(os.environ["RESEARCHOS_DATA_DIR"])
        s.allow_weak_sandbox = _bool("RESEARCHOS_ALLOW_WEAK_SANDBOX", s.allow_weak_sandbox)
        s.embedded_worker = _bool("RESEARCHOS_EMBEDDED_WORKER", s.embedded_worker)
        s.auto_migrate = _bool("RESEARCHOS_AUTO_MIGRATE", s.auto_migrate)
        s.training_enabled = _bool("RESEARCHOS_TRAINING_ENABLED", s.training_enabled)
        s.inference_device = os.environ.get("RESEARCHOS_INFERENCE_DEVICE", s.inference_device)
        if "RESEARCHOS_ADMIN_EMAILS" in os.environ:
            s.admin_emails = tuple(e.strip().lower() for e in os.environ["RESEARCHOS_ADMIN_EMAILS"].split(",") if e.strip())
        if "RESEARCHOS_MODELS_DIR" in os.environ:
            s.models_dir = Path(os.environ["RESEARCHOS_MODELS_DIR"])
        s.demo_corpus = _bool("RESEARCHOS_DEMO_CORPUS", s.demo_corpus)
        s.cookie_secure = _bool("RESEARCHOS_COOKIE_SECURE", s.cookie_secure)
        for attr, var in _INT_ENV.items():  # numeric limits an operator may tune
            if var in os.environ:
                setattr(s, attr, int(os.environ[var]))
        s.enable_docs = _bool("RESEARCHOS_ENABLE_DOCS", s.enable_docs)
        s.hsts = _bool("RESEARCHOS_HSTS", s.cookie_secure)
        s.rate_limits = rates_from_env()
        if "RESEARCHOS_ALLOWED_ORIGINS" in os.environ:
            s.allowed_origins = tuple(o.strip() for o in os.environ["RESEARCHOS_ALLOWED_ORIGINS"].split(",") if o.strip())
            if any(o == "*" or "*" in o for o in s.allowed_origins):
                raise ValueError("RESEARCHOS_ALLOWED_ORIGINS must list exact origins; a wildcard would let any site make credentialed writes")
        return s
