"""Application factory. `create_app(settings)` builds an isolated app (tests pass their own settings and index)."""
from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import Callable
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.config import Settings
from api.db import Base, make_engine, make_sessionmaker
from api import migrate
from api.corpus_service import CorpusManager, Scanner
from api.experiments_service import ExperimentService
from api.jobs import JobQueue, Worker
from api.model_service import InferenceService, ModelRegistry, TrainingService
from api.routes import agents, approvals, auth, claims, documents, experiments, models, projects, reports, runs, streams
from api.runtime import EngineRuntime
from api.security import RateLimiter
from api.sse import StreamGate
from research.bm25.index import BM25Index

log = logging.getLogger("researchos.api")
def prepare_schema(engine, settings: Settings) -> None:
    """In-memory databases (tests) are created directly; every other database is managed by Alembic migrations."""
    if ":memory:" in str(engine.url):
        Base.metadata.create_all(engine)
    elif settings.auto_migrate:
        rev = migrate.upgrade(engine)
        log.info("database schema at revision %s", rev)
    elif migrate.current_revision(engine) != migrate.head_revision():
        raise migrate.SchemaMismatch("the database is not at the latest migration and RESEARCHOS_AUTO_MIGRATE is off: "
                                     "run `alembic upgrade head` first")


UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def create_app(settings: Settings | None = None, index_provider: Callable[[str], BM25Index] | None = None,
               scanner: Scanner | None = None, sandbox_factory=None) -> FastAPI:
    settings = settings or Settings.from_env()
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.embedded_worker:
            app.state.worker.start()  # also picks up jobs left queued by a previous process
        yield
        app.state.worker.stop()  # running jobs are handed back, not lost
        app.state.training.stop_all()  # training children must not outlive the API: they checkpoint and exit, or are killed

    app = FastAPI(title="RESEARCHOS API", version="0.1.0", docs_url="/docs", redoc_url=None, lifespan=lifespan)
    engine = make_engine(settings.database_url)
    prepare_schema(engine, settings)
    app.state.settings = settings
    app.state.db_engine = engine
    app.state.sessionmaker = make_sessionmaker(engine)
    app.state.limiter = RateLimiter()
    app.state.corpora = CorpusManager(settings, scanner)
    app.state.experiments = ExperimentService(settings, app.state.sessionmaker, sandbox_factory)
    app.state.runtime = EngineRuntime(settings, index_provider, app.state.corpora)
    app.state.runtime.jobs = JobQueue(app.state.sessionmaker, settings, app.state.runtime.store)
    app.state.worker = Worker(settings, app.state.runtime, app.state.runtime.jobs, app.state.sessionmaker)
    app.state.model_registry = ModelRegistry(settings, settings.data_dir / settings.training_dir_name)
    app.state.training = TrainingService(settings, app.state.sessionmaker, app.state.model_registry)
    app.state.inference = InferenceService(settings)
    app.state.training.recover()
    app.state.streams = StreamGate(settings.sse_max_streams_per_user, settings.sse_max_streams_total, settings.sse_max_seconds)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        if request.method in UNSAFE:  # cross-site writes are refused even before the CSRF token is checked
            origin = request.headers.get("origin")
            if origin is not None:
                host = urlparse(origin)
                if f"{host.scheme}://{host.netloc}" not in settings.allowed_origins:
                    return JSONResponse({"detail": "origin not allowed"}, status_code=403)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        log.error("unhandled error request_id=%s path=%s", rid, request.url.path, exc_info=exc)
        return JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500,
                            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})

    @app.get("/health", tags=["system"])
    def health():
        return {"status": "ok", "service": "researchos-api", "workflow": settings.workflow}

    for module in (auth, projects, documents, runs, approvals, experiments, reports, models, streams, agents, claims):
        app.include_router(module.router)
    return app
