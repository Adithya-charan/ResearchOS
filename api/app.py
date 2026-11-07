"""Application factory. `create_app(settings)` builds an isolated app (tests pass their own settings and index)."""
from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from api.config import Settings
from api.db import Base, make_engine, make_sessionmaker
from api import migrate
from api.corpus_service import CorpusManager, Scanner
from api.experiments_service import ExperimentService
from api.jobs import JobQueue, Worker
from api.model_service import InferenceService, ModelRegistry, TrainingService
from api.modellab_service import ModelLab
from api.routes import agents, approvals, auth, claims, documents, experiments, modellab, models, overview, projects, reports, runs, streams
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
    engine = make_engine(settings.database_url, settings.db_pool_size, settings.db_max_overflow)
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
    app.state.modellab = ModelLab(settings, app.state.sessionmaker, app.state.model_registry, app.state.inference)
    app.state.runtime.provider_resolver = app.state.modellab.provider_for_project
    app.state.training.recover()
    app.state.streams = StreamGate(settings.sse_max_streams_per_user, settings.sse_max_streams_total, settings.sse_max_seconds)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        if request.method in UNSAFE:  # cross-site writes are refused even before the CSRF token is checked
            origin = request.headers.get("origin")
            if origin is not None:
                host = urlparse(origin)
                claimed = f"{host.scheme}://{host.netloc}"
                # the bundled Model Lab is served by this app, so a browser's same-origin request names this server itself; a
                # cross-site page cannot make a browser send this server's own origin, so this does not widen the CSRF surface
                served_here = claimed == f"{request.url.scheme}://{request.headers.get('host', '')}"
                if claimed not in settings.allowed_origins and not served_here:
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

    for module in (auth, projects, documents, runs, approvals, experiments, reports, models, modellab, streams, agents, claims, overview):
        app.include_router(module.router)

    # Model Lab: three fixed files, looked up in a whitelist (the request path is never joined onto a directory)
    lab_dir = Path(__file__).parent / "static" / "lab"
    lab_files = {"": ("index.html", "text/html; charset=utf-8"), "lab.js": ("lab.js", "text/javascript; charset=utf-8"),
                 "lab.css": ("lab.css", "text/css; charset=utf-8")}
    lab_headers = {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
                   "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"}

    def lab_response(key: str):
        name, media = lab_files[key]
        path = lab_dir / name
        if not path.is_file():
            return JSONResponse({"detail": "Model Lab files are not installed"}, status_code=404)
        return Response(path.read_bytes(), media_type=media, headers=lab_headers)

    @app.get("/lab", include_in_schema=False)
    @app.get("/lab/", include_in_schema=False)
    def lab_index():
        return lab_response("")

    @app.get("/lab/lab.js", include_in_schema=False)
    def lab_js():
        return lab_response("lab.js")

    @app.get("/lab/lab.css", include_in_schema=False)
    def lab_css():
        return lab_response("lab.css")

    # Web application (P9): the built single-page app in frontend/dist, served same-origin under /app so the HttpOnly session cookie and
    # CSRF token work unchanged. Paths are resolved inside the build directory only; unknown paths fall back to index.html (client routing).
    spa_dir = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    spa_headers = {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
                                              "form-action 'self'; object-src 'none'",
                   "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"}

    @app.get("/app", include_in_schema=False)
    @app.get("/app/{spa_path:path}", include_in_schema=False)
    def spa(spa_path: str = ""):
        index = spa_dir / "index.html"
        if not index.is_file():
            return JSONResponse({"detail": "the web app is not built: run `npm ci && npm run build` in frontend/"}, status_code=404)
        target = (spa_dir / spa_path).resolve() if spa_path else index
        if spa_path and not (target.is_file() and target.is_relative_to(spa_dir.resolve())):
            if "." in Path(spa_path).name:  # a missing asset is a 404, not the app shell
                return JSONResponse({"detail": "not found"}, status_code=404)
            target = index
        return Response(target.read_bytes(), media_type=_media_type(target.suffix), headers=spa_headers)

    return app


_MEDIA = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".svg": "image/svg+xml", ".json": "application/json", ".png": "image/png", ".ico": "image/x-icon", ".woff2": "font/woff2",
          ".map": "application/json"}


def _media_type(suffix: str) -> str:
    return _MEDIA.get(suffix.lower(), "application/octet-stream")
