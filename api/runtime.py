"""Bridge between the API and the custom agent engine. Runs execute in workers (api/jobs.py), never in a request handler; all
durable state lives in the engine's Store and the job table, so a restarted process can recover and continue a run."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from agent_engine.contracts import RunLimits, RunStatus
from agent_engine.registry.defaults import build_engine
from agent_engine.scheduler.engine import ResearchEngine
from agent_engine.state.store import Store
from api.config import Settings
from research.bm25.index import BM25Index, passages_from_text

log = logging.getLogger("researchos.runtime")
TERMINAL = {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}


class NoCorpus(Exception):
    pass


def default_index(settings: Settings) -> BM25Index:
    files = sorted(settings.default_corpus_dir.glob("*.txt")) if settings.default_corpus_dir.is_dir() else []
    if not files:
        raise NoCorpus(f"no documents are indexed (looked in {settings.default_corpus_dir})")
    index = BM25Index()
    for f in files[: settings.max_corpus_docs]:
        index.add_passages(passages_from_text(f.stem, f.read_text(encoding="utf-8", errors="replace")))
    return index


class EngineRuntime:
    def __init__(self, settings: Settings, index_provider: Callable[[str], BM25Index] | None = None, corpora=None):
        self.settings = settings
        self.corpora = corpora  # CorpusManager: per-project document libraries
        self.store = Store(str(settings.engine_db))  # a SQLite file, ":memory:" or a postgresql:// URL
        self._index_provider = index_provider
        self._default_index: BM25Index | None = None
        self.provider_resolver = None  # project_id -> (backend | None, RolePolicy | None); set by the app (the Model Lab selection)
        self.jobs = None  # JobQueue, attached by the app (execution happens in workers, never in request handlers)
        self._lock = threading.Lock()

    def index_for(self, project_id: str) -> BM25Index:
        if self._index_provider is not None:
            return self._index_provider(project_id)
        if self.corpora is not None:
            idx = self.corpora.index_for(project_id)
            if idx is not None:
                return idx
            if not self.settings.demo_corpus:
                raise NoCorpus("this project has no searchable documents; upload some first")
        with self._lock:
            if self._default_index is None:
                self._default_index = default_index(self.settings)
            return self._default_index

    def engine(self, project_id: str) -> ResearchEngine:
        """An engine that searches this project's documents (citations are re-verified against the stored originals)."""
        index = self.index_for(project_id)
        catalog = verifier = None
        mgr = self.corpora
        if self._index_provider is None and mgr is not None and mgr.exists(project_id):
            def catalog(doc_id: str):
                with mgr.lock(project_id):
                    d = mgr.get(project_id).get_document(doc_id)
                return None if d is None else {"title": d.title, "year": d.year, "doi": d.doi, "license": d.license}

            def verifier(passage_id: str):
                with mgr.lock(project_id):
                    v = mgr.get(project_id).verify_passage(passage_id)
                return v.ok, v.message
        backend, role_policy = (self.provider_resolver(project_id) if self.provider_resolver is not None else (None, None))
        return build_engine(self.store, index, workflow=self.settings.workflow, catalog=catalog, verifier=verifier,
                            poll_interval=0.05, backend=backend, role_policy=role_policy)

    def controller(self) -> ResearchEngine:
        """Pause/resume/cancel/recover only touch stored state, so they need no index (and must not rebuild one)."""
        return build_engine(self.store, BM25Index(), workflow=self.settings.workflow, poll_interval=0.05)

    def clamp(self, requested: dict | None) -> RunLimits:
        """User-requested limits can only tighten the server's ceilings."""
        s = self.settings
        data = dict(requested or {})
        data["max_tasks"] = min(int(data.get("max_tasks", 40)), s.run_max_tasks)
        data["max_runtime_s"] = min(float(data.get("max_runtime_s", 300.0)), s.run_max_runtime_s)
        data["max_iterations"] = min(int(data.get("max_iterations", 2)), s.run_max_iterations)
        return RunLimits(**data)

    def create(self, project_id: str, question: str, limits: RunLimits) -> str:
        eng = self.engine(project_id)
        return eng.create_run(eng.create_project(question[:80], question), question, limits)

    def status(self, engine_run_id: str) -> dict:
        from api.jobs import lifecycle  # local import: jobs imports this module
        run = self.store.get_run(engine_run_id)
        job = self.jobs.get(engine_run_id) if self.jobs is not None else None
        tasks = self.store.list_tasks(engine_run_id)
        counts: dict[str, int] = {}
        for t in tasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        return {"status": run["status"], "lifecycle": lifecycle(job, run, tasks), "stop_reason": run["stop_reason"],
                "control": run["control"], "elapsed_s": run["elapsed_s"], "task_counts": counts,
                "model_tokens_used": run["model_tokens_used"], "active": self.is_active(engine_run_id),
                "job": None if job is None else {"status": job["status"], "attempts": job["attempts"],
                                                 "max_attempts": job["max_attempts"], "worker_id": job["worker_id"],
                                                 "error": job["error"]}}

    def is_active(self, engine_run_id: str) -> bool:
        """True while the run is queued for, or executing on, a worker (any worker, in any process)."""
        return self.jobs is not None and self.jobs.is_active(engine_run_id)
