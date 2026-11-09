"""Run a research worker as its own process:  python scripts/run_worker.py [--concurrency N]

It claims queued research jobs from the application database and executes them with the agent engine. Run it next to an API that has
RESEARCHOS_EMBEDDED_WORKER=false (or alongside an embedded one: claiming is atomic, so a job is never executed twice). SQLite is
shared through files, so the API and worker must see the same RESEARCHOS_DATABASE_URL / RESEARCHOS_ENGINE_DB / RESEARCHOS_DATA_DIR.
SIGINT/SIGTERM stop it gracefully: running jobs are handed back to the queue and another worker resumes them."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, help="runs executed at the same time (default: RESEARCHOS worker_concurrency)")
    args = ap.parse_args()
    from api.app import prepare_schema
    from api.config import Settings
    from api.corpus_service import CorpusManager
    from api.db import make_engine, make_sessionmaker
    from api.jobs import JobQueue, Worker
    from api.runtime import EngineRuntime

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    if args.concurrency:
        settings.worker_concurrency = args.concurrency
    engine = make_engine(settings.database_url, settings.db_pool_size, settings.db_max_overflow)
    prepare_schema(engine, settings)
    sessions = make_sessionmaker(engine)
    runtime = EngineRuntime(settings, None, CorpusManager(settings, None))
    runtime.jobs = JobQueue(sessions, settings, runtime.store)
    worker = Worker(settings, runtime, runtime.jobs, sessions)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    worker.start()
    logging.getLogger("researchos.worker").info("worker %s started (concurrency %d)", worker.worker_id, settings.worker_concurrency)
    # Liveness for container health checks: the file is touched only while the worker's loop thread is alive, so a wedged or dead
    # worker stops refreshing it. (It proves the loop is running, not that jobs are progressing: see docs/DEPLOYMENT.md.)
    beat = os.environ.get("RESEARCHOS_WORKER_HEARTBEAT", "/tmp/researchos-worker.alive")
    while not done.wait(5.0):
        if worker._thread is not None and worker._thread.is_alive():
            try:
                Path(beat).write_text(str(time.time()))
            except OSError:
                pass
    worker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
