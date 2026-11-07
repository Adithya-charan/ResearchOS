"""Starts an isolated real backend for the Playwright run: fresh SQLite files, embedded worker, SPA served from ../dist."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import uvicorn  # noqa: E402

from api.app import create_app  # noqa: E402
from api.config import Settings  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="researchos-e2e-"))
settings = Settings(database_url=f"sqlite:///{(work / 'app.db').as_posix()}", engine_db=str(work / "engine.db"),
                    data_dir=work / "data", corpus_isolate=False, training_enabled=False)
uvicorn.run(create_app(settings), host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8793, log_level="warning")
