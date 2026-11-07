"""Apply (or check) the schema migrations of both databases - the API's and the agent engine's - for whichever backend the
RESEARCHOS_* environment selects (SQLite files by default, PostgreSQL when RESEARCHOS_DATABASE_URL is a postgresql:// URL).

    python scripts/migrate_db.py upgrade     # create or upgrade both schemas (the compose `migrate` service runs this)
    python scripts/migrate_db.py check       # exit 1 when either schema is not at the latest revision; changes nothing

On PostgreSQL the work is serialised by advisory locks, so running this next to API and worker processes that are starting up is safe.
A schema this code does not recognise (tables without migration history, or a revision from newer code) is refused with exit code 2 and
is never altered. Nothing here drops or resets data. The command waits up to --wait seconds for the database to accept connections."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["upgrade", "check"])
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the database to accept connections")
    args = ap.parse_args()
    from agent_engine.state import dburl
    from agent_engine.state import migrate as engine_migrate
    from agent_engine.state.store import StoreUnavailable
    from api import migrate as api_migrate
    from api.config import Settings
    from api.db import make_engine

    settings = Settings.from_env()
    engine_target = str(settings.engine_db)
    deadline = time.monotonic() + args.wait
    while True:
        try:
            api_engine = make_engine(settings.database_url, 2, 0)
            try:
                api_now = api_migrate.current_revision(api_engine)
                engine_now = None if engine_target == ":memory:" else (
                    engine_migrate.current_revision(engine_target)
                    if dburl.is_postgres(engine_target) or Path(engine_target).exists() else None)
                if args.action == "check":
                    print(f"api    revision {api_now}  (head {api_migrate.head_revision()})")
                    print(f"engine revision {engine_now}  (head {engine_migrate.head_revision()})")
                    return 0 if (api_now == api_migrate.head_revision() and engine_now == engine_migrate.head_revision()) else 1
                print("api    schema at", api_migrate.upgrade(api_engine))
                if engine_target != ":memory:":
                    print("engine schema at", engine_migrate.upgrade(engine_target))
                return 0
            finally:
                api_engine.dispose()
        except (api_migrate.SchemaMismatch, engine_migrate.SchemaMismatch) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            transient = type(exc).__name__ in ("OperationalError", "StoreUnavailable", "InterfaceError") or isinstance(exc, StoreUnavailable)
            if not transient or time.monotonic() > deadline:
                print(f"failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 3
            time.sleep(1.0)  # the database is still starting


if __name__ == "__main__":
    sys.exit(main())
