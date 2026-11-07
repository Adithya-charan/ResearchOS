"""Copy an existing SQLite installation into PostgreSQL.

    python scripts/migrate_database.py --to postgresql://user:password@host:5432/researchos \\
        --api-db artifacts/api/researchos.db --engine-db artifacts/api/engine.db

The API tables go to schema `api` and the engine tables to schema `engine` of the target database (override with --api-schema /
--engine-schema). Source files are only read. The target must be empty: the script refuses to merge into data that is already there,
and each database is copied in one transaction that is committed only after the row counts match. Read the password from the
environment rather than the command line where you can:  --to "$RESEARCHOS_PG_URL".
See docs/DATABASE.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", required=True, help="postgresql:// URL of the target database")
    ap.add_argument("--api-db", help="SQLite file (or sqlite:/// URL) of the API database")
    ap.add_argument("--engine-db", help="SQLite file of the agent-engine database")
    ap.add_argument("--api-schema", default="api")
    ap.add_argument("--engine-schema", default="engine")
    args = ap.parse_args()
    from agent_engine.state import dburl
    from api import dbcopy

    if not dburl.is_postgres(args.to):
        ap.error("--to must be a postgresql:// URL")
    if not (args.api_db or args.engine_db):
        ap.error("give --api-db and/or --engine-db")
    report: dict = {}
    try:
        if args.api_db:
            report["api"] = dbcopy.copy_api(args.api_db, dburl.with_schema(args.to, args.api_schema))
        if args.engine_db:
            report["engine"] = dbcopy.copy_engine(args.engine_db, dburl.with_schema(args.to, args.engine_schema))
    except dbcopy.CopyRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"copied_rows": report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
