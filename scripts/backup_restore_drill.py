"""Backup and restore drill on a THROWAWAY PostgreSQL database (never run it against real user data).

    python scripts/backup_restore_drill.py [--postgres postgresql://user:pw@host:port/postgres] [--keep]

It proves the documented procedure (docs/BACKUP_RECOVERY.md) end to end:
  1. create a scratch database, migrate it, and create real application data through the HTTP API (account, project, document, search);
  2. take a custom-format `pg_dump`, and verify the archive with `pg_restore --list`;
  3. restore it into a SECOND scratch database, compare every table's row count, check migration state with `migrate_db.py check`;
  4. start the application against the restored database and sign in with the original credentials; the project and the uploaded document's
     search hit must be there (the document files are restored from a copy of the data directory: the database alone does not hold them).
Without --postgres it starts the pip-installable PostgreSQL used by the tests (pgserver) and uses its pg_dump/pg_restore. Exit code 0 only if every
check passed. The result is printed as JSON."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PW = "correct horse battery 42"


def pg_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    import pgserver
    exe = Path(pgserver.__file__).parent / "pginstall" / "bin" / (name + (".exe" if os.name == "nt" else ""))
    if exe.is_file():
        return str(exe)
    raise SystemExit(f"{name} not found: install the PostgreSQL client tools")


def with_db(url: str, db: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path=f"/{db}", query=""))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd[:3])}...\n{r.stderr[-800:]}")
    return r


def counts(url: str) -> dict[str, int]:
    import psycopg
    from agent_engine.state import dburl
    out: dict[str, int] = {}
    with psycopg.connect(dburl.psycopg_dsn(url)[0], autocommit=True) as c:
        rows = c.execute("SELECT schemaname, tablename FROM pg_tables WHERE schemaname IN ('api','engine') ORDER BY 1,2").fetchall()
        for sch, tab in rows:
            out[f"{sch}.{tab}"] = c.execute(f'SELECT COUNT(*) FROM "{sch}"."{tab}"').fetchone()[0]
    return out


def app_env(db_url: str, data_dir: Path) -> dict[str, str]:
    return {**os.environ, "RESEARCHOS_DATABASE_URL": db_url, "RESEARCHOS_DATABASE_SCHEMA": "api", "RESEARCHOS_ENGINE_DATABASE_SCHEMA": "engine",
            "RESEARCHOS_DATA_DIR": str(data_dir), "RESEARCHOS_EMBEDDED_WORKER": "false", "RESEARCHOS_AUTO_MIGRATE": "false"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--postgres")
    ap.add_argument("--keep", action="store_true", help="keep the scratch databases and files")
    a = ap.parse_args()
    import psycopg
    from agent_engine.state import dburl
    if a.postgres:
        base = a.postgres
    else:
        sys.path.insert(0, str(ROOT))
        from tests import pg_support
        base = pg_support.base_url()
    suffix = uuid.uuid4().hex[:8]
    src_db, dst_db = f"drill_src_{suffix}", f"drill_restore_{suffix}"
    work = Path(tempfile.mkdtemp(prefix="researchos-drill-"))
    checks: dict[str, object] = {}
    admin = psycopg.connect(dburl.psycopg_dsn(base)[0], autocommit=True)
    try:
        admin.execute(f'CREATE DATABASE "{src_db}"')
        src_url, dst_url = with_db(base, src_db), with_db(base, dst_db)
        src_data = work / "data_src"
        env = app_env(src_url, src_data)
        run([sys.executable, str(ROOT / "scripts" / "migrate_db.py"), "upgrade"], env=env)

        # 1. real data through the real application
        from fastapi.testclient import TestClient

        from api.app import create_app
        from api.config import Settings
        os.environ.update({k: v for k, v in env.items() if k.startswith("RESEARCHOS_")})
        app = create_app(Settings.from_env())
        with TestClient(app, base_url="http://testserver") as c:
            r = c.post("/auth/register", json={"email": "drill@example.org", "password": PW})
            h = {"X-CSRF-Token": r.json()["csrf_token"]}
            pid = c.post("/projects", headers=h, json={"title": "Drill project", "question": "Does restore preserve data?"}).json()["id"]
            up = c.post(f"/projects/{pid}/documents", headers=h, data={"license": "CC0-1.0", "source": "drill", "allowed_use": "search,quote"},
                        files={"file": ("note.txt", b"Abstract\n\nRestored libraries keep the exact passage text used for citations.\n", "text/plain")})
            assert up.status_code == 201, up.text
            hits = c.get(f"/projects/{pid}/search", params={"q": "exact passage citations"}).json()["hits"]
            checks["source_search_hits"] = len(hits)
        app.state.db_engine.dispose()
        before = counts(src_url)

        # 2. backup + verify the archive
        dump = work / "researchos.dump"
        t = time.perf_counter()
        run([pg_bin("pg_dump"), "--format=custom", "--no-owner", "--file", str(dump), src_url])
        checks["backup_seconds"] = round(time.perf_counter() - t, 2)
        checks["backup_bytes"] = dump.stat().st_size
        listing = run([pg_bin("pg_restore"), "--list", str(dump)]).stdout
        checks["archive_lists_tables"] = "TABLE api users" in listing and "TABLE engine runs" in listing
        shutil.copytree(src_data, work / "data_backup")  # the document files are not in the database

        # 3. restore into a second database
        admin.execute(f'CREATE DATABASE "{dst_db}"')
        t = time.perf_counter()
        run([pg_bin("pg_restore"), "--no-owner", "--exit-on-error", "--dbname", dst_url, str(dump)])
        checks["restore_seconds"] = round(time.perf_counter() - t, 2)
        after = counts(dst_url)
        checks["row_counts_equal"] = before == after
        checks["tables_compared"] = len(before)
        checks["rows_compared"] = sum(before.values())
        rest_data = work / "data_restored"
        shutil.copytree(work / "data_backup", rest_data)
        renv = app_env(dst_url, rest_data)
        chk = subprocess.run([sys.executable, str(ROOT / "scripts" / "migrate_db.py"), "check"], env=renv, capture_output=True, text=True)
        checks["migrations_at_head_after_restore"] = chk.returncode == 0

        # 4. the application runs on the restored database
        os.environ.update({k: v for k, v in renv.items() if k.startswith("RESEARCHOS_")})
        app2 = create_app(Settings.from_env())
        with TestClient(app2, base_url="http://testserver") as c2:
            li = c2.post("/auth/login", json={"email": "drill@example.org", "password": PW})
            checks["login_after_restore"] = li.status_code == 200
            projects = c2.get("/projects").json()
            checks["project_after_restore"] = any(p["title"] == "Drill project" for p in projects)
            hits2 = c2.get(f"/projects/{pid}/search", params={"q": "exact passage citations"}).json()["hits"]
            checks["search_after_restore"] = len(hits2) >= 1 and hits2[0]["text"] == hits[0]["text"]
        app2.state.db_engine.dispose()
    finally:
        if not a.keep:
            for db in (src_db, dst_db):
                try:
                    admin.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
                except Exception:  # noqa: BLE001
                    pass
            shutil.rmtree(work, ignore_errors=True)
        admin.close()
    bools = [v for v in checks.values() if isinstance(v, bool)]
    checks["all_passed"] = bool(bools) and all(bools)
    print(json.dumps(checks, indent=2))
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
