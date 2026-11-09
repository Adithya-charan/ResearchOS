"""P7 deployment files and tooling. Docker is not installed on the development machine, so the compose files are checked statically; the
database behaviour behind them is covered by tests/test_p7_postgres.py against a real PostgreSQL server."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests import pg_support as pg

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
PGTEXT = (ROOT / "docker-compose.postgres.yml").read_text(encoding="utf-8")
PG = yaml.safe_load(PGTEXT)
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


def test_the_default_compose_file_still_runs_on_sqlite_alone():
    assert set(BASE["services"]) == {"api"}
    assert "postgres" not in yaml.safe_dump(BASE).lower().replace("postgresql / redis", "")  # the base file is unchanged by P7


def test_postgres_override_defines_the_expected_services_and_ordering():
    svc = PG["services"]
    assert {"postgres", "migrate", "api", "worker"} <= set(svc)
    for name in ("api", "worker"):
        deps = svc[name]["depends_on"]
        assert deps["postgres"]["condition"] == "service_healthy"
        assert deps["migrate"]["condition"] == "service_completed_successfully"  # nothing starts before the schema is at head
    assert svc["migrate"]["command"] == ["python", "scripts/migrate_db.py", "upgrade"] and svc["migrate"]["restart"] == "no"
    assert svc["worker"]["command"] == ["python", "scripts/run_worker.py"]
    assert "pg_isready" in " ".join(svc["postgres"]["healthcheck"]["test"])


def test_database_is_persistent_private_and_has_no_hardcoded_credentials():
    pg_svc = PG["services"]["postgres"]
    assert any(v.startswith("researchos-pgdata:/var/lib/postgresql/data") for v in pg_svc["volumes"]) and "researchos-pgdata" in PG["volumes"]
    assert "ports" not in pg_svc  # reachable from the other services only
    # every credential is an interpolation; the password has no default and the file refuses to start without it
    assert "${POSTGRES_PASSWORD:?" in PGTEXT
    assert not re.search(r"(?<!\{)POSTGRES_PASSWORD\s*[:=]\s*[^$\s'\"\n]", PGTEXT) and "POSTGRES_PASSWORD:-" not in PGTEXT
    url = PG["x-researchos-env"]["RESEARCHOS_DATABASE_URL"]
    assert re.fullmatch(r"postgresql://\$\{POSTGRES_USER:-researchos\}:\$\{POSTGRES_PASSWORD:\?[^}]+\}@postgres:5432/\$\{POSTGRES_DB:-researchos\}", url)
    assert "POSTGRES_PASSWORD=" in (ROOT / ".env.example").read_text(encoding="utf-8")
    assert not re.search(r"POSTGRES_PASSWORD=\S", (ROOT / ".env.example").read_text(encoding="utf-8"))  # blank in the template
    assert ".env" in (ROOT / ".dockerignore").read_text(encoding="utf-8").split()


def test_application_services_are_hardened_and_shut_down_gracefully():
    for name in ("migrate", "worker"):
        s = PG["services"][name]
        assert s["read_only"] is True and s["cap_drop"] == ["ALL"] and "no-new-privileges:true" in s["security_opt"]
    assert "docker.sock" not in PGTEXT
    assert PG["services"]["worker"]["stop_grace_period"] == "40s" and PG["services"]["worker"]["init"] is True
    env = PG["x-researchos-env"]
    assert env["RESEARCHOS_EMBEDDED_WORKER"] == "false" and env["RESEARCHOS_AUTO_MIGRATE"] == "false"
    assert env["RESEARCHOS_DATABASE_SCHEMA"] == "api" and env["RESEARCHOS_ENGINE_DATABASE_SCHEMA"] == "engine"


def test_the_image_installs_the_postgresql_driver_and_defaults_to_sqlite():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r'pip install (?:-c \S+ )?"\.\[[^\]]*postgres[^\]]*\]"', dockerfile)  # optional pinned-constraints flag
    assert "RESEARCHOS_DATABASE_URL=sqlite:////data/" in dockerfile  # still a working single-container default
    assert "postgres" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def sqlite_env(tmp_path):
    return {**ENV, "RESEARCHOS_DATABASE_URL": f"sqlite:///{(tmp_path / 'a.db').as_posix()}", "RESEARCHOS_ENGINE_DB": str(tmp_path / "e.db")}


def run_migrate(env, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "migrate_db.py"), *args], capture_output=True, text=True, env=env, timeout=180)


def test_migrate_db_script_on_sqlite(tmp_path):
    env = sqlite_env(tmp_path)
    assert run_migrate(env, "check", "--wait", "1").returncode == 1  # nothing created yet
    up = run_migrate(env, "upgrade")
    assert up.returncode == 0, up.stderr
    assert run_migrate(env, "upgrade").returncode == 0  # idempotent
    assert run_migrate(env, "check").returncode == 0


@pytest.mark.postgres
@pytest.mark.skipif(not pg.available(), reason="needs a PostgreSQL server")
def test_migrate_db_script_on_postgres_and_it_refuses_foreign_schemas():
    db = pg.new_database()
    env = {**ENV, "RESEARCHOS_DATABASE_URL": db}
    for k in ("RESEARCHOS_ENGINE_DATABASE_URL", "RESEARCHOS_DATABASE_SCHEMA", "RESEARCHOS_ENGINE_DATABASE_SCHEMA", "RESEARCHOS_ENGINE_DB"):
        env.pop(k, None)
    assert run_migrate(env, "check", "--wait", "5").returncode == 1
    up = run_migrate(env, "upgrade")
    assert up.returncode == 0, up.stderr
    assert "api" in up.stdout and "engine" in up.stdout
    assert run_migrate(env, "check").returncode == 0
    # a schema that holds someone else's `runs` table is refused with exit code 2 and left alone
    env2 = {**env, "RESEARCHOS_DATABASE_URL": pg.new_database()}
    base = env2["RESEARCHOS_DATABASE_URL"]
    with pg.connect(base) as c:
        c.execute('CREATE SCHEMA "api"')
        c.execute('CREATE TABLE "api".users (id TEXT, note TEXT)')
        c.execute("INSERT INTO \"api\".users VALUES ('1', 'keep')")
    bad = run_migrate(env2, "upgrade", "--wait", "5")
    assert bad.returncode == 2 and "refused" in bad.stderr
    with pg.connect(base) as c:
        assert c.execute('SELECT note FROM "api".users').fetchall() == [("keep",)]
