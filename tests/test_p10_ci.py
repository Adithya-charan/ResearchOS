"""P10: static checks of the GitHub Actions workflow. They show the workflow is well formed and has the required jobs and safeguards;
they do not show that it passes on GitHub (it had not been run there when this was written)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
CI = yaml.safe_load(TEXT)
JOBS = CI["jobs"]


def steps_text(job: str) -> str:
    return "\n".join(str(s.get("run", "")) + str(s.get("uses", "")) for s in JOBS[job]["steps"])


def test_required_jobs_exist():
    assert {"backend", "frontend", "e2e", "postgres", "security", "hygiene", "docker"} <= set(JOBS)


def test_frontend_job_runs_types_lint_unit_tests_and_build():
    t = steps_text("frontend")
    for cmd in ("npx tsc --noEmit", "npx eslint .", "npx vitest run", "npm run build"):
        assert cmd in t, cmd


def test_backend_and_postgres_jobs_run_the_suites_and_migration_checks():
    assert "python -m pytest" in steps_text("backend") and "constraints.txt" in steps_text("backend")
    pg = steps_text("postgres")
    assert "tests/test_p7_postgres.py" in pg and "RESEARCHOS_TEST_ON_POSTGRES" in TEXT
    assert "migrate_db.py upgrade" in pg and "migrate_db.py check" in pg
    assert JOBS["postgres"]["services"]["postgres"]["image"].startswith("postgres:")


def test_security_job_scans_dependencies_code_and_secrets():
    t = steps_text("security")
    assert "pip-audit" in t and "bandit" in t and "npm audit --omit=dev --audit-level=critical" in t and "gitleaks" in t
    assert JOBS["security"]["steps"][0]["with"]["fetch-depth"] == 0  # whole history is scanned


def test_docker_job_builds_the_image_and_exercises_the_production_stack():
    t = steps_text("docker")
    assert "docker build" in t and "docker-compose.prod.yml" in t and "restart api worker" in t and "5432" in t


def test_workflow_uses_no_repository_secrets_has_least_privilege_and_never_deploys():
    assert "secrets." not in TEXT
    assert CI["permissions"] == {"contents": "read"}
    lowered = TEXT.lower()
    assert "docker push" not in lowered and "kubectl" not in lowered and "ssh " not in lowered and "environment: production" not in lowered
