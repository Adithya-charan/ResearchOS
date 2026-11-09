"""P10 static checks of the production Docker artifacts. These do NOT prove the image builds or the stack starts: Docker is not
available on the development machine, so those steps are verified in CI / on a Docker host (docs/DOCKER.md). They guard the properties
that can be checked from the files: multi-stage build, non-root runtime, no secrets, internal database network, health checks."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
PROD = yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8"))
PROD_TEXT = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")


def test_dockerfile_is_multi_stage_and_builds_the_frontend_into_the_image():
    stages = re.findall(r"^FROM (\S+) AS (\w+)", DOCKERFILE, re.M)
    assert [n for _, n in stages] == ["frontend", "pybuild", "runtime"]
    assert "npm ci" in DOCKERFILE and "npm run build" in DOCKERFILE
    assert "COPY --from=frontend /build/dist ./frontend/dist" in DOCKERFILE
    runtime = DOCKERFILE.split("AS runtime", 1)[1]
    assert "node" not in runtime.lower().replace("node_modules", "") and "npm" not in runtime  # no Node toolchain in the final image


def test_runtime_image_is_unprivileged_with_healthcheck_and_signal_handling():
    runtime = DOCKERFILE.split("AS runtime", 1)[1]
    assert re.search(r"^USER 10001:10001", runtime, re.M)
    assert "HEALTHCHECK" in runtime and "/health" in runtime
    assert 'ENTRYPOINT ["tini", "--"]' in runtime and "STOPSIGNAL SIGTERM" in runtime
    cmd = re.search(r"^CMD (.+)$", runtime, re.M).group(1)
    assert "serve_api.py" in cmd and "--reload" not in cmd and "--reload" not in DOCKERFILE


def test_dockerfile_embeds_no_secrets_and_installs_pinned_dependencies():
    assert not re.search(r"(?i)^(ENV|ARG)\s+\w*(PASSWORD|SECRET|TOKEN|API_KEY)\w*", DOCKERFILE, re.M)
    assert "-c constraints.txt" in DOCKERFILE and (ROOT / "requirements" / "constraints.txt").is_file()
    pins = [l for l in (ROOT / "requirements" / "constraints.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert pins and all("==" in l for l in pins)
    assert not any(l.lower().startswith(("torch", "nvidia")) for l in pins)  # torch comes from the CPU wheel index


def test_dockerignore_keeps_secrets_state_and_node_modules_out_of_the_build_context():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for must in (".env", ".git", ".venv", "frontend/node_modules", "*.db", "*.safetensors"):
        assert must in ignored, must


def test_prod_compose_database_is_internal_and_never_published():
    svc = PROD["services"]
    assert "ports" not in svc["postgres"] and svc["postgres"]["networks"] == ["internal"]
    assert PROD["networks"]["internal"]["internal"] is True
    assert svc["worker"]["networks"] == ["internal"] and svc["migrate"]["networks"] == ["internal"]
    assert set(svc["api"]["networks"]) == {"internal", "default"}
    assert svc["api"]["ports"] == ["127.0.0.1:${RESEARCHOS_PORT:-8000}:8000"]
    assert "researchos-pgdata:/var/lib/postgresql/data" in svc["postgres"]["volumes"]
    assert "docker.sock" not in PROD_TEXT


def test_prod_compose_orders_startup_and_has_health_checks_and_restarts():
    svc = PROD["services"]
    assert svc["postgres"]["healthcheck"]["test"][0] == "CMD-SHELL" and svc["postgres"]["restart"] == "unless-stopped"
    for name in ("api", "worker"):
        dep = svc[name]["depends_on"]
        assert dep["postgres"]["condition"] == "service_healthy" and dep["migrate"]["condition"] == "service_completed_successfully"
        assert svc[name]["restart"] == "unless-stopped" and svc[name]["init"] is True
    assert svc["migrate"]["restart"] == "no" and svc["migrate"]["command"][-1] == "upgrade"
    assert "healthcheck" in svc["worker"] and svc["worker"]["stop_grace_period"] == "40s"
    assert svc["worker"]["command"] == ["python", "scripts/run_worker.py"]


def test_prod_compose_secrets_are_required_from_the_environment_and_nothing_weak_is_default():
    assert "${POSTGRES_PASSWORD:?" in PROD_TEXT and "${RESEARCHOS_ALLOWED_ORIGINS:?" in PROD_TEXT
    env = PROD["x-researchos-env"]
    assert env["RESEARCHOS_AUTO_MIGRATE"] == "false" and env["RESEARCHOS_EMBEDDED_WORKER"] == "false"
    assert env["RESEARCHOS_ALLOW_WEAK_SANDBOX"] == "false" and env["RESEARCHOS_TRAINING_ENABLED"] == "false"
    api_env = PROD["services"]["api"]["environment"]
    assert api_env["RESEARCHOS_COOKIE_SECURE"].endswith(":-true}") and api_env["RESEARCHOS_HSTS"].endswith(":-true}")
    for name in ("api", "worker", "migrate"):
        s = PROD["services"][name]
        assert s["read_only"] is True and s["cap_drop"] == ["ALL"] and "no-new-privileges:true" in s["security_opt"]


def test_env_example_has_no_real_credentials():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^POSTGRES_PASSWORD=$", text, re.M)  # blank on purpose
    assert not re.search(r"(?i)(secret|password|token|key)\s*=\s*[A-Za-z0-9+/_\-]{16,}", text)
