"""Deployment files. Docker is not installed on every development machine, so these tests check what can be checked statically
(every COPY source exists and is not ignored, the container runs unprivileged, the compose file grants nothing extra) and that the
app starts from container-style environment variables with directories that do not exist yet. CI builds and runs the real image."""
import fnmatch
import re
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings
from scripts.serve_api import server_options

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def ignored(path: str) -> bool:
    patterns = [ln.strip() for ln in (ROOT / ".dockerignore").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    hit = False
    for pat in patterns:
        negate = pat.startswith("!")
        pat = pat.lstrip("!").rstrip("/")
        if fnmatch.fnmatch(path, pat) or path.startswith(pat + "/") or fnmatch.fnmatch(Path(path).name, pat):
            hit = not negate
    return hit


def test_every_copied_path_exists_and_is_not_dockerignored():
    sources = [ln.split()[1] for ln in DOCKERFILE.splitlines() if ln.startswith("COPY ") and not ln.startswith("COPY --from")]
    assert sources and "pyproject.toml" in sources
    for src in sources:
        assert (ROOT / src).exists(), f"COPY {src}: not in the repository"
        assert not ignored(src.rstrip("/")), f"COPY {src}: excluded by .dockerignore, so the build would fail"
    for sub in (ROOT / "artifacts" / "tokenizer").iterdir():
        assert not ignored(f"artifacts/tokenizer/{sub.name}")


def test_image_runs_unprivileged_and_ships_no_secrets_or_weights():
    users = re.findall(r"^USER\s+(\S+)", DOCKERFILE, re.M)
    assert users and users[-1] not in ("root", "0") and users[-1].split(":")[0].isdigit()
    assert "HEALTHCHECK" in DOCKERFILE and "EXPOSE 8000" in DOCKERFILE
    for needle in (".env", "checkpoints", "datasets", ".db"):
        assert not any(needle in ln for ln in DOCKERFILE.splitlines() if ln.startswith("COPY")), needle
    assert re.search(r"RESEARCHOS_DATABASE_URL=sqlite:////data/", DOCKERFILE)  # state lives on the volume, not in the image layer


def test_compose_grants_nothing_extra():
    api = COMPOSE["services"]["api"]
    flat = yaml.safe_dump(COMPOSE)
    assert "docker.sock" not in flat and not api.get("privileged") and "network_mode" not in api
    assert api["read_only"] is True and api["cap_drop"] == ["ALL"] and "no-new-privileges:true" in api["security_opt"]
    assert all(p.startswith("127.0.0.1:") for p in api["ports"])  # loopback only; TLS proxy is the public entry point
    assert any(v.endswith(":/models:ro") for v in api["volumes"]) and any(v.startswith("researchos-data:/data") for v in api["volumes"])
    env = api["environment"]
    assert env["RESEARCHOS_ALLOW_WEAK_SANDBOX"].endswith(":-false}") and env["RESEARCHOS_TRAINING_ENABLED"].endswith(":-false}")
    assert set(COMPOSE["services"]) == {"api"}  # no PostgreSQL/Redis service until the app actually uses them


def test_proxy_headers_are_trusted_only_from_configured_addresses():
    default = server_options("0.0.0.0", 8000, {})
    assert default["factory"] and default["proxy_headers"] and default["forwarded_allow_ips"] == "127.0.0.1"
    custom = server_options("0.0.0.0", 8000, {"RESEARCHOS_TRUSTED_PROXIES": "10.0.0.5", "RESEARCHOS_LOG_LEVEL": "warning"})
    assert custom["forwarded_allow_ips"] == "10.0.0.5" and custom["log_level"] == "warning"


def test_app_starts_from_container_style_environment_with_missing_directories(tmp_path, monkeypatch):
    data = tmp_path / "data"  # does not exist yet, like a fresh volume
    for k, v in {"RESEARCHOS_DATABASE_URL": f"sqlite:///{(data / 'researchos.db').as_posix()}",
                 "RESEARCHOS_ENGINE_DB": str(data / "engine.db"), "RESEARCHOS_DATA_DIR": str(data / "files"),
                 "RESEARCHOS_MODELS_DIR": str(tmp_path / "models-not-mounted"), "RESEARCHOS_ALLOWED_ORIGINS": "https://app.example.org"}.items():
        monkeypatch.setenv(k, v)
    s = Settings.from_env()
    assert s.models_dir == tmp_path / "models-not-mounted" and s.allowed_origins == ("https://app.example.org",)
    with TestClient(create_app(s), base_url="http://testserver") as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/auth/register", json={"email": "ops@example.org", "password": "correct horse battery 42"})
        assert r.status_code == 201
        assert c.get("/models").json() == []  # no operator models mounted: an empty registry, not an error
        assert c.get("/models/health").json()["status"] == "degraded"
    assert (data / "researchos.db").exists() and (data / "engine.db").exists()
