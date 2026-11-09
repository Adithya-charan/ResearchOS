# Docker

> **Verification status: NOT VERIFIED.** Docker is not installed on the machine this project was developed on, so the image has **never been built**
> and the compose stack has **never been started** here. What exists is static validation (`tests/test_p10_deploy.py`, YAML parsed and checked for the
> properties below) and a CI job that builds and exercises everything on GitHub's runners (which had not run when this was written). Treat the commands
> below as a checklist to run on a machine with Docker, and record the outcome.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | one multi-stage image used by the API, worker and migration job |
| `docker-compose.prod.yml` | **production** stack: PostgreSQL + migration + API + worker |
| `docker-compose.yml` | development / single host, SQLite, API only |
| `docker-compose.postgres.yml` | override that adds PostgreSQL + worker to `docker-compose.yml` (development) |
| `requirements/constraints.txt` | exact versions of the 36 runtime Python packages (everything except PyTorch) |
| `.dockerignore` | keeps `.env`, `.git`, `.venv`, databases, weights and `node_modules` out of the build context |

## Image

1. **frontend** (`node:20-bookworm-slim`): `npm ci` then `npm run build` (`tsc --noEmit && vite build`) produces `dist/`.
2. **pybuild** (`python:3.11-slim-bookworm`): virtualenv; CPU-only PyTorch from the PyTorch CPU wheel index, then
   `pip install -c requirements/constraints.txt ".[api,corpus,postgres]"`.
3. **runtime** (`python:3.11-slim-bookworm`): copies the virtualenv, the application source and `frontend/dist`. Only `tini` is added with apt.
   No Node, compiler, pip cache or tests. Runs as UID 10001, `ENTRYPOINT ["tini","--"]`, `STOPSIGNAL SIGTERM`, `HEALTHCHECK` on `/health`,
   `CMD` is `scripts/serve_api.py` (uvicorn, no reload).

Reproducibility: Python packages are pinned by the constraints file; the base images and PyTorch are **not** digest-pinned (tags float within
`3.11-slim-bookworm` / `20-bookworm-slim`; PyTorch resolves `>=2.4` from the CPU index). For fully reproducible builds pin the base image digests
and the torch version, and regenerate `constraints.txt` on Linux (it was generated on Windows/Python 3.11 and is only a constraint, not an install list).
No secrets are baked in: configuration is `RESEARCHOS_*` environment variables at run time.

## Build and run on a Docker host

```bash
git clone https://github.com/Adithya-charan/ResearchOS.git && cd ResearchOS
docker build -t researchos:local .
docker run --rm --entrypoint sh researchos:local -c 'id -u; test -f /app/frontend/dist/index.html && echo web-app-present; command -v node || echo no-node'
```

Expected: `10001`, `web-app-present`, `no-node`.

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"      # paste into POSTGRES_PASSWORD (URL-safe characters only)
# set RESEARCHOS_ALLOWED_ORIGINS=https://your.host  (or http://127.0.0.1:8000 for a local trial, with RESEARCHOS_COOKIE_SECURE=false RESEARCHOS_HSTS=false)
docker compose -f docker-compose.prod.yml --env-file .env up -d --build
docker compose -f docker-compose.prod.yml ps          # postgres, api, worker healthy; migrate exited (0)
curl -fsS http://127.0.0.1:8000/health
```

## Verification checklist (record results; none has been run yet)

1. `docker build` succeeds; image runs as UID 10001; no Node in the image.
2. `up` brings `postgres` and `api` and `worker` to `healthy`; `migrate` exits 0.
3. `curl -I http://127.0.0.1:8000/app/` shows the CSP; the app loads in a browser.
4. PostgreSQL is not published: `nc -z 127.0.0.1 5432` fails; `docker compose exec api python -c "import socket; socket.create_connection(('postgres',5432),3)"` succeeds.
5. Register, create a project, upload a document, start a research run in the UI; the worker log shows the claim and the run completes; the SSE stream updates live.
6. Generate a report; `docker compose restart api worker postgres`; sign in again: project, document and report are still there.
7. `docker compose stop worker` during a run, then `start`: the run resumes (lease expiry hands the job to the restarted worker).
8. `docker compose down` keeps data; `docker compose down -v` deletes it (do not run on real data).

The CI job `docker` in `.github/workflows/ci.yml` automates steps 1-4 and a persistence check across a restart.

## Notes

* **Experiments:** the Docker sandbox cannot be used from inside the API container (the Docker socket is deliberately not mounted), so experiment execution
  is unavailable in this deployment unless `RESEARCHOS_ALLOW_WEAK_SANDBOX=true`, which is **not isolating** and is forced off in the production file.
* **Worker health** is a file refreshed every 5 s while the worker's loop thread lives (`scripts/run_worker.py`). It detects a dead or wedged loop, not a stuck job.
* **Graceful shutdown:** `stop_grace_period` is 30 s (API) / 40 s (worker); on SIGTERM the worker hands running jobs back to the queue.
* **GPU:** the image is CPU-only on purpose. Model training is disabled (`RESEARCHOS_TRAINING_ENABLED=false`) in the production file.
