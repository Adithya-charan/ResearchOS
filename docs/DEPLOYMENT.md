# Deployment

Target: one Linux host with Docker, a TLS-terminating reverse proxy, PostgreSQL in a container, and the API + worker from one image. See `DOCKER.md`
for the container details and its **verification status (not verified: Docker was unavailable during development)**.

## Architecture

```
browser --https--> reverse proxy (TLS, HSTS, optional shared rate limit) --http--> api :8000 (loopback)
                                                                                    |  serves /app (React build), /health, JSON API, SSE
                                                       internal network (no egress) |
                          postgres:5432  <---------  api, worker, migrate (one-shot)
                          volumes: researchos-pgdata, researchos-data (/data, shared by api and worker)
```

* One origin: the API serves the web app at `/app`, so cookies and CSRF work unchanged and no CORS is required.
* `migrate` runs `scripts/migrate_db.py upgrade` once before `api` and `worker` start; they refuse to run on a schema that is behind.
* The worker executes research runs; the API never runs them (`RESEARCHOS_EMBEDDED_WORKER=false`). Add worker replicas with
  `docker compose ... up -d --scale worker=2` (claiming is atomic) and tune `RESEARCHOS_WORKER_CONCURRENCY`.

## First deployment

```bash
cp .env.example .env && chmod 600 .env
# edit .env: POSTGRES_PASSWORD (python3 -c "import secrets;print(secrets.token_urlsafe(32))"), RESEARCHOS_ALLOWED_ORIGINS=https://research.example.org,
#            RESEARCHOS_ADMIN_EMAILS (optional), RESEARCHOS_TRUSTED_PROXIES (the proxy's address as the API sees it)
docker compose -f docker-compose.prod.yml --env-file .env up -d --build
docker compose -f docker-compose.prod.yml ps
curl -fsS http://127.0.0.1:8000/health
```

Then configure the reverse proxy (Caddy shown; nginx/Traefik are equivalent). Caddy obtains and renews certificates itself:

```
research.example.org {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1          # do not buffer server-sent events
    }
}
```

For nginx: `proxy_buffering off;` (or `X-Accel-Buffering: no`) and a long `proxy_read_timeout` (>= 600 s) for `/runs/*/stream`; pass `X-Forwarded-For`/`-Proto`.
Set `RESEARCHOS_TRUSTED_PROXIES` to the proxy address so the client IP (used for rate limits and audit) comes from `X-Forwarded-For`; forwarded headers from any other
address are ignored.

Upgrade: `git pull && docker compose -f docker-compose.prod.yml up -d --build` (take a backup first, see `BACKUP_RECOVERY.md`). Logs:
`docker compose -f docker-compose.prod.yml logs -f api worker`.

## HTTPS requirements

* Production must be served over HTTPS only. The production compose file defaults `RESEARCHOS_COOKIE_SECURE=true` (the session cookie is then never sent over
  http) and `RESEARCHOS_HSTS=true`. With a Secure cookie, signing in over plain `http://` will not work (except on `localhost` in most browsers).
* For a local trial without TLS set `RESEARCHOS_COOKIE_SECURE=false RESEARCHOS_HSTS=false RESEARCHOS_ALLOWED_ORIGINS=http://127.0.0.1:8000`.
* The API logs a warning at startup for risky combinations (https origin with an insecure cookie, weak sandbox, short passwords) and refuses a wildcard origin.
* **Not verified:** the Secure-cookie and HSTS behaviour over real HTTPS.

## Environment variables

All are optional unless marked. Booleans accept `1/true/yes/on`. Defaults are for development; the production compose file overrides several (shown).

**Database**

| Variable | Default | Meaning |
|---|---|---|
| `RESEARCHOS_DATABASE_URL` | SQLite file under `artifacts/api/` | `postgresql://user:pw@host:5432/db` for production. Add `?sslmode=require` when the database is not on a private network |
| `RESEARCHOS_DATABASE_SCHEMA` / `RESEARCHOS_ENGINE_DATABASE_SCHEMA` | `api` / `engine` | PostgreSQL schemas (API and engine tables share names, so they need separate schemas) |
| `RESEARCHOS_ENGINE_DATABASE_URL` | same server | optional: engine state in another database |
| `RESEARCHOS_ENGINE_DB` | `artifacts/api/engine.db` | engine SQLite file (SQLite mode) |
| `RESEARCHOS_DB_POOL_SIZE`, `RESEARCHOS_DB_MAX_OVERFLOW` | 10, 20 | PostgreSQL connections per process |
| `RESEARCHOS_AUTO_MIGRATE` | `true` (prod compose: `false`) | apply migrations at start; `false` = refuse to start unless at head |
| `POSTGRES_USER`, `POSTGRES_DB`, **`POSTGRES_PASSWORD`** (required) | `researchos`, `researchos`, none | used by the compose `postgres` service and to build the URL |

**Security and web**

| Variable | Default | Meaning |
|---|---|---|
| **`RESEARCHOS_ALLOWED_ORIGINS`** (required in prod compose) | localhost origins | exact, comma-separated origins allowed to make state-changing requests; wildcards are refused |
| `RESEARCHOS_COOKIE_SECURE` | `false` (prod compose: `true`) | `Secure` flag on the session cookie |
| `RESEARCHOS_HSTS` | follows `COOKIE_SECURE` | send `Strict-Transport-Security` |
| `RESEARCHOS_ENABLE_DOCS` | `true` (prod compose: `false`) | serve `/docs` and `/openapi.json` |
| `RESEARCHOS_TRUSTED_PROXIES` | `127.0.0.1` | addresses whose `X-Forwarded-*` headers are honoured |
| `RESEARCHOS_ADMIN_EMAILS` | none | accounts that may deploy models / change system-model lifecycle |
| `RESEARCHOS_SESSION_TTL_S` | 43200 | session lifetime |
| `RESEARCHOS_MIN_PASSWORD_LENGTH` | 12 | |
| `RESEARCHOS_LOGIN_ATTEMPTS`, `RESEARCHOS_LOGIN_WINDOW_S`, `RESEARCHOS_REGISTER_ATTEMPTS` | 5, 900, 10 (per hour per IP) | |
| `RESEARCHOS_RATE_<NAME>` | see `SECURITY.md` | `count/window_seconds` or `off`; names: `UPLOAD, RUN_CREATE, SEARCH, VERIFY, REPORT, EXPERIMENT, EVALUATE, REPROCESS` |

**Capacity and limits**

| Variable | Default | Meaning |
|---|---|---|
| `RESEARCHOS_MAX_UPLOAD_BYTES` | 26214400 | largest accepted file |
| `RESEARCHOS_MAX_DOCUMENTS_PER_PROJECT`, `RESEARCHOS_MAX_CORPUS_BYTES` | 200, 524288000 | per-project library caps |
| `RESEARCHOS_MAX_PROJECTS_PER_USER`, `RESEARCHOS_MAX_ACTIVE_RUNS_PER_USER` | 50, 2 | |
| `RESEARCHOS_SSE_MAX_STREAMS_PER_USER`, `RESEARCHOS_SSE_MAX_STREAMS_TOTAL` | 4, 16 | concurrent live streams |
| `RESEARCHOS_EMBEDDED_WORKER` | `true` (prod compose: `false`) | run a worker thread inside the API process |
| `RESEARCHOS_WORKER_CONCURRENCY` | 2 | runs one worker executes at once |
| `RESEARCHOS_DATA_DIR`, `RESEARCHOS_MODELS_DIR` | `artifacts/api/data`, `artifacts/checkpoints` | `/data/files`, `/models` in the image |
| `RESEARCHOS_TRAINING_ENABLED` | `false` | allow training jobs on this host (off in prod) |
| `RESEARCHOS_ALLOW_WEAK_SANDBOX` | `false` | non-isolating experiment fallback: development only |
| `RESEARCHOS_DEMO_CORPUS` | `false` | search synthetic demo text when a project has no documents |
| `RESEARCHOS_INFERENCE_DEVICE` | `cpu` | |

**Logging**

| Variable | Default | Meaning |
|---|---|---|
| `RESEARCHOS_LOG_LEVEL` | `info` | uvicorn log level (`debug`, `info`, `warning`, `error`); the worker logs at INFO |
| `RESEARCHOS_WORKER_HEARTBEAT` | `/tmp/researchos-worker.alive` | liveness file the worker refreshes |

Logs go to stdout/stderr (collect them with the container runtime). Each request has an `X-Request-ID`; unhandled errors log the id with the stack
trace and return only the id to the client. Passwords, tokens and document text are never written to audit records.

## File upload limits

Defaults: 25 MB per file, 200 documents and 500 MB per project, PDF/DOCX/TXT/MD/HTML only (type detected from content), parsing in a timed subprocess,
120 uploads per 10 minutes per user. Put the same ceiling at the proxy (`client_max_body_size 30m` in nginx; Caddy: `request_body { max_size 30MB }`).

## Production checklist

- [ ] `POSTGRES_PASSWORD` is long and random; `.env` is `chmod 600` and not committed
- [ ] `RESEARCHOS_ALLOWED_ORIGINS` is the exact https origin; cookies Secure; HSTS on
- [ ] PostgreSQL is not published (`docker compose ps` shows no host port for it)
- [ ] reverse proxy terminates TLS, does not buffer SSE, sets `X-Forwarded-*`; `RESEARCHOS_TRUSTED_PROXIES` matches it
- [ ] `RESEARCHOS_ENABLE_DOCS=false`, training off, weak sandbox off
- [ ] backups (database **and** `/data`) scheduled, copied off the host, and a restore has been rehearsed
- [ ] PyMuPDF licence decision recorded (`SECURITY.md`); a LICENSE file exists
- [ ] CI is green on the commit being deployed (`CI_CD.md`)
- [ ] a human has read the startup log for `configuration:` warnings

## Known limitations of this deployment

Single host, no high availability; in-process rate limiting (per API process); no experiment execution (no Docker sandbox inside the container);
CPU-only inference; the bundled models are demo-scale and not used for research output by default (see P8 reports).
