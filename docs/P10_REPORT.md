# P10 final report: production hardening, security, Docker, CI/CD

Nothing is committed or pushed. All P10 changes are in the working tree.

## VERIFIED (run on this machine)

* **Security fixes, each with tests**
  * The DOCX DTD check only scanned the first 4 KiB of a member, so a long comment prolog could hide a DTD. Fixed, with a regression test that fails on the old code.
  * Added configurable per-user rate limits on upload, run creation, search, claim verify, report, experiment, evaluate and reprocess (`RESEARCHOS_RATE_<NAME>`). An unauthenticated call never uses up budget.
  * Added baseline security headers: CSP, nosniff, frame protection, Referrer-Policy, Permissions-Policy, COOP/CORP and HSTS (when enabled).
  * `/docs` can now be switched off; the production compose file turns it off.
  * Wildcard origins are refused.
  * Agent-graph keyboard focus fixed: edges were 100+ invisible tab stops.
  * A behavioural test shows every documented operation except `/health`, `/auth/login` and `/auth/register` returns 401 without a session (more than 90 operations).
* **Dependency and secret scans**
  * `pip-audit` on the pinned runtime set: no known vulnerabilities.
  * `bandit`: 17 medium notes, reviewed as false positives (parameterised SQL with fixed fragments).
  * `gitleaks` over the history: no real secrets. Three false positives are allow-listed narrowly.
  * `npm audit fix` applied (react-router).
* **Frontend**
  * Build, `tsc` and `eslint --max-warnings 0` pass.
  * Vitest: 31 of 31 pass.
  * Playwright full-flow, axe A/AA scan (16 routes, light and dark) and the new keyboard/focus/mobile spec all pass.
* **Backup and restore drill:** passed on a throwaway local PostgreSQL 16.
  * 27 tables and 9 rows matched after restore.
  * The migration check passed on the restored database, and the app started on it and signed in.
  * The archive was 63 KB, backup took 0.3–0.7 s and restore 0.65 s.
* **Performance:** measured and recorded in `docs/PERFORMANCE.md` and `artifacts/evaluation/*.json`. The numbers are from an idle machine after discarding an earlier run that overlapped with builds.
  * Simple endpoints: p50 4–24 ms.
  * Upload: p50 about 350 ms, dominated by isolated-subprocess parsing.
  * Chunk search p50 12–30 ms from 25 to 250 documents.
  * 1, 4 and 8 concurrent users: 0 errors.
  * A research run completes in about 1.6–1.8 s, with SSE ending cleanly.
  * Cold page load: DCL 88–122 ms, LCP 344–504 ms.
* **Measured optimisations**
  * Phrase pre-filter: 145 to 110 ms p50 on 3000 passages, results identical.
  * Lazy corpus imports: child start about 375 to 195 ms in isolation. End-to-end upload stays about 350 ms.
  * Charts library no longer preloaded: first-paint JS about 752 to 366 KB raw.
  * gzip on SPA assets: JS wire about 30 % of decoded.

## IMPLEMENTED BUT UNVERIFIED

* **Multi-stage Dockerfile** (Node frontend stage, pinned Python install, tini, non-root UID 10001, healthcheck): never built. Docker is not installed here.
* **`docker-compose.prod.yml`** (internal-only PostgreSQL, one-shot migrate, API on loopback, worker with heartbeat healthcheck, read-only root, dropped capabilities): never run. Only static tests exist, and `docker compose config` was never executed.
* **GitHub Actions CI** (backend, frontend, e2e, PostgreSQL, security, Docker): never run on GitHub. Only static tests exist.
* **HSTS and Secure cookie** over real HTTPS.
* **Docker-based backup commands.**
* **`docs/DOCKER.md`** has the exact verification checklist.

## PENDING / OPEN

* **Full backend regression:** running now, about 43 % through when last checked. The baseline was 954 passed and 1 skipped. I won't call P1–P9 regression-safe until it finishes.
* **PyMuPDF licence:** AGPL-3.0 or Artifex commercial. This needs your or counsel's decision. Alternatives (pypdf, pdfminer.six, pypdfium2) were not evaluated. The repo has no LICENSE file.
* **Deferred breaking upgrades:**
  * zod (high, unreachable DoS).
  * react-router v7 (moderate).
  * vite/esbuild and vitest/tinypool (dev-only, including a critical in the test runner).
  * Deprecated: eslint 9, recharts 2.x.
* **Screen-reader testing:** not done.
  * Also not done: zoom/reflow at 200–400 % and non-Edge browsers.
* **Rate limiter:** in-process, so the limit applies per API process.
* **Experiments:** cannot run inside the container (no Docker sandbox there).
* **Benchmarks:** synthetic corpus, loopback, single machine, one run each. Docker, real literature scale and multi-worker throughput were not measured.

## Production deployment commands (unverified: no Docker here)

```bash
cp .env.example .env && chmod 600 .env
# edit .env: POSTGRES_PASSWORD (python3 -c "import secrets;print(secrets.token_urlsafe(32))"),
#            RESEARCHOS_ALLOWED_ORIGINS=https://your.domain, RESEARCHOS_TRUSTED_PROXIES
docker compose -f docker-compose.prod.yml --env-file .env config -q
docker compose -f docker-compose.prod.yml --env-file .env up -d --build
docker compose -f docker-compose.prod.yml ps
curl -fsS http://127.0.0.1:8000/health
```

Put a TLS reverse proxy in front, and set `flush_interval -1` so SSE isn't buffered. Back up the database and `/data` before every upgrade (`docs/BACKUP_RECOVERY.md`).

## Files

* **New**
  * `api/ratelimit.py`
  * `docker-compose.prod.yml`
  * `requirements/constraints.txt`
  * `.gitleaks.toml`
  * `scripts/bench_api.py`
  * `scripts/backup_restore_drill.py`
  * `frontend/e2e/keyboard.spec.ts`
  * `frontend/e2e/perf.spec.ts`
  * `tests/test_p10_{security,deploy,ci,backup}.py`
  * Docs: `SECURITY`, `DOCKER`, `DEPLOYMENT`, `CI_CD`, `PERFORMANCE`, `BACKUP_RECOVERY`, `P10_AUDIT`, `P10_REPORT`
  * Evaluation artifacts: `api_bench*.json`, `page_load.json`, `backup_restore_drill.json`
* **Modified**
  * `api/app.py`, `api/config.py` and the route files
  * `research/corpus/*`, `research/bm25/index.py`
  * `scripts/run_worker.py`
  * `Dockerfile`, `.dockerignore`, `.env.example`
  * `.github/workflows/ci.yml`
  * `frontend` (vite config, `Agents.tsx`, `index.css`, playwright config, lockfile)
  * `README.md`, `PHASE_REPORTS.md`, `IMPLEMENTATION_AUDIT.md`, `FRONTEND_TESTING.md`
