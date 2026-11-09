# P10 audit: what was verified at the start, what is complete, incomplete or unverified

Method: the repository, git history, working tree, reports and configuration were inspected directly on 2026-11-09, and claims in earlier reports were
re-checked against the code or by running the relevant tests, not assumed. Start point: `main` at `b9f2f96` (P7-P9 committed and pushed).

## State at the start of P10

| Item | Finding | Evidence |
|---|---|---|
| P1-P9 code present | present; P8 is demo-scale (see P8 report), P9 is Vite + React | git log, test suites |
| Test baseline | 954 passed, 1 skipped | full `pytest` run before P10 changes |
| Frontend | build, `tsc`, `eslint`, vitest (31), Playwright full-flow and axe scan all passing | run before P10 |
| API architecture | FastAPI, routes at root, session cookie + CSRF, SSE, embedded or separate worker | code |
| PostgreSQL | integration-tested against a real PostgreSQL 16 (pgserver); compose file never run | P7 report |
| SQLite development mode | working (default) | tests |
| Dockerfile | **existed, single-stage, did not build the frontend** (the web app was missing from the image), no pinned dependencies, no `tini` | file |
| Compose | `docker-compose.yml` (SQLite) and a PostgreSQL override; **no self-contained production file**, no worker health check, no internal network | files |
| CI | one workflow: backend tests, PostgreSQL tests, hygiene, a Docker job; **no frontend job, no e2e, no dependency or secret scanning, no migration validation** | `ci.yml` |
| Rate limits | login, registration, model inference only | code |
| Docker on this machine | **not installed** | `docker: command not found` |
| Git history secrets | none real (3 hash/fixture false positives found by gitleaks, now allow-listed narrowly) | gitleaks |

## P10 status by item

| Required item | Status | Where |
|---|---|---|
| Security review (backend, frontend) | **done and documented** | `SECURITY.md` |
| Critical security issues resolved | **done** (no critical found; 1 high, 4 medium fixed) | `SECURITY.md` |
| Rate limiting | **implemented, tested, configurable** | `api/ratelimit.py`, `tests/test_p10_security.py` |
| Dependency audit (npm, Python, licences) | **done**; many dev-only npm advisories **not** fixed (breaking upgrades), two production ones triaged | `SECURITY.md` |
| PyMuPDF licensing | **investigated; decision open** (needs owner / counsel) | `SECURITY.md` |
| Security headers | **implemented, tested locally**; HSTS / Secure cookie **unverified over real HTTPS** | `SECURITY.md` |
| Production Dockerfile (multi-stage, frontend built, non-root, health, tini) | **written, statically tested, NOT built** | `Dockerfile`, `DOCKER.md` |
| Production compose | **written, statically tested, NOT run** | `docker-compose.prod.yml` |
| Real Docker verification | **NOT DONE: Docker unavailable**; commands and checklist provided | `DOCKER.md` |
| CI workflows | **written and statically tested; NOT yet run on GitHub** | `.github/workflows/ci.yml`, `CI_CD.md` |
| Performance measurements | **measured on this machine** (SQLite and PostgreSQL, loopback), limits stated | `PERFORMANCE.md`, `artifacts/evaluation/api_bench*.json` |
| Backup and recovery | **procedure documented; restore drill passed on a throwaway local PostgreSQL**; Docker-based commands unverified | `BACKUP_RECOVERY.md` |
| Production configuration / env docs / `.env.example` | **done** | `DEPLOYMENT.md`, `.env.example` |
| Accessibility | automated keyboard, focus, modal and mobile axe checks **pass**; screen-reader testing **pending** | `FRONTEND_TESTING.md` |
| Regression (P1-P9) | see the final report for the full-suite result | `PHASE_REPORTS.md` |
