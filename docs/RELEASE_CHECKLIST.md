# Release checklist and quality-gate status

Updated 2026-11-10 at commit `dfad702` (local; `origin/main` is at `4842df1`, so `dfad702` is **not pushed**). A failed or blocked gate stays visible here.
Status words: **PASS** (run, with evidence), **BLOCKED** (cannot be run in the current environment), **OPEN** (not done), **PARTIAL**.

| Gate | Status | Evidence / what is missing |
|---|---|---|
| Backend regression | **PASS** | 991 passed, 0 failed, 1 skipped (the "app not built" case), 1 third-party warning; pytest exit 0 (run before commit `dfad702`, tree unchanged since) |
| Frontend (tsc, eslint 0 warnings, vitest 31, build) | **PASS** at `4842df1` | re-run 2026-11-09; frontend code unchanged since; not re-run on 2026-11-10 |
| E2E (Playwright full-flow), axe A/AA, keyboard/focus | **PASS** at `4842df1` | same note |
| Cross-user isolation | **PASS** | `tests/test_p10_isolation.py` (about 95 id-taking operations; mutation-checked) |
| Security tools | **PARTIAL** | pip-audit, bandit, gitleaks (history and staged), `npm audit` run on 2026-11-09 and triaged in `SECURITY.md`; dev-only high/critical npm advisories and two production ones (zod, react-router) are **open** (breaking upgrades deferred) |
| Database: fresh migration, repeat, check | **PASS** (SQLite, PostgreSQL in tests) | `migrate_db.py upgrade` twice then `check` = 0. Upgrade from an **old pre-migration dev database is refused by design** (backup and migrate by hand) |
| Backup and restore | **PASS** on a throwaway local PostgreSQL | `backup_restore_drill.py`; Docker-based commands unverified |
| **Docker image build and run** | **BLOCKED** | no Docker on this machine (`docker: command not found`). WSL 2 with Ubuntu is installed, which could host Docker Engine, but installing it was not done without your approval |
| **Production compose runtime** | **BLOCKED** | same; `docker compose config/build/up` never executed |
| **GitHub Actions run** | **BLOCKED** | `gh` is not signed in; `dfad702` is not pushed; the workflow has never run on GitHub (first run happened only if the `4842df1` push triggered it, and its result has not been read) |
| **HTTPS (HSTS, Secure cookie in a real browser/TLS)** | **BLOCKED / partial** | cookie attributes and HSTS header verified in tests over an `https://` test client only; no real TLS endpoint |
| Real-corpus research evaluation | **OPEN** | needs a licensed corpus you choose (see below); only synthetic evaluations exist (`artifacts/evaluation/*`), which do not demonstrate real scientific capability |
| Model Lab (P8) | **EXPERIMENTAL, unchanged** | `small_v1` (5.4 M parameters) is worse than the random baseline on all 7 synthetic probes; no real-text training |
| PyMuPDF licence | **OPEN (owner decision)** | `SECURITY.md`; pdfminer.six feasibility spike (word similarity 0.949; 22/22 headings detectable) but **no replacement made**; no LICENSE file exists |
| Dependabot PR #14 (`actions/setup-python` 5 to 7) | **OPEN** | cannot be read without `gh` auth; not merged |
| Documentation matches implementation | **PARTIAL** | README has genuine screenshots (`docs/screenshots/`, seeded demo data); `SECURITY.md`, `DEPLOYMENT.md`, `PERFORMANCE.md`, `BACKUP_RECOVERY.md` current; a consolidated release-notes file does not exist yet |
| Git | **PARTIAL** | working tree has the screenshots, README, spike and this file staged but uncommitted; `dfad702` unpushed |

## What unblocks each item

1. **Docker / Compose**: run `docker --version`, then the commands in `docs/DOCKER.md` on a Docker-enabled machine (Docker Desktop on Windows, or Docker Engine inside the installed WSL Ubuntu). Tell me which and I will do it here.
2. **GitHub Actions**: run `gh auth login` (or set `GH_TOKEN`), approve the push of `dfad702`, then I read the workflow run and fix real failures.
3. **Real corpus**: choose a source and licence (for example a few open-access arXiv or PubMed Central CC-BY papers per domain). I will not download or label a corpus without that decision.
4. **PyMuPDF**: decide to keep (and comply with AGPL-3.0 or buy a commercial licence) or approve a replacement trial against real papers.
5. **HTTPS**: needs a real TLS endpoint (a reverse proxy with a certificate, or a tunnel).
