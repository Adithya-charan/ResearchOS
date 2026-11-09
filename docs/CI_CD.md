# CI/CD

Workflow: `.github/workflows/ci.yml`, run on every pull request and on pushes to `main`. **Nothing deploys**: there is no deploy job, no registry push, no
SSH and no use of repository secrets (`tests/test_p10_ci.py` asserts that). Permissions are `contents: read`.

> **Status: not yet run on GitHub.** The workflow was written and checked statically (parsed as YAML, required jobs and safeguards asserted) and its
> commands were exercised locally where possible, but GitHub Actions had not executed it when this document was written. The first run is the verification;
> expect to fix small environment differences (Ubuntu paths, Playwright browser, Docker versions).

| Job | What it does | Blocking |
|---|---|---|
| `backend` | Python 3.11, CPU PyTorch, pinned install, `ruff` correctness rules (`E9,F63,F7,F82`), the whole pytest suite on SQLite | yes |
| `frontend` | `npm ci`, `tsc --noEmit`, `eslint --max-warnings 0`, `vitest`, production build, bundle size report, uploads `dist/` | yes |
| `e2e` | builds the SPA, installs Chromium, runs Playwright `full-flow`, `a11y` and `keyboard` against a **real** backend (`frontend/e2e/serve.py`) | yes |
| `postgres` | PostgreSQL 16 service: `tests/test_p7_postgres.py`; the existing API/engine/agent suites re-run on PostgreSQL (`RESEARCHOS_TEST_ON_POSTGRES=1`); migrations applied to an empty PostgreSQL and SQLite database, applied twice (idempotent) and `check`ed | yes |
| `security` | `pip-audit` on the pinned packages, `bandit` (high severity), `npm audit --omit=dev --audit-level=critical`, full `npm audit` (report only), `gitleaks` over the full history with `.gitleaks.toml` | yes, except the report-only `npm audit` |
| `hygiene` | no model weights/databases, nothing over 5 MB, no `.env` or private keys tracked | yes |
| `docker` | validates the three compose setups, builds the production image, checks it runs as UID 10001 with the web app and no Node, starts the production stack, waits for API and worker health, checks headers, confirms PostgreSQL is not reachable from the host, registers a user, creates a project, restarts API and worker and verifies the data survived | yes |

## Secrets and credentials

* The PostgreSQL service in `postgres` uses `POSTGRES_HOST_AUTH_METHOD=trust` for a throwaway container that lives only for the job.
* The `docker` job generates a random database password at run time (`openssl rand`) and exports it to later steps of the same job; it is never printed.
  If a real secret is ever needed, store it as a GitHub Actions secret and reference it as `${{ secrets.NAME }}` (GitHub masks it in logs); do not echo it.
* `gitleaks` runs over the whole history. Documented false positives are allow-listed narrowly in `.gitleaks.toml` (a tokenizer hash, a throwaway e2e password).

## Running the same checks locally

```bash
python -m pytest                                              # backend
cd frontend && npx tsc --noEmit && npx eslint . && npx vitest run && npm run build
cd frontend && npx playwright test full-flow a11y keyboard   # needs the built SPA; uses Edge locally (PW_CHANNEL / PW_PYTHON override)
pip-audit -r requirements/constraints.txt --no-deps --disable-pip
gitleaks detect --config .gitleaks.toml
```

## Known gaps

* No automated deployment (by design) and no release/tagging workflow.
* `ruff` runs with correctness rules only; there is no formatter or full lint configuration.
* The Docker job needs GitHub's hosted runner; it cannot run on the development machine.
