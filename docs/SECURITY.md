# Security review and hardening (P10)

Scope: the API (`api/`), the agent engine and corpus code it runs, the web app (`frontend/`), and the container/CI configuration. This is an
engineering review done by reading the code and running tools; it is **not** a penetration test or a third-party audit. Reporting policy:
see the repository-level `SECURITY.md`.

## What was checked, and how

| Area | Method | Result |
|---|---|---|
| Authentication | read `api/security.py`, `api/routes/auth.py` | Argon2id hashes; a dummy hash is verified for unknown accounts (no timing oracle); password policy (length >= 12, common-password and email-name checks); session token is 256-bit random, only its SHA-256 is stored; cookie `HttpOnly`, `SameSite=Lax`, `Secure` when `RESEARCHOS_COOKIE_SECURE=true`; 12 h TTL; logout deletes the row |
| CSRF | read `api/deps.py`, `api/app.py`; tests in `tests/test_api.py` | every unsafe method needs `X-CSRF-Token` equal (constant time) to the session's token, and a cross-site `Origin` is refused before that |
| Authorization | AST scan of all 108 route handlers + behavioural test | every handler requires a session except `/health`, `/auth/login`, `/auth/register`; the 11 handlers that take an id but showed no ownership call delegate to helpers (`_control`, `_resolve`, `_stop`) that check ownership. **Regression guard:** `tests/test_p10_security.py::test_every_api_route_requires_authentication...` calls every documented operation without a session and fails if any answers anything but 401 |
| SSE authorization | read `api/routes/streams.py`, `api/sse.py` | a stream authenticates like any request, is scoped to the caller's own run/job, re-checks the session every `sse_reauth_s`, and is capped per user and in total |
| SQL injection | `bandit`, grep for string-built SQL | all queries are parameterised. 17 bandit "medium" B608 notes are f-strings that interpolate only fixed fragments or operator-controlled table/schema names (schema names are validated by regex in `dburl.with_schema`); none include request data |
| File upload / parsing | read `api/corpus_service.py`, `research/corpus/*` | type is detected from content, not the name; size, page and document-count limits; parsing happens in a subprocess with a timeout; stored under a server-generated name; document text is treated as untrusted data and is never executed |
| **Path traversal** | read artifact download, SPA route; tests | artifacts are served only if listed in the run manifest, not symlinks, inside the output directory and hash-checked; `/app` resolves paths inside the build directory only (tested with `..`, encoded and nested forms) |
| Command execution | grep | `subprocess` is used with argument lists (never `shell=True`) for extraction, training children and the experiment sandbox; no `eval`/`exec`/`pickle`; `torch.load(..., weights_only=True)` |
| SSRF | grep for HTTP clients | the server makes **no outbound HTTP requests** (no `requests`, `urllib.request`, `httpx` calls in application code), so there is no server-side fetch of user-supplied URLs. A document's `url` field is stored as text only |
| Errors and logging | read `api/app.py` | unhandled errors return a generic message with a request id; details go to the server log. Audit entries never store passwords, tokens or document text. A secret-scan of the full git history (gitleaks) found nothing real (see below) |
| Secrets | gitleaks, `.env` handling | `.env` is git-ignored and docker-ignored; compose files require secrets from the environment (`${VAR:?}`); the image embeds none; `.env.example` holds no values |
| Frontend XSS | grep, tests | no `dangerouslySetInnerHTML`, `innerHTML` or `eval`; report/document Markdown goes through `react-markdown` with `rehype-sanitize` (test in `src/components/ui.test.tsx`); CSP forbids inline script and `eval` |
| Frontend auth state | read `src/lib/api.ts`; e2e test | the CSRF token lives in memory only; `localStorage` holds only theme, nav state, reading size and the selected project id (the e2e test asserts no token/password in storage) |

## Findings and fixes

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | High | Only login, registration and model inference were rate limited; uploads, run creation, search, verification, report export, experiments and evaluation had no limits | **Fixed**: per-user server-side limits (below) |
| 2 | Medium | DOCX parsing refused XML DTDs by scanning only the first 4 KiB, so a long comment before `<!DOCTYPE>` bypassed the check (entity-expansion risk; the parser already runs in a size-capped, time-limited subprocess) | **Fixed**: the whole member is scanned; regression test `test_docx_dtd_hidden_behind_a_long_comment_prolog_is_still_refused` fails on the old code and passes now |
| 3 | Medium | No `Permissions-Policy`, HSTS, COOP/CORP, `Referrer-Policy` or CSP on ordinary API responses (only on the SPA and `/lab`) | **Fixed**: baseline headers on every response (below) |
| 4 | Medium | `/docs` and `/openapi.json` were always public | **Fixed**: `RESEARCHOS_ENABLE_DOCS` (default on for development; the production compose file sets it off) |
| 5 | Medium | Wildcard `RESEARCHOS_ALLOWED_ORIGINS` would have been accepted | **Fixed**: refused at startup. Risky combinations (https origin with insecure cookie, weak sandbox, short passwords) are logged as warnings |
| 6 | Low | Graph nodes were keyboard-focusable without a visible focus ring; graph edges were 100+ invisible tab stops | **Fixed** (see accessibility in `FRONTEND_TESTING.md`) |
| 7 | Low | Frontend dependencies with advisories | **Triaged** below |
| 8 | Info | The experiment sandbox's non-Docker fallback is **not isolating** | Unchanged: it is refused unless `RESEARCHOS_ALLOW_WEAK_SANDBOX=true`, which the production compose file forces to `false`. Real isolation needs the Docker sandbox, which cannot run from inside the API container (the Docker socket is deliberately not mounted), so experiment execution is **unavailable in the compose deployment** |
| 9 | Info | The in-process rate limiter counts per API process | Documented: behind several API replicas the ceiling multiplies; add a shared limit at the reverse proxy for a hard cap |
| 10 | Info | The repository has **no LICENSE file** | Open: see PyMuPDF below; choosing a licence is the owner's decision |
| 11 | Medium | Login throttling was keyed per (IP, email) only, so one address could try 5 passwords against *every* account (credential stuffing) | Fixed: an added per-IP failure budget across all emails (`RESEARCHOS_LOGIN_IP_ATTEMPTS`, default 30 per window); tested in `tests/test_p10_security.py`. Trade-off: while an address is throttled its legitimate logins are refused too |
| 12 | Low | `RateLimiter` created a key on every check and never removed stale ones, so a flood of random emails grew memory without bound | Fixed: checks no longer create keys; stale keys are swept every 60 s; tested |
| 13 | Low | Logout cleared the session cookie without the `Secure`/`HttpOnly` attributes; browsers can ignore a non-Secure cookie that tries to replace a Secure one (server-side session was always revoked) | Fixed: the clearing cookie carries the same attributes; tested over https |
| 14 | Low | Job reaper counted a requeue it did not perform when two reapers raced (state stayed consistent; the count was wrong; found as a flaky test) | Fixed in `api/jobs.py` with a deterministic regression test |
| - | - | Cross-user access and malformed input were only covered per feature | Added `tests/test_p10_isolation.py`: a second signed-in user calls every id-taking operation (about 95) with the owner's ids and gets no 2xx/5xx and no leak; hostile ids/bodies on every write operation never produce a 5xx. Mutation-checked: removing the project ownership check makes it fail |

## Rate limits (server side, per signed-in user)

Implemented in `api/ratelimit.py` as a dependency that runs after authentication (an unauthenticated request never spends budget), over the
existing sliding-window limiter. A limited request gets `429` with `Retry-After`. Override with `RESEARCHOS_RATE_<NAME>="count/window_seconds"` or `off`.

| Name | Applies to | Default |
|---|---|---|
| `upload` | document upload | 120 / 10 min |
| `reprocess` | re-extract a document | 30 / 10 min |
| `run_create` | create a research run (also capped by `max_active_runs_per_user`, `max_runs_per_project`) | 20 / hour |
| `search` | library search (all three levels) | 120 / min |
| `verify` | claim verification, claim extraction, novelty assessment | 60 / 10 min |
| `report` | report generation and MD/PDF/JSON export | 30 / 10 min |
| `experiment` | execute / reproduce an experiment | 10 / hour |
| `evaluate` | model evaluation job | 10 / hour |

Already in place and unchanged: login failures 5 / 15 min per (IP, email); registrations 10 / hour per IP; model generation 20 / min per user;
SSE streams 4 per user and 16 in total. The defaults are sized for one researcher working interactively (a 100-paper library import fits inside the
upload budget); `tests/test_p10_security.py` checks that limits are per user (another user is unaffected), unauthenticated calls do not consume
budget, and `off` disables a limit. Limits are server-side only; nothing relies on the browser.

## Security headers

Added to **every** response by the middleware in `api/app.py` (`baseline_headers`); the SPA and `/lab` keep their own, stricter-for-their-assets CSP
(`setdefault` never overrides them).

| Header | Value | Notes |
|---|---|---|
| `Content-Security-Policy` | API responses: `default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'`. SPA: `default-src 'self'; img-src 'self' data:; style-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'` | no `unsafe-inline` / `unsafe-eval` anywhere; fonts and scripts are self-hosted. `/docs` (Swagger UI loads from a CDN) is the one path without it, and can be switched off |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | only when `RESEARCHOS_HSTS=true` (implied by `RESEARCHOS_COOKIE_SECURE`); never send it over plain HTTP |
| `X-Content-Type-Options` | `nosniff` | |
| `Referrer-Policy` | `no-referrer` | |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=(), payment=(), usb=()` | |
| `X-Frame-Options` | `DENY` | belt and braces next to `frame-ancestors` |
| `Cross-Origin-Opener-Policy` / `Cross-Origin-Resource-Policy` | `same-origin` | |
| `Cache-Control` | `no-store` for API and pages; `public, max-age=31536000, immutable` for fingerprinted `/app/assets/*` | |

Compatibility: the SPA runs under this CSP (end-to-end and accessibility tests pass against the built app); SSE responses carry the same headers and
are unaffected (the stream tests pass); cookies are unchanged. Header presence is tested in `tests/test_p10_security.py` and `tests/test_p9_web.py`.
**Not verified:** HSTS and the `Secure` cookie flag over real HTTPS (needs a TLS proxy; see `DEPLOYMENT.md`).

## Dependency audit (2026-11-09)

### Python (`pip-audit`, 78 installed packages; the 36 pinned runtime packages in `requirements/constraints.txt` separately)

* Pinned runtime packages: **no known vulnerabilities**.
* Environment scan: 1 package, `setuptools 78.1.0` (a build tool, not a runtime dependency of the application or of the Docker image's virtualenv): PYSEC-2025-49 (path
  traversal in `PackageIndex`, fixed 78.1.1), PYSEC-2026-3447 (fixed 83.0.0), GHSA-xmj3-9gf7-h52w (Windows-only wheel handling, fixed 84.1.0). Not reachable by
  application input; update the build tooling on developer machines.
* `bandit` (high severity): none. Medium notes reviewed above.

### JavaScript (`npm audit`; 19 advisories before, after `npm audit fix` (lockfile only, no `package.json` change): see table)

| Package | Severity | Production? | Issue | Decision |
|---|---|---|---|---|
| `react-router` / `react-router-dom` 6.30.6 | moderate (was high) | yes | open redirect via backslash in `<Link>`/`useNavigate`; `deserializeErrors()` constructor injection in SSR hydration | `npm audit fix` took it to 6.30.6; the remaining advisory is fixed only in v7 (breaking). **Not exploitable here**: the app never navigates to a user-controlled URL (deep links use query parameters, not targets) and has no SSR. Upgrade to v7 is scheduled work, not done |
| `zod` 3.x | high | yes | memory exhaustion validating oversized constrained collections | fix is zod 4 (major, and needs `@hookform/resolvers` 5). Zod only validates short form fields and our own server's session response, never untrusted bulk data, so the DoS path is not reachable. **Not upgraded** (compatibility risk); tracked |
| `vite` 5.x, `esbuild`, `vite-node`, `@vitejs/plugin-react` | high / moderate | **no** (dev server and build) | dev-server request handling issues | affects only `npm run dev` on a developer machine; the production image serves static files. Upgrade needs Vite 6+ (and vitest). Not done |
| `vitest`, `tinypool` | critical | **no** (test runner) | prototype pollution gadget in worker options | runs only in tests/CI on trusted code. Upgrade is a major version; not done |
| `typescript-eslint` family, `minimatch` | moderate | no | ReDoS-class advisories in lint tooling | dev only; major upgrade; not done |
| `@playwright/test`, `playwright` | high | no | browser download without certificate verification | the project uses the installed Edge / `playwright install chromium` in CI on GitHub's network only |

CI blocks on **critical** advisories in **production** dependencies (`npm audit --omit=dev --audit-level=critical`) — none today — and reports the rest.
Raising that bar to `high` needs the zod 4 / react-router 7 upgrades first.

### Deprecated / incompatible

* JavaScript (from `package-lock.json`): `eslint` 9.x (version no longer supported; dev only), `recharts` 2.x ("1.x and 2.x branches are no longer active", moves to
  v3; runtime dependency, no advisory open against it, upgrade is a breaking change), `whatwg-encoding` (transitive, test environment).
* Python: `pip-audit` does not report deprecation, and no separate deprecation review was done. `StarletteDeprecationWarning` (use `httpx2` with the test client)
  appears in test output; it affects tests only.

### Licences

Declared licences of the distributions installed in the development virtualenv (a superset of the runtime set; counted from package metadata): MIT 33, BSD-family about 20, Apache-2.0 14, PSF 2, MPL-2.0 1, LGPL-3.0 2, and a few that only say "see metadata". Flagged for attention:

| Package | Licence | Note |
|---|---|---|
| **pymupdf 1.28.2** | **AGPL-3.0 or Artifex Commercial** | see below |
| psycopg / psycopg-binary 3.3.8 | LGPL-3.0 | used as an unmodified, dynamically installed library; keep it replaceable (do not vendor a patched copy into a proprietary bundle). Confirm with counsel for your distribution model |
| pgserver 0.1.4 | declared in metadata only | test-only dependency (`postgres-test` extra); not in the image |
| torch 2.14.0 | BSD-style | declared in package metadata/NOTICE rather than a short string |

## PyMuPDF licensing (for legal review; not legal advice)

**Facts established from the installed package** (`pymupdf` 1.28.2; `pyproject.toml` pins `>=1.28,<2`):
* Metadata states: *"Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License"*; author Artifex. Its README says the open-source option is
  AGPL v3, "free for open-source projects", and that separate commercial licences from Artifex exist "for proprietary applications".
* It is used only in `research/corpus/extract_pdf.py` (and in tests) for text extraction; it is imported inside the isolated extraction subprocess.
* The repository currently declares **no licence of its own**, and the application is offered to users over a network (a web service).

**What the licence terms mean in outline** (from the AGPL text, to be confirmed by counsel): the AGPL is a strong copyleft licence whose
section 13 adds an obligation for modified versions run as a network service to offer users the corresponding source. Whether linking PyMuPDF
into this application makes the whole application a "work based on" it, and whether running it as a hosted service triggers that obligation, is a
legal question that depends on how the software is distributed and used, and that this review does not decide.

**Consequences to decide on**
1. If the project is and stays open source under an AGPL-compatible licence, using PyMuPDF under the AGPL is straightforward; add a LICENSE file and a notice.
2. If the project (or a hosted service built from it) is to be proprietary or commercial, either obtain an Artifex commercial licence **before** distributing
   or offering it, or replace PyMuPDF.
3. Alternatives with permissive licences exist (for example `pypdf` BSD-3, `pdfminer.six` MIT, `pypdfium2` Apache-2.0/BSD-3). **They have not been evaluated**
   against this project's PDF fixtures, page/heading/header-footer detection or citation-offset requirements, so switching is real engineering work.
   Isolation already exists (the extractor is one function behind `extract_pdf`), which keeps a swap contained.

Open item for the owner: legal review and a decision; until then the status is **unresolved**.
