# Frontend API integration

The web app (`frontend/`, served at `/app`) talks only to the existing REST API, same origin. It adds no endpoints of its own except the two
read-only aggregates described below. Nothing is simulated: where the API has no endpoint for something, the page says so.

## Client (`frontend/src/lib/api.ts`)

* Session is the HttpOnly cookie set by `/auth/login` / `/auth/register`. The CSRF token is returned in the response body and held **in memory only**
  (never `localStorage`/`sessionStorage`) and sent as `X-CSRF-Token` on `POST/PUT/PATCH/DELETE` only.
* A failed GET is retried once; a mutation is never retried. A 401 notifies a listener that signs the user out.
* Error bodies (`{detail}`) are surfaced as readable messages. Uploads go through `api.upload` (multipart).
* Server state is cached by TanStack Query. Live run updates use `lib/sse.ts` (fetch-based SSE parser, `Last-Event-ID` resume, de-duplication,
  stops on `end` or a 4xx) via `lib/useRunStream.ts`.
* Types for responses are in `lib/types.ts` and mirror the actual payloads.

## Endpoints added for the UI (P9)

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /overview` | dashboard aggregates | scoped to the caller; counts are computed from stored records; run statuses/task counts cover the latest 200 runs and say so in `note` |
| `GET /audit` | the caller's own audit entries | filter `action`, `limit` 1-200, `offset`; never another user's entries; no credentials in details |
| `GET /app`, `/app/{path}` | serves `frontend/dist` | strict CSP (`default-src 'self'`, no inline script/style), path traversal refused, unknown extension-less paths fall back to the shell, missing assets 404, clear message when not built |

Tests: `tests/test_p9_web.py`.

## Route to API matrix

| UI route | Endpoints used |
|---|---|
| `/login` | `POST /auth/login`, `POST /auth/register`, `GET /auth/me`, `POST /auth/logout` |
| shell (every page) | `GET /projects`, `GET /approvals?status=pending` (bell badge, 60 s poll), `POST /auth/logout` |
| `/` Overview | `GET /overview`, `GET /audit?limit=8`, `GET /projects/{id}/reports` |
| `/projects`, `/projects/:id` | `GET/POST /projects`, `GET/PATCH/DELETE /projects/{id}`, `GET /projects/{id}/corpus`, `GET /projects/{id}/reports` |
| `/workspace`, `/workspace/:runId` | `GET/POST /projects/{id}/runs`, `GET /runs/{id}`, `GET /runs/{id}/tasks`, `GET /runs/{id}/stream` (SSE), `POST /runs/{id}/{pause\|resume\|cancel\|recover}`, `POST /runs/{id}/reports`, `GET /projects/{id}/claims?run_id=`, `GET /projects/{id}/reports`, `GET /reports/{id}/structured`, `GET /approvals`, `POST /approvals/{kind}/{id}/{approve\|reject}` |
| `/agents`, `/agents/:runId` | `GET /agents`, `GET /runs/{id}`, `GET /runs/{id}/tasks`, `GET /runs/{id}/tasks/{taskId}`, SSE as above |
| `/library` | `GET/POST /projects/{id}/documents` (upload), `DELETE /projects/{id}/documents/{doc}`, `POST .../reprocess`, `GET .../documents/{doc}/passages`, `GET /projects/{id}/corpus`, `GET /projects/{id}/search` (levels passage / document / metadata; phrase, filters) |
| `/gaps`, `/hypotheses` | derived read-only from the `gap` / `hypothesis` agent task outputs (`GET /runs/{id}/tasks`, `/tasks/{taskId}`) |
| `/experiments` | `GET /projects/{id}/experiments`, `GET /experiments/{id}`, `GET /experiments/{id}/logs`, `POST /experiments/{id}/request-approval`, `POST /experiments/{id}/execute`; proposed plans from the `experiment_design` task output |
| `/evidence` | `GET/POST /projects/{id}/claims`, `GET .../claims/{id}`, `GET .../claims/{id}/verifications`, `POST .../verify`, `POST/DELETE .../evidence`, `POST /runs/{id}/claims/extract`, `GET /projects/{id}/documents` |
| `/novelty` | `GET/POST /projects/{id}/novelty` |
| `/reports`, `/reports/:id` | `GET /projects/{id}/reports`, `GET /reports/{id}/structured`, `GET /reports/{id}/versions`, `GET /reports/{id}/export.md`, `GET /reports/{id}/export.pdf` (JSON is saved client-side from `/structured`) |
| `/model-lab` | `GET /modellab/overview`, `/modellab/provider`, `/modellab/hardware-benchmarks`, `/modellab/models/{id}`, `/modellab/models/{id}/evaluation` (read-only) |
| `/activity` | `GET /audit`, `GET /approvals`, `GET /runs/{id}/events`, `GET /projects/{id}/verifications` |
| `/settings` | `GET /agents`, `GET /models/health`, `GET /auth/me` |

## Known gaps (reported, not hidden)

* No password-change endpoint exists, so Settings does not offer one.
* Gaps, hypotheses and experiment plans are not stored entities; the pages derive them read-only from agent task outputs.
* No UI for creating experiments (code submission stays on the API/CLI).
* Model Lab is read-only; training is not started from the web app.
