# ResearchOS - Implementation Audit

Audit date: 2026-11-06. Method: every statement below was checked against the repository (`git status`, file reads, greps, a full test run).
Nothing in this document is a plan presented as a fact; "exists" means present, imported and covered by a passing test unless stated.

## 1. Baseline

| Item | Result |
|---|---|
| Git | clean working tree, `main` equals `origin/main` (35 commits). Nothing uncommitted was at risk. |
| Test suite (`python -m pytest`, Windows, Python 3.11, CPU/CUDA torch 2.14) | **589 passed, 0 failed, 0 skipped** locally in 267 s. A fresh clone gave 571 passed + 7 skipped (tests that need git-ignored checkpoints/data skip cleanly). |
| Code size (Python lines) | model 2.2k, agent_engine 2.7k, research 2.5k, api 3.6k, scripts 1.3k, tests 7.2k |
| GitHub Actions | workflow exists (tests, hygiene, Docker build/run). **Its first run on GitHub has not been inspected**; treat CI as unverified until its result is read. |
| Docker | **Not installed on the development machine.** The Dockerfile, compose file and CI Docker job have never been executed. |
| PostgreSQL | **Not installed; no driver (`psycopg`) in the environment or dependencies.** Nothing has ever run against PostgreSQL. |
| Frontend | **Does not exist.** No Next.js project, no `package.json`, no Node toolchain used. |

## 2. Existing architecture

```
model/            tokenizer (byte BPE) | Transformer (RoPE/RMSNorm/SwiGLU/SDPA, KV cache) | data pipeline | trainer | inference | eval
agent_engine/     contracts | planner | policies | scheduler (asyncio) | SQLite Store | memory | messaging | agents | reporting
research/         bm25 | corpus (extract/segment/quality/metadata/store/facade) | experiments + sandbox | knowledge_graph | formal
api/              FastAPI app, SQLAlchemy models, services, routes, SSE, Alembic migrations
scripts/          CLIs (train, evaluate, generate, run_research, manage_corpus, run_experiment, export_report, serve_api, ...)
```

Two separate databases exist:

* **Application DB** (SQLAlchemy 2 + Alembic): users, sessions, projects, runs (a thin row pointing at an engine run id), audit log,
  experiments, reports + versions, training jobs. Migration `0001` covers all nine tables; a drift test guards it.
* **Engine DB** (`agent_engine/state/store.py`, sqlite3 with hand-written `CREATE TABLE IF NOT EXISTS`, `schema_info` version): runs, tasks,
  events, approvals, memory. **Not under Alembic.**

Run execution today: `api/runtime.py` starts `asyncio.run(engine.run(...))` in a **daemon thread inside the API process**. There is no
separate worker, no claim/lease, and nothing re-launches a run after the API restarts (a user must call `POST /runs/{id}/recover`).

## 3. Requirement-by-requirement status

Legend: DONE = implemented and tested; PARTIAL = exists but differs from the requirement; MISSING = not present.

### Backend / database

| Requirement | Status | Detail |
|---|---|---|
| Auth, sessions, profile | PARTIAL | register/login/logout/me done (Argon2id, CSRF, rate limits). **No profile update endpoint.** |
| Projects, runs, tasks, events | DONE | owner-scoped, quotas, 404 for others. |
| Documents / corpus | DONE | PDF/DOCX/TXT/MD/HTML; dedup by content hash; chunking; quality flags. Stored in per-project SQLite, not PostgreSQL. |
| Evidence and citation records | PARTIAL | citations (doc/page/passage + embedded source text) live inside the **report JSON snapshot**; no queryable `evidence`/`claims` tables. |
| Approvals, experiments, reports (+versions), model registry, training jobs | DONE | |
| Real-time streaming | DONE (tests) | SSE for runs, training, experiments, token generation. Wall-clock incremental delivery against a live server **not verified** (test client buffers). |
| Migrations | PARTIAL | app DB: baseline + drift + legacy adoption tests (SQLite only). **Engine DB not integrated. PostgreSQL untested.** |
| Foreign keys / indexes / constraints | PARTIAL | present on app tables; engine DB has no ownership concept (ownership enforced in the app DB). |

### Durable execution engine

| Requirement | Status | Detail |
|---|---|---|
| Task graph, dependency-aware scheduling | DONE | planner validates graphs; scheduler honours `depends_on`. |
| Retries, task timeouts, limits, pause/resume/cancel, approval gate | DONE | retries are counted with exponential backoff (`retry_backoff_s * 2**(attempt-1)`, `RetryPolicy`), no jitter. |
| Persisted task inputs/outputs, events | DONE | |
| Crash recovery | PARTIAL | tasks left `running` are re-queued by `engine.recover`, but only on explicit request. |
| Run lifecycle `queued -> planning -> running -> reviewing -> completed` | **MISSING** | engine statuses are `pending, running, paused, completed, failed, cancelled, stopped_limit`. No `queued/planning/reviewing`. |
| Worker that claims queued jobs; no double execution across workers | **DONE (P1)** | `research_jobs` table, atomic claim, lease + heartbeat, reaper with backoff, graceful hand-back; embedded or `scripts/run_worker.py`. Tested incl. 8-thread claim race. Not yet run as a live second process; PostgreSQL untested. |
| Execution logs, token usage, time and resource tracking | PARTIAL | events + elapsed time + model tokens counter exist; no resource (CPU/RAM) accounting, no separate log store. |

### Model

| Requirement | Status | Detail |
|---|---|---|
| Custom BPE tokenizer, versioning, special tokens | DONE | |
| Decoder-only Transformer, configurable sizes | DONE | |
| Training (accumulation, clipping, bf16, checkpoints, exact resume, metrics, val loss/perplexity) | DONE | |
| Inference (temperature, top-k/top-p, repetition penalty, KV cache, streaming, context management) | DONE | |
| Untrained / partially trained / trained distinction | **PARTIAL** | the registry marks models by checkpoint and notice; there is **no explicit readiness state** (`untrained/partial/trained`) and no API-level "not ready for reliable research" gate. |
| Model drives agent text | **MISSING (by design so far)** | agents are deterministic and extractive; the Transformer is not used to generate findings. This is honest for a 5M-parameter synthetic-trained model, but the requirement says agents should be model-powered "where applicable". |

### Agents

| Required agent | Present as | Status |
|---|---|---|
| Supervisor | `SupervisorAgent` | DONE (task graph, dependency-aware) |
| Literature Discovery | `DiscoveryAgent` | DONE (BM25) |
| Literature Analysis | `LiteratureAgent` | DONE (extractive) |
| Research Gap | `GapAgent` | DONE (heuristic, labelled candidate) |
| Hypothesis | `HypothesisAgent` | DONE (template-based, labelled untested) |
| Experiment (plans) | `ExperimentAgent` | **PARTIAL**: runs an *approved* spec; does **not** create experiment plans, and is not wired into API runs |
| Critical Reviewer | `CriticAgent` | PARTIAL: lexical quality gate on evidence, not a reviewer of conclusions |
| Verification | `VerifierAgent` | PARTIAL: statuses are `verified/unsupported/needs_review/hypothesis/candidate_gap` (lexical) |
| Synthesis | `SynthesisAgent` | DONE (report dict) |

All are deterministic; they are labelled as such. Typed contracts, tool permissions, memory and logging exist.

### Retrieval, evidence, verification, report

| Requirement | Status | Detail |
|---|---|---|
| BM25, source attribution, citation metadata | DONE | title/authors/year/DOI/license with provenance of each field. URL field does not exist. |
| PostgreSQL full-text search | MISSING | |
| Retrieval evaluation | **MISSING** | no precision/recall/MRR harness. |
| Claim statuses `supported / partially supported / contradicted / insufficient evidence / unverified` | **MISSING (different taxonomy)**; contradicting-evidence detection does not exist (documented limitation). |
| SymPy verification | **MISSING** (SymPy is importable but unused by the verifier) |
| Z3 | PARTIAL: optional adapter in `research/formal.py`; **z3-solver not installed**, so never exercised. |
| 16-section report | **PARTIAL**: the report has question, scope, closest documents, findings, needs-review, unsupported claims, gaps, hypotheses, uncertainties, limitations - not the requested 16 sections (no abstract, background, methodology, proposed experiments, critical review, conclusion, future work, references). |
| Report storage / editing / versioning / export | DONE | immutable snapshot + editable layer + versions + MD/PDF/JSON. |

### Frontend, Docker, tests

| Requirement | Status |
|---|---|
| 17 frontend screens | **MISSING entirely** |
| Docker: API | PARTIAL (written, unexecuted) |
| Docker: PostgreSQL, Redis, worker, frontend services | **MISSING** |
| `docker compose config/build/up/ps/logs` verified | **NOT POSSIBLE on this machine** (no Docker) - CI is the only route |
| PostgreSQL integration tests | **MISSING** |
| End-to-end acceptance test (register -> run -> report -> restart -> data persists) | **PARTIAL (P1 added `test_end_to_end_workflow_survives_an_application_restart`)**: register -> project -> run on a worker -> tasks -> SSE replay -> report -> restart -> data intact is tested; frontend and PostgreSQL legs are not. |
| Docker startup test | CI job only, unobserved |

## 4. Technical debt and risks

1. **Execution lives in the API process** (daemon thread). A restart kills it silently; nothing resumes it. Highest-impact gap.
2. ~~Two schema owners~~ (resolved in P2: the engine DB now has its own Alembic tree).
3. **No PostgreSQL ever exercised**: SQLite-isms (JSON columns, `PRAGMA`, batch ALTER) may fail. Cannot be fixed without a PostgreSQL server (installable locally or via Docker CI service).
4. **Status vocabularies differ** from the requested lifecycle and claim statuses; changing them is an API/contract change touching reports, tests and docs.
5. **Evidence/claims are not rows**, only JSON inside report snapshots, so "evidence inspection" across runs needs new tables.
6. **Unverified deliverables on record**: CI result, Docker image, live SSE delivery.
7. `README` previously contained mangled commands (fixed); `docs/PHASE_REPORTS.md` is long and append-only.
7b. Run start/recovery are user-triggered API calls rather than automatic.
8. No LICENSE chosen; PyMuPDF is AGPL.
9. Local machine: C: drive nearly full, no Docker and no PostgreSQL. **Node 24.7 and npm 11.5 are installed**, so a Next.js frontend can be built and tested locally; Docker and PostgreSQL need the user to install them (or CI to provide them).

## 5. Constraints that shape the plan (stated honestly)

* **Cannot be verified here:** Docker (`compose build/up`) and PostgreSQL, until those tools are installed on this machine. Where verification is only possible in GitHub CI, the result is *unverified* until CI output is read; reports must say so.
* **The prompt forbids commits/pushes without authorisation.** Nothing in this audit was committed. Previous explicit authorisations (the user asked for the earlier push) do not carry over.
* A 5M-parameter model trained on synthetic template text cannot produce reliable research findings. Making agents "model-powered" must keep outputs labelled untrusted, as today.

## 6. Implementation priorities and dependencies

Ordered by dependency (each unlocks the next) and by risk reduction:

| # | Work item | Depends on | Why this order |
|---|---|---|---|
| P1 (**DONE**, uncommitted) | **Durable worker + job queue**: add `queued/planning/reviewing` run states, a `jobs` table with claim/lease + heartbeat, a `worker` entry point, auto-resume of interrupted jobs at startup, exponential backoff with jitter, API enqueues instead of spawning threads | none | The central acceptance criterion ("runs must actually execute", "restart recovery") and the biggest current gap |
| P2 (**DONE**, uncommitted) | **Put the engine DB under Alembic** (single migration strategy), plus migration tests | P1 (schema changes) | Avoid migrating twice |
| P3 | **Evidence / claims / verification tables** and the new claim-status taxonomy (+contradiction heuristics, SymPy numeric/algebraic checks, Z3 only where applicable, explicit "formally verified" gating) | P2 | Needed by the report and the evidence UI |
| P4 | **16-section report builder** from persisted records; Experiment-plan agent (plans only, never fabricated results); Critical Reviewer upgrade | P3 | |
| P5 | **Model readiness state** (untrained / partial / trained from checkpoint + metrics thresholds) surfaced in API and in run limitations | none (parallel) | |
| P6 | **Retrieval evaluation** harness (precision/recall/MRR on a labelled synthetic set) | none (parallel) | |
| P7 | **PostgreSQL**: driver, compose service, run migrations + API tests against it (CI service container, and locally if the user installs PostgreSQL), full-text search where it helps | P2 | Verification needs a PostgreSQL instance |
| P8 | **End-to-end acceptance test** covering restart persistence | P1-P4 | |
| P9 | **Docker**: worker, PostgreSQL, (Redis only if a concrete need appears: the job queue can use PostgreSQL row locking, so Redis is **not justified** by the current architecture), frontend; verify with real `compose` commands | P1, P7, Docker installed | |
| P10 | **Frontend** (17 screens against the real API, SSE) | stable API (P1-P4), Node installed | Largest item; should start only once the API contract settles |

**Renumbering (second master prompt, P3-P10).** The newer prompt orders work as: P3 durable execution (= my P1+P2 plus transitions,
execution records, extra events: **done**), P4 orchestration/agents (reviews the nine agents; extra agents Literature Analysis and
Experiment are the main gaps), P5 knowledge system / retrieval evaluation (= my P6 plus ingestion review), P6 evidence verification and
reports (= my P3+P4), P7 PostgreSQL (= my P7), P8 model training/inference/registry (= my P5 readiness plus provider interface),
P9 frontend (= my P10), P10 security, Docker, CI, performance, final docs (= my P9 plus the rest). The table above keeps the old numbers.

Redis decision (rule 4 of the prompt: "if justified"): not justified now. PostgreSQL `SELECT ... FOR UPDATE SKIP LOCKED` (and a
leased row on SQLite) covers job claiming; rate limiting remains in-process and is documented as single-process.

## 7. Proposed first increment

P1 (durable worker and job queue) with its tests, because every other acceptance step (live updates, restart persistence, report after
restart) depends on runs executing outside the request process and surviving restarts. No commits will be made unless you authorise them.

## Status log (second master prompt)

* P3 durable execution: done (624 tests at the time).
* P4 multi-agent orchestration: done. Nine required roles are present (Experiment Design and Critical Reviewer added), declared specs
  checked against real outputs, supervisor replanning (discovery failures only), duplicate prevention, per-agent timeout/retry ceilings.
  Full suite: 637 passed plus one test whose agent-type whitelist was updated for the two new agents (then re-run green).
  Open: reviewer/plans not yet in the rendered report (P6); agents deterministic, no model-written analysis.
* P5 knowledge system: done except PostgreSQL full-text search (P7) and a bundled malware scanner (hook only). Added declared publication
  date/URL/document type, phrase / document-level / metadata search, citation records, corpus schema v2 upgrade and the retrieval
  evaluation harness (22 queries: lexical MRR 1.0, synonym MRR 0.0 - the limitation is measured and documented).
  Closure pass: 18 further tests (deletion/index cleanup, restart persistence, combined filters, agent evidence traceability,
  authorization at every search level, evaluation reproducibility and synonym-stem disjointness). See `docs/RETRIEVAL_SYSTEM.md`.
  Lexical only: no semantic synonym matching; no coverage claim beyond the uploaded library.
* P6 evidence verification and research intelligence: implemented, **uncommitted** at the time of writing. Persisted claims, claim
  evidence, verification history and novelty assessments (API migration 0003); deterministic lexical verification engine with five
  statuses and explicit contradiction handling; bounded novelty assessment; 18-section structured report; Verifier/Supervisor/Reviewer
  integration with bounded evidence follow-up rounds; authenticated project-scoped claim, verification, novelty and structured-report
  endpoints. Evaluation is on a synthetic dataset only (18/18 on non-limitation cases, 0/3 on documented known limitations); no
  real-literature validation yet. Not semantic, not a proof. See `docs/EVIDENCE_VERIFICATION.md`.
