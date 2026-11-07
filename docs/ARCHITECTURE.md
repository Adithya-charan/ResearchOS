# Architecture

RESEARCHOS is a from-scratch research platform: a custom Transformer and tokenizer (`model/`), a custom multi-agent engine
(`agent_engine/`), a document library with lexical retrieval (`research/`), and an HTTP API (`api/`). No pretrained weights or
embeddings, no hosted AI inference, no agent frameworks.

```
 client ──HTTP──▶ API process(es)  ───────────────┐            ┌──────── worker process(es)
                  api/app.py, routes/              │            │         scripts/run_worker.py
                  auth, projects, runs, corpus,    ▼            ▼
                  reports, claims, models, SSE   ┌────────────────────────────┐
                                                 │  API database               │  users, projects, runs, job queue (leases),
                                                 │  (SQLAlchemy + Alembic)     │  reports, claims, evidence, verifications, ...
                                                 └────────────────────────────┘
                  EngineRuntime ─────────────────▶ ┌────────────────────────────┐
                  api/runtime.py                   │  Engine database            │  runs, tasks, executions, events,
                  agent_engine/                    │  (Store + Alembic)          │  approvals, memory
                                                   └────────────────────────────┘
                  document libraries (files + SQLite corpus index) under the data directory
```

* **API process**: authentication, validation, authorisation (every project-scoped route resolves the project through its owner), enqueueing.
  It never executes a research run in a request handler.
* **Worker process**: claims a job from the queue (atomic claim with a lease and heartbeat), runs the agent engine for it, hands the job back on
  shutdown and is recovered by lease expiry if it dies. `embedded_worker=true` runs the same worker inside the API process for development.
* **Agent engine** (`agent_engine/`, see `docs/AGENT_SYSTEM.md`): task graph, scheduler, policy, planner, nine+ agents, validated state transitions,
  execution records, approval gate. State lives only in the engine database, so a run survives any process.
* **Evidence layer** (`research/evidence/`, see `docs/EVIDENCE_VERIFICATION.md`) and **retrieval** (`research/`, `docs/RETRIEVAL_SYSTEM.md`).

## Persistence (P7)

Two databases, different schemas, each on **SQLite (default, development)** or **PostgreSQL (production)**. On PostgreSQL they share one
database in two schemas, `api` and `engine`. Schema management is by versioned Alembic migrations for both layers; concurrent starters are
serialised by advisory locks; unknown or newer schemas are refused, never altered. Job claiming uses `FOR UPDATE SKIP LOCKED` plus conditional
updates, leases are renewed by heartbeat and reclaimed on expiry, and the engine store validates every status transition on a locked row.
Details, the migration procedure from an existing SQLite installation, and the measured behaviour: `docs/DATABASE.md`.

## Deployment

* single host, SQLite: `docker-compose.yml` (unchanged);
* PostgreSQL: `docker compose -f docker-compose.yml -f docker-compose.postgres.yml up` — postgres (persistent volume, health check, no published
  port), one-shot `migrate`, `api`, `worker`; credentials only from the environment (`.env`, see `.env.example`).

## Boundaries that remain

Single API node per rate limiter / SSE gate, no hosted model calls, experiments need an isolating sandbox (Docker socket is never mounted),
the model is a small from-scratch demonstration model and the agents are deterministic and lexical.
