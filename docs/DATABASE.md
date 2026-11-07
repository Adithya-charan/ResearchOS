# Database (SQLite and PostgreSQL)

RESEARCHOS has **two databases with different schemas** and each can run on SQLite (default, local development, lightweight tests) or
on PostgreSQL (production). They are not the same schema and must not be treated as one: both define tables called `runs` and
`projects`, but they mean different things.

| | API database | Agent-engine database |
|---|---|---|
| Code | SQLAlchemy models `api/models.py`, Alembic tree `api/migrations/` | raw SQL behind `agent_engine/state/store.py`, Alembic tree `agent_engine/state/migrations/` |
| Holds | users, sessions, projects, runs (API view), job queue (`research_jobs`), reports and versions, claims, claim evidence, verification history, novelty assessments, experiments, training jobs, audit log | engine projects/runs, tasks, task executions, events, approvals, memory |
| Timestamps / JSON | native `TIMESTAMP` / `JSON` | ISO-8601 `TEXT` / JSON `TEXT` (identical on both backends, so rows copy 1:1) |
| Revisions at P7 | `0001`–`0003` | `0001`–`0002` |

## Separate schemas in one PostgreSQL database

Decision: on PostgreSQL **one server, one database, two schemas** — `api` and `engine` — each with its own `alembic_version`
table. Schemas keep the identically named tables apart, give one backup/restore unit and one connection endpoint, and need no
cross-database queries (there are none). Overrides:

| Variable | Meaning |
|---|---|
| `RESEARCHOS_DATABASE_URL=postgresql://user:pw@host:5432/db` | selects PostgreSQL for the API (SQLite stays the default when unset) |
| `RESEARCHOS_DATABASE_SCHEMA` | API schema, default `api` (or `?schema=` in the URL) |
| `RESEARCHOS_ENGINE_DATABASE_URL` | optional: put the engine in a different database/server (must be PostgreSQL) |
| `RESEARCHOS_ENGINE_DATABASE_SCHEMA` | engine schema, default `engine` |
| `RESEARCHOS_DB_POOL_SIZE`, `RESEARCHOS_DB_MAX_OVERFLOW` | API connection pool (defaults 10 and 20) |

Settings refuse a configuration in which both layers would share one namespace. The schema is applied as `search_path` on every
connection (`agent_engine/state/dburl.py`); the `schema` URL parameter is removed before libpq sees it. Credentials come from the
environment only; error messages and logs redact the password.

The engine uses its own connection (psycopg 3, autocommit, explicit `BEGIN`/`COMMIT`), the API uses a bounded SQLAlchemy pool with
`pool_pre_ping` and `pool_recycle`.

## Migrations

* Versioned Alembic migrations for both layers. Engine migrations carry a PostgreSQL variant of each statement list (`UP_POSTGRES`:
  identity columns instead of `AUTOINCREMENT`, `DOUBLE PRECISION`, `BIGINT`); the API migrations are dialect-neutral and produce a schema
  that `compare_metadata` reports as identical to `api/models.py` on PostgreSQL (tested).
* **Fresh installation**: the first process creates the schema (and the PostgreSQL schema namespace) and stamps head. Running again does nothing.
* **Existing SQLite installation, staying on SQLite**: unchanged behaviour — upgraded in place, a pre-migration database is adopted only if it
  matches the baseline exactly.
* **Existing SQLite installation → PostgreSQL**: `python scripts/migrate_database.py --to postgresql://… --api-db … --engine-db …`
  (`api/dbcopy.py`). The source must already be at head and is only read. The target is brought to head by the normal migrations and must be
  **empty** (a non-empty table aborts with exit code 2; nothing is merged or overwritten). Each database is copied in one PostgreSQL
  transaction that commits only after every table's row count equals the source's. Identity sequences are advanced past the copied ids.
  Tested with a SQLite installation created by the application itself (users, project, run, claims, engine events) — rows, JSON values and
  event order are identical afterwards and a new event gets a fresh id.
* **Incompatible schemas are detected before anything changes**: tables that look like ours but have no migration history are refused
  (`SchemaMismatch`, exit code 2 in `scripts/migrate_db.py`), as is a revision this code does not know (a database written by newer code).
  Neither is altered. Nothing in P7 drops, truncates or resets data.
* **Concurrent startup**: on PostgreSQL every upgrade runs under a session-level advisory lock on its own connection
  (`agent_engine/state/pglock.py`), keyed per layer and schema; the loser waits, then finds the schema at head. The lock disappears if the process
  dies. Tested with five simultaneous processes per layer plus six threads.
* **Failed migration**: PostgreSQL DDL is transactional, so a failing revision leaves the schema as it was (tested with a deliberately failing revision).
* `scripts/migrate_db.py upgrade|check` runs both layers (the compose `migrate` service uses it) and waits for the database to accept connections.
  The API and worker still self-migrate under the lock unless `RESEARCHOS_AUTO_MIGRATE=false`, which the compose file sets so that only the
  `migrate` service changes schemas.

## Transactions, locking and isolation

Everything runs at PostgreSQL's default **READ COMMITTED**. Correctness comes from row locks and conditional updates, not from raising the
isolation level:

| Operation | Mechanism |
|---|---|
| Job claim (`api/jobs.py`) | candidate chosen with `SELECT … FOR UPDATE SKIP LOCKED`, then a conditional `UPDATE … WHERE status='queued'`; concurrent workers never queue behind each other and never receive the same job |
| Lease renewal | single conditional `UPDATE` (owner and `running` status in the `WHERE`) |
| Lease expiry / requeue | `SELECT … FOR UPDATE` on the job row, then re-check the lease inside the lock: a worker that renewed in time is never requeued; two reapers cannot both requeue |
| Duplicate enqueue | `run_id` is unique; the losing `INSERT` is caught and reported as "already queued" |
| Task and run status changes | read the current status with `FOR UPDATE` and validate the transition (`agent_engine/state/transitions.py`) in the same transaction as the write — two writers cannot both move a task out of `pending` |
| Run numbers | `FOR UPDATE` on the project row before counting existing runs |
| Token budget reservation, approval decision | read under `FOR UPDATE`, change in the same transaction |
| Event ordering | sequence values are allocated before commit, so concurrent writers could commit out of id order and a poller using `id > last_seen` would skip events; a per-run `pg_advisory_xact_lock` held until commit makes a run's events commit in id order (tested with a continuously polling reader and six racing writers) |
| Multi-statement units | `Store.transaction()`; any exception rolls back everything in the block (tested) |

On SQLite the same code paths use `BEGIN IMMEDIATE` (one writer at a time) and `FOR UPDATE` is omitted.

Connection loss: the engine store replaces a connection already known to be dead, retries a **read** that failed mid-flight, and raises
`StoreUnavailable` for a **write** or an interrupted `COMMIT` rather than guessing (the statement may have been applied). The next call reconnects.
The worker loop already catches errors and tries again; a job whose worker died is recovered by lease expiry (tested with a worker process killed
with no cleanup, followed by `reap_expired` and a claim by another worker).

## Testing

`tests/test_p7_postgres.py` (42 tests) and `tests/test_p7_deploy.py` (7) run against a **real PostgreSQL 16 server**: the `pgserver` pip package
starts an unmodified PostgreSQL on a local TCP port (no Docker, no admin rights), or set `RESEARCHOS_TEST_POSTGRES_URL` to use any server
(CI uses a `postgres:16` service container). Every test uses fresh schemas or databases. Nothing is mocked or replaced by SQLite.

The existing suite is also run on PostgreSQL unchanged: `RESEARCHOS_TEST_ON_POSTGRES=1 python -m pytest tests/test_api*.py tests/test_jobs.py …`
redirects every `Store(...)` and `make_engine(...)` the tests create to a PostgreSQL schema (`tests/conftest.py`). Results are in
`docs/PHASE_REPORTS.md`. The only test that cannot run that way asserts the SQLite exception class for a foreign-key violation; the PostgreSQL
equivalent is in `tests/test_p7_postgres.py`.

## Performance observations (indicative only)

`python scripts/bench_database.py` (results in `artifacts/evaluation/database_bench.json`): one laptop, PostgreSQL 16 over loopback, short single runs.

| | SQLite | PostgreSQL |
|---|---|---|
| event insert, 1 connection | ~420/s | ~690/s |
| task status update | ~445/s | ~610–760/s |
| 4 concurrent writers, events | ~390/s | ~475/s |
| `list_tasks` (200 tasks) | ~170/s | ~215/s |
| `list_events(after_id)` | ~300/s | ~155/s |
| enqueue job | ~64/s | ~240/s |
| claim, 4 workers: median / p95 | 28 ms / 234 ms | 28 ms / 44 ms |

Read as: nothing is pathologically slow, PostgreSQL's tail latency under contention is much better (no writer lock), single-row reads are slightly
slower over a socket. These do not predict behaviour on a networked database, with real data volumes, or under a real research workload.

## Limitations

* Docker is not installed on the development machine: `docker-compose.postgres.yml` is validated statically (tests) and by `docker compose config` in CI, **not run end to end here**.
* Lease times use the application clock (`utcnow()`); workers on hosts with badly skewed clocks can expire each other's leases early. Use NTP.
* No connection-pool proxy (PgBouncer) was tested; the advisory locks are session-level and need a session-mode pool.
* Timestamps in the engine DB are ISO text, so range queries by time rely on ISO ordering (UTC, fixed width).
* The in-process rate limiter and SSE gates are still per process; running several API processes multiplies those limits.
* Row-level security, read replicas, backups and point-in-time recovery are operator concerns and not configured here.
* The SQLite → PostgreSQL copy is offline: stop writers first. There is no live replication and no PostgreSQL → SQLite path.
