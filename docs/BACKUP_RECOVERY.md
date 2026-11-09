# Backup and recovery

RESEARCHOS state lives in three places. **A database backup alone is not a full backup.**

| State | Where | Back up with |
|---|---|---|
| Accounts, projects, runs, claims, reports, audit, job queue, agent-engine state | PostgreSQL, one database, schemas `api` and `engine` (or two SQLite files in the single-host setup) | `pg_dump` (below) |
| Uploaded documents, extracted text, search indexes, experiment outputs | the `/data` volume (`researchos-data`) | volume copy / filesystem snapshot |
| Model checkpoints (operator-trained) | `/models` (read-only mount, `RESEARCHOS_MODELS_PATH`) | normal file backup; they can be regenerated from `configs/` + data only by re-training |

The database refers to documents by project/document id; restoring the database without the matching `/data` leaves records whose files are missing,
and restoring `/data` without the database leaves orphan files. **Back both up at the same moment** (stop the API and worker briefly, or snapshot the volume
and take the dump back to back; a dump is transactionally consistent by itself, and the window between it and the file copy should be small and
preferably quiet).

## PostgreSQL backup

```bash
# from the host, against the compose stack (the database is on the internal network, so run pg_dump inside the postgres container)
docker compose -f docker-compose.prod.yml exec -T postgres \
  sh -c 'pg_dump --format=custom --no-owner -U "$POSTGRES_USER" "$POSTGRES_DB"' > researchos-$(date +%F-%H%M).dump

# the uploaded files (volume name is <project>_researchos-data; check with `docker volume ls`)
docker run --rm -v researchos_researchos-data:/data:ro -v "$PWD":/backup alpine tar czf /backup/researchos-data-$(date +%F-%H%M).tgz -C /data .
```

Keep dumps off the Docker host (another machine or object storage), encrypted at rest, and rotate them (for example daily for a week, weekly for a month).
Dumps contain password hashes, session hashes, and every user's research data: protect them like the database itself.

**Verify every backup** (an unverified backup is a hope, not a backup):

```bash
pg_restore --list researchos-2026-11-09-1200.dump | head       # must list tables such as "TABLE api users" and "TABLE engine runs"
tar tzf researchos-data-2026-11-09-1200.tgz | head
```

and periodically do the full drill below.

## PostgreSQL restore

Restore into a **new, empty** database first; only replace production after checking the result.

```bash
docker compose -f docker-compose.prod.yml stop api worker                       # nothing may write during a restore
docker compose -f docker-compose.prod.yml exec -T postgres sh -c 'createdb -U "$POSTGRES_USER" researchos_restore'
docker compose -f docker-compose.prod.yml exec -T postgres sh -c 'pg_restore --no-owner --exit-on-error -U "$POSTGRES_USER" -d researchos_restore' < researchos-2026-11-09-1200.dump
# check, then point RESEARCHOS_DATABASE_URL / POSTGRES_DB at the restored database (or rename databases), restore the data volume, and start the stack
docker compose -f docker-compose.prod.yml run --rm migrate python scripts/migrate_db.py check     # exit 0 = schema at the latest revision
docker compose -f docker-compose.prod.yml up -d
```

Never run a restore, `DROP DATABASE`, or `docker compose down -v` against a production database you have not backed up. The commands above never overwrite
the live database in place.

## Migration recovery

* Migrations run only in the `migrate` service (`RESEARCHOS_AUTO_MIGRATE=false` elsewhere). They are serialised with PostgreSQL advisory locks and are idempotent.
* A schema the code does not recognise (tables without migration history, or a revision newer than the code) is **refused** with exit code 2 and never altered.
* **Before upgrading the application, take a backup.** There is no automated downgrade path: if a new release's migration misbehaves, restore the dump taken
  before the upgrade and run the previous image.
* `python scripts/migrate_db.py check` is read-only and safe at any time.

## Restart and worker recovery

* `restart: unless-stopped` restarts crashed services; containers shut down on SIGTERM within `stop_grace_period`.
* The worker hands running jobs back to the queue when it stops. If it is killed, a job's lease expires (30 s) and another worker claims it; a run left
  mid-task is re-queued by `POST /runs/{id}/recover`. A job is attempted up to `job_max_attempts` (3) times before it fails.
* PostgreSQL data survives `docker compose down` (named volume `researchos-pgdata`); only `down -v` removes it.
* SSE clients reconnect with `Last-Event-ID` and resume without losing events.

## Persistent volume considerations

Use a named volume or a bind mount on reliable storage with its own backup; do not put `/data` on an ephemeral filesystem. Size it for
`max_corpus_bytes` (500 MB default per project) times your projects. PostgreSQL's volume should be on local fast storage, not a network share.
For point-in-time recovery use WAL archiving or `pg_basebackup` (not set up here); the procedure above gives recovery to the time of each dump.

## The drill that was run

`scripts/backup_restore_drill.py` automates the procedure on a **throwaway** database (it refuses nothing, because it only ever creates and drops its own
scratch databases; never point it at a database with real data):

```bash
python scripts/backup_restore_drill.py        # starts a test PostgreSQL (pgserver) unless --postgres is given
```

It creates a scratch database, migrates it, creates a user, a project and a document through the real HTTP application, takes a custom-format `pg_dump`,
lists the archive, restores into a second scratch database, compares the row counts of all 27 tables, runs `migrate_db.py check` on the restored database,
starts the application on it, signs in with the original credentials and searches the restored document. **Result on the development machine
(PostgreSQL 16, local):** all checks passed (`artifacts/evaluation/backup_restore_drill.json`); archive 63 KB; backup 0.3-0.7 s; restore 0.65 s.

**Limits of that evidence:** 9 rows, not production-sized data; the dump/restore ran against a local PostgreSQL, not the compose stack; the
uploaded-files step was a directory copy, not a Docker volume backup; no encryption, off-host copy, or WAL archiving was exercised; the Docker-based
commands above are **unverified** (no Docker here). Test the full procedure on your own environment before relying on it.
