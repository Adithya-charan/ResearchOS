# Model operations

How the custom model is managed, secured and used by agents. Code: `api/modellab_service.py`, `api/routes/modellab.py`,
`model/lifecycle.py`, `agent_engine/model_provider.py`, UI in `api/static/lab/` (served at `/lab`).

## Lifecycle

`INITIALIZED -> TRAINING -> TRAINED -> VALIDATED -> DEPLOYED`, plus `ARCHIVED` (terminal) and `FAILED`. The server **derives** the evidence
from artifacts on disk (`run.json`/`summary.json`, checkpoint verification, `splits.json` leakage record, the evaluation report bound
to the best checkpoint's SHA-256); a request body only names the target state and an optional note. A refused transition returns 409
with the reasons and is audit-logged.

| Transition | Requires |
|---|---|
| TRAINING -> TRAINED | completed run, >= 1 applied optimizer step, finite final loss, verified `random_init` lineage |
| TRAINED -> VALIDATED | verified checkpoint, verified lineage, finite validation loss that **beats the random-initialised baseline**, leakage check clean, evaluation made for the current best checkpoint |
| VALIDATED -> DEPLOYED | an administrator (`RESEARCHOS_ADMIN_EMAILS`); the authorising user is recorded |
| anything -> ARCHIVED | allowed from every non-terminal state except TRAINING |

A model is never TRAINED because it was merely initialised. System (operator) models can only be moved by an administrator.

## Provider modes for agents

Exactly two providers exist: **deterministic** (no language model; the default and the fallback) and **custom Transformer** (the model this
project trained). No hosted AI API exists in the code and none is used as a fallback.

* *production*: needs a DEPLOYED model **and** a measured integration stage >= 3.
* *experimental*: an explicit per-user opt-in (`allow_experimental`) and a model with trained weights. A model that has no trained weights
  (INITIALIZED/TRAINING) additionally needs a second flag that the UI does not expose; it exists for pipeline tests.
* An ineligible selection is refused with the reason (409); at run time an unavailable model degrades to the deterministic provider and
  the run records that. Nothing switches silently.
* Model text is always an untrusted draft note. A guard drops any note containing citation-like or source-like text (`[1]`, `(Doe, 2019)`,
  DOIs, URLs, passage ids). The model cannot create evidence, citations or verdicts; those come from the deterministic evidence layer
  (`docs/EVIDENCE_VERIFICATION.md`). Sandbox execution is never delegated to model text.

### Integration stages (measured, not asserted)

1. loads, provider interface works, generation reproducible. 2. structured-output pass rate >= 0.8 over 8 seeded prompts and invalid output
   rejected. 3. a recorded instruction-evaluation artifact on permitted data that beats the deterministic baseline. 4. and 5. are
   decisions, not measurements, and are **not enabled**. The stage a model has reached is recomputed by the evaluation and shown in the UI.

Current standing: the bundled models reach stage 1 at most (structured-output pass rate 0 of 8 for `small_v1`); there is no
instruction-tuning data or run, so stage 3 is unreachable today and production mode is not available.

## API summary (all require a session; writes require the CSRF header; cross-site origins are refused)

* `/modellab/overview`, `/modellab/profiles`, `/modellab/hardware-benchmarks`, `/modellab/models/{id}`
* `/modellab/models/{id}/lifecycle` (GET, POST), `/modellab/models/{id}/evaluate` (POST, 202), `/modellab/models/{id}/evaluation`
* `/modellab/datasets` (GET, POST), `/modellab/datasets/{id}` (GET, DELETE), `/modellab/datasets/{id}/prepare`, `/modellab/datasets/{id}/tokenizer`
* `/modellab/provider` (GET), `/modellab/selection` (PUT)
* existing: `/models*`, `/models/{id}/generate[/stream]`, `/models/{id}/checkpoints[/{name}/verify]`, `/training/configs`, `/training/jobs*`
* `DELETE /modellab/models/{id}/checkpoints/{name}`: own training jobs only, never best/latest, never operator checkpoints

## Security notes

* **No filesystem paths from clients.** Requests carry ids and small numeric/enum options; the server resolves directories. Checkpoint
  names are validated against the server's own listing; dataset ids must belong to the caller (404 otherwise).
* **Authorization.** Datasets, training jobs and trained models are owner-scoped; operator models are read-only for users. DEPLOYED and
  system-model changes need an administrator. Lifecycle and selection changes are audit-logged, including refusals.
* **Limits.** Dataset documents/characters/vocabulary/count, training steps and concurrent jobs, prompt size, new tokens, per-user request
  rate, concurrent streams, one evaluation at a time, evaluation window cap, upload sizes.
* **Safe serialization.** Weights are safetensors; JSON for everything else; no pickle is ever loaded from user-controlled data.
* **Rights.** Dataset upload requires a license from a fixed list, attribution and a rights confirmation; the manifest is hashed.
  Nothing is scraped or downloaded by the platform.
* **UI hardening.** The lab is three whitelisted static files served with `Content-Security-Policy: default-src 'self'` (no inline script or
  style, no eval, no framing), server text is inserted only as text nodes, and a test asserts no external URLs and that every endpoint the
  UI calls exists. Same-origin browser writes are accepted because the request's `Origin` equals this server; any other origin not in
  `RESEARCHOS_ALLOWED_ORIGINS` is refused.
* **Not reviewed.** No external penetration test or dependency audit was performed; the review is a code-level pass with tests.

## Configuration

`RESEARCHOS_TRAINING_ENABLED` (default off), `RESEARCHOS_ADMIN_EMAILS` (comma-separated), `RESEARCHOS_ALLOWED_ORIGINS`,
`RESEARCHOS_MODELS_DIR`, `RESEARCHOS_INFERENCE_DEVICE`, `RESEARCHOS_DATA_DIR`. Migration `0004_model_lab` adds `model_entries`,
`training_datasets` and `model_selections`; it runs with the other Alembic revisions (SQLite and PostgreSQL).

## Status vocabulary used in the reports

*Implemented* (code exists) < *Unit tested* < *Smoke tested* (a short real run) < *Hardware tested* (measured on named hardware) <
*Successfully trained* (a completed run with recorded loss) < *Experimentally evaluated* (evaluation report exists). A feature is only
listed at the highest level it has evidence for.
