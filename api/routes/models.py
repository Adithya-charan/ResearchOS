"""Model management: registry of checkpoints, training jobs, metrics, local inference and health."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from sqlalchemy import func, select

from api.deps import DB, CurrentAuth, audit
from api.db import utcnow
from api.model_service import (ACTIVE, FINISHED, NOTICE, RESUMABLE, SYSTEM_PREFIX, Busy, InferenceService, JobRefused, ModelNotFound,
                               ModelRegistry, TrainingService, describe, list_checkpoints, pick_checkpoint, read_metrics,
                               read_samples, tokenizer_dir_for, verify_checkpoint)
from api.models import TrainingJob, new_id
from api.schemas import GenerateIn, TrainingJobIn

router = APIRouter(tags=["models"])
LOG_LINES = 300


def _svc(request: Request) -> tuple[TrainingService, ModelRegistry, InferenceService]:
    s = request.app.state
    return s.training, s.model_registry, s.inference


def owned_job(db, job_id: str, user_id: str) -> TrainingJob:
    j = db.get(TrainingJob, job_id)
    if j is None or j.owner_id != user_id:
        raise HTTPException(404, "training job not found")
    return j


def _job_view(training: TrainingService, j: TrainingJob) -> dict:
    run_dir = training.root / j.id / "run"
    cks = list_checkpoints(run_dir) if run_dir.is_dir() else []
    last = (read_metrics(run_dir)["train"] or [{}])[-1]
    cfg = j.config or {}
    return {"id": j.id, "name": j.name, "base_config": j.base_config, "status": j.status, "pending_request": j.control or None,
            "config": cfg, "segments": j.segments, "created_at": j.created_at, "updated_at": j.updated_at,
            "finished_at": j.finished_at, "error": j.error, "summary": j.summary, "checkpoints": len(cks),
            "step": last.get("step"), "max_steps": cfg.get("max_steps"), "last_loss": last.get("loss"),
            "process_alive": training.is_alive(j.id), "resumable": j.status in RESUMABLE and bool(cks),
            "model_id": j.id if cks else None, "notice": NOTICE}


def _resolve(request: Request, db, auth, model_id: str) -> tuple[Path, str, dict]:
    """(run_dir, kind, extra) for a registry id: sys_<name> = operator checkpoints, trn_<id> = the caller's own job."""
    _, registry, _ = _svc(request)
    try:
        if model_id.startswith(SYSTEM_PREFIX):
            return registry.system_dir(model_id), "system", {"owner": "operator"}
        if model_id.startswith("trn_"):
            job = owned_job(db, model_id, auth.user.id)
            run_dir = registry.job_run_dir(job.id)
            if not run_dir.is_dir():
                raise ModelNotFound("this job has no checkpoints yet")
            return run_dir, "trained", {"owner": "you", "name": job.name, "job_status": job.status}
    except ModelNotFound as e:
        raise HTTPException(404, str(e)) from e
    raise HTTPException(404, "model not found")


# ------------------------------------------------------------------ registry
@router.get("/models/health")
def models_health(request: Request, auth: CurrentAuth):
    import torch
    s = request.app.state.settings
    training, registry, inference = _svc(request)
    tok_ok = (s.tokenizer_dir / "vocab.json").is_file() and (s.tokenizer_dir / "merges.txt").is_file()
    cuda = torch.cuda.is_available()
    return {"status": "ok" if tok_ok and registry.system_models() else "degraded",
            "tokenizer": {"present": tok_ok, "dir": s.tokenizer_dir.name},
            "system_models": [mid for mid, _ in registry.system_models()],
            "inference": {"device": s.inference_device, "loaded": inference.loaded(), "max_new_tokens": s.infer_max_new_tokens,
                          "timeout_s": s.infer_timeout_s},
            "training": {"enabled": s.training_enabled, "running_processes": training.running_count(),
                         "capacity": s.max_running_training_jobs, "max_steps": s.max_train_steps},
            "hardware": {"cuda_available": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None, "torch": torch.__version__},
            "notes": ["Per-checkpoint integrity is verified by GET /models/{id}/health (it hashes the weights).",
                      NOTICE]}


@router.get("/models")
def list_models(request: Request, auth: CurrentAuth, db: DB):
    _, registry, _ = _svc(request)
    out = [describe(mid, d, "system", {"owner": "operator"}) for mid, d in registry.system_models()]
    for j in db.scalars(select(TrainingJob).where(TrainingJob.owner_id == auth.user.id).order_by(TrainingJob.created_at.desc())):
        run_dir = registry.job_run_dir(j.id)
        if run_dir.is_dir() and any(run_dir.glob("step_*")):
            out.append(describe(j.id, run_dir, "trained", {"owner": "you", "name": j.name, "job_status": j.status}))
    return out


@router.get("/models/{model_id}")
def read_model(model_id: str, request: Request, auth: CurrentAuth, db: DB):
    run_dir, kind, extra = _resolve(request, db, auth, model_id)
    return describe(model_id, run_dir, kind, extra)


@router.get("/models/{model_id}/checkpoints")
def checkpoints(model_id: str, request: Request, auth: CurrentAuth, db: DB):
    run_dir, _, _ = _resolve(request, db, auth, model_id)
    return list_checkpoints(run_dir)


@router.get("/models/{model_id}/checkpoints/{name}/verify")
def verify(model_id: str, name: str, request: Request, auth: CurrentAuth, db: DB):
    run_dir, _, _ = _resolve(request, db, auth, model_id)
    if name not in {c["name"] for c in list_checkpoints(run_dir)}:  # only names we listed: never a path from the client
        raise HTTPException(404, "checkpoint not found")
    return {"checkpoint": name, **verify_checkpoint(run_dir / name)}


@router.get("/models/{model_id}/health")
def model_health(model_id: str, request: Request, auth: CurrentAuth, db: DB):
    run_dir, _, _ = _resolve(request, db, auth, model_id)
    s = request.app.state.settings
    out = {"id": model_id, "checkpoints": {}}
    ok = True
    for which in ("best", "latest"):
        try:
            ck = pick_checkpoint(run_dir, which)
        except ModelNotFound:
            out["checkpoints"][which] = {"ok": False, "problem": "no checkpoint"}
            ok = False
            continue
        v = verify_checkpoint(ck)
        out["checkpoints"][which] = {"name": ck.name, **v}
        ok = ok and v["ok"]
    try:
        from model.tokenizer.bpe import BPETokenizer
        from model.training.checkpoint import read_meta
        tok = BPETokenizer.load(s.tokenizer_dir)
        expected = (read_meta(pick_checkpoint(run_dir, "best")).get("tokenizer") or {}).get("fingerprint")
        match = expected is None or expected == tok.fingerprint
        out["tokenizer"] = {"ok": match, "fingerprint": tok.fingerprint[:12], "expected": (expected or "")[:12] or None}
        ok = ok and match
    except Exception as e:  # tokenizer files missing/corrupt: report, do not 500
        out["tokenizer"] = {"ok": False, "problem": type(e).__name__}
        ok = False
    out["status"] = "ok" if ok else "unhealthy"
    return out


def admit_generation(request: Request, auth, body: GenerateIn) -> None:
    """Size limits and the per-user rate limit, shared by the plain and the streaming generate endpoints."""
    s = request.app.state.settings
    if len(body.prompt) > s.infer_max_prompt_chars:
        raise HTTPException(422, f"prompt is limited to {s.infer_max_prompt_chars} characters")
    if body.max_new_tokens > s.infer_max_new_tokens:
        raise HTTPException(422, f"max_new_tokens is limited to {s.infer_max_new_tokens} on this server")
    key, limiter = f"gen:{auth.user.id}", request.app.state.limiter
    if limiter.blocked(key, s.infer_per_minute, 60):
        raise HTTPException(429, "too many generation requests; slow down")
    limiter.hit(key, 60)


@router.post("/models/{model_id}/generate")
def generate(model_id: str, body: GenerateIn, request: Request, auth: CurrentAuth, db: DB):
    _, _, inference = _svc(request)
    run_dir, _, _ = _resolve(request, db, auth, model_id)
    admit_generation(request, auth, body)
    from model.inference.runtime import ContextOverflowError, TokenizerMismatchError
    from model.training.checkpoint import CheckpointError
    try:
        ck = pick_checkpoint(run_dir, body.checkpoint)
        result = inference.generate(ck, body.prompt, body.max_new_tokens, body.temperature, body.top_k, body.top_p,
                                    body.repetition_penalty, body.seed,
                                    tokenizer_dir=tokenizer_dir_for(request.app.state.settings, run_dir))
    except ModelNotFound as e:
        raise HTTPException(404, str(e)) from e
    except Busy as e:
        raise HTTPException(429, str(e)) from e
    except ContextOverflowError as e:
        raise HTTPException(422, str(e)) from e
    except (TokenizerMismatchError, CheckpointError) as e:
        raise HTTPException(409, f"this checkpoint cannot be served: {e}") from e
    except (FileNotFoundError, ValueError) as e:  # e.g. tokenizer files missing, ids outside the vocabulary
        raise HTTPException(503, f"model is not available: {e}") from e
    audit(db, request, "model_generate", auth.user.id, model_id, prompt_chars=len(body.prompt), new_tokens=result["new_tokens"])
    return {"model_id": model_id, "checkpoint": ck.name, **result}


# ------------------------------------------------------------------ training
@router.get("/training/configs")
def training_configs(request: Request, auth: CurrentAuth):
    training, _, _ = _svc(request)
    s = request.app.state.settings
    return {"enabled": s.training_enabled, "max_steps": s.max_train_steps, "configs": training.base_configs(),
            "note": "A job starts from one of these operator-controlled configs; only the listed numeric hyper-parameters can be changed."}


def _require_enabled(request: Request) -> None:
    if not request.app.state.settings.training_enabled:
        raise HTTPException(403, "training is disabled on this server (the operator enables it with RESEARCHOS_TRAINING_ENABLED=true)")


def _refusal(e: JobRefused) -> HTTPException:
    return HTTPException(e.status, str(e))


@router.post("/training/jobs", status_code=201)
def start_job(body: TrainingJobIn, request: Request, auth: CurrentAuth, db: DB):
    _require_enabled(request)
    training, _, _ = _svc(request)
    s = request.app.state.settings
    if db.scalar(select(func.count()).select_from(TrainingJob).where(TrainingJob.owner_id == auth.user.id)) >= s.max_training_jobs_per_user:
        raise HTTPException(429, f"training job quota reached ({s.max_training_jobs_per_user}); delete finished jobs")
    if training.running_count() >= s.max_running_training_jobs:
        raise HTTPException(429, "the training capacity of this server is in use; wait for the running job to finish")
    overrides = body.model_dump(exclude={"name", "base_config", "dataset_id"}, exclude_none=True)
    dataset_dir = None
    if body.dataset_id:
        from api.models import TrainingDataset
        ds = db.get(TrainingDataset, body.dataset_id)
        if ds is None or ds.owner_id != auth.user.id:
            raise HTTPException(404, "dataset not found")
        if ds.status != "prepared":
            raise HTTPException(409, f"this dataset is {ds.status}; prepare it first")
        dataset_dir = request.app.state.modellab.dataset_dir(ds.id)
    job = TrainingJob(id=new_id("trn"), owner_id=auth.user.id, name=body.name, base_config=body.base_config, config={}, status="running")
    job_dir = training.root / job.id
    try:
        job_dir.mkdir(parents=True)
        job.config = training.build_config(body.base_config, overrides, job_dir, dataset_dir, body.dataset_id)
        job.segments = 1
        db.add(job)
        db.commit()
        training.start_process(job.id, resume=False)
    except JobRefused as e:
        db.rollback()
        _discard(training, job.id)
        raise _refusal(e) from e
    except Exception:
        db.rollback()
        _discard(training, job.id)
        raise
    audit(db, request, "training_start", auth.user.id, job.id, base=body.base_config, overrides=overrides)
    return _job_view(training, job)


def _discard(training: TrainingService, job_id: str) -> None:
    import shutil
    shutil.rmtree(training.root / job_id, ignore_errors=True)


@router.get("/training/jobs")
def list_jobs(request: Request, auth: CurrentAuth, db: DB):
    training, _, _ = _svc(request)
    rows = db.scalars(select(TrainingJob).where(TrainingJob.owner_id == auth.user.id).order_by(TrainingJob.created_at.desc())).all()
    return [_job_view(training, j) for j in rows]


@router.get("/training/jobs/{job_id}")
def read_job(job_id: str, request: Request, auth: CurrentAuth, db: DB):
    training, _, _ = _svc(request)
    return _job_view(training, owned_job(db, job_id, auth.user.id))


@router.post("/training/jobs/{job_id}/pause")
def pause(job_id: str, request: Request, auth: CurrentAuth, db: DB):
    return _stop(request, auth, db, job_id, "pause", "pausing")


@router.post("/training/jobs/{job_id}/cancel")
def cancel(job_id: str, request: Request, auth: CurrentAuth, db: DB):
    return _stop(request, auth, db, job_id, "cancel", "cancelling")


def _stop(request: Request, auth, db, job_id: str, action: str, status: str) -> dict:
    training, _, _ = _svc(request)
    j = owned_job(db, job_id, auth.user.id)
    if action == "cancel" and j.status in RESUMABLE:  # a paused job has no process: cancelling just closes it
        j.status, j.finished_at = "cancelled", utcnow()
        db.commit()
    elif j.status == "running":
        j.status, j.control = status, action
        db.commit()
        training.request_stop(j.id)
    elif j.status == "pausing" and action == "cancel":  # upgrade a pending pause to a cancel
        j.status, j.control = status, action
        db.commit()
    else:
        raise HTTPException(409, f"a {j.status} job cannot be {action}d")
    audit(db, request, f"training_{action}", auth.user.id, j.id)
    return _job_view(training, j)


@router.post("/training/jobs/{job_id}/resume")
def resume(job_id: str, request: Request, auth: CurrentAuth, db: DB):
    _require_enabled(request)
    training, _, _ = _svc(request)
    j = owned_job(db, job_id, auth.user.id)
    if j.status not in RESUMABLE:
        raise HTTPException(409, f"only a paused or interrupted job can be resumed (this one is {j.status})")
    if not (training.root / j.id / "run" / "latest.json").is_file():
        raise HTTPException(409, "there is no checkpoint to resume from")
    try:
        training.start_process(j.id, resume=True)
    except JobRefused as e:
        raise _refusal(e) from e
    j.status, j.control, j.error, j.finished_at, j.segments = "running", "", "", None, j.segments + 1
    db.commit()
    audit(db, request, "training_resume", auth.user.id, j.id)
    return _job_view(training, j)


@router.get("/training/jobs/{job_id}/metrics")
def job_metrics(job_id: str, request: Request, auth: CurrentAuth, db: DB, after_step: int = Query(0, ge=0),
                limit: int = Query(500, ge=1, le=2000)):
    training, _, _ = _svc(request)
    j = owned_job(db, job_id, auth.user.id)
    return {"job_id": j.id, "status": j.status, **read_metrics(training.root / j.id / "run", after_step, limit),
            "note": "train.loss is the batch loss; eval.val_loss is measured on held-out documents of the training corpus."}


@router.get("/training/jobs/{job_id}/samples")
def job_samples(job_id: str, request: Request, auth: CurrentAuth, db: DB):
    training, _, _ = _svc(request)
    j = owned_job(db, job_id, auth.user.id)
    return read_samples(training.root / j.id / "run")


@router.get("/training/jobs/{job_id}/log")
def job_log(job_id: str, request: Request, auth: CurrentAuth, db: DB, lines: int = Query(100, ge=1, le=LOG_LINES)):
    training, _, _ = _svc(request)
    j = owned_job(db, job_id, auth.user.id)
    return {"job_id": j.id, "lines": training._tail(j.id, lines, 40_000).splitlines()}


@router.delete("/training/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, request: Request, auth: CurrentAuth, db: DB):
    training, _, inference = _svc(request)
    j = owned_job(db, job_id, auth.user.id)
    if j.status not in FINISHED or training.is_alive(j.id):
        raise HTTPException(409, "stop the job (pause or cancel) before deleting it")
    db.delete(j)
    db.commit()
    _discard(training, job_id)
    audit(db, request, "training_delete", auth.user.id, job_id)
