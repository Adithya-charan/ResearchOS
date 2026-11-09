"""Model Lab: lifecycle state, evaluation, user datasets + tokenizer preparation, provider selection and integration status.

Everything is derived from real artifacts (see api/modellab_service.py). Clients send ids and small numeric/enum options; they never
send paths, and lifecycle evidence is never taken from a request."""
from __future__ import annotations

from fastapi import Depends, APIRouter, HTTPException, Request
from sqlalchemy import select

from agent_engine.model_provider import (AGENT_ROLES, STAGE_DESCRIPTIONS, NoValidatedModelError, ProviderSelection, capability_of,
                                         deterministic_status)
from api.ratelimit import rate_limit
from api.deps import DB, CurrentAuth, audit
from api.model_service import (NOTICE, ModelNotFound, _read_json, list_checkpoints, pick_checkpoint, read_metrics, tokenizer_dir_for)
from api.modellab_service import LICENSES, LabError, ModelLab, is_admin, selection_view
from api.models import ModelEntry, ModelSelection, TrainingDataset, TrainingJob
from api.routes.models import _resolve
from api.schemas import DatasetIn, LifecycleIn, PrepareIn, SelectionIn
from model.lifecycle import ModelState
from model.profiles import PROFILES, check_profile
from model.training.memory import estimate_training_memory, hardware_report

router = APIRouter(prefix="/modellab", tags=["modellab"])


def lab(request: Request) -> ModelLab:
    return request.app.state.modellab


def _fail(e: LabError) -> HTTPException:
    return HTTPException(e.status, str(e))


def _entry_for(request: Request, db, auth, model_id: str) -> tuple[ModelEntry, object]:
    run_dir, kind, _ = _resolve(request, db, auth, model_id)  # ownership/404 rules are the existing ones
    ml = lab(request)
    row = ml.entry(db, model_id, None if kind == "system" else auth.user.id)
    job = db.get(TrainingJob, model_id) if model_id.startswith("trn_") else None
    ml.refresh(db, row, job)
    return row, run_dir


def _owned_dataset(db, auth, dataset_id: str) -> TrainingDataset:
    ds = db.get(TrainingDataset, dataset_id)
    if ds is None or ds.owner_id != auth.user.id:
        raise HTTPException(404, "dataset not found")
    return ds


def _dataset_view(ds: TrainingDataset) -> dict:
    return {"id": ds.id, "name": ds.name, "license": ds.license, "attribution": ds.attribution, "status": ds.status,
            "documents": ds.documents, "chars": ds.chars, "manifest_sha256": ds.manifest_sha256, "stats": ds.stats, "error": ds.error,
            "created_at": ds.created_at, "updated_at": ds.updated_at}


# ------------------------------------------------------------------ overview, profiles
@router.get("/overview")
def overview(request: Request, auth: CurrentAuth, db: DB):
    import torch
    ml, s = lab(request), request.app.state.settings
    _, registry, inference = request.app.state.training, request.app.state.model_registry, request.app.state.inference
    entries = []
    ids = [(mid, None) for mid, _ in registry.system_models()]
    ids += [(j.id, j) for j in db.scalars(select(TrainingJob).where(TrainingJob.owner_id == auth.user.id)) if registry.job_run_dir(j.id).is_dir()
            and any(registry.job_run_dir(j.id).glob("step_*"))]
    for mid, job in ids:
        row = ml.entry(db, mid, None if job is None else auth.user.id)
        ml.refresh(db, row, job)
        entries.append({"model_id": mid, "state": row.state, "capability": capability_of(row.state), "stage_reached": row.stage_reached})
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["state"]] = counts.get(e["state"], 0) + 1
    sel = ml.selection_of(db, auth.user.id)
    return {"hardware": hardware_report(), "models": entries, "lifecycle_counts": counts, "selection": selection_view(sel),
            "training_enabled": s.training_enabled, "inference": {"device": s.inference_device, "loaded": inference.loaded()},
            "datasets": db.query(TrainingDataset).filter(TrainingDataset.owner_id == auth.user.id).count(),
            "is_admin": is_admin(s, auth.user.email), "notice": NOTICE}


@router.get("/profiles")
def profiles(auth: CurrentAuth):
    out = []
    for name, p in PROFILES.items():
        cfg = p.model_config()
        chk = check_profile(p)
        out.append({"name": name, "description": p.description, "target_hardware": p.target_hardware, "parameters": cfg.estimate_params(),
                    "model": p.model, "training_defaults": p.train, "min_memory_gb_planning": p.min_memory_gb, "notes": list(p.notes),
                    "fit_check": {"device": chk["device"], "estimated_bytes": chk["estimated_bytes"], "budget_bytes": chk["budget_bytes"],
                                  "fits_by_estimate": chk["fits_by_estimate"], "basis": chk["basis"]},
                    "measured_here": "see GET /modellab/hardware-benchmarks (only measured fit counts as a compatibility claim)"})
    return out


@router.get("/hardware-benchmarks")
def hardware_benchmarks(request: Request, auth: CurrentAuth):
    from model.common import repo_root
    path = repo_root() / "artifacts" / "evaluation" / "model_hardware.json"
    data = _read_json(path)
    return data or {"status": "not_run", "note": "run `python scripts/bench_model.py` on this machine to record measured figures"}


# ------------------------------------------------------------------ model detail and lifecycle
@router.get("/models/{model_id}")
def model_detail(model_id: str, request: Request, auth: CurrentAuth, db: DB):
    row, run_dir = _entry_for(request, db, auth, model_id)
    ml = lab(request)
    cks = list_checkpoints(run_dir)
    config = None
    try:
        meta = _read_json(pick_checkpoint(run_dir, "best") / "meta.json") or {}
        from model.architecture.config import ModelConfig
        mc = ModelConfig.from_dict(meta["model_config"])
        tc = meta.get("train_config") or {}
        est = estimate_training_memory(mc, int(tc.get("batch_size", 1)), mixed_precision=tc.get("mixed_precision") in ("bf16", "auto"),
                                       activation_checkpointing=bool(tc.get("activation_checkpointing")), device="cpu")
        config = {"model": meta["model_config"], "parameters": mc.estimate_params(), "training": tc,
                  "tokenizer": meta.get("tokenizer"), "precision": meta.get("precision"), "lineage_root": (meta.get("lineage") or {}).get("root"),
                  "memory_estimate": {"basis": "estimate", **est.to_dict()}}
    except (ModelNotFound, KeyError, ValueError, TypeError):
        pass
    return {"model_id": model_id, "lifecycle": ml.view(row), "capability": capability_of(row.state), "config": config,
            "run": _read_json(run_dir / "run.json") or _read_json(run_dir / "summary.json"), "checkpoints": cks,
            "evaluation": row.evaluation, "notice": NOTICE}


@router.get("/models/{model_id}/lifecycle")
def lifecycle(model_id: str, request: Request, auth: CurrentAuth, db: DB):
    row, _ = _entry_for(request, db, auth, model_id)
    return lab(request).view(row)


@router.post("/models/{model_id}/lifecycle")
def change_lifecycle(model_id: str, body: LifecycleIn, request: Request, auth: CurrentAuth, db: DB):
    row, _ = _entry_for(request, db, auth, model_id)
    ml = lab(request)
    before = row.state
    try:
        ml.change_state(db, row, body.target, auth.user, body.note)
    except LabError as e:
        audit(db, request, "model_lifecycle_refused", auth.user.id, model_id, to_state=body.target, reason=str(e)[:300])
        raise _fail(e) from e
    audit(db, request, "model_lifecycle", auth.user.id, model_id, before=before, after=row.state, note=body.note[:200])
    return ml.view(row)


# ------------------------------------------------------------------ evaluation
@router.post("/models/{model_id}/evaluate", status_code=202, dependencies=[Depends(rate_limit("evaluate"))])
def evaluate(model_id: str, request: Request, auth: CurrentAuth, db: DB):
    _entry_for(request, db, auth, model_id)
    try:
        out = lab(request).start_evaluation(model_id)
    except LabError as e:
        raise _fail(e) from e
    except ModelNotFound as e:
        raise HTTPException(404, str(e)) from e
    audit(db, request, "model_evaluate", auth.user.id, model_id)
    return out


@router.get("/models/{model_id}/evaluation")
def evaluation(model_id: str, request: Request, auth: CurrentAuth, db: DB):
    _entry_for(request, db, auth, model_id)
    return lab(request).evaluation_status(model_id)


# ------------------------------------------------------------------ checkpoints
@router.delete("/models/{model_id}/checkpoints/{name}", status_code=204)
def delete_checkpoint(model_id: str, name: str, request: Request, auth: CurrentAuth, db: DB):
    """Owner-only, and only for the caller's own training jobs: operator checkpoints are never deletable through the API."""
    import shutil
    if not model_id.startswith("trn_"):
        raise HTTPException(403, "operator checkpoints cannot be deleted through the API")
    run_dir, _, _ = _resolve(request, db, auth, model_id)
    training = request.app.state.training
    if training.is_alive(model_id):
        raise HTTPException(409, "stop the training job before deleting its checkpoints")
    cks = {c["name"]: c for c in list_checkpoints(run_dir)}
    if name not in cks:  # only names we listed: never a path from the client
        raise HTTPException(404, "checkpoint not found")
    if cks[name]["is_best"] or cks[name]["is_latest"]:
        raise HTTPException(409, "the best and latest checkpoints are what the model is served from and cannot be deleted")
    shutil.rmtree(run_dir / name)
    audit(db, request, "checkpoint_delete", auth.user.id, model_id, checkpoint=name)


# ------------------------------------------------------------------ provider selection and integration status
def _selection_from(body: SelectionIn) -> ProviderSelection:
    return ProviderSelection(body.provider, body.model_id, body.mode, body.allow_experimental, False, "error")  # type: ignore[arg-type]


@router.get("/provider")
def provider(request: Request, auth: CurrentAuth, db: DB):
    ml = lab(request)
    row = ml.selection_of(db, auth.user.id)
    view = selection_view(row)
    if row is None or row.provider == "deterministic":
        status = deterministic_status().to_dict()
        error = None
    else:
        sel = ProviderSelection("custom_transformer", row.model_id, row.mode, bool(row.allow_experimental), False, "error")  # type: ignore[arg-type]
        try:
            _, st = ml.resolve(db, auth.user.id, sel, load=False)
            status, error = st.to_dict(), None
        except NoValidatedModelError as e:
            status, error = deterministic_status(reason=str(e)).to_dict(), str(e)
    return {"selection": view, "effective": status, "error": error, "never_uses_hosted_api": True,
            "stages": {str(k): v for k, v in STAGE_DESCRIPTIONS.items()},
            "roles": list(AGENT_ROLES) if status["provider"] == "custom_transformer" else [],
            "note": "Model text is always an untrusted draft note. Evidence, citations and verdicts come only from the deterministic evidence layer."}


@router.put("/selection")
def select_provider(body: SelectionIn, request: Request, auth: CurrentAuth, db: DB):
    ml = lab(request)
    if body.provider == "custom_transformer":
        if not body.model_id:
            raise HTTPException(422, "model_id is required for custom_transformer")
        _entry_for(request, db, auth, body.model_id)  # ownership + refresh before judging eligibility
        try:
            ml.resolve(db, auth.user.id, _selection_from(body), load=False)  # eligibility only; nothing is loaded
        except NoValidatedModelError as e:
            audit(db, request, "model_select_refused", auth.user.id, body.model_id, reason=str(e)[:300])
            raise HTTPException(409, str(e)) from e
    row = db.get(ModelSelection, auth.user.id) or ModelSelection(owner_id=auth.user.id)
    row.provider, row.model_id = body.provider, body.model_id if body.provider == "custom_transformer" else None
    row.mode, row.allow_experimental = body.mode, int(body.allow_experimental)
    db.add(row)
    db.commit()
    audit(db, request, "model_select", auth.user.id, body.model_id or "", provider=body.provider, mode=body.mode,
          allow_experimental=body.allow_experimental)
    return selection_view(row)


# ------------------------------------------------------------------ datasets and tokenizer
@router.get("/datasets")
def list_datasets(auth: CurrentAuth, db: DB):
    rows = db.scalars(select(TrainingDataset).where(TrainingDataset.owner_id == auth.user.id).order_by(TrainingDataset.created_at.desc()))
    return {"licenses": list(LICENSES), "datasets": [_dataset_view(r) for r in rows],
            "policy": "Only text you have the right to train on. Nothing is downloaded or scraped; every upload carries a license, "
                      "attribution and your rights confirmation, recorded in a hashed manifest."}


@router.post("/datasets", status_code=201)
def create_dataset(body: DatasetIn, request: Request, auth: CurrentAuth, db: DB):
    try:
        ds = lab(request).create_dataset(db, auth.user.id, body.name, body.license, body.attribution, body.rights_confirmed,
                                         [d.model_dump() for d in body.documents])
    except LabError as e:
        raise _fail(e) from e
    audit(db, request, "dataset_create", auth.user.id, ds.id, license=ds.license, documents=ds.documents, chars=ds.chars)
    return _dataset_view(ds)


@router.get("/datasets/{dataset_id}")
def read_dataset(dataset_id: str, auth: CurrentAuth, db: DB):
    return _dataset_view(_owned_dataset(db, auth, dataset_id))


@router.post("/datasets/{dataset_id}/prepare", status_code=202)
def prepare(dataset_id: str, body: PrepareIn, request: Request, auth: CurrentAuth, db: DB):
    """Tokenizer training and packing: clean, deduplicate, split by document, fit BPE on the training split only, write token files."""
    ds = _owned_dataset(db, auth, dataset_id)
    s = request.app.state.settings
    if body.vocab_size > s.dataset_max_vocab:
        raise HTTPException(422, f"vocab_size is limited to {s.dataset_max_vocab} on this server")
    if ds.status == "preparing":
        raise HTTPException(409, "this dataset is already being prepared")
    try:
        lab(request).prepare_dataset(ds.id, body.vocab_size, body.pretokenizer, body.min_chars)
    except LabError as e:
        raise _fail(e) from e
    ds.status, ds.error = "preparing", ""
    db.commit()
    audit(db, request, "dataset_prepare", auth.user.id, ds.id, vocab_size=body.vocab_size, pretokenizer=body.pretokenizer)
    return _dataset_view(ds)


@router.get("/datasets/{dataset_id}/tokenizer")
def dataset_tokenizer(dataset_id: str, request: Request, auth: CurrentAuth, db: DB):
    ds = _owned_dataset(db, auth, dataset_id)
    if ds.status != "prepared":
        raise HTTPException(409, f"this dataset is {ds.status}; there is no tokenizer yet")
    d = lab(request).dataset_dir(ds.id) / "tokenizer"
    manifest = _read_json(d / "manifest.json")
    return {"manifest": manifest, "statistics": _read_json(d / "vocab_statistics.json"),
            "trained_on": "training split only (validation and test text never influence the tokenizer)"}


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, request: Request, auth: CurrentAuth, db: DB):
    ds = _owned_dataset(db, auth, dataset_id)
    try:
        lab(request).delete_dataset(db, ds)
    except LabError as e:
        raise _fail(e) from e
    audit(db, request, "dataset_delete", auth.user.id, dataset_id)
