"""Atomic, verifiable checkpoints: safetensors weights + optimizer state + JSON metadata."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from model.architecture.config import ModelConfig
from model.architecture.transformer import Transformer
from model.common import atomic_write_json, sha256_file

CHECKPOINT_FORMAT = 1
MODEL_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.pt"
META_FILE = "meta.json"


class CheckpointError(RuntimeError):
    pass


def _state_for_save(model: Transformer) -> dict[str, torch.Tensor]:
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if model.cfg.tie_embeddings:  # shares storage with tok_emb.weight; rebuilt on load
        sd.pop("lm_head.weight", None)
    return sd


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model: Transformer,
    optimizer: torch.optim.Optimizer | None,
    meta: dict,
    keep_last: int = 3,
) -> Path:
    """Write run_dir/step_XXXXXXX atomically: build in a temp dir, then rename into place."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / f"step_{step:07d}"
    tmp = run_dir / f".tmp_step_{step:07d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    save_file(_state_for_save(model), str(tmp / MODEL_FILE), metadata={"format": "pt"})
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / OPTIMIZER_FILE)
    meta = dict(meta)
    meta.update(
        checkpoint_format=CHECKPOINT_FORMAT,
        step=step,
        model_config=model.cfg.to_dict(),
        model_sha256=sha256_file(tmp / MODEL_FILE),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        torch_version=torch.__version__,
    )
    (tmp / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    for f in tmp.iterdir():  # flush file contents to disk before the rename makes them visible
        with open(f, "r+b") as fh:  # Windows requires a writable handle for fsync
            os.fsync(fh.fileno())
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    atomic_write_json(run_dir / "latest.json", {"path": final.name, "step": step})
    _prune(run_dir, keep_last)
    return final


def mark_best(run_dir: str | Path, ckpt_dir: Path, val_loss: float) -> None:
    atomic_write_json(Path(run_dir) / "best.json", {"path": ckpt_dir.name, "val_loss": val_loss})


def _protected(run_dir: Path) -> set[str]:
    best = run_dir / "best.json"
    return {json.loads(best.read_text())["path"]} if best.exists() else set()


def _prune(run_dir: Path, keep_last: int) -> None:
    ckpts = sorted(p for p in run_dir.glob("step_*") if p.is_dir())
    protected = _protected(run_dir)
    for p in ckpts[: max(0, len(ckpts) - keep_last)]:
        if p.name not in protected:
            shutil.rmtree(p, ignore_errors=True)


def resolve_checkpoint(path: str | Path, prefer: str = "latest") -> Path:
    """Accept a checkpoint dir, or a run dir (uses best.json / latest.json)."""
    p = Path(path)
    if (p / META_FILE).exists():
        return p
    pointer = p / f"{prefer}.json"
    if not pointer.exists():
        pointer = p / ("latest.json" if prefer == "best" else "best.json")
    if not pointer.exists():
        raise CheckpointError(f"no checkpoint found at {p}")
    return p / json.loads(pointer.read_text())["path"]


def read_meta(ckpt_dir: str | Path) -> dict:
    return json.loads((Path(ckpt_dir) / META_FILE).read_text(encoding="utf-8"))


def load_weights(model: Transformer, ckpt_dir: str | Path, verify: bool = True) -> dict:
    """Load weights from a checkpoint into `model` (same architecture). Returns the metadata."""
    ckpt_dir = Path(ckpt_dir)
    meta = read_meta(ckpt_dir)
    weights = ckpt_dir / MODEL_FILE
    if verify and sha256_file(weights) != meta["model_sha256"]:
        raise CheckpointError(f"{weights} is corrupt: sha256 does not match meta.json")
    if ModelConfig.from_dict(meta["model_config"]) != model.cfg:
        raise CheckpointError("checkpoint architecture does not match the model being loaded into")
    result = model.load_state_dict(load_file(str(weights)), strict=False)
    allowed_missing = {"lm_head.weight"} if model.cfg.tie_embeddings else set()
    if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
        raise CheckpointError(f"state mismatch: missing={result.missing_keys} unexpected={result.unexpected_keys}")
    return meta


def load_model(ckpt_dir: str | Path, device: str | torch.device = "cpu", verify: bool = True) -> tuple[Transformer, dict]:
    ckpt_dir = resolve_checkpoint(ckpt_dir)
    meta = read_meta(ckpt_dir)
    model = Transformer(ModelConfig.from_dict(meta["model_config"]))
    load_weights(model, ckpt_dir, verify=verify)
    return model.to(device).eval(), meta


def load_optimizer_state(optimizer: torch.optim.Optimizer, ckpt_dir: str | Path) -> None:
    path = Path(ckpt_dir) / OPTIMIZER_FILE
    if not path.exists():
        raise CheckpointError(f"{path} missing; cannot resume optimizer state")
    optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
