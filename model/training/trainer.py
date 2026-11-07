"""Training loop for the from-scratch Transformer (next-token prediction, cross-entropy)."""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import platform
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from model.architecture.transformer import Transformer
from model.common import atomic_write_json
from model.data.dataset import TokenDataset
from model.evaluation.metrics import evaluate_loss
from model.training.memory import check_budget, estimate_training_memory, hardware_report
from model.training.checkpoint import (
    load_optimizer_state,
    load_weights,
    mark_best,
    resolve_checkpoint,
    save_checkpoint,
)


@dataclass
class TrainConfig:
    run_name: str = "run"
    seed: int = 42
    batch_size: int = 16
    grad_accum_steps: int = 2
    max_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_interval: int = 500
    keep_last: int = 3
    log_interval: int = 25
    mixed_precision: str = "auto"  # auto | bf16 | none
    activation_checkpointing: bool = False
    early_stopping_patience: int = 0  # evaluations without improvement; 0 disables
    device: str = "auto"
    max_nonfinite_steps: int = 5
    max_memory_gb: float = 0.0  # >0: refuse to start if the estimated peak exceeds this; 0 = check against free device memory
    max_epochs: float = 0.0  # >0: also stop after this many passes over the training tokens (0 disables)
    enforce_memory_check: bool = True  # estimate before the first step and refuse an over-budget configuration

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        return cls(**data)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to learning_rate * min_lr_ratio."""
    floor = cfg.learning_rate * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return floor
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return floor + (cfg.learning_rate - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_autocast(device: torch.device, mode: str):
    """Return (context_manager, description). bf16 needs no loss scaling."""
    if mode == "auto":
        mode = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "none"
    if mode == "none":
        return contextlib.nullcontext(), "fp32"
    if mode == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16), "bf16"
    raise ValueError(f"mixed_precision must be auto|bf16|none, got {mode!r}")


def build_optimizer(model: Transformer, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]  # biases, norm gains
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=device.type == "cuda"
    )


class Trainer:
    def __init__(
        self,
        model: Transformer,
        cfg: TrainConfig,
        train_data: TokenDataset,
        val_data: TokenDataset | None,
        run_dir: str | Path,
        meta_base: dict | None = None,
        on_eval: Callable[[int, Transformer, torch.device], dict] | None = None,
        log: Callable[[str], None] = print,
        should_stop: Callable[[], bool] | None = None,
    ):
        self.model, self.cfg = model, cfg
        self.train_data, self.val_data = train_data, val_data
        self.run_dir = Path(run_dir)
        self.meta_base = dict(meta_base or {})
        self.on_eval, self.log = on_eval, log
        self.should_stop = should_stop  # cooperative pause/cancel: polled once per optimizer step, then a checkpoint is saved
        self.device = resolve_device(cfg.device)
        self.history: list[dict] = []
        self.run_record: dict = {}
        self.run_id = str(self.meta_base.get("run_id") or f"run_{uuid.uuid4().hex[:12]}")

    def _now(self) -> str:
        return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    def _write_run_record(self, **fields_) -> None:
        """run.json: machine-readable identity and state of this run, rewritten atomically at start and at every exit."""
        self.run_record.update(fields_)
        atomic_write_json(self.run_dir / "run.json", self.run_record)

    @staticmethod
    def _peak_ram_mb() -> float | None:
        try:
            import psutil

            mi = psutil.Process().memory_info()
            return getattr(mi, "peak_wset", mi.rss) / 2**20  # peak working set on Windows, current RSS elsewhere
        except Exception:  # pragma: no cover
            return None

    def _emit(self, record: dict) -> None:
        self.history.append(record)
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def fit(self, resume: str | Path | None = None, stop_at: int | None = None) -> dict:
        """Train until `max_steps` (or `stop_at`, a pause point that keeps the same LR schedule)."""
        cfg, model, device = self.cfg, self.model, self.device
        limit = cfg.max_steps if stop_at is None else min(cfg.max_steps, stop_at)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.run_dir / "train_config.json", asdict(cfg))
        model.to(device).train()
        model.gradient_checkpointing = cfg.activation_checkpointing
        autocast, precision = make_autocast(device, cfg.mixed_precision)
        seq_len = self.train_data.seq_len
        started_at, t_start = self._now(), time.perf_counter()
        n_params = model.num_parameters()
        epoch_tokens = int(len(self.train_data.tokens)) if hasattr(self.train_data, "tokens") else 0
        est = estimate_training_memory(
            model.cfg, cfg.batch_size, seq_len, mixed_precision=precision != "fp32",
            activation_checkpointing=cfg.activation_checkpointing, device=device.type, params=n_params,
        ) if hasattr(model, "cfg") else None
        self.run_record = {
            "run_id": self.run_id,
            "run_name": cfg.run_name,
            "status": "running",
            "completion_state": None,
            "started_at": started_at,
            "finished_at": None,
            "seed": cfg.seed,
            "parameters": n_params,
            "model_config": model.cfg.to_dict() if hasattr(model, "cfg") else None,
            "train_config": asdict(cfg),
            "optimizer": {"name": "AdamW", "lr": cfg.learning_rate, "betas": [cfg.beta1, cfg.beta2],
                          "weight_decay": cfg.weight_decay, "grad_clip_norm": cfg.grad_clip_norm},
            "tokenizer_fingerprint": self.meta_base.get("tokenizer_fingerprint"),
            "dataset_fingerprint": self.meta_base.get("dataset_fingerprint") or self.meta_base.get("data_fingerprint"),
            "precision": precision,
            "device": str(device),
            "hardware": hardware_report(),
            "memory_estimate": est.to_dict() if est else None,
            "software": {"torch": torch.__version__, "python": platform.python_version()},
        }
        if cfg.enforce_memory_check and est is not None:
            budget = int(cfg.max_memory_gb * 2**30) if cfg.max_memory_gb > 0 else None
            try:
                check_budget(est, budget, label="this training configuration")
            except Exception as exc:
                self._write_run_record(status="failed", completion_state="refused_memory_budget", error=str(exc),
                                       finished_at=self._now())
                raise
        self._write_run_record()
        optimizer = build_optimizer(model, cfg, device)

        step, best_val, bad_evals, tokens_seen = 0, math.inf, 0, 0
        first_loss: float | None = None
        last_loss: float | None = None
        if resume is not None:
            ckpt = resolve_checkpoint(resume)
            meta = load_weights(model, ckpt)
            load_optimizer_state(optimizer, ckpt)
            step = int(meta["step"])
            state = meta.get("trainer", {})
            best_val = state.get("best_val_loss", math.inf)
            bad_evals, tokens_seen = state.get("bad_evals", 0), state.get("tokens_seen", 0)
            self.log(f"resumed from {ckpt} at step {step}")
        self.log(
            f"training {model.num_parameters():,} params on {device} ({precision}); "
            f"{cfg.batch_size}x{cfg.grad_accum_steps} seqs of {seq_len} = "
            f"{cfg.batch_size * cfg.grad_accum_steps * seq_len:,} tokens/step"
        )

        tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * seq_len
        last_eval: dict = {}
        saved_steps: set[int] = set()
        last_ckpt: Path | None = None
        bad_in_a_row, stopped = 0, ("max_steps" if limit == cfg.max_steps else "stop_at")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_mark, steps_since_mark = time.perf_counter(), 0

        def save(at_step: int) -> Path:
            nonlocal last_ckpt
            meta = {
                **self.meta_base,
                "train_config": asdict(cfg),
                "trainer": {"best_val_loss": best_val, "bad_evals": bad_evals, "tokens_seen": tokens_seen},
                "metrics": last_eval,
                "precision": precision,
            }
            last_ckpt = save_checkpoint(self.run_dir, at_step, model, optimizer, meta, cfg.keep_last)
            saved_steps.add(at_step)
            return last_ckpt

        try:
            while step < limit:
                torch.manual_seed(cfg.seed * 1_000_003 + step)  # dropout masks depend only on (seed, step)
                lr = lr_at(step, cfg)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=device)
                for micro in range(cfg.grad_accum_steps):
                    x, y = self.train_data.train_batch(cfg.seed, step, micro, cfg.batch_size)
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with autocast:
                        out = model(x, targets=y)
                    (out.loss / cfg.grad_accum_steps).backward()
                    loss_sum += out.loss.detach() / cfg.grad_accum_steps
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    bad_in_a_row = 0
                else:  # skip the update rather than corrupt the weights
                    optimizer.zero_grad(set_to_none=True)
                    bad_in_a_row += 1
                    self.log(f"step {step + 1}: non-finite gradient norm, update skipped ({bad_in_a_row})")
                    if bad_in_a_row > cfg.max_nonfinite_steps:
                        raise FloatingPointError("too many consecutive non-finite gradient steps")
                step += 1
                steps_since_mark += 1
                tokens_seen += tokens_per_step

                if step % cfg.log_interval == 0 or step == 1:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    rec = {
                        "type": "train",
                        "step": step,
                        "loss": float(loss_sum),
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "tokens_per_sec": steps_since_mark * tokens_per_step / max(now - t_mark, 1e-9),
                        "tokens_seen": tokens_seen,
                    }
                    if device.type == "cuda":
                        rec["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
                    if first_loss is None:
                        first_loss = rec["loss"]
                    last_loss = rec["loss"]
                    self._emit(rec)
                    self.log(
                        f"step {step:>6} | loss {rec['loss']:.4f} | lr {lr:.2e} | gnorm {rec['grad_norm']:.2f} "
                        f"| {rec['tokens_per_sec']:,.0f} tok/s"
                    )
                    t_mark, steps_since_mark = now, 0

                if self.val_data is not None and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
                    ev = evaluate_loss(model, self.val_data, cfg.batch_size, cfg.eval_batches, device, autocast)
                    last_eval = {"val_loss": ev["loss"], "val_perplexity": ev["perplexity"], "step": step}
                    rec = {"type": "eval", "step": step, **last_eval}
                    if self.on_eval is not None:
                        rec["extra"] = self.on_eval(step, model, device)
                        model.train()
                    self._emit(rec)
                    self.log(f"  eval @ {step}: val_loss {ev['loss']:.4f} ppl {ev['perplexity']:.2f}")
                    if ev["loss"] < best_val:
                        best_val, bad_evals = ev["loss"], 0
                        mark_best(self.run_dir, save(step), best_val)
                    else:
                        bad_evals += 1
                        if cfg.early_stopping_patience and bad_evals >= cfg.early_stopping_patience:
                            stopped = "early_stopping"
                            self.log(f"early stopping at step {step}")
                            break
                    t_mark, steps_since_mark = time.perf_counter(), 0
                if step % cfg.checkpoint_interval == 0 and step not in saved_steps:
                    save(step)
                if cfg.max_epochs > 0 and epoch_tokens and tokens_seen / epoch_tokens >= cfg.max_epochs:
                    stopped = "max_epochs"
                    self.log(f"reached {cfg.max_epochs} epochs at step {step}")
                    break
                if self.should_stop is not None and self.should_stop():
                    stopped = "requested"
                    self.log(f"stop requested - checkpointing at step {step}")
                    break
        except KeyboardInterrupt:
            stopped = "interrupted"
            self.log("interrupted - saving checkpoint")
        except Exception as exc:
            # The last periodic checkpoint is still on disk and resumable; record the failure instead of losing it.
            self._write_run_record(
                status="failed", completion_state="failed", finished_at=self._now(), steps=step,
                duration_seconds=time.perf_counter() - t_start, error=f"{type(exc).__name__}: {exc}"[:2000],
                traceback_tail=traceback.format_exc()[-2000:], last_checkpoint=str(last_ckpt) if last_ckpt else None,
                tokens_seen=tokens_seen,
            )
            raise
        if step not in saved_steps:
            save(step)
        completion = {
            "max_steps": "completed", "early_stopping": "completed", "max_epochs": "completed",
            "stop_at": "paused", "requested": "stopped_by_request", "interrupted": "interrupted",
        }[stopped]
        duration = time.perf_counter() - t_start
        summary = {
            "run_id": self.run_id,
            "completion_state": completion,
            "started_at": started_at,
            "finished_at": self._now(),
            "duration_seconds": duration,
            "initial_loss": first_loss,
            "final_loss": last_loss,
            "steps": step,
            "stopped": stopped,
            "best_val_loss": None if math.isinf(best_val) else best_val,
            "last_eval": last_eval,
            "tokens_seen": tokens_seen,
            "last_checkpoint": str(last_ckpt),
            "precision": precision,
        }
        if device.type == "cuda":
            summary["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        summary["peak_ram_mb"] = self._peak_ram_mb()
        summary["tokens_per_sec_overall"] = tokens_seen / max(duration, 1e-9)
        summary["epochs"] = (tokens_seen / epoch_tokens) if epoch_tokens else None
        atomic_write_json(self.run_dir / "summary.json", summary)
        self._write_run_record(
            status="finished" if completion == "completed" else completion, completion_state=completion,
            finished_at=summary["finished_at"], steps=step, duration_seconds=duration, initial_loss=first_loss,
            final_loss=last_loss, best_val_loss=summary["best_val_loss"], tokens_seen=tokens_seen,
            last_checkpoint=summary["last_checkpoint"], peak_gpu_mem_mb=summary.get("peak_gpu_mem_mb"),
            peak_ram_mb=summary["peak_ram_mb"], tokens_per_sec_overall=summary["tokens_per_sec_overall"],
        )
        return summary
