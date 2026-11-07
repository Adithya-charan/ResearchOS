"""Hardware-oriented model/training profiles.

A profile is a *starting point* (architecture + training defaults) sized for a class of machine. Whether it really fits the
machine at hand is decided by `check_profile`, which estimates memory against what the device has free **right now**, and by
`measure_profile`, which actually runs forward/backward/optimizer steps and reports the measured peak. A profile is never
described as compatible with hardware it has not been measured on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from model.architecture.config import ModelConfig
from model.architecture.transformer import Transformer
from model.training.memory import (
    MemoryEstimate,
    available_budget,
    estimate_training_memory,
    hardware_report,
)
from model.training.trainer import TrainConfig, build_optimizer, make_autocast, resolve_device


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    target_hardware: str
    model: dict
    train: dict
    min_memory_gb: float  # planning figure from the estimator, not a guarantee
    device_hint: str  # "cpu" | "cuda"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def model_config(self, vocab_size: int | None = None) -> ModelConfig:
        d = dict(self.model)
        if vocab_size is not None:
            d["vocab_size"] = vocab_size
        return ModelConfig.from_dict(d)

    def train_config(self, **overrides) -> TrainConfig:
        return TrainConfig.from_dict({**self.train, **overrides})


PROFILES: dict[str, Profile] = {
    "CPU_DEBUG": Profile(
        name="CPU_DEBUG",
        description="Tiny model for debugging, CI-sized smoke runs and machines without a GPU.",
        target_hardware="any CPU, < 1 GiB RAM",
        model=dict(vocab_size=512, max_seq_len=64, d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0),
        train=dict(run_name="cpu_debug", batch_size=4, grad_accum_steps=1, max_steps=60, learning_rate=3e-3,
                   warmup_steps=5, eval_interval=20, eval_batches=5, checkpoint_interval=30, log_interval=10,
                   mixed_precision="none", device="cpu"),
        min_memory_gb=0.1,
        device_hint="cpu",
        notes=("Not a useful language model: exists to exercise the pipeline quickly.",),
    ),
    "CONSUMER_SMALL": Profile(
        name="CONSUMER_SMALL",
        description="~5M-parameter model (the original blueprint starter) for an entry-level GPU or a patient CPU.",
        target_hardware="4-6 GiB GPU (bf16) or 8+ GiB RAM CPU",
        model=dict(vocab_size=4096, max_seq_len=256, d_model=256, n_layers=6, n_heads=8, d_ff=1024, dropout=0.1,
                   attention_impl="sdpa"),
        train=dict(run_name="consumer_small", batch_size=16, grad_accum_steps=2, max_steps=3000, learning_rate=6e-4,
                   warmup_steps=150, eval_interval=250, eval_batches=40, checkpoint_interval=500, log_interval=25,
                   early_stopping_patience=6),
        min_memory_gb=1.5,
        device_hint="cuda",
        notes=("Trained only on the synthetic template corpus unless a real licensed corpus is supplied.",),
    ),
    "CONSUMER_MEDIUM": Profile(
        name="CONSUMER_MEDIUM",
        description="~30M-parameter modern-style decoder (RoPE, RMSNorm, SwiGLU) sized for a 6 GiB consumer GPU.",
        target_hardware="6+ GiB GPU with bf16 support",
        model=dict(vocab_size=8192, max_seq_len=512, d_model=512, n_layers=8, n_heads=8, d_ff=1536, dropout=0.05,
                   normalization="rmsnorm", positional_encoding="rope", mlp="swiglu", bias=False, attention_impl="sdpa"),
        train=dict(run_name="consumer_medium", batch_size=4, grad_accum_steps=8, max_steps=3000, learning_rate=4e-4,
                   warmup_steps=200, eval_interval=250, eval_batches=20, checkpoint_interval=500, log_interval=25,
                   activation_checkpointing=True, early_stopping_patience=6),
        min_memory_gb=2.5,
        device_hint="cuda",
        notes=("Needs far more real text than the bundled synthetic corpus to be meaningful.",
               "Fit on a given machine is only claimed after measure_profile() has run there."),
    ),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name.upper()]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}") from None


def profile_estimate(profile: Profile, device: str | None = None, vocab_size: int | None = None) -> MemoryEstimate:
    device = device or ("cuda" if torch.cuda.is_available() and profile.device_hint == "cuda" else "cpu")
    t = profile.train_config()
    mp = t.mixed_precision != "none" and device.startswith("cuda")
    return estimate_training_memory(
        profile.model_config(vocab_size), t.batch_size, mixed_precision=mp,
        activation_checkpointing=t.activation_checkpointing, device=device,
    )


def check_profile(profile: Profile, device: str | None = None) -> dict:
    """Estimated (not measured) fit against the device's free memory right now."""
    est = profile_estimate(profile, device)
    budget = available_budget(est.device)
    return {
        "profile": profile.name,
        "device": est.device,
        "estimated_bytes": est.total,
        "budget_bytes": budget,
        "fits_by_estimate": bool(budget and est.total <= budget),
        "basis": "estimate",
        "estimate": est.to_dict(),
    }


def measure_profile(profile: Profile, device: str | None = None, steps: int = 3) -> dict:
    """Run real optimizer steps at the profile's batch size and report the measured peak. This is the only function that may
    be used to claim a profile fits this hardware. Returns `{"ok": False, "error": ...}` on out-of-memory instead of raising."""
    est = profile_estimate(profile, device)
    dev = resolve_device(est.device)
    t = profile.train_config()
    result: dict = {"profile": profile.name, "device": str(dev), "basis": "measurement", "hardware": hardware_report(),
                    "estimated_bytes": est.total}
    model = None
    try:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        cfg = profile.model_config()
        t0 = time.perf_counter()
        model = Transformer(cfg).to(dev).train()
        model.gradient_checkpointing = t.activation_checkpointing
        result["init_seconds"] = time.perf_counter() - t0
        result["parameters"] = model.num_parameters()
        opt = build_optimizer(model, t, dev)
        autocast, precision = make_autocast(dev, t.mixed_precision)
        result["precision"] = precision
        x = torch.randint(6, cfg.vocab_size, (t.batch_size, cfg.max_seq_len), device=dev)
        y = torch.roll(x, -1, dims=1)
        times = []
        for _ in range(steps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            s = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with autocast:
                out = model(x, targets=y)
            out.loss.backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - s)
        steady = times[1:] or times
        sec = sum(steady) / len(steady)
        result.update(
            ok=True,
            step_seconds=sec,
            tokens_per_second=t.batch_size * cfg.max_seq_len / sec,
            micro_batch=t.batch_size,
            seq_len=cfg.max_seq_len,
            final_loss_finite=bool(torch.isfinite(out.loss).item()),
        )
        if dev.type == "cuda":
            result["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
            result["estimate_over_measured"] = est.total / max(result["peak_gpu_bytes"], 1)
    except torch.cuda.OutOfMemoryError as e:
        result.update(ok=False, error=f"out of memory: {str(e)[:200]}")
    finally:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return result
