"""Memory estimation and budget enforcement for training, plus a hardware report.

The estimate is deliberately conservative and is a *planning* number, not a measurement. `model.profiles` and
`scripts/bench_model.py` record the real peak next to the estimate so the error is visible; nothing here guarantees that a
configuration that passes the check will fit (other processes, fragmentation and driver overhead are not modelled).
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass

import torch

from model.architecture.config import ModelConfig

GIB = 2**30
CUDA_CONTEXT_OVERHEAD = int(0.5 * GIB)  # driver + cuBLAS workspaces; measured ~0.3-0.5 GiB on the development GPU
SAFETY_FACTOR = 1.25


class MemoryBudgetError(RuntimeError):
    """The requested configuration is estimated to need more memory than the budget allows."""


@dataclass
class MemoryEstimate:
    params: int
    weights: int
    gradients: int
    optimizer_state: int
    activations: int
    logits: int
    overhead: int
    total: int  # includes SAFETY_FACTOR on the dynamic parts
    batch_size: int
    seq_len: int
    mixed_precision: bool
    activation_checkpointing: bool
    device: str

    def to_dict(self) -> dict:
        return asdict(self)

    def human(self) -> str:
        return (
            f"{self.total / GIB:.2f} GiB (weights {self.weights / GIB:.2f}, grads {self.gradients / GIB:.2f}, "
            f"optimizer {self.optimizer_state / GIB:.2f}, activations {self.activations / GIB:.2f}, "
            f"logits {self.logits / GIB:.2f}, overhead {self.overhead / GIB:.2f})"
        )


def estimate_training_memory(
    cfg: ModelConfig,
    batch_size: int,
    seq_len: int | None = None,
    *,
    mixed_precision: bool = False,
    activation_checkpointing: bool = False,
    device: str = "cpu",
    params: int | None = None,
) -> MemoryEstimate:
    """Estimate peak training memory for one micro-batch (gradient accumulation does not add memory).

    Weights, gradients and AdamW's two moments are fp32 (4+4+8 bytes per parameter) because autocast keeps master weights
    in fp32. Activation bytes per token per layer follow the standard transformer accounting (~34*d bytes for fp16/bf16 with
    a fused attention kernel, doubled when fp32), plus the explicit attention matrices when `attention_impl="manual"`.
    """
    seq = seq_len or cfg.max_seq_len
    n = params if params is not None else cfg.estimate_params()
    act_bytes = 2 if mixed_precision else 4
    tokens = batch_size * seq
    mlp_factor = 1.35 if cfg.mlp == "swiglu" else 1.0
    per_layer = tokens * cfg.d_model * 17 * act_bytes * mlp_factor  # 17 = 34 bytes/(2 bytes) per hidden-state element
    if cfg.attention_impl == "manual":
        per_layer += batch_size * cfg.n_heads * seq * seq * act_bytes * 3  # scores, softmax, dropout mask
    if activation_checkpointing:
        # keep each block's input only, then recompute one block at a time
        activations = cfg.n_layers * tokens * cfg.d_model * act_bytes + per_layer
    else:
        activations = cfg.n_layers * per_layer
    # lm_loss casts logits to fp32 (4B) and the backward pass holds a gradient of the same size; the low-precision copy adds 2B
    logits = tokens * cfg.vocab_size * (4 + 4 + (2 if mixed_precision else 0))
    weights, grads, opt = n * 4, n * 4, n * 8
    overhead = CUDA_CONTEXT_OVERHEAD if device.startswith("cuda") else 0
    total = int((weights + grads + opt + activations + logits) * SAFETY_FACTOR) + overhead
    return MemoryEstimate(
        params=n, weights=weights, gradients=grads, optimizer_state=opt, activations=int(activations), logits=logits,
        overhead=overhead, total=total, batch_size=batch_size, seq_len=seq, mixed_precision=mixed_precision,
        activation_checkpointing=activation_checkpointing, device=device,
    )


def hardware_report() -> dict:
    """What this machine actually has right now (no assumptions)."""
    info: dict = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_bytes"] = vm.total
        info["ram_available_bytes"] = vm.available
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        info["ram_total_bytes"] = info["ram_available_bytes"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.update(
            gpu_name=props.name, gpu_total_bytes=int(total), gpu_free_bytes=int(free),
            gpu_compute_capability=f"{props.major}.{props.minor}", bf16_supported=bool(torch.cuda.is_bf16_supported()),
            cuda_version=torch.version.cuda,
        )
    return info


def available_budget(device: str, headroom: float = 0.9) -> int:
    """Bytes a training process may plan to use on `device` (free memory times `headroom`)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        return int(free * headroom)
    try:
        import psutil

        return int(psutil.virtual_memory().available * headroom)
    except ImportError:  # pragma: no cover
        return 0


def check_budget(est: MemoryEstimate, budget_bytes: int | None = None, *, label: str = "training") -> None:
    """Raise `MemoryBudgetError` if the estimate exceeds the budget (default: what the device has free right now)."""
    limit = budget_bytes if budget_bytes is not None else available_budget(est.device)
    if limit and est.total > limit:
        raise MemoryBudgetError(
            f"{label} needs an estimated {est.total / GIB:.2f} GiB but only {limit / GIB:.2f} GiB is allowed on "
            f"{est.device}; reduce batch_size/max_seq_len/model size, enable activation_checkpointing or mixed precision. "
            f"Breakdown: {est.human()}"
        )


def max_safe_batch_size(
    cfg: ModelConfig, budget_bytes: int, seq_len: int | None = None, *, upper: int = 256, **kw
) -> int:
    """Largest micro-batch whose estimate fits `budget_bytes` (0 if even batch 1 does not). Never suggests more than `upper`."""
    best = 0
    lo, hi = 1, upper
    params = kw.pop("params", None) or cfg.estimate_params()
    while lo <= hi:
        mid = (lo + hi) // 2
        if estimate_training_memory(cfg, mid, seq_len, params=params, **kw).total <= budget_bytes:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best
