"""Measure the three hardware profiles on THIS machine and record the figures to artifacts/evaluation/model_hardware.json.

Everything written is measured here, now: parameter count, initialisation time, training step time and throughput at the profile's
micro-batch, peak GPU memory, inference latency with and without the KV cache, and checkpoint size. Inference speed is measured on a
freshly initialised model of the profile's architecture: speed depends on shapes, not on the learned weights, and the file says so.
A profile that does not fit is recorded as such (with the error), never omitted.

    python scripts/bench_model.py                  # all profiles on the best available device
    python scripts/bench_model.py --device cpu     # CPU only
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from model.architecture.transformer import Transformer  # noqa: E402
from model.common import atomic_write_json  # noqa: E402
from model.evaluation.suite import benchmark_inference  # noqa: E402
from model.profiles import PROFILES, measure_profile  # noqa: E402
from model.training.memory import hardware_report  # noqa: E402


def rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def checkpoint_bytes(model: Transformer) -> int:
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in _unique(model).items()}, str(p))
        return p.stat().st_size


def _unique(model: Transformer) -> dict:
    """State dict without tied duplicates (safetensors refuses shared storage)."""
    seen, out = set(), {}
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        out[k] = v
    return out


def bench_inference(profile, device: str) -> dict:
    dev = torch.device(device)
    model = Transformer(profile.model_config()).to(dev).eval()
    try:
        res = benchmark_inference(model, dev)
        res["basis"] = "measured on freshly initialised weights of this architecture (speed does not depend on learned weights)"
        res["checkpoint_bytes_fp32"] = checkpoint_bytes(model)
        return res
    except torch.cuda.OutOfMemoryError as e:
        return {"ok": False, "error": f"out of memory: {str(e)[:160]}"}
    finally:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cpu | cuda (default: each profile's preferred device if available)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "evaluation" / "model_hardware.json"))
    args = ap.parse_args()
    report = {"status": "measured", "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hardware": hardware_report(), "profiles": {},
              "notes": ["Training figures use random token batches at the profile's micro-batch size: they measure memory and throughput, not learning.",
                        "A profile is only said to fit this machine where `ok` is true here."]}
    for name, p in PROFILES.items():
        devices = [args.device] if args.device else (["cuda", "cpu"] if torch.cuda.is_available() and p.device_hint == "cuda" else ["cpu"])
        report["profiles"][name] = {}
        for dev in devices:
            if name != "CPU_DEBUG" and dev == "cpu" and args.device is None:
                continue  # medium/small on CPU are not a supported planning target; request with --device cpu if wanted
            print(f"[{name}] {dev}: training step benchmark", flush=True)
            train = measure_profile(p, device=dev, steps=args.steps)
            train.pop("hardware", None)
            print(f"[{name}] {dev}: inference benchmark", flush=True)
            inf = bench_inference(p, dev) if train.get("ok") else {"skipped": "training step did not fit"}
            report["profiles"][name][dev] = {"training": train, "inference": inf, "process_rss_mb_after": rss_mb()}
            print(json.dumps({k: train.get(k) for k in ("ok", "parameters", "tokens_per_second", "peak_gpu_bytes", "error")}), flush=True)
    atomic_write_json(Path(args.out), report)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
