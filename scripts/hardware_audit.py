"""Phase 1: record the machine, verify PyTorch, and measure which model sizes actually fit.

Writes artifacts/reports/hardware_report.json. Re-run it whenever drivers/PyTorch change.
"""
import argparse
import ctypes
import gc
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import torch
from model.architecture import ModelConfig, build_model
from model.common import atomic_write_json, repo_root


def _cim(cls: str, props: str) -> list[dict]:
    if platform.system() != "Windows":
        return []
    cmd = f"Get-CimInstance {cls} | Select-Object {props} | ConvertTo-Json -Compress"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _ram_gb() -> float | None:
    if platform.system() == "Windows":
        class MEM(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
        m = MEM()
        m.l = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.total / 2**30, 1)
    return None


def machine_info() -> dict:
    root = repo_root()
    du = shutil.disk_usage(root)
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cim("Win32_Processor", "Name,NumberOfCores,NumberOfLogicalProcessors"),
        "ram_gb": _ram_gb(),
        "disk_project_drive_free_gb": round(du.free / 2**30, 1),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads_torch": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": p.name,
            "vram_gb": round(p.total_memory / 2**30, 2),
            "compute_capability": f"{p.major}.{p.minor}",
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "multi_processors": p.multi_processor_count,
        }
    return info


def matmul_bench(device: str, dtype: torch.dtype, n: int = 2048, iters: int = 10) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(2):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t) / 1e12  # TFLOPS


CANDIDATES = {
    "small_6M": dict(d_model=256, n_layers=6, n_heads=8, d_ff=1024),
    "base_16M": dict(d_model=384, n_layers=8, n_heads=6, d_ff=1536),
    "medium_34M": dict(d_model=512, n_layers=10, n_heads=8, d_ff=2048),
    "large_88M": dict(d_model=768, n_layers=12, n_heads=12, d_ff=3072),
}


def profile_training(name: str, shape: dict, batch: int, seq: int, steps: int = 4) -> dict:
    cfg = ModelConfig(vocab_size=4096, max_seq_len=seq, dropout=0.1, attention_impl="sdpa", **shape)
    row = {"config": name, "params_M": round(cfg.estimate_params() / 1e6, 1), "batch": batch, "seq": seq}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = opt = x = loss = None
    try:
        model = build_model(cfg, 0).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        x = torch.randint(0, 4096, (batch, seq), device="cuda")
        for i in range(steps + 1):
            if i == 1:
                torch.cuda.synchronize()
                t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, targets=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        row["tokens_per_s"] = round(batch * seq * steps / (time.perf_counter() - t))
        row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except torch.cuda.OutOfMemoryError:
        row["result"] = "OOM"
    finally:
        model = opt = x = loss = None  # drop references so the next candidate starts from a clean GPU
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()

    report = {"machine": machine_info()}
    print(json.dumps(report["machine"], indent=2))
    report["matmul_tflops"] = {"cpu_fp32": round(matmul_bench("cpu", torch.float32, 1024, 5), 3)}
    if torch.cuda.is_available():
        report["matmul_tflops"]["gpu_fp32"] = round(matmul_bench("cuda", torch.float32), 2)
        report["matmul_tflops"]["gpu_bf16"] = round(matmul_bench("cuda", torch.bfloat16), 2)
        rows = [profile_training(n, s, b, args.seq) for n, s in CANDIDATES.items() for b in args.batches]
        report["training_profile_bf16_adamw"] = rows
        print(f"\n{'config':<12}{'params(M)':>10}{'batch':>7}{'peak GB':>9}{'tok/s':>10}")
        for r in rows:
            print(f"{r['config']:<12}{r['params_M']:>10}{r['batch']:>7}{r.get('peak_mem_gb', 'OOM'):>9}{r.get('tokens_per_s', '-'):>10}")
    else:
        print("CUDA not available: CPU-only run. Install a CUDA build of PyTorch to use the GPU.")
    print("matmul TFLOPS:", report["matmul_tflops"])
    out = repo_root() / "artifacts" / "reports" / "hardware_report.json"
    atomic_write_json(out, report)
    print("report ->", out)


if __name__ == "__main__":
    main()
