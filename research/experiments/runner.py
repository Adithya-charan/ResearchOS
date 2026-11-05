"""Run an approved experiment in a sandbox and record everything needed to reproduce and review it.

Code contract: the program reads `config.json` (and any input files) from its working directory, writes artifacts to
`out/`, uses `EXPERIMENT_SEED` for randomness, and prints one final line `RESULT {json}` with its metrics."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from research.experiments.sandbox import Sandbox, SandboxRefused, SandboxResult
from research.experiments.spec import ExperimentSpec

MANIFEST_VERSION = 1
MAX_RUNS = 200
MAX_ROOT_BYTES = 500_000_000


class QuotaExceeded(Exception):
    pass


class NotApproved(Exception):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def dataset_hash(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name].encode("utf-8") + b"\0")
    return h.hexdigest()


def check_quota(root: Path, max_runs: int = MAX_RUNS, max_bytes: int = MAX_ROOT_BYTES) -> None:
    runs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
    if len(runs) >= max_runs:
        raise QuotaExceeded(f"{len(runs)} stored experiment runs (limit {max_runs}); delete old runs first")
    if root.is_dir() and dir_bytes(root) >= max_bytes:
        raise QuotaExceeded(f"experiment storage is full ({max_bytes} bytes); delete old runs first")


def parse_metrics(stdout: str) -> tuple[dict, list[str]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            try:
                m = json.loads(line[7:])
            except json.JSONDecodeError:
                return {}, ["the final RESULT line is not valid JSON; no metrics recorded"]
            if isinstance(m, dict):
                return m, []
            return {}, ["the RESULT line is not a JSON object; no metrics recorded"]
    return {}, ["no RESULT line was printed; no metrics recorded"]


def _versions(packages: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for p in sorted(set(packages)):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _hardware() -> dict:
    import os
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count()}


def manifest_digest(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return sha256_bytes(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def run_experiment(spec: ExperimentSpec, sandbox: Sandbox, root: Path, *, approved_by: str,
                   run_id: str | None = None) -> tuple[Path, dict]:
    """Execute `spec` and return (run_dir, manifest). Refuses without a named approver; the sandbox may refuse too."""
    if not approved_by or not approved_by.strip():
        raise NotApproved("experiments run only after explicit human approval (approved_by is required)")
    root.mkdir(parents=True, exist_ok=True)
    check_quota(root)
    run_id = run_id or f"exp-{uuid.uuid4().hex[:10]}"
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment run {run_id} already exists")
    run_dir.mkdir()
    try:
        (run_dir / "main.py").write_bytes(spec.code.encode("utf-8"))  # bytes: no newline translation, so hashes match
        (run_dir / "config.json").write_text(json.dumps(spec.config, sort_keys=True, indent=2), encoding="utf-8")
        for name, text in spec.files.items():
            (run_dir / name).write_bytes(text.encode("utf-8"))
        (run_dir / "out").mkdir()
        (run_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        started = datetime.now(timezone.utc).isoformat()
        res: SandboxResult = sandbox.run(spec, run_dir)
    except SandboxRefused:
        shutil.rmtree(run_dir, ignore_errors=True)  # a refused run leaves nothing behind
        raise
    (run_dir / "stdout.log").write_text(res.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(res.stderr, encoding="utf-8")

    metrics, warnings = parse_metrics(res.stdout)
    artifacts = [{"path": p.relative_to(run_dir / "out").as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in sorted((run_dir / "out").rglob("*")) if p.is_file()]
    if res.timed_out:
        status = "timed_out"
    elif res.exit_code != 0:
        status = "failed"
    else:
        status = "succeeded"
    if res.truncated:
        warnings.append("output exceeded the size limit and was truncated")
    manifest = {
        "manifest_version": MANIFEST_VERSION, "id": run_id, "name": spec.name, "hypothesis_id": spec.hypothesis_id,
        "status": status, "started_at": started, "approved_by": approved_by,
        "code_sha256": sha256_bytes(spec.code.encode("utf-8")), "dataset_sha256": dataset_hash(spec.files),
        "input_files": sorted(spec.files), "config": spec.config, "seed": spec.seed,
        "dependencies": _versions(spec.packages), "hardware": _hardware(),
        "limits": {"timeout_s": spec.timeout_s, "max_output_bytes": spec.max_output_bytes, "memory_mb": spec.memory_mb,
                   "cpus": spec.cpus},
        "isolation": {"sandbox": sandbox.name, "level": res.isolation, "network_blocked": res.network_blocked,
                      "notes": res.notes},
        "result": {"exit_code": res.exit_code, "timed_out": res.timed_out, "truncated": res.truncated,
                   "duration_s": round(res.duration_s, 3)},
        "resources": {"stdout_bytes": len(res.stdout.encode()), "stderr_bytes": len(res.stderr.encode()),
                      "artifact_bytes": sum(a["bytes"] for a in artifacts), "run_dir_bytes": dir_bytes(run_dir)},
        "metrics": metrics, "artifacts": artifacts, "warnings": warnings,
        "generated_by_model": spec.generated_by_model,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir, manifest


def load_manifest(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))


def verify_manifest(run_dir: Path) -> list[str]:
    """Recompute every hash. Returns problems (empty = the stored run is exactly what the manifest describes)."""
    run_dir = Path(run_dir)
    problems: list[str] = []
    m = load_manifest(run_dir)
    if m.get("manifest_sha256") != manifest_digest(m):
        problems.append("manifest content does not match its recorded hash")
    if sha256_file(run_dir / "main.py") != m["code_sha256"]:
        problems.append("main.py was modified after the run")
    files = {n: (run_dir / n).read_bytes().decode("utf-8") for n in m["input_files"] if (run_dir / n).is_file()}
    if len(files) != len(m["input_files"]) or dataset_hash(files) != m["dataset_sha256"]:
        problems.append("input files were modified or are missing")
    for a in m["artifacts"]:
        p = run_dir / "out" / a["path"]
        if not p.is_file() or sha256_file(p) != a["sha256"]:
            problems.append(f"artifact changed or missing: {a['path']}")
    return problems
