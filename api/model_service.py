"""Model management: the checkpoint registry, training jobs (each an OS process driving scripts/train_model.py) and local inference.

Nothing here talks to an external model or service. A training job can only start from an operator-controlled config file,
may only override a short list of numeric hyper-parameters, and writes only inside its own job folder."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import yaml

from api.config import ROOT, Settings
from api.db import utcnow
from api.models import TrainingJob

log = logging.getLogger("researchos.models")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
SYSTEM_PREFIX = "sys_"
ACTIVE = ("running", "pausing", "cancelling")
RESUMABLE = ("paused", "interrupted")
FINISHED = ("succeeded", "failed", "cancelled", "paused", "interrupted")
OVERRIDABLE = {"max_steps", "learning_rate", "batch_size", "grad_accum_steps", "seed", "eval_interval", "checkpoint_interval",
               "warmup_steps"}
NOTICE = ("Experimental small language model trained from random initialisation on a limited corpus. Its output is not "
          "evidence: it is unverified, may be fluent but wrong, and must not be cited as a source.")


class ModelNotFound(Exception):
    pass


class JobRefused(Exception):
    """A request the service will not carry out (surfaced to the client as 409/422/429/503 by the route)."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ checkpoint inspection (pure file reads)
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def list_checkpoints(run_dir: Path) -> list[dict]:
    best, latest = _read_json(run_dir / "best.json") or {}, _read_json(run_dir / "latest.json") or {}
    out = []
    for d in sorted(p for p in run_dir.glob("step_*") if p.is_dir()):
        meta = _read_json(d / "meta.json")
        if meta is None:
            continue
        out.append({"name": d.name, "step": meta.get("step"), "val_loss": (meta.get("metrics") or {}).get("val_loss"),
                    "model_sha256": meta.get("model_sha256"), "created_at": meta.get("created_at"), "precision": meta.get("precision"),
                    "size_bytes": _dir_size(d), "is_best": best.get("path") == d.name, "is_latest": latest.get("path") == d.name})
    return out


def pick_checkpoint(run_dir: Path, prefer: str) -> Path:
    for pointer in (f"{prefer}.json", "latest.json", "best.json"):
        data = _read_json(run_dir / pointer)
        if data and (run_dir / data["path"] / "meta.json").is_file():
            return run_dir / data["path"]
    raise ModelNotFound("this model has no checkpoint yet")


def describe(model_id: str, run_dir: Path, kind: str, extra: dict | None = None) -> dict:
    cks = list_checkpoints(run_dir)
    info = {"id": model_id, "kind": kind, "checkpoints": len(cks), "notice": NOTICE, **(extra or {})}
    if cks:
        try:
            meta = _read_json(pick_checkpoint(run_dir, "best") / "meta.json") or {}
        except ModelNotFound:
            meta = {}
        cfg = meta.get("model_config") or {}
        info.update({
            "step": meta.get("step"), "best_val_loss": next((c["val_loss"] for c in cks if c["is_best"]), None),
            "architecture": {k: cfg.get(k) for k in ("vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads", "d_ff",
                                                     "normalization", "positional_encoding", "mlp")},
            "lineage_root": (meta.get("lineage") or {}).get("root"), "trained_on_tokens": (meta.get("data") or {}).get("tokens"),
            "tokenizer_fingerprint": (meta.get("tokenizer") or {}).get("fingerprint"),
            "description": meta.get("description", ""), "disclaimer": meta.get("disclaimer", NOTICE)})
    return info


def read_metrics(run_dir: Path, after_step: int = 0, limit: int = 500) -> dict:
    train, evals = [], []
    path = run_dir / "metrics.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a half-written last line while training is running
            if not isinstance(rec, dict) or int(rec.get("step", 0)) <= after_step:
                continue
            (train if rec.get("type") == "train" else evals).append(rec)
    return {"train": train[:limit], "eval": evals[:limit], "truncated": len(train) > limit or len(evals) > limit}


def read_samples(run_dir: Path, limit: int = 20) -> list[dict]:
    path = run_dir / "samples.jsonl"
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out.append({"step": rec.get("step"), "repetition_3gram": rec.get("repetition_3gram"),
                        "text": rec.get("text"), "label": "model output - unverified sample generated during training"})
    return out


def verify_checkpoint(ckpt: Path) -> dict:
    from model.common import sha256_file
    meta = _read_json(ckpt / "meta.json")
    if meta is None:
        return {"ok": False, "problem": "meta.json is missing or unreadable"}
    weights = ckpt / "model.safetensors"
    if not weights.is_file():
        return {"ok": False, "problem": "model.safetensors is missing"}
    actual = sha256_file(weights)
    if actual != meta.get("model_sha256"):
        return {"ok": False, "problem": "model.safetensors does not match the hash recorded when it was saved", "actual": actual}
    from model.training.lineage import LineageError, assert_random_root
    try:
        assert_random_root(meta)
    except LineageError as e:
        return {"ok": False, "problem": str(e), "model_sha256": actual}
    return {"ok": True, "model_sha256": actual, "step": meta.get("step"), "lineage_root": meta["lineage"]["root"]}


def tokenizer_dir_for(settings: Settings, run_dir: Path) -> Path:
    """The tokenizer a model must be served with. Operator models use the operator tokenizer; a model trained on a user dataset
    uses that dataset's own tokenizer (found only inside the managed datasets folder, never from a path in a request)."""
    try:
        ck = pick_checkpoint(run_dir, "best")
    except ModelNotFound:
        return settings.tokenizer_dir
    processed = ((_read_json(ck / "meta.json") or {}).get("data") or {}).get("processed_dir")
    if processed:
        root = (settings.data_dir / settings.datasets_dir_name).resolve()
        p = Path(processed).resolve()
        if root in p.parents and (p.parent / "tokenizer" / "merges.txt").is_file():
            return p.parent / "tokenizer"
    return settings.tokenizer_dir


# ------------------------------------------------------------------ registry
class ModelRegistry:
    def __init__(self, settings: Settings, training_root: Path):
        self.settings, self.training_root = settings, training_root

    def system_models(self) -> list[tuple[str, Path]]:
        root = self.settings.models_dir
        if not root.is_dir():
            return []
        return [(SYSTEM_PREFIX + d.name, d) for d in sorted(root.iterdir())
                if d.is_dir() and NAME_RE.match(d.name) and any(d.glob("step_*"))]

    def system_dir(self, model_id: str) -> Path:
        name = model_id[len(SYSTEM_PREFIX):]
        d = self.settings.models_dir / name
        if not NAME_RE.match(name) or not d.is_dir():
            raise ModelNotFound("model not found")
        return d

    def job_run_dir(self, job_id: str) -> Path:
        return self.training_root / job_id / "run"


# ------------------------------------------------------------------ training jobs
class TrainingService:
    def __init__(self, settings: Settings, sessionmaker, registry: ModelRegistry):
        self.settings, self.sessions, self.registry = settings, sessionmaker, registry
        self.root = registry.training_root
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()

    # -- configuration -------------------------------------------------
    def base_configs(self) -> list[dict]:
        out = []
        d = self.settings.train_configs_dir
        for p in sorted(d.glob("*.yaml")) if d.is_dir() else []:
            if not NAME_RE.match(p.stem):
                continue
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            t, m = raw.get("training", {}), raw.get("model", {})
            out.append({"name": p.stem, "description": " ".join(str(raw.get("description", "")).split()),
                        "model": {k: m.get(k) for k in ("max_seq_len", "d_model", "n_layers", "n_heads", "d_ff")},
                        "defaults": {k: t.get(k) for k in sorted(OVERRIDABLE)}, "device": t.get("device", "auto")})
        return out

    def build_config(self, base: str, overrides: dict, job_dir: Path, dataset_dir: Path | None = None, dataset_id: str | None = None) -> dict:
        path = self.settings.train_configs_dir / f"{base}.yaml"
        if not NAME_RE.match(base) or not path.is_file():
            raise JobRefused(f"unknown base config '{base}'", 422)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        training = dict(raw["training"])
        for key, value in overrides.items():
            if key not in OVERRIDABLE:
                raise JobRefused(f"'{key}' cannot be overridden", 422)
            training[key] = value
        s = self.settings
        if int(training["max_steps"]) > s.max_train_steps:
            raise JobRefused(f"max_steps is limited to {s.max_train_steps} on this server", 422)
        if not 0 < float(training["learning_rate"]) <= 0.1:
            raise JobRefused("learning_rate must be in (0, 0.1]", 422)
        if int(training["batch_size"]) > 64 or int(training["grad_accum_steps"]) > 16:
            raise JobRefused("batch_size is limited to 64 and grad_accum_steps to 16", 422)
        training["run_name"] = "run"
        raw["training"] = training
        data = dict(raw.get("data", {}))
        data["checkpoint_root"] = str(job_dir)  # checkpoints land inside the job folder, never next to operator checkpoints
        if dataset_dir is not None:  # a prepared dataset of the caller's own: its processed data and ITS tokenizer
            meta = _read_json(dataset_dir / "processed" / "meta.json") or {}
            seq = int(raw["model"]["max_seq_len"])
            toks = meta.get("tokens") or {}
            if toks.get("train", 0) < seq + 2 or toks.get("val", 0) < seq + 2:
                raise JobRefused(f"this dataset is too small for a context of {seq} tokens (train {toks.get('train', 0)}, "
                                 f"validation {toks.get('val', 0)}); add documents or choose a config with a shorter context", 422)
            data["processed_dir"] = str(dataset_dir / "processed")
            data["tokenizer_dir"] = str(dataset_dir / "tokenizer")
        raw["data"] = data
        try:  # validate exactly as the trainer will
            from model.training.run_config import load_run_config
            tmp = job_dir / ".validate.yaml"
            tmp.write_text(yaml.safe_dump(raw), encoding="utf-8")
            try:
                load_run_config(tmp, vocab_size_from_tokenizer=1024)
            finally:
                tmp.unlink(missing_ok=True)
        except (ValueError, TypeError, KeyError) as e:
            raise JobRefused(f"invalid training configuration: {e}", 422) from e
        (job_dir / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
        if dataset_id:
            training = {**training, "dataset_id": dataset_id}
        return training

    # -- process control -----------------------------------------------
    def running_count(self) -> int:
        with self._lock:
            return sum(1 for p in self._procs.values() if p.poll() is None)

    def start_process(self, job_id: str, resume: bool) -> None:
        with self._lock:
            if self.running_count() >= self.settings.max_running_training_jobs:
                raise JobRefused("the training capacity of this server is in use; wait for a running job to finish", 429)
            job_dir = self.root / job_id
            run_dir = job_dir / "run"
            (job_dir / "STOP").unlink(missing_ok=True)
            (run_dir / "summary.json").unlink(missing_ok=True)  # never mistake an earlier segment's summary for this one's
            cmd = [sys.executable, "-u", str(ROOT / "scripts" / "train_model.py"), "--config", str(job_dir / "config.yaml"),
                   "--stop-file", str(job_dir / "STOP")]
            if resume:
                cmd += ["--resume", str(run_dir)]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            log_file = open(job_dir / "train.log", "ab")
            try:
                proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        creationflags=flags)
            finally:
                log_file.close()
            self._procs[job_id] = proc
        threading.Thread(target=self._watch, args=(job_id, proc), daemon=True, name=f"train-{job_id}").start()

    def request_stop(self, job_id: str) -> None:
        (self.root / job_id / "STOP").write_text("stop", encoding="utf-8")
        proc = self._procs.get(job_id)
        if proc is not None:
            t = threading.Timer(self.settings.train_stop_grace_s, lambda: proc.poll() is None and proc.kill())
            t.daemon = True
            t.start()

    def stop_all(self, wait_s: float = 15.0) -> None:
        with self._lock:
            procs = dict(self._procs)
        for job_id, p in procs.items():
            if p.poll() is None:
                (self.root / job_id / "STOP").write_text("stop", encoding="utf-8")
        deadline = time.time() + wait_s
        for p in procs.values():
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()

    def is_alive(self, job_id: str) -> bool:
        p = self._procs.get(job_id)
        return p is not None and p.poll() is None

    def _tail(self, job_id: str, n: int = 25, limit: int = 3000) -> str:
        p = self.root / job_id / "train.log"
        if not p.is_file():
            return ""
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])[-limit:]

    def _watch(self, job_id: str, proc: subprocess.Popen) -> None:
        code = proc.wait()
        try:
            with self.sessions() as db:
                job = db.get(TrainingJob, job_id)
                if job is None:
                    return
                run_dir = self.root / job_id / "run"
                summary = _read_json(run_dir / "summary.json")
                has_ckpt = (run_dir / "latest.json").is_file()
                if code == 0 and summary is not None:
                    if summary.get("stopped") == "requested":
                        job.status = "cancelled" if job.control == "cancel" else "paused"
                    else:
                        job.status = "succeeded"
                    job.summary, job.error = summary, ""
                elif job.control == "cancel":
                    job.status = "cancelled"
                elif job.control == "pause" and has_ckpt:
                    job.status, job.error = "paused", "the process had to be killed after the grace period; resumed runs start from the last checkpoint"
                else:
                    job.status, job.error = "failed", f"training process exited with code {code}\n{self._tail(job_id)}"
                job.control, job.finished_at = "", utcnow()
                db.commit()
        except Exception:  # never let the watcher die silently: the job would stay "running" forever
            log.exception("could not record the outcome of training job %s", job_id)

    def recover(self) -> int:
        """After an API restart no child process is ours any more: jobs that were active become 'interrupted'."""
        n = 0
        with self.sessions() as db:
            for job in db.query(TrainingJob).filter(TrainingJob.status.in_(ACTIVE)).all():
                if not self.is_alive(job.id):
                    job.status = "interrupted"
                    job.error = "the API restarted while this job was running; resume continues from the last checkpoint"
                    job.control, job.finished_at = "", utcnow()
                    n += 1
            db.commit()
        return n


# ------------------------------------------------------------------ inference
class Busy(Exception):
    pass


class InferenceService:
    """Holds at most `infer_cache_size` loaded checkpoints; one generation at a time (the host's CPU/GPU is shared)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: OrderedDict[tuple[str, str], object] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._gen = threading.Semaphore(1)

    def _runtime(self, ckpt: Path, tokenizer_dir: Path | None = None):
        from model.inference import ResearchOSModel
        meta = _read_json(ckpt / "meta.json")
        if meta is None:
            raise ModelNotFound("checkpoint metadata is missing")
        tok_dir = tokenizer_dir or self.settings.tokenizer_dir
        key = (str(ckpt.resolve()), str(meta.get("model_sha256")), str(Path(tok_dir).resolve()))
        with self._cache_lock:
            rt = self._cache.get(key)
            if rt is not None:
                self._cache.move_to_end(key)
                return rt
        rt = ResearchOSModel(device=self.settings.inference_device)
        rt.load_checkpoint(ckpt, tokenizer_dir=tok_dir)  # verifies weights hash and tokenizer fingerprint
        with self._cache_lock:
            self._cache[key] = rt
            while len(self._cache) > max(1, self.settings.infer_cache_size):
                self._cache.popitem(last=False)
        return rt

    def runtime_for(self, ckpt: Path, tokenizer_dir: Path | None = None):
        return self._runtime(ckpt, tokenizer_dir)

    def acquire_slot(self, timeout: float = 5.0) -> None:
        if not self._gen.acquire(timeout=timeout):
            raise Busy("another generation is running; try again shortly")

    def release_slot(self) -> None:
        self._gen.release()

    def loaded(self) -> list[dict]:
        with self._cache_lock:
            return [{"checkpoint": k[0], "model_sha256": k[1], "device": str(rt.device)} for k, rt in self._cache.items()]

    def generate(self, ckpt: Path, prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float,
                 repetition_penalty: float, seed: int | None, tokenizer_dir: Path | None = None) -> dict:
        rt = self._runtime(ckpt, tokenizer_dir)
        if not self._gen.acquire(timeout=5):
            raise Busy("another generation is running; try again shortly")
        cancel = threading.Event()
        timer = threading.Timer(self.settings.infer_timeout_s, cancel.set)
        timer.daemon = True
        timer.start()
        try:
            res = rt.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                                   repetition_penalty=repetition_penalty, seed=seed, cancel_event=cancel)
        finally:
            timer.cancel()
            self._gen.release()
        return {"text": res.text, "finish_reason": res.finish_reason, "prompt_tokens": res.prompt_tokens,
                "new_tokens": len(res.token_ids), "tokens_per_s": round(res.tokens_per_s, 2), "seconds": round(res.elapsed_s, 3),
                "timed_out": res.finish_reason == "cancelled", "device": str(rt.device),
                "model": {"step": rt.meta.get("step"), "model_sha256": rt.meta.get("model_sha256"),
                          "lineage_root": (rt.meta.get("lineage") or {}).get("root"),
                          "tokenizer_fingerprint": rt.tokenizer.fingerprint if rt.tokenizer else None},
                "notice": NOTICE}
