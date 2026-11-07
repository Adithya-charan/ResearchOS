"""Model Lab services: lifecycle registry, user datasets (rights-declared, prepared by the same pipeline as the operator corpus),
evaluation runs and agent-provider selection.

Authority rules: lifecycle evidence is always derived here from artifacts on disk (run.json / summary.json, checkpoint hashes,
splits.json, evaluation reports) and never accepted from a request body; DEPLOYED and every change to a system model need an
administrator (settings.admin_emails); clients never name filesystem paths - only ids this module issued."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import shutil
import threading
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from agent_engine.model_provider import (
    CustomTransformerBackend,
    ModelRecordView,
    NoValidatedModelError,
    ProviderSelection,
    RolePolicy,
    deterministic_status,
    evaluate_stage,
    resolve_provider,
)
from api.config import ROOT, Settings
from api.db import utcnow
from api.model_service import (SYSTEM_PREFIX, InferenceService, ModelNotFound, ModelRegistry, _read_json, list_checkpoints,
                               pick_checkpoint, tokenizer_dir_for, verify_checkpoint)
from api.models import ModelEntry, ModelSelection, TrainingDataset, TrainingJob, new_id
from model.common import atomic_write_json, canonical_json, sha256_bytes
from model.lifecycle import ALLOWED, Evidence, LifecycleError, ModelState, check_transition, transition

log = logging.getLogger("researchos.modellab")
LICENSES = ("CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "MIT", "public-domain", "user-owned")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,79}$")


class LabError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def is_admin(settings: Settings, email: str) -> bool:
    return email.lower() in settings.admin_emails


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ evidence from artifacts
def _legacy_completion(summary: dict | None) -> str | None:
    """Runs before P8 have no completion_state; their summary's `stopped` field says how they ended."""
    if not summary:
        return None
    if summary.get("completion_state"):
        return summary["completion_state"]
    return {"max_steps": "completed", "early_stopping": "completed"}.get(summary.get("stopped"))


def leakage_from_splits(processed_dir: Path | None) -> tuple[bool, bool]:
    """(checked, found): re-derived from the split manifest the preparation step wrote. Missing manifest = not checked."""
    if processed_dir is None:
        return False, True
    data = _read_json(processed_dir / "splits.json")
    if not isinstance(data, dict) or not {"train", "val", "test"} <= set(data):
        return False, True
    sets = {k: set(v) for k, v in data.items()}
    names = list(sets)
    found = any(sets[a] & sets[b] for i, a in enumerate(names) for b in names[i + 1:])
    return True, found


class ModelLab:
    def __init__(self, settings: Settings, sessionmaker, registry: ModelRegistry, inference: InferenceService):
        self.settings, self.sessions, self.registry, self.inference = settings, sessionmaker, registry, inference
        self.eval_root = settings.data_dir / "evaluations"
        self.datasets_root = settings.data_dir / settings.datasets_dir_name
        self._eval_lock = threading.Lock()
        self._eval_state: dict[str, dict] = {}
        self._prep_lock = threading.Lock()

    # -------------------------------------------------------------- locating artifacts
    def run_dir(self, model_id: str) -> Path:
        if model_id.startswith(SYSTEM_PREFIX):
            return self.registry.system_dir(model_id)
        if model_id.startswith("trn_"):
            d = self.registry.job_run_dir(model_id)
            if not d.is_dir():
                raise ModelNotFound("this job has no checkpoints yet")
            return d
        raise ModelNotFound("model not found")

    def processed_dir_of(self, run_dir: Path) -> Path | None:
        try:
            meta = _read_json(pick_checkpoint(run_dir, "best") / "meta.json") or {}
        except ModelNotFound:
            return None
        raw = (meta.get("data") or {}).get("processed_dir")
        if not raw:
            return None
        p = Path(raw)
        p = p if p.is_absolute() else ROOT / p  # relative paths are repo-relative
        p = p.resolve()
        allowed = [(ROOT / "datasets").resolve(), self.datasets_root.resolve()]
        return p if any(a == p or a in p.parents for a in allowed) and p.is_dir() else None

    # -------------------------------------------------------------- evidence
    def gather_evidence(self, model_id: str, entry: ModelEntry, authorized_by: str | None = None) -> Evidence:
        run_dir = self.run_dir(model_id)
        run = _read_json(run_dir / "run.json") or {}
        summary = _read_json(run_dir / "summary.json")
        steps = int(run.get("steps") or (summary or {}).get("steps") or 0)
        completion = run.get("completion_state") or _legacy_completion(summary)
        final = run.get("final_loss") if run.get("final_loss") is not None else (summary or {}).get("final_loss")
        if final is None:  # legacy runs: the last logged batch loss
            from api.model_service import read_metrics
            tail = read_metrics(run_dir)["train"]
            final = tail[-1].get("loss") if tail else None
        ck_ok, lineage_ok = False, False
        try:
            ck = pick_checkpoint(run_dir, "best")
            v = verify_checkpoint(ck)
            ck_ok = lineage_ok = bool(v["ok"])  # verify_checkpoint covers the weights hash and the random-init lineage
            try:
                from model.tokenizer.bpe import BPETokenizer
                tok = BPETokenizer.load(tokenizer_dir_for(self.settings, run_dir))
                want = ((_read_json(ck / "meta.json") or {}).get("tokenizer") or {}).get("fingerprint")
                ck_ok = ck_ok and (want is None or want == tok.fingerprint)
            except Exception:
                ck_ok = False
        except ModelNotFound:
            pass
        ev = entry.evaluation or {}
        checked, found = leakage_from_splits(self.processed_dir_of(run_dir))
        return Evidence(
            steps_completed=steps, completion_state=completion, final_loss=final, checkpoint_verified=ck_ok,
            lineage_verified=lineage_ok, val_loss=ev.get("validation_loss"), baseline_val_loss=ev.get("baseline_validation_loss"),
            leakage_checked=checked, leakage_found=found, authorized_by=authorized_by,
            extra={"evaluation_checkpoint_sha256": ev.get("checkpoint_sha256")},
        )

    # -------------------------------------------------------------- entries
    def entry(self, db: Session, model_id: str, owner_id: str | None) -> ModelEntry:
        row = db.get(ModelEntry, model_id)
        if row is None:
            row = ModelEntry(model_id=model_id, owner_id=owner_id, state=ModelState.INITIALIZED.value, stage_reached=0, evidence={},
                             history=[{"at": _now(), "from": None, "to": "INITIALIZED", "by": "system", "note": "registered"}],
                             created_at=utcnow(), updated_at=utcnow())
            db.add(row)
            db.commit()
        return row

    def _apply(self, db: Session, row: ModelEntry, target: ModelState, ev: Evidence, by: str, note: str) -> None:
        new = transition(row.state, target, ev)  # raises LifecycleError with every reason
        row.history = [*row.history, {"at": _now(), "from": row.state, "to": new.value, "by": by, "note": note[:300]}]
        row.state = new.value
        row.evidence = {"steps_completed": ev.steps_completed, "completion_state": ev.completion_state, "final_loss": ev.final_loss,
                        "checkpoint_verified": ev.checkpoint_verified, "lineage_verified": ev.lineage_verified,
                        "val_loss": ev.val_loss, "baseline_val_loss": ev.baseline_val_loss,
                        "leakage_checked": ev.leakage_checked, "leakage_found": ev.leakage_found, "authorized_by": ev.authorized_by}
        row.updated_at = utcnow()
        db.commit()

    def refresh(self, db: Session, row: ModelEntry, job: TrainingJob | None = None) -> ModelEntry:
        """Move the state forward when the artifacts prove it (training started, finished or failed). Never grants VALIDATED/DEPLOYED."""
        try:
            ev = self.gather_evidence(row.model_id, row)
        except ModelNotFound:
            ev = Evidence()
        state = ModelState(row.state)
        status = job.status if job is not None else None
        try:
            if status in ("running", "pausing", "cancelling", "paused", "interrupted") and state in (
                    ModelState.INITIALIZED, ModelState.TRAINED, ModelState.VALIDATED, ModelState.FAILED):
                self._apply(db, row, ModelState.TRAINING, ev, "system", f"training job {status}")
            elif status == "succeeded" and state in (ModelState.INITIALIZED, ModelState.FAILED):
                self._apply(db, row, ModelState.TRAINING, ev, "system", "training job finished")
                self._apply(db, row, ModelState.TRAINED, ev, "system", "training job completed")
            elif status == "succeeded" and state == ModelState.TRAINING:
                self._apply(db, row, ModelState.TRAINED, ev, "system", "training job completed")
            elif status == "failed" and state != ModelState.FAILED:
                if state != ModelState.TRAINING:
                    self._apply(db, row, ModelState.TRAINING, ev, "system", "training job failed")
                self._apply(db, row, ModelState.FAILED, ev, "system", "training job failed")
            elif status == "cancelled" and state == ModelState.TRAINING:
                target = ModelState.INITIALIZED if ev.steps_completed == 0 else ModelState.FAILED
                self._apply(db, row, target, ev, "system", "training job cancelled")
            elif job is None and state == ModelState.INITIALIZED and ev.steps_completed > 0:  # operator checkpoint folders
                self._apply(db, row, ModelState.TRAINING, ev, "system", "checkpoints found")
                if ev.completion_state == "completed":
                    self._apply(db, row, ModelState.TRAINED, ev, "system", "completed run found")
            elif job is None and state == ModelState.TRAINING and ev.completion_state == "completed":
                self._apply(db, row, ModelState.TRAINED, ev, "system", "completed run found")
        except LifecycleError as e:  # evidence insufficient: stay where we are and say why
            row.evidence = {**(row.evidence or {}), "last_refusal": str(e)[:300]}
            db.commit()
        return row

    def view(self, row: ModelEntry) -> dict:
        state = ModelState(row.state)
        return {"model_id": row.model_id, "state": state.value, "stage_reached": row.stage_reached,
                "allowed_next": sorted(s.value for s in ALLOWED[state]), "history": row.history, "evidence": row.evidence,
                "evaluation": row.evaluation, "updated_at": row.updated_at}

    def change_state(self, db: Session, row: ModelEntry, target: str, user, note: str) -> ModelEntry:
        try:
            tgt = ModelState(target)
        except ValueError as e:
            raise LabError(f"unknown state {target!r}", 422) from e
        admin = is_admin(self.settings, user.email)
        if row.model_id.startswith(SYSTEM_PREFIX) and not admin:
            raise LabError("system models can only be changed by an administrator", 403)
        if tgt in (ModelState.DEPLOYED,) and not admin:
            raise LabError("deploying a model needs an administrator", 403)
        if tgt in (ModelState.TRAINED, ModelState.INITIALIZED, ModelState.FAILED) and ModelState(row.state) == ModelState.TRAINING:
            raise LabError("TRAINING ends by itself when the run does; it cannot be forced from here", 409)
        ev = self.gather_evidence(row.model_id, row, authorized_by=user.id if admin else None)
        # a validation result only counts if it was produced for the checkpoint that is being validated now
        if tgt == ModelState.VALIDATED:
            ck = pick_checkpoint(self.run_dir(row.model_id), "best")
            sha = (_read_json(ck / "meta.json") or {}).get("model_sha256")
            if (row.evaluation or {}).get("checkpoint_sha256") != sha:
                ev.val_loss = ev.baseline_val_loss = None  # stale or missing evaluation
        try:
            self._apply(db, row, tgt, ev, user.id, note)
        except LifecycleError as e:
            raise LabError(str(e), 409) from e
        return row

    # -------------------------------------------------------------- evaluation
    def evaluation_status(self, model_id: str) -> dict:
        st = self._eval_state.get(model_id, {})
        # never read the report while an evaluation is replacing it (on Windows that races with the atomic rename)
        report = None if st.get("state") == "running" else _read_json(self.eval_root / model_id / "report.json")
        return {"state": st.get("state", "done" if report else "none"), "error": st.get("error"), "started_at": st.get("started_at"),
                "report": report}

    def start_evaluation(self, model_id: str) -> dict:
        self.run_dir(model_id)  # 404 if there is nothing to evaluate
        if not self._eval_lock.acquire(blocking=False):
            raise LabError("another evaluation is running; wait for it to finish", 429)
        self._eval_state[model_id] = {"state": "running", "started_at": _now()}
        threading.Thread(target=self._evaluate, args=(model_id,), daemon=True, name=f"eval-{model_id}").start()
        return {"state": "running"}

    def _evaluate(self, model_id: str) -> None:
        try:
            import numpy as np

            from model.data.dataset import TokenDataset
            from model.evaluation.suite import run_evaluation
            run_dir = self.run_dir(model_id)
            ck = pick_checkpoint(run_dir, "best")
            rt = self.inference.runtime_for(ck, tokenizer_dir_for(self.settings, run_dir))
            proc = self.processed_dir_of(run_dir)
            if proc is None:
                raise LabError("the dataset this model was trained on is not available for evaluation", 409)
            seq = rt.model.cfg.max_seq_len
            toks = {k: np.asarray(TokenDataset(proc / f"{k}.bin", seq).tokens) for k in ("train", "val", "test") if (proc / f"{k}.bin").exists()}
            out = self.eval_root / model_id
            out.mkdir(parents=True, exist_ok=True)
            report = run_evaluation(
                rt.model, rt.tokenizer, rt.device, val_tokens=toks.get("val"), test_tokens=toks.get("test"),
                train_tokens=toks.get("train", np.zeros(0))[:2_000_000], checkpoint=ck,
                max_windows=self.settings.eval_max_windows,
            )
            sha = (_read_json(ck / "meta.json") or {}).get("model_sha256")
            backend = CustomTransformerBackend(rt, model_id=model_id)
            stage = evaluate_stage(backend)
            report["agent_integration"] = stage
            report["checkpoint_sha256"] = sha
            atomic_write_json(out / "report.json", report)
            with self.sessions() as db:
                row = db.get(ModelEntry, model_id)
                if row is not None:
                    row.stage_reached = stage["stage_reached"]
                    row.evaluation = {
                        "at": _now(), "checkpoint_sha256": sha, "validation_loss": report.get("validation_loss"),
                        "baseline_validation_loss": report.get("baseline_validation_loss"),
                        "beats_random_baseline": report.get("beats_random_baseline"), "memorization_risk": report.get("memorization_risk"),
                        "validation_perplexity": (report["splits"].get("validation") or {}).get("model", {}).get("perplexity"),
                        "stage_reached": stage["stage_reached"]}
                    db.commit()
            self._eval_state[model_id] = {"state": "done", "started_at": self._eval_state[model_id]["started_at"]}
        except Exception as e:
            log.exception("evaluation of %s failed", model_id)
            self._eval_state[model_id] = {"state": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}",
                                          "started_at": self._eval_state.get(model_id, {}).get("started_at")}
        finally:
            self._eval_lock.release()

    # -------------------------------------------------------------- datasets
    def dataset_dir(self, dataset_id: str) -> Path:
        if not re.fullmatch(r"dts_[0-9a-f]{16}", dataset_id):
            raise ModelNotFound("dataset not found")
        return self.datasets_root / dataset_id

    def create_dataset(self, db: Session, owner_id: str, name: str, license_: str, attribution: str, rights_confirmed: bool,
                       documents: list[dict]) -> TrainingDataset:
        s = self.settings
        if not rights_confirmed:
            raise LabError("you must confirm that you have the right to train a model on these documents", 422)
        if license_ not in LICENSES:
            raise LabError(f"license must be one of {', '.join(LICENSES)}", 422)
        if not NAME_RE.match(name):
            raise LabError("name must be 1-80 characters: letters, digits, space, _ . -", 422)
        if license_ != "user-owned" and not attribution.strip():
            raise LabError("attribution (source/author) is required for licensed material", 422)
        if not 1 <= len(documents) <= s.dataset_max_docs:
            raise LabError(f"a dataset holds 1 to {s.dataset_max_docs} documents", 422)
        total = sum(len(d["text"]) for d in documents)
        if any(len(d["text"]) > s.dataset_max_doc_chars for d in documents):
            raise LabError(f"each document is limited to {s.dataset_max_doc_chars} characters", 422)
        if total > s.dataset_max_total_chars:
            raise LabError(f"a dataset is limited to {s.dataset_max_total_chars} characters in total", 422)
        owned = len(db.scalars(select(TrainingDataset.id).where(TrainingDataset.owner_id == owner_id)).all())
        if owned >= s.dataset_max_per_user:
            raise LabError(f"dataset quota reached ({s.dataset_max_per_user}); delete one first", 429)
        ds = TrainingDataset(id=new_id("dts"), owner_id=owner_id, name=name, license=license_, attribution=attribution.strip(),
                             status="uploaded", documents=len(documents), chars=total)
        d = self.datasets_root / ds.id
        raw = d / "raw"
        raw.mkdir(parents=True)
        try:
            from model.data.manifest import DocRecord, manifest_sha256, write_manifest
            records = []
            for i, doc in enumerate(documents):
                text = doc["text"]
                fn = f"d{i:04d}.txt"
                (raw / fn).write_text(text, encoding="utf-8", newline="\n")
                records.append(DocRecord(
                    doc_id=f"d{i:04d}", path=f"raw/{fn}", sha256=hashlib.sha256((raw / fn).read_bytes()).hexdigest(),
                    source=f"user-upload:{owner_id}:{(doc.get('title') or fn)[:80]}", license=license_,
                    acquired=dt.date.today().isoformat(), allowed_use=["training"], bytes=len(text.encode("utf-8")),
                    notes=attribution.strip()[:200]))
            write_manifest(d / "manifest.jsonl", records)
            ds.manifest_sha256 = manifest_sha256(records)
            db.add(ds)
            db.commit()
        except Exception:
            db.rollback()
            shutil.rmtree(d, ignore_errors=True)
            raise
        return ds

    def prepare_dataset(self, ds_id: str, vocab_size: int, pretokenizer: str, min_chars: int) -> None:
        """Run the standard preparation (clean, dedupe, split by document, fit the tokenizer on train only, pack) in a thread."""
        if not self._prep_lock.acquire(blocking=False):
            raise LabError("another dataset is being prepared; wait for it to finish", 429)
        threading.Thread(target=self._prepare, args=(ds_id, vocab_size, pretokenizer, min_chars), daemon=True,
                         name=f"prep-{ds_id}").start()

    def _prepare(self, ds_id: str, vocab_size: int, pretokenizer: str, min_chars: int) -> None:
        try:
            from model.data.pipeline import PrepareConfig, prepare_dataset
            d = self.dataset_dir(ds_id)
            shutil.rmtree(d / "processed", ignore_errors=True)
            shutil.rmtree(d / "tokenizer", ignore_errors=True)
            cfg = PrepareConfig(manifest_path=str(d / "manifest.jsonl"), data_root=str(d), out_dir=str(d / "processed"),
                                tokenizer_dir=str(d / "tokenizer"), vocab_size=vocab_size, min_chars=min_chars, pretokenizer=pretokenizer,
                                val_frac=0.1, test_frac=0.1)  # small user corpora need larger held-out shares than the operator corpus
            meta = prepare_dataset(cfg, log=lambda m: None)
            stats = {k: meta[k] for k in ("tokens", "documents", "dropped_total", "dropped_reasons", "vocab_size", "tokenizer_fingerprint",
                                          "tokenizer_corpus_sha256", "manifest_sha256")}
            stats["pretokenizer"] = pretokenizer
            tok_stats = _read_json(d / "tokenizer" / "vocab_statistics.json")
            if tok_stats:
                stats["tokenizer_sample"] = tok_stats.get("sample")
            status, error = "prepared", ""
        except Exception as e:
            stats, status, error = {}, "failed", f"{type(e).__name__}: {str(e)[:400]}"
            log.warning("dataset %s preparation failed: %s", ds_id, error)
        finally:
            self._prep_lock.release()
        with self.sessions() as db:
            row = db.get(TrainingDataset, ds_id)
            if row is not None:
                row.status, row.stats, row.error = status, stats, error
                db.commit()

    def delete_dataset(self, db: Session, ds: TrainingDataset) -> None:
        if ds.status == "preparing":
            raise LabError("this dataset is being prepared", 409)
        in_use = [j.id for j in db.scalars(select(TrainingJob).where(TrainingJob.owner_id == ds.owner_id)) if (j.config or {}).get("dataset_id") == ds.id
                  and j.status in ("running", "pausing", "cancelling")]
        if in_use:
            raise LabError("a running training job uses this dataset", 409)
        shutil.rmtree(self.dataset_dir(ds.id), ignore_errors=True)
        db.delete(ds)
        db.commit()

    # -------------------------------------------------------------- provider selection
    def selection_of(self, db: Session, user_id: str) -> ModelSelection | None:
        return db.get(ModelSelection, user_id)

    def lookup_for(self, db: Session, user_id: str):
        def lookup(model_id: str) -> ModelRecordView | None:
            try:
                self.run_dir(model_id)
            except ModelNotFound:
                return None
            if model_id.startswith("trn_"):
                job = db.get(TrainingJob, model_id)
                if job is None or job.owner_id != user_id:
                    return None
            row = db.get(ModelEntry, model_id)
            if row is None:
                return None
            return ModelRecordView(model_id, ModelState(row.state), row.stage_reached)
        return lookup

    def resolve(self, db: Session, user_id: str, sel: ProviderSelection, load: bool = True):
        def loader(model_id: str):
            if not load:
                return _Probe()
            run_dir = self.run_dir(model_id)
            ck = pick_checkpoint(run_dir, "best")
            rt = self.inference.runtime_for(ck, tokenizer_dir_for(self.settings, run_dir))
            return CustomTransformerBackend(rt, model_id=model_id, max_new_tokens_limit=self.settings.infer_max_new_tokens)
        return resolve_provider(sel, self.lookup_for(db, user_id), loader)

    def provider_for_project(self, project_id: str):
        """Called by the engine runtime for every run: the owner's selection, or deterministic. A selection that is no longer
        eligible degrades to deterministic and says so in the log; GET /modellab/provider reports it as an error to the user."""
        from api.models import Project
        with self.sessions() as db:
            proj = db.get(Project, project_id)
            if proj is None:
                return None, None
            row = db.get(ModelSelection, proj.owner_id)
            if row is None or row.provider != "custom_transformer":
                return None, None
            sel = ProviderSelection("custom_transformer", row.model_id, row.mode, bool(row.allow_experimental), False, "error")
            try:
                backend, status = self.resolve(db, proj.owner_id, sel)
            except NoValidatedModelError as e:
                log.warning("project %s: selected model unavailable, running deterministically: %s", project_id, e)
                return None, None
            return backend, RolePolicy.for_status(status)


class _Probe:
    """Stands in for a backend when a selection is only being validated (nothing is loaded)."""
    model_id = ""
    state = ModelState.INITIALIZED
    mode = "experimental"


def selection_view(sel: ModelSelection | None) -> dict:
    if sel is None:
        return {"provider": "deterministic", "model_id": None, "mode": None, "allow_experimental": False}
    return {"provider": sel.provider, "model_id": sel.model_id, "mode": sel.mode, "allow_experimental": bool(sel.allow_experimental)}


__all__ = ["ModelLab", "LabError", "LICENSES", "is_admin", "selection_view", "deterministic_status", "canonical_json", "sha256_bytes",
           "list_checkpoints"]
