"""Evaluation suite for the from-scratch language model.

Everything here is measured on this machine from real checkpoints; nothing is estimated or compared with other models.

* Token-level, token-weighted loss/perplexity/accuracy (padding and prefix tokens carry IGNORE_INDEX and are not scored).
* Fixed val/test splits from the prepared dataset plus small task-specific probe sets (`datasets/evaluation/task_sets`).
* A random-initialised baseline of the same architecture scored on identical data, so "trained" always means "better than
  chance-level initialisation", never "good".
* A train-overlap check (8-gram hashing against the training stream) so memorisation is visible instead of silently
  counted as capability.
* Inference latency/throughput/peak memory, cached and uncached.

Limits that stay attached to every report: tiny probe sets, synthetic training text unless stated otherwise, and no comparison
with any other model (proprietary or otherwise) is made or implied.
"""
from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.architecture.transformer import IGNORE_INDEX, Transformer, build_model
from model.common import atomic_write_json, repo_root, sha256_file
from model.tokenizer.bpe import BOS_ID, EOS_ID, BPETokenizer
from model.training.memory import hardware_report

TASK_SET_DIR = repo_root() / "datasets" / "evaluation" / "task_sets"
OVERLAP_N = 8
MEMORIZATION_THRESHOLD = 0.25  # fraction of eval 8-grams also present in training data above which we flag it

CAVEATS = (
    "Probe sets are tiny (4-10 items each); results are smoke-level signals, not benchmarks.",
    "Perplexity is only comparable between models that share a tokenizer and evaluation text.",
    "No comparison with any other model, proprietary or open, is made or implied.",
    "Low loss on text overlapping the training data measures memorisation, not capability; see train_overlap.",
)


@dataclass
class ScoredText:
    ids: list[int]
    targets: list[int]  # same length as ids; IGNORE_INDEX where a position is not scored


class TaskSetError(ValueError):
    pass


# --------------------------------------------------------------------------- scoring primitives
def _score_batches(model: Transformer, rows: Sequence[ScoredText], device: torch.device, batch_size: int = 8) -> dict:
    """Token-weighted NLL and accuracy over `rows` (right-padded; padding is ignored)."""
    was_training = model.training
    model.eval()
    nll, correct, tokens = 0.0, 0, 0
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            width = max(len(r.ids) for r in chunk)
            x = torch.zeros(len(chunk), width, dtype=torch.long)
            y = torch.full((len(chunk), width), IGNORE_INDEX, dtype=torch.long)
            for j, r in enumerate(chunk):
                x[j, :len(r.ids)] = torch.tensor(r.ids)
                y[j, :len(r.targets)] = torch.tensor(r.targets)
            x, y = x.to(device), y.to(device)
            logits = model(x).logits.float()
            valid = y != IGNORE_INDEX
            n = int(valid.sum())
            if n == 0:
                continue
            nll += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"))
            correct += int(((logits.argmax(-1) == y) & valid).sum())
            tokens += n
    model.train(was_training)
    if tokens == 0:
        raise TaskSetError("no scorable tokens")
    loss = nll / tokens
    return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "token_accuracy": correct / tokens, "tokens": tokens}


def _window_rows(ids: Sequence[int], seq_len: int, stride: int | None = None) -> list[ScoredText]:
    """Non-overlapping next-token windows over a flat token stream (every target scored once)."""
    stride = stride or seq_len
    rows = []
    for s in range(0, len(ids) - 1, stride):
        chunk = list(ids[s:s + seq_len + 1])
        if len(chunk) < 2:
            break
        rows.append(ScoredText(chunk[:-1], chunk[1:]))
    return rows


def score_token_stream(model: Transformer, ids: Sequence[int], device, seq_len: int | None = None, batch_size: int = 8,
                       max_windows: int | None = None) -> dict:
    seq_len = seq_len or model.cfg.max_seq_len
    rows = _window_rows(ids, seq_len)
    if max_windows is not None:
        rows = rows[:max_windows]
    return _score_batches(model, rows, device, batch_size)


def encode_text_item(tok: BPETokenizer, item: dict, max_len: int) -> ScoredText | None:
    """Turn a task item into scored positions. Plain text: every token after BOS is scored. Completion: only the
    continuation is scored (prefix targets are ignored). Over-long items are left-truncated so the scored part survives."""
    if item["kind"] == "completion":
        prefix = tok.encode_ordinary(item["prefix"])
        cont = tok.encode_ordinary(item["continuation"]) + [EOS_ID]
        seq = [BOS_ID] + prefix + cont
        ignored = len(prefix)  # targets that merely reproduce the prefix
    else:
        seq = [BOS_ID] + tok.encode_ordinary(item["text"]) + [EOS_ID]
        ignored = 0
    inputs, targets = seq[:-1], list(seq[1:])  # targets[i] is the token that follows inputs[i]
    for i in range(ignored):
        targets[i] = IGNORE_INDEX
    if len(inputs) > max_len:
        cut = len(inputs) - max_len
        inputs, targets = inputs[cut:], targets[cut:]
    if not any(t != IGNORE_INDEX for t in targets):
        return None
    return ScoredText(inputs, targets)


# --------------------------------------------------------------------------- task sets
def load_task_sets(directory: str | Path = TASK_SET_DIR) -> dict[str, dict]:
    """Load the committed probe sets and verify each file against MANIFEST.json (a changed file is an error)."""
    d = Path(directory)
    manifest = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for name, meta in manifest["sets"].items():
        path = d / meta["file"]
        if sha256_file(path) != meta["sha256"]:
            raise TaskSetError(f"task set {name} does not match its manifest hash; regenerate with scripts/make_task_sets.py")
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[name] = {**meta, "items": items}
    return out


def _rolling_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    if len(ids) < n:
        return np.zeros(0, dtype=np.uint64)
    ids = ids.astype(np.uint64)
    h = np.zeros(len(ids) - n + 1, dtype=np.uint64)
    p = np.uint64(1000003)
    mul = np.uint64(1)
    with np.errstate(over="ignore"):
        for k in range(n):
            h += ids[k:len(ids) - n + 1 + k] * mul
            mul = mul * p
    return h


def train_overlap(eval_rows: Sequence[ScoredText], train_tokens: np.ndarray | None, n: int = OVERLAP_N) -> dict:
    """Fraction of the evaluation 8-grams that also occur in the training token stream."""
    if train_tokens is None:
        return {"checked": False, "reason": "no training token file supplied"}
    grams: set[int] = set()
    total = 0
    for r in eval_rows:
        hs = _rolling_hashes(np.asarray(r.ids, dtype=np.int64), n)
        grams.update(int(x) for x in hs)
        total += len(hs)
    if total == 0:
        return {"checked": True, "ngram": n, "eval_ngrams": 0, "overlap_fraction": 0.0, "memorization_risk": False}
    train_h = _rolling_hashes(np.asarray(train_tokens), n)
    hit = np.isin(np.fromiter(grams, dtype=np.uint64, count=len(grams)), train_h)
    frac = float(hit.mean())
    return {"checked": True, "ngram": n, "eval_ngrams": len(grams), "overlap_fraction": frac,
            "memorization_risk": frac > MEMORIZATION_THRESHOLD}


def evaluate_task_sets(model: Transformer, tok: BPETokenizer, device, train_tokens: np.ndarray | None = None,
                       directory: str | Path = TASK_SET_DIR) -> dict:
    results: dict[str, dict] = {}
    for name, spec in load_task_sets(directory).items():
        rows = [r for r in (encode_text_item(tok, it, model.cfg.max_seq_len) for it in spec["items"]) if r is not None]
        if not rows:
            results[name] = {"error": "no scorable items"}
            continue
        res = _score_batches(model, rows, device)
        res.update(items=len(rows), kind=spec["kind"], train_overlap=train_overlap(rows, train_tokens))
        results[name] = res
    return results


# --------------------------------------------------------------------------- baseline and benchmarks
def random_baseline(model: Transformer, seed: int = 12345) -> Transformer:
    """A freshly initialised model with the same architecture. Scored identically it shows what 'no learning' looks like."""
    return build_model(model.cfg, seed).eval()


def benchmark_inference(model: Transformer, device, prompt_len: int = 32, new_tokens: int = 64, runs: int = 3) -> dict:
    """Measured generation speed with and without the KV cache, plus peak memory. Greedy decoding of random token ids
    (speed does not depend on content). Returns medians over `runs` after one warm-up."""
    cfg = model.cfg
    prompt_len = min(prompt_len, cfg.max_seq_len - 2)
    new_tokens = max(1, min(new_tokens, cfg.max_seq_len - prompt_len))
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(6, cfg.vocab_size, (1, prompt_len), generator=g).to(device)
    model.eval()

    def run(use_cache: bool) -> tuple[float, float]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids, past, first = prompt, None, None
        with torch.inference_mode():
            for _ in range(new_tokens):
                if use_cache:
                    out = model(ids if past is None else ids[:, -1:], past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
                else:
                    out = model(ids, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                if first is None:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    first = time.perf_counter() - t0
                ids = torch.cat([ids, nxt], dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, first or 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    res: dict = {"prompt_tokens": prompt_len, "new_tokens": new_tokens, "runs": runs}
    for label, flag in (("cached", True), ("uncached", False)):
        run(flag)  # warm-up
        samples = [run(flag) for _ in range(runs)]
        total = float(np.median([s[0] for s in samples]))
        ttft = float(np.median([s[1] for s in samples]))
        res[label] = {"seconds": total, "tokens_per_second": new_tokens / total, "ms_per_token": 1000 * total / new_tokens,
                      "time_to_first_token_ms": 1000 * ttft}
    res["cache_speedup"] = res["uncached"]["seconds"] / res["cached"]["seconds"]
    if device.type == "cuda":
        res["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
    return res


def checkpoint_size_bytes(ckpt: str | Path) -> int:
    p = Path(ckpt)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


# --------------------------------------------------------------------------- full report
def run_evaluation(
    model: Transformer,
    tok: BPETokenizer,
    device: torch.device | str,
    *,
    val_tokens: np.ndarray | None = None,
    test_tokens: np.ndarray | None = None,
    train_tokens: np.ndarray | None = None,
    checkpoint: str | Path | None = None,
    baseline_seed: int = 12345,
    max_windows: int | None = 200,
    benchmark: bool = True,
    out_path: str | Path | None = None,
) -> dict:
    """Score the trained model and a random-initialised baseline on the same data and return one report dict."""
    device = torch.device(device)
    model = model.to(device).eval()
    base = random_baseline(model, baseline_seed).to(device)

    def both(fn):
        return {"model": fn(model), "random_baseline": fn(base)}

    report: dict = {
        "schema": "researchos-model-evaluation-v1",
        "parameters": model.num_parameters(),
        "model_config": model.cfg.to_dict(),
        "tokenizer_fingerprint": tok.fingerprint,
        "vocab_size": tok.vocab_size,
        "hardware": hardware_report(),
        "platform": platform.platform(),
        "baseline": {"kind": "random_init_same_architecture", "seed": baseline_seed},
        "caveats": list(CAVEATS),
        "comparison_with_other_models": "none (no proprietary or third-party model was run or compared)",
        "splits": {},
    }
    if checkpoint is not None:
        report["checkpoint_bytes"] = checkpoint_size_bytes(checkpoint)
    for name, toks in (("validation", val_tokens), ("test", test_tokens)):
        if toks is not None:
            ids = np.asarray(toks, dtype=np.int64).tolist()
            report["splits"][name] = both(lambda m, ids=ids: score_token_stream(m, ids, device, max_windows=max_windows))
            report["splits"][name]["train_overlap"] = train_overlap(_window_rows(ids, model.cfg.max_seq_len)[:max_windows or None],
                                                                    train_tokens)
    tasks_model = evaluate_task_sets(model, tok, device, train_tokens)
    tasks_base = evaluate_task_sets(base, tok, device, None)
    report["tasks"] = {k: {"model": tasks_model[k], "random_baseline": tasks_base.get(k)} for k in tasks_model}
    # headline comparison used by lifecycle validation
    primary = report["splits"].get("validation")
    if primary:
        report["validation_loss"] = primary["model"]["loss"]
        report["baseline_validation_loss"] = primary["random_baseline"]["loss"]
        report["beats_random_baseline"] = bool(primary["model"]["loss"] < primary["random_baseline"]["loss"])
    report["memorization_risk"] = any(
        s.get("train_overlap", {}).get("memorization_risk") for s in report["splits"].values()
    ) or any(t["model"].get("train_overlap", {}).get("memorization_risk") for t in report["tasks"].values())
    if benchmark:
        report["inference"] = benchmark_inference(model, device)
    if out_path is not None:
        atomic_write_json(out_path, report)
    return report
