import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.architecture import build_model
from model.common import sha256_file
from model.data.dataset import TokenDataset, write_token_file
from model.evaluation.metrics import evaluate_loss, repetition_rate
from model.training.checkpoint import (
    CheckpointError,
    load_model,
    read_meta,
    resolve_checkpoint,
    save_checkpoint,
)
from model.training.lineage import LineageError, assert_random_root, child_lineage, new_lineage
from model.training.trainer import TrainConfig, Trainer, lr_at
from tests.conftest import tiny_config

SEQ = 16


def make_data(tmp_path: Path, n_tokens=6000, vocab=64, seed=0):
    if (tmp_path / "train.bin").exists():  # a live memmap of this file blocks rewriting it on Windows
        return TokenDataset(tmp_path / "train.bin", SEQ), TokenDataset(tmp_path / "val.bin", SEQ)
    rng = np.random.default_rng(seed)
    # learnable structure: next token = (prev * 3 + 1) % vocab with 10% noise
    toks = np.zeros(n_tokens, dtype=np.int64)
    for i in range(1, n_tokens):
        toks[i] = (toks[i - 1] * 3 + 1) % vocab if rng.random() > 0.1 else rng.integers(0, vocab)
    write_token_file(tmp_path / "train.bin", toks)
    write_token_file(tmp_path / "val.bin", toks[:2000])
    return TokenDataset(tmp_path / "train.bin", SEQ), TokenDataset(tmp_path / "val.bin", SEQ)


def make_trainer(tmp_path, steps, run="run", **cfg_over):
    train, val = make_data(tmp_path)
    mcfg = tiny_config(max_seq_len=SEQ, dropout=0.1)
    model = build_model(mcfg, 0)
    cfg = TrainConfig(
        run_name=run, seed=3, batch_size=8, grad_accum_steps=2, max_steps=steps, learning_rate=3e-3, warmup_steps=5,
        eval_interval=10, eval_batches=3, checkpoint_interval=10, log_interval=5, device="cpu", mixed_precision="none",
        **cfg_over,
    )
    meta = {"lineage": new_lineage(model, 0), "tokenizer": {"fingerprint": "t"}}
    return Trainer(model, cfg, train, val, tmp_path / run, meta_base=meta, log=lambda *_: None), model


# ---------------------------------------------------------------- schedule
def test_lr_schedule_shape():
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    lrs = [lr_at(s, cfg) for s in range(100)]
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10]))  # warmup strictly increasing
    assert lrs[9] == pytest.approx(1e-3)  # peak when warmup ends
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))  # cosine decay never rises
    assert lr_at(100, cfg) == pytest.approx(1e-4)  # floor


# ---------------------------------------------------------------- training
def test_training_reduces_loss_and_writes_artifacts(tmp_path):
    trainer, _ = make_trainer(tmp_path, steps=60)
    summary = trainer.fit()
    train_losses = [r["loss"] for r in trainer.history if r["type"] == "train"]
    evals = [r["val_loss"] for r in trainer.history if r["type"] == "eval"]
    assert train_losses[-1] < train_losses[0] - 0.5
    assert evals[-1] < evals[0] and summary["best_val_loss"] == min(evals)
    run = tmp_path / "run"
    for f in ("metrics.jsonl", "train_config.json", "summary.json", "latest.json", "best.json"):
        assert (run / f).exists()
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert {"loss", "lr", "grad_norm", "tokens_per_sec"} <= set(rows[0])
    assert not list(run.glob(".tmp_*"))  # no half-written checkpoints left behind


def test_checkpoint_reload_matches_exactly(tmp_path):
    trainer, model = make_trainer(tmp_path, steps=20)
    trainer.fit()
    loaded, meta = load_model(tmp_path / "run")
    x = torch.randint(0, 64, (2, SEQ))
    with torch.no_grad():
        assert torch.equal(model.cpu().eval()(x).logits, loaded(x).logits)
    assert meta["step"] == 20 and meta["model_config"] == model.cfg.to_dict()
    assert_random_root(meta)


def test_corrupt_checkpoint_is_rejected(tmp_path):
    trainer, _ = make_trainer(tmp_path, steps=10)
    trainer.fit()
    ckpt = resolve_checkpoint(tmp_path / "run")
    blob = bytearray((ckpt / "model.safetensors").read_bytes())
    blob[-1] ^= 0xFF
    (ckpt / "model.safetensors").write_bytes(bytes(blob))
    with pytest.raises(CheckpointError, match="corrupt"):
        load_model(ckpt)


def test_resume_reproduces_uninterrupted_run_bit_for_bit(tmp_path):
    straight, m_straight = make_trainer(tmp_path / "a", steps=20, run="r")
    straight.fit()

    first, _ = make_trainer(tmp_path / "b", steps=20, run="r")  # same schedule, paused at step 10
    first.fit(stop_at=10)
    second, m_second = make_trainer(tmp_path / "b", steps=20, run="r")
    second.fit(resume=tmp_path / "b" / "r")

    for (n, a), (_, b) in zip(m_straight.state_dict().items(), m_second.state_dict().items()):
        assert torch.equal(a.cpu(), b.cpu()), f"{n} differs after resume"


def test_keep_last_prunes_but_protects_best(tmp_path):
    trainer, _ = make_trainer(tmp_path, steps=50, keep_last=2)
    trainer.fit()
    run = tmp_path / "run"
    best = json.loads((run / "best.json").read_text())["path"]
    kept = sorted(p.name for p in run.glob("step_*"))
    assert best in kept and len(kept) <= 3


def test_early_stopping(tmp_path):
    trainer, _ = make_trainer(tmp_path, steps=200, early_stopping_patience=2)
    trainer.cfg.learning_rate = 0.0  # loss cannot improve after the first evaluation
    summary = trainer.fit()
    assert summary["stopped"] == "early_stopping" and summary["steps"] < 200


def test_nonfinite_gradients_skip_the_update(tmp_path):
    trainer, model = make_trainer(tmp_path, steps=3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.tok_emb.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        trainer.cfg.max_nonfinite_steps = 1
        trainer.cfg.max_steps = 10
        trainer.fit()
    # weights other than the poisoned entry were never touched by an optimizer step
    assert torch.equal(model.blocks[0].attn.qkv.weight, before["blocks.0.attn.qkv.weight"])


def test_evaluation_is_deterministic(tmp_path):
    _, val = make_data(tmp_path)
    m = build_model(tiny_config(max_seq_len=SEQ), 0)
    a = evaluate_loss(m, val, 4, 5, "cpu")
    b = evaluate_loss(m, val, 4, 5, "cpu")
    assert a == b and a["perplexity"] == pytest.approx(np.exp(a["loss"]))


def test_repetition_rate():
    assert repetition_rate(list(range(20))) == 0.0
    assert repetition_rate([1, 2, 3] * 20) > 0.9
    assert repetition_rate([1]) == 0.0


# ---------------------------------------------------------------- lineage
def test_lineage_tracks_descent_from_random_init(tmp_path):
    model = build_model(tiny_config(), 11)
    lin = new_lineage(model, 11)
    assert lin["root"] == "random_init" and lin["parent_checkpoint_sha256"] is None
    ckpt = save_checkpoint(tmp_path, 5, model, None, {"lineage": lin})
    parent = read_meta(ckpt)
    child = child_lineage(parent)
    assert child["init_weights_sha256"] == lin["init_weights_sha256"]
    assert child["parent_checkpoint_sha256"] == sha256_file(ckpt / "model.safetensors")
    assert len(child["ancestry"]) == 1
    with pytest.raises(LineageError):
        assert_random_root({"lineage": {"root": "downloaded"}})
    with pytest.raises(LineageError):
        assert_random_root({})


def test_cooperative_stop_checkpoints_and_resumes_exactly(tmp_path):
    straight, m_straight = make_trainer(tmp_path / "a", steps=20, run="r")
    straight.fit()

    first, _ = make_trainer(tmp_path / "b", steps=20, run="r")
    first.should_stop = lambda: first.history and first.history[-1]["step"] >= 5 or False  # polled once per step
    summary = first.fit()
    assert summary["stopped"] == "requested" and 5 <= summary["steps"] < 20
    assert (tmp_path / "b" / "r" / "latest.json").exists()  # a checkpoint was saved at the stop point

    second, m_second = make_trainer(tmp_path / "b", steps=20, run="r")
    second.fit(resume=tmp_path / "b" / "r")
    for (n, a), (_, b) in zip(m_straight.state_dict().items(), m_second.state_dict().items()):
        assert torch.equal(a.cpu(), b.cpu()), f"{n} differs after a requested stop and resume"
