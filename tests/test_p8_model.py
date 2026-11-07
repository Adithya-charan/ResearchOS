"""P8 Gates B/C/E: profiles and memory guards, trainer run records, lifecycle rules, evaluation suite.
Everything uses tiny models and in-memory/tmp data on CPU; nothing here downloads data or trains anything large."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from model.architecture.config import ModelConfig
from model.architecture.transformer import IGNORE_INDEX, Transformer, build_model
from model.data.dataset import TokenDataset, write_token_file
from model.evaluation import suite
from model.lifecycle import (
    ALLOWED,
    Evidence,
    LifecycleError,
    ModelState,
    check_transition,
    experimental_eligible,
    production_eligible,
    transition,
)
from model.profiles import PROFILES, check_profile, get_profile, measure_profile, profile_estimate
from model.tokenizer.bpe import BOS_ID, EOS_ID, BPETokenizer
from model.training.memory import (
    MemoryBudgetError,
    available_budget,
    check_budget,
    estimate_training_memory,
    hardware_report,
    max_safe_batch_size,
)
from model.training.trainer import TrainConfig, Trainer

TEXT = (
    "Claim C1: conductivity increased with temperature.\nStatus: supported.\nEvidence: doc-1, page 3.\n"
    "Research question: How does temperature affect the ionic conductivity of the material?\n"
) * 60


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train([TEXT, "A confounder is a variable that influences both the exposure and the outcome."], vocab_size=400)


@pytest.fixture(scope="module")
def data_files(tmp_path_factory, tok):
    d = tmp_path_factory.mktemp("p8data")
    ids = tok.encode_ordinary(TEXT)
    n = len(ids)
    paths = {}
    for name, part in (("train", ids[: int(n * 0.8)]), ("val", ids[int(n * 0.8):int(n * 0.9)]), ("test", ids[int(n * 0.9):])):
        paths[name] = d / f"{name}.bin"
        write_token_file(paths[name], part)
    return paths


def tiny_cfg(vocab, **kw):
    base = dict(vocab_size=vocab, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def make_trainer(tmp_path, tok, data_files, **cfg_kw):
    cfgd = dict(run_name="t", seed=3, batch_size=4, grad_accum_steps=1, max_steps=20, learning_rate=3e-3, warmup_steps=2,
                eval_interval=10, eval_batches=2, checkpoint_interval=10, log_interval=5, mixed_precision="none", device="cpu")
    cfgd.update(cfg_kw)
    model = build_model(tiny_cfg(tok.vocab_size), 3)
    train = TokenDataset(data_files["train"], 32)
    val = TokenDataset(data_files["val"], 32)
    return Trainer(model, TrainConfig(**cfgd), train, val, tmp_path / "run",
                   meta_base={"tokenizer_fingerprint": tok.fingerprint, "dataset_fingerprint": "ds-test"}, log=lambda m: None)


# ------------------------------------------------------------------ profiles / memory
def test_three_profiles_exist_and_build():
    assert set(PROFILES) == {"CPU_DEBUG", "CONSUMER_SMALL", "CONSUMER_MEDIUM"}
    sizes = {n: p.model_config().estimate_params() for n, p in PROFILES.items()}
    assert sizes["CPU_DEBUG"] < sizes["CONSUMER_SMALL"] < sizes["CONSUMER_MEDIUM"]
    assert get_profile("cpu_debug").name == "CPU_DEBUG"
    with pytest.raises(ValueError):
        get_profile("HUGE")


def test_profile_vocab_override_and_train_overrides():
    p = get_profile("CPU_DEBUG")
    assert p.model_config(300).vocab_size == 300
    assert p.train_config(max_steps=3).max_steps == 3
    with pytest.raises(ValueError):
        p.train_config(not_a_field=1)


def test_cpu_debug_profile_really_runs_one_step():
    r = measure_profile(get_profile("CPU_DEBUG"), device="cpu", steps=2)
    assert r["ok"] and r["basis"] == "measurement"
    assert r["final_loss_finite"] and r["parameters"] > 0 and r["tokens_per_second"] > 0
    assert "peak_gpu_bytes" not in r


@pytest.mark.parametrize("profile,config", [("CPU_DEBUG", "cpu_debug"), ("CONSUMER_MEDIUM", "consumer_medium"), ("CONSUMER_SMALL", "small")])
def test_shipped_yaml_configs_match_their_profile(profile, config):
    """The profile table and the YAML a training job starts from must describe the same model and hyper-parameters."""
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / f"{config}.yaml").read_text(encoding="utf-8"))
    p = get_profile(profile)
    assert raw["model"]["vocab_size"] == "auto"
    got = ModelConfig.from_dict({**raw["model"], "vocab_size": 1000})
    want = p.model_config(1000)
    assert got == want, f"{config}.yaml model differs from profile {profile}"
    for k, v in p.train.items():
        if k != "run_name":
            assert raw["training"].get(k, TrainConfig.from_dict({}).__dict__.get(k)) == v or k in ("mixed_precision", "device"), (profile, k)
    assert TrainConfig.from_dict(raw["training"]).max_steps == p.train_config().max_steps


def test_check_profile_is_labelled_as_estimate():
    r = check_profile(get_profile("CONSUMER_SMALL"), device="cpu")
    assert r["basis"] == "estimate" and r["estimated_bytes"] > 0


def test_memory_estimate_monotonic_and_checkpointing_helps():
    cfg = tiny_cfg(400, max_seq_len=128, n_layers=6)
    a = estimate_training_memory(cfg, 4)
    b = estimate_training_memory(cfg, 16)
    assert b.total > a.total and b.activations > a.activations
    c = estimate_training_memory(cfg, 16, activation_checkpointing=True)
    assert c.activations < b.activations
    d = estimate_training_memory(cfg, 16, mixed_precision=True)
    assert d.activations < b.activations
    manual = estimate_training_memory(tiny_cfg(400, attention_impl="manual"), 4)
    fused = estimate_training_memory(tiny_cfg(400, attention_impl="sdpa"), 4)
    assert manual.activations > fused.activations


def test_budget_check_refuses_and_names_remedies():
    est = estimate_training_memory(tiny_cfg(400), 4)
    check_budget(est, est.total + 1)  # fits
    with pytest.raises(MemoryBudgetError, match="batch_size"):
        check_budget(est, est.total // 2)


def test_max_safe_batch_size_respects_budget_and_cap():
    cfg = tiny_cfg(400, max_seq_len=64)
    budget = estimate_training_memory(cfg, 10).total
    n = max_safe_batch_size(cfg, budget)
    assert n >= 10 and estimate_training_memory(cfg, n).total <= budget
    assert estimate_training_memory(cfg, n + 1).total > budget or n == 256
    assert max_safe_batch_size(cfg, 1) == 0
    assert max_safe_batch_size(cfg, 10**15, upper=32) == 32  # never suggests beyond the cap


def test_hardware_report_has_real_fields():
    h = hardware_report()
    assert h["torch"] and h["cpu_count"] and isinstance(h["cuda_available"], bool)
    assert available_budget("cpu") >= 0


# ------------------------------------------------------------------ trainer run record
def test_run_record_and_summary_fields(tmp_path, tok, data_files):
    tr = make_trainer(tmp_path, tok, data_files)
    summary = tr.fit()
    rec = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert rec["status"] == "finished" and rec["completion_state"] == "completed"
    for key in ("run_id", "started_at", "finished_at", "seed", "parameters", "model_config", "train_config", "optimizer",
                "tokenizer_fingerprint", "dataset_fingerprint", "hardware", "memory_estimate", "duration_seconds",
                "initial_loss", "final_loss", "tokens_seen", "peak_ram_mb"):
        assert key in rec, key
    assert rec["tokenizer_fingerprint"] == tok.fingerprint and rec["dataset_fingerprint"] == "ds-test"
    assert rec["final_loss"] < rec["initial_loss"]
    assert summary["completion_state"] == "completed" and summary["steps"] == 20
    assert summary["run_id"] == rec["run_id"]
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert any(x["type"] == "eval" for x in lines) and any(x["type"] == "train" for x in lines)


def test_memory_budget_refusal_is_recorded_and_runs_no_step(tmp_path, tok, data_files):
    tr = make_trainer(tmp_path, tok, data_files, max_memory_gb=1e-6)
    with pytest.raises(MemoryBudgetError):
        tr.fit()
    rec = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert rec["status"] == "failed" and rec["completion_state"] == "refused_memory_budget"
    assert not list((tmp_path / "run").glob("step_*"))


def test_failure_is_recorded_and_last_checkpoint_survives(tmp_path, tok, data_files):
    calls = {"n": 0}

    def boom(step, model, device):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("synthetic failure")
        return {}

    tr = make_trainer(tmp_path, tok, data_files)
    tr.on_eval = boom
    with pytest.raises(RuntimeError, match="synthetic failure"):
        tr.fit()
    rec = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert rec["status"] == "failed" and "synthetic failure" in rec["error"]
    assert rec["last_checkpoint"] and Path(rec["last_checkpoint"]).exists()
    # the failed run can be resumed from that checkpoint and finishes
    tr2 = make_trainer(tmp_path, tok, data_files)
    out = tr2.fit(resume=tmp_path / "run")
    assert out["completion_state"] == "completed" and out["steps"] == 20


def test_cancellation_and_pause_states(tmp_path, tok, data_files):
    tr = make_trainer(tmp_path, tok, data_files)
    tr.should_stop = lambda: True
    s = tr.fit()
    assert s["stopped"] == "requested" and s["completion_state"] == "stopped_by_request" and s["steps"] == 1
    rec = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert rec["status"] == "stopped_by_request"  # a stopped run is never reported as finished


def test_max_epochs_stops_early(tmp_path, tok, data_files):
    tr = make_trainer(tmp_path, tok, data_files, max_steps=10_000, max_epochs=0.5, eval_interval=10_000, checkpoint_interval=10_000)
    s = tr.fit()
    assert s["stopped"] == "max_epochs" and 0.5 <= s["epochs"] < 1.5 and s["completion_state"] == "completed"


def test_eval_perplexity_is_token_weighted():
    from model.evaluation.metrics import evaluate_loss

    class FakeData:
        seq_len = 4

        def eval_batches(self, bs, mb):
            yield torch.zeros(1, 4, dtype=torch.long), torch.tensor([[1, 1, 1, 1]])
            yield torch.zeros(1, 4, dtype=torch.long), torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 1]])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, targets=None):
            self.calls += 1
            from types import SimpleNamespace
            return SimpleNamespace(loss=torch.tensor(1.0 if self.calls == 1 else 3.0))

    r = evaluate_loss(FakeModel(), FakeData(), 1, None, "cpu")
    assert r["tokens"] == 5
    assert r["loss"] == pytest.approx((1.0 * 4 + 3.0 * 1) / 5)  # not the unweighted mean 2.0
    assert r["perplexity"] == pytest.approx(math.exp(r["loss"]))


# ------------------------------------------------------------------ lifecycle
def good_validation(**kw):
    base = dict(steps_completed=20, completion_state="completed", final_loss=1.2, checkpoint_verified=True,
                lineage_verified=True, val_loss=1.5, baseline_val_loss=6.0, leakage_checked=True, leakage_found=False)
    base.update(kw)
    return Evidence(**base)


def test_initialized_model_cannot_jump_to_trained_or_validated():
    for target in ("TRAINED", "VALIDATED", "DEPLOYED"):
        assert check_transition("INITIALIZED", target, good_validation(authorized_by="admin"))  # not an allowed arrow


def test_trained_needs_real_completed_run():
    ok = good_validation()
    assert transition("TRAINING", "TRAINED", ok) == ModelState.TRAINED
    assert check_transition("TRAINING", "TRAINED", good_validation(steps_completed=0))
    assert check_transition("TRAINING", "TRAINED", good_validation(completion_state="stopped_by_request"))
    assert check_transition("TRAINING", "TRAINED", good_validation(final_loss=float("nan")))
    assert check_transition("TRAINING", "TRAINED", good_validation(lineage_verified=False))


def test_training_back_to_initialized_only_without_steps():
    assert not check_transition("TRAINING", "INITIALIZED", Evidence(steps_completed=0))
    assert check_transition("TRAINING", "INITIALIZED", Evidence(steps_completed=3))


def test_validated_requires_successful_validation():
    assert transition("TRAINED", "VALIDATED", good_validation()) == ModelState.VALIDATED
    cases = [
        dict(checkpoint_verified=False), dict(lineage_verified=False), dict(val_loss=None), dict(val_loss=float("inf")),
        dict(baseline_val_loss=None), dict(val_loss=7.0), dict(leakage_checked=False), dict(leakage_found=True),
    ]
    for kw in cases:
        assert check_transition("TRAINED", "VALIDATED", good_validation(**kw)), kw


def test_deploy_needs_authorisation_and_validated_state():
    assert check_transition("VALIDATED", "DEPLOYED", good_validation())
    assert transition("VALIDATED", "DEPLOYED", good_validation(authorized_by="admin_1")) == ModelState.DEPLOYED
    assert check_transition("TRAINED", "DEPLOYED", good_validation(authorized_by="admin_1"))


def test_archived_is_terminal_and_failed_can_retry():
    assert ALLOWED[ModelState.ARCHIVED] == frozenset()
    assert not check_transition("FAILED", "TRAINING")
    assert check_transition("ARCHIVED", "TRAINING")
    with pytest.raises(LifecycleError):
        transition("ARCHIVED", "VALIDATED", good_validation())


def test_eligibility_policy():
    assert production_eligible("DEPLOYED") and not production_eligible("VALIDATED")
    assert not experimental_eligible("INITIALIZED") and not experimental_eligible("TRAINING")
    assert experimental_eligible("TRAINED") and experimental_eligible("VALIDATED")
    assert not experimental_eligible("FAILED") and not experimental_eligible("ARCHIVED")


# ------------------------------------------------------------------ evaluation suite
def test_task_sets_load_verify_and_detect_tampering(tmp_path):
    sets = suite.load_task_sets()
    assert set(sets) == {"general_lm", "scientific_vocabulary", "research_question_completion", "abstract_continuation",
                         "citation_format", "structured_output", "research_terminology"}
    assert all(len(s["items"]) >= 4 for s in sets.values())
    copy = tmp_path / "ts"
    copy.mkdir()
    for f in suite.TASK_SET_DIR.iterdir():
        (copy / f.name).write_bytes(f.read_bytes())
    (copy / "general_lm.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(suite.TaskSetError, match="manifest hash"):
        suite.load_task_sets(copy)


def test_completion_items_score_only_the_continuation(tok):
    item = {"kind": "completion", "prefix": "Research question: How does", "continuation": " temperature matter?"}
    r = suite.encode_text_item(tok, item, 64)
    scored = [t for t in r.targets if t != IGNORE_INDEX]
    cont = tok.encode_ordinary(item["continuation"]) + [EOS_ID]
    assert scored == cont
    assert r.ids[0] == BOS_ID and len(r.ids) == len(r.targets)


def test_plain_items_score_every_token_after_bos(tok):
    r = suite.encode_text_item(tok, {"kind": "text", "text": "hello world"}, 64)
    body = tok.encode_ordinary("hello world") + [EOS_ID]
    assert r.targets == body and r.ids == [BOS_ID] + body[:-1]


def test_long_items_are_left_truncated_keeping_the_scored_tail(tok):
    item = {"kind": "text", "text": "word " * 100}
    r = suite.encode_text_item(tok, item, 16)
    assert len(r.ids) == 16 and r.targets[-1] == EOS_ID


def test_scoring_matches_manual_cross_entropy_and_ignores_padding(tok):
    model = build_model(tiny_cfg(tok.vocab_size), 1).eval()
    a = suite.ScoredText([BOS_ID, 10, 11], [10, 11, 12])
    b = suite.ScoredText([BOS_ID, 10, 11, 13, 14, 15], [10, 11, 13, 14, 15, 16])  # forces padding of `a`
    res = suite._score_batches(model, [a, b], torch.device("cpu"), batch_size=2)
    alone = [suite._score_batches(model, [x], torch.device("cpu")) for x in (a, b)]
    manual = sum(r["loss"] * r["tokens"] for r in alone) / sum(r["tokens"] for r in alone)
    assert res["tokens"] == 9
    assert res["loss"] == pytest.approx(manual, rel=1e-4)  # padding contributes nothing


def test_random_baseline_is_near_chance_and_deterministic(tok):
    model = build_model(tiny_cfg(tok.vocab_size), 99)
    b1, b2 = suite.random_baseline(model, 5), suite.random_baseline(model, 5)
    assert all(torch.equal(x, y) for x, y in zip(b1.state_dict().values(), b2.state_dict().values()))
    ids = tok.encode_ordinary(TEXT)[:600]
    r = suite.score_token_stream(b1, ids, torch.device("cpu"))
    assert abs(r["loss"] - math.log(tok.vocab_size)) < 0.5 and r["token_accuracy"] < 0.2


def test_trained_model_beats_baseline_and_report_is_complete(tmp_path, tok, data_files):
    tr = make_trainer(tmp_path, tok, data_files, max_steps=60, eval_interval=30, checkpoint_interval=30, learning_rate=5e-3)
    tr.fit()
    out = tmp_path / "report.json"
    train_tokens = np.asarray(TokenDataset(data_files["train"], 32).tokens)
    rep = suite.run_evaluation(
        tr.model, tok, "cpu", val_tokens=np.asarray(TokenDataset(data_files["val"], 32).tokens),
        test_tokens=np.asarray(TokenDataset(data_files["test"], 32).tokens), train_tokens=train_tokens,
        checkpoint=tmp_path / "run", max_windows=20, out_path=out,
    )
    assert rep["beats_random_baseline"] and rep["validation_loss"] < rep["baseline_validation_loss"]
    assert set(rep["splits"]) == {"validation", "test"}
    assert len(rep["tasks"]) == 7
    assert rep["checkpoint_bytes"] > 0 and rep["parameters"] > 0 and rep["hardware"]["cpu_count"]
    inf = rep["inference"]
    assert inf["cached"]["tokens_per_second"] > 0 and inf["uncached"]["ms_per_token"] > 0 and "cache_speedup" in inf
    assert "none" in rep["comparison_with_other_models"] and rep["caveats"]
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "researchos-model-evaluation-v1"
    # this synthetic stream repeats itself, so heavy overlap must be flagged rather than reported as capability
    assert rep["splits"]["validation"]["train_overlap"]["memorization_risk"] is True
    assert rep["memorization_risk"] is True


def test_train_overlap_detects_shared_ngrams_and_clean_text():
    rng = np.random.default_rng(0)
    train = rng.integers(10, 5000, size=5000)
    shared = suite.ScoredText(train[100:140].tolist(), train[101:141].tolist())
    fresh = suite.ScoredText(rng.integers(6000, 7000, size=40).tolist(), [0] * 40)
    assert suite.train_overlap([shared], train)["overlap_fraction"] == pytest.approx(1.0)
    assert suite.train_overlap([fresh], train)["overlap_fraction"] == 0.0
    assert suite.train_overlap([fresh], None)["checked"] is False
    assert suite.train_overlap([suite.ScoredText([1, 2], [2, 3])], train)["eval_ngrams"] == 0  # shorter than n


def test_task_sets_do_not_appear_in_the_bundled_training_corpus():
    """The probe sets are written separately from the synthetic corpus: no probe text may be a training document."""
    train_bin = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "synthetic_v1" / "train.bin"
    tok_dir = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer"
    if not (train_bin.exists() and tok_dir.exists()):
        pytest.skip("bundled synthetic corpus not prepared")
    tk = BPETokenizer.load(tok_dir)
    train = np.memmap(train_bin, dtype=np.uint16, mode="r")
    for name, spec in suite.load_task_sets().items():
        rows = [r for r in (suite.encode_text_item(tk, it, 256) for it in spec["items"]) if r]
        ov = suite.train_overlap(rows, np.asarray(train[:2_000_000]))
        assert ov["overlap_fraction"] < 0.5, (name, ov)
