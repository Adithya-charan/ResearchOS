import json
import threading

import pytest
import torch

from model.architecture import build_model
from model.inference import ContextOverflowError, ResearchOSModel, TokenizerMismatchError
from model.inference.sampling import (
    apply_repetition_penalty,
    sample_next,
    top_k_filter,
    top_p_filter,
)
from model.inference.runtime import _StopMatcher
from model.tokenizer.bpe import BOS_ID, DOC_ID, EOS_ID, PAD_ID, UNK_ID, BPETokenizer
from model.training.checkpoint import save_checkpoint
from model.training.lineage import new_lineage
from tests.conftest import VARIANTS, tiny_config

PROMPT = [DOC_ID, 10, 11, 12]


def make_runtime(**cfg_over):
    cfg = tiny_config(max_seq_len=48, **cfg_over)
    rt = ResearchOSModel(device="cpu")
    rt.attach(build_model(cfg, 3), device="cpu")
    return rt


@pytest.fixture(params=["baseline", "rope_rms_swiglu", "sdpa"])
def rt(request):
    return make_runtime(**VARIANTS[request.param])


# ---------------------------------------------------------------- determinism and cache
def test_same_seed_same_output_different_seed_differs(rt):
    kw = dict(max_new_tokens=24, temperature=1.0, top_k=0, stop_token_ids=())
    a = rt.generate(PROMPT, seed=1, **kw).token_ids
    assert a == rt.generate(PROMPT, seed=1, **kw).token_ids
    assert a != rt.generate(PROMPT, seed=2, **kw).token_ids


def test_greedy_is_deterministic_without_seed(rt):
    kw = dict(max_new_tokens=12, temperature=0.0, stop_token_ids=())
    assert rt.generate(PROMPT, **kw).token_ids == rt.generate(PROMPT, **kw).token_ids


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_kv_cache_generation_equals_uncached(rt, temperature):
    kw = dict(max_new_tokens=30, temperature=temperature, top_k=20, seed=5, stop_token_ids=())
    cached = rt.generate(PROMPT, use_cache=True, **kw)
    plain = rt.generate(PROMPT, use_cache=False, **kw)
    assert cached.token_ids == plain.token_ids and len(cached.token_ids) == 30


def test_stream_matches_generate(rt):
    kw = dict(max_new_tokens=16, temperature=1.0, seed=9, stop_token_ids=())
    assert list(rt.stream(PROMPT, **kw)) == rt.generate(PROMPT, **kw).token_ids


# ---------------------------------------------------------------- stopping
def test_stop_token_ends_generation_and_is_excluded(rt):
    kw = dict(max_new_tokens=20, temperature=0.0)
    full = rt.generate(PROMPT, stop_token_ids=(), **kw).token_ids
    stop = full[2]
    k = full.index(stop)
    res = rt.generate(PROMPT, stop_token_ids=(stop,), **kw)
    assert res.token_ids == full[:k] and res.finish_reason == "eos"


def test_stop_sequence_is_trimmed_and_streaming_agrees(rt):
    kw = dict(max_new_tokens=20, temperature=0.0, stop_token_ids=())
    full = rt.generate(PROMPT, **kw).token_ids
    seq = (full[3], full[4])
    first = next(i for i in range(len(full) - 1) if (full[i], full[i + 1]) == seq)
    res = rt.generate(PROMPT, stop_sequences=[seq], **kw)
    assert res.token_ids == full[:first] and res.finish_reason == "stop_sequence"
    assert list(rt.stream(PROMPT, stop_sequences=[seq], **kw)) == full[:first]


def test_stop_matcher_never_emits_part_of_a_stop_sequence():
    m = _StopMatcher([(1, 2, 3)])
    out = []
    for t in (9, 1, 2, 7):  # (1,2) looked like a stop prefix but 7 broke it
        emit, stopped = m.push(t)
        out += emit
        assert not stopped
    assert out + m.flush() == [9, 1, 2, 7]
    m = _StopMatcher([(1, 2, 3)])
    out, stopped = [], False
    for t in (5, 1, 2, 3, 8):
        emit, stopped = m.push(t)
        out += emit
        if stopped:
            break
    assert stopped and out == [5]


def test_length_and_context_limits(rt):
    res = rt.generate(PROMPT, max_new_tokens=5, stop_token_ids=())
    assert res.finish_reason == "length" and len(res.token_ids) == 5
    long_prompt = [DOC_ID] + [10] * 45  # 46 of 48 positions used
    res = rt.generate(long_prompt, max_new_tokens=50, stop_token_ids=())
    assert res.finish_reason == "context" and len(res.token_ids) == 2


def test_prompt_validation(rt):
    with pytest.raises(ContextOverflowError):
        rt.generate([5] * 48, max_new_tokens=1)
    with pytest.raises(ValueError):
        rt.generate([], max_new_tokens=1)
    with pytest.raises(ValueError):
        rt.generate([10_000], max_new_tokens=1)


def test_cancellation(rt):
    ev = threading.Event()
    ev.set()
    res = rt.generate(PROMPT, max_new_tokens=30, cancel_event=ev, stop_token_ids=())
    assert res.finish_reason == "cancelled" and res.token_ids == []
    ev = threading.Event()
    got = []
    for tok in rt.stream(PROMPT, max_new_tokens=30, cancel_event=ev, stop_token_ids=(), seed=1):
        got.append(tok)
        if len(got) == 3:
            ev.set()
    assert len(got) == 3


def test_model_never_emits_reserved_input_tokens(rt):
    out = rt.generate(PROMPT, max_new_tokens=40, temperature=2.0, top_k=0, seed=3, stop_token_ids=()).token_ids
    assert not {PAD_ID, UNK_ID, BOS_ID} & set(out)


# ---------------------------------------------------------------- sampling primitives
def test_top_k_keeps_exactly_k():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    out = top_k_filter(logits, 2)
    assert torch.isfinite(out).sum() == 2 and out[1] == 5.0 and out[3] == 4.0
    assert torch.equal(top_k_filter(logits, 0), logits)


def test_top_p_keeps_smallest_nucleus_and_always_the_best():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    assert torch.isfinite(top_p_filter(logits, 0.7)).sum() == 2  # 0.5 + 0.3 >= 0.7
    assert torch.isfinite(top_p_filter(logits, 0.01)).sum() == 1
    assert torch.equal(top_p_filter(logits, 1.0), logits)


def test_repetition_penalty_lowers_seen_tokens_only():
    logits = torch.tensor([2.0, -2.0, 1.0])
    out = apply_repetition_penalty(logits, [0, 1], 2.0)
    assert out[0] == 1.0 and out[1] == -4.0 and out[2] == 1.0
    assert logits[0] == 2.0  # input not mutated


def test_sample_next_greedy_and_banned():
    logits = torch.tensor([0.0, 9.0, 1.0, 2.0])
    assert sample_next(logits, temperature=0.0, banned=()) == 1
    assert sample_next(logits, temperature=0.0, banned=(1,)) == 3
    g = torch.Generator().manual_seed(0)
    draws = {sample_next(torch.zeros(10), temperature=1.0, top_k=0, banned=(0, 1, 2), generator=g) for _ in range(200)}
    assert draws.isdisjoint({0, 1, 2}) and len(draws) > 3


# ---------------------------------------------------------------- checkpoints and identity
def test_load_checkpoint_and_tokenizer_identity(tmp_path):
    tok = BPETokenizer.train(["some training text for the tokenizer " * 20], vocab_size=300)
    tok.save(tmp_path / "tok")
    cfg = tiny_config(vocab_size=tok.vocab_size, max_seq_len=48)
    model = build_model(cfg, 1)
    meta = {"lineage": new_lineage(model, 1), "tokenizer": {"fingerprint": tok.fingerprint}}
    save_checkpoint(tmp_path / "run", 7, model, None, meta)

    rt = ResearchOSModel(device="cpu")
    rt.load_checkpoint(tmp_path / "run", tokenizer_dir=tmp_path / "tok")
    out = rt.generate_text("some training", max_new_tokens=8, seed=1, stop_token_ids=())
    assert out.text is not None and len(out.token_ids) == 8
    logits = rt.forward([DOC_ID, 7, 8])
    assert logits.shape == (1, 3, tok.vocab_size)

    rt.save_runtime_state(tmp_path / "state.json")
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["step"] == 7 and state["tokenizer_fingerprint"] == tok.fingerprint
    assert state["lineage"]["root"] == "random_init"

    other = BPETokenizer.train(["completely different corpus words here " * 20], vocab_size=300)
    other.save(tmp_path / "tok2")
    with pytest.raises(TokenizerMismatchError):
        ResearchOSModel(device="cpu").load_checkpoint(tmp_path / "run", tokenizer_dir=tmp_path / "tok2")


def test_runtime_requires_a_loaded_model():
    with pytest.raises(RuntimeError):
        ResearchOSModel(device="cpu").generate([1, 2], max_new_tokens=1)
