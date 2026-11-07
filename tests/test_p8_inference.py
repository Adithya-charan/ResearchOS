"""P8 Gate D: KV-cache correctness across requests, context boundaries, concurrency and batching."""
import threading

import pytest
import torch

from model.architecture import build_model
from model.inference import ContextOverflowError, ResearchOSModel
from model.tokenizer.bpe import DOC_ID, EOS_ID
from tests.conftest import VARIANTS, tiny_config

MAXLEN = 40


def make_rt(**over):
    rt = ResearchOSModel(device="cpu")
    rt.attach(build_model(tiny_config(max_seq_len=MAXLEN, **over), 5), device="cpu")
    return rt


@pytest.fixture(params=list(VARIANTS))
def rt(request):
    return make_rt(**VARIANTS[request.param])


def prompt(n, base=10):
    return [DOC_ID] + [6 + (base + i * 7) % 56 for i in range(n - 1)]  # ids 6..61 (vocab is 64)


def test_cached_and_uncached_logits_agree_at_every_position(rt):
    """Incremental decoding with the cache must give the same next-token logits as a full forward at each step."""
    model = rt.model
    ids = torch.tensor([prompt(MAXLEN)])
    with torch.inference_mode():
        full = model(ids).logits
        out = model(ids[:, :5], use_cache=True)
        past, collected = out.past_key_values, [out.logits]
        for t in range(5, MAXLEN):
            out = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            collected.append(out.logits)
    assert torch.allclose(torch.cat(collected, dim=1), full, atol=1e-4, rtol=1e-4)


def test_cache_with_uneven_chunks_matches_full_forward(rt):
    model = rt.model
    ids = torch.tensor([prompt(30)])
    with torch.inference_mode():
        full = model(ids).logits
        past, parts = None, []
        for a, b in ((0, 3), (3, 4), (4, 17), (17, 30)):
            out = model(ids[:, a:b], past_key_values=past, use_cache=True)
            past = out.past_key_values
            parts.append(out.logits)
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-4, rtol=1e-4)


def test_cache_cannot_grow_past_context(rt):
    ids = torch.tensor([prompt(MAXLEN)])
    with torch.inference_mode():
        out = rt.model(ids, use_cache=True)
        with pytest.raises(ValueError, match="max_seq_len"):
            rt.model(ids[:, :1], past_key_values=out.past_key_values, use_cache=True)


def test_prompt_at_context_boundary(rt):
    with pytest.raises(ContextOverflowError):
        rt.generate(prompt(MAXLEN), 4, temperature=0)
    res = rt.generate(prompt(MAXLEN - 1), 10, temperature=0)  # exactly one token of room
    assert len(res.token_ids) <= 1 and res.finish_reason in ("context", "eos")
    one = rt.generate(prompt(MAXLEN - 2), 10, temperature=0)
    assert len(one.token_ids) <= 2


def test_no_state_leaks_between_consecutive_requests(rt):
    """A request's output is the same whether or not other requests ran before it (a long one, then a short one)."""
    a = rt.generate(prompt(6), 12, temperature=0).token_ids
    rt.generate(prompt(30, base=100), 8, temperature=0)  # a longer, different request in between
    rt.generate(prompt(3, base=200), 5, temperature=0.9, seed=1)
    assert rt.generate(prompt(6), 12, temperature=0).token_ids == a


def test_interleaved_streams_do_not_contaminate_each_other(rt):
    solo_a = rt.generate(prompt(5), 10, temperature=0.8, seed=11).token_ids
    solo_b = rt.generate(prompt(9, base=60), 10, temperature=0.8, seed=22).token_ids
    sa = rt.stream(prompt(5), 10, temperature=0.8, seed=11)
    sb = rt.stream(prompt(9, base=60), 10, temperature=0.8, seed=22)
    out_a, out_b = [], []
    for ta, tb in zip(sa, sb):  # advance both generators in lock-step
        out_a.append(ta)
        out_b.append(tb)
    out_a.extend(sa)
    out_b.extend(sb)
    assert out_a == solo_a and out_b == solo_b


def test_concurrent_threads_get_independent_correct_results():
    rt = make_rt()
    prompts = [prompt(4 + i, base=10 + 13 * i) for i in range(6)]
    expected = [rt.generate(p, 10, temperature=0.7, seed=100 + i).token_ids for i, p in enumerate(prompts)]
    got = [None] * len(prompts)

    def work(i):
        got[i] = rt.generate(prompts[i], 10, temperature=0.7, seed=100 + i).token_ids

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert got == expected


def test_cancellation_stops_generation_and_next_request_is_clean(rt):
    ev = threading.Event()
    ev.set()
    res = rt.generate(prompt(5), 20, temperature=0, cancel_event=ev)
    assert res.finish_reason == "cancelled" and res.token_ids == []
    assert rt.generate(prompt(5), 6, temperature=0).finish_reason in ("length", "eos")


def test_batch_matches_sequential_for_equal_length_prompts(rt):
    prompts = [prompt(6, base=10), prompt(6, base=40), prompt(6, base=90)]
    seeds = [1, 2, 3]
    seq = [rt.generate(p, 12, temperature=0.8, seed=s).token_ids for p, s in zip(prompts, seeds)]
    bat = rt.generate_batch(prompts, 12, temperature=0.8, seeds=seeds)
    assert [r.token_ids for r in bat] == seq


def test_batch_greedy_handles_mixed_lengths_in_input_order(rt):
    prompts = [prompt(5), prompt(9, base=70), prompt(5, base=33), prompt(12, base=20)]
    seq = [rt.generate(p, 8, temperature=0).token_ids for p in prompts]
    res = rt.generate_batch(prompts, 8, temperature=0)
    assert [r.token_ids for r in res] == seq
    assert [r.prompt_tokens for r in res] == [5, 9, 5, 12]


def test_batch_validation_and_limits(rt):
    assert rt.generate_batch([]) == []
    with pytest.raises(ValueError, match="max_batch_size"):
        rt.generate_batch([prompt(4)] * 3, max_batch_size=2)
    with pytest.raises(ValueError):
        rt.generate_batch([[]])
    with pytest.raises(ValueError, match="vocabulary"):
        rt.generate_batch([[10**6]])
    with pytest.raises(ContextOverflowError):
        rt.generate_batch([prompt(MAXLEN)])
    with pytest.raises(ValueError, match="seeds"):
        rt.generate_batch([prompt(4)], seeds=[1, 2])


def test_batch_cancellation_marks_all_rows():
    rt = make_rt()
    ev = threading.Event()
    ev.set()
    res = rt.generate_batch([prompt(5), prompt(5, base=30)], 10, cancel_event=ev)
    assert [r.finish_reason for r in res] == ["cancelled", "cancelled"]


def test_batch_stop_token_ends_that_row_only():
    rt = make_rt()
    free = rt.generate(prompt(5), 6, temperature=0).token_ids
    stop = free[2]
    res = rt.generate_batch([prompt(5)], 6, temperature=0, stop_token_ids=(stop,))
    assert res[0].finish_reason == "eos" and res[0].token_ids == free[:free.index(stop)]


def test_stop_token_default_is_eos():
    rt = make_rt()
    r = rt.generate_batch([prompt(5)], 4, temperature=0)[0]
    assert EOS_ID not in r.token_ids
