import math

import pytest
import torch
import torch.nn.functional as F

from model.architecture import IGNORE_INDEX, ModelConfig, Transformer, build_model, lm_loss, weights_fingerprint
from model.architecture.transformer import apply_rope, build_rope_tables
from tests.conftest import tiny_config


def _ids(cfg, B=3, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (B, T), generator=g)


def test_output_shape(variant_config):
    m = build_model(variant_config, 0).eval()
    x = _ids(variant_config)
    out = m(x)
    assert out.logits.shape == (3, 16, variant_config.vocab_size)
    assert out.loss is None and out.past_key_values is None


def test_causal_mask_blocks_future_tokens(variant_config):
    m = build_model(variant_config, 1).eval()
    x = _ids(variant_config, seed=1)
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 7) % variant_config.vocab_size  # change only the future
    a, b = m(x).logits, m(x2).logits
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-5)


def test_padding_is_excluded_from_loss(variant_config):
    m = build_model(variant_config, 2).eval()
    x = _ids(variant_config, B=2, T=12, seed=2)
    y = _ids(variant_config, B=2, T=12, seed=3)
    y_padded = y.clone()
    y_padded[:, 8:] = IGNORE_INDEX
    logits = m(x).logits
    manual = F.cross_entropy(logits[:, :8].reshape(-1, logits.size(-1)), y[:, :8].reshape(-1))
    assert torch.allclose(lm_loss(logits, y_padded), manual, atol=1e-6)
    y_other = y_padded.clone()  # ignored positions can hold anything else that is also ignored
    assert torch.allclose(lm_loss(logits, y_other), lm_loss(logits, y_padded))
    assert float(lm_loss(logits, torch.full_like(y, IGNORE_INDEX)).detach()) == 0.0  # all-pad batch is not NaN


def test_initial_loss_is_near_log_vocab(variant_config):
    m = build_model(variant_config, 3).eval()
    x, y = _ids(variant_config, seed=4), _ids(variant_config, seed=5)
    assert abs(float(m(x, targets=y).loss.detach()) - math.log(variant_config.vocab_size)) < 0.3


def test_gradients_are_finite_and_nonzero(variant_config):
    cfg = variant_config
    m = build_model(cfg, 4).train()
    out = m(_ids(cfg), targets=_ids(cfg, seed=9))
    out.loss.backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_one_batch_overfit():
    cfg = tiny_config(vocab_size=48, max_seq_len=16, d_model=48, n_layers=2, n_heads=4, d_ff=96)
    m = build_model(cfg, 0).train()
    g = torch.Generator().manual_seed(123)
    seq = torch.randint(0, cfg.vocab_size, (4, 17), generator=g)
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = m(x, targets=y).loss
        first = first if first is not None else float(loss.detach())
        loss.backward()
        opt.step()
    assert first > 3.0
    final = float(loss.detach())
    assert final < 0.3, f"failed to overfit one batch: {first:.3f} -> {final:.3f}"


def test_seeded_init_is_reproducible_and_random():
    cfg = tiny_config()
    a, b, c = build_model(cfg, 7), build_model(cfg, 7), build_model(cfg, 8)
    assert weights_fingerprint(a) == weights_fingerprint(b)
    assert weights_fingerprint(a) != weights_fingerprint(c)
    state = torch.random.get_rng_state()
    build_model(cfg, 99)
    assert torch.equal(state, torch.random.get_rng_state())  # global RNG untouched


def test_parameter_count_estimate_is_exact(variant_config):
    m = build_model(variant_config, 0)
    assert variant_config.estimate_params() == m.num_parameters()


def test_weight_tying_shares_storage():
    tied = build_model(tiny_config(tie_embeddings=True), 0)
    untied = build_model(tiny_config(tie_embeddings=False), 0)
    assert tied.lm_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()
    assert untied.num_parameters() - tied.num_parameters() == 64 * 32


def test_blueprint_starter_config_size():
    cfg = ModelConfig()  # vocab 4096, d_model 256, 6 layers, 8 heads, d_ff 1024
    assert 5_000_000 < cfg.estimate_params() < 7_000_000


def test_kv_cache_matches_full_forward(variant_config):
    cfg = variant_config
    m = build_model(cfg, 5).eval()
    x = _ids(cfg, B=2, T=20, seed=6)
    with torch.no_grad():
        full = m(x).logits
        out = m(x[:, :11], use_cache=True)  # prefill
        pieces, past = [out.logits], out.past_key_values
        for t in range(11, 20):  # then one token at a time
            out = m(x[:, t : t + 1], past_key_values=past, use_cache=True)
            pieces.append(out.logits)
            past = out.past_key_values
        # multi-token continuation with a cache (T>1 and S>T) must also be correct
        out2 = m(x[:, 11:20], past_key_values=m(x[:, :11], use_cache=True).past_key_values, use_cache=True)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-4)
    assert torch.allclose(out2.logits, full[:, 11:], atol=1e-4)
    assert past[0][0].shape[2] == 20


def test_logits_to_keep(variant_config):
    m = build_model(variant_config, 5).eval()
    x = _ids(variant_config)
    assert torch.allclose(m(x, logits_to_keep=1).logits, m(x).logits[:, -1:], atol=1e-6)
    with pytest.raises(ValueError):
        m(x, targets=x, logits_to_keep=1)


def test_exceeding_context_raises():
    m = build_model(tiny_config(), 0).eval()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))


def test_manual_attention_equals_sdpa():
    cfg_m, cfg_s = tiny_config(), tiny_config(attention_impl="sdpa")
    manual = build_model(cfg_m, 0).eval()
    fused = build_model(cfg_s, 0).eval()
    fused.load_state_dict(manual.state_dict())
    x = _ids(cfg_m)
    assert torch.allclose(manual(x).logits, fused(x).logits, atol=1e-5)


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = build_rope_tables(8, 64, 10000.0)
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(1, 1, 1, 8, generator=g), torch.randn(1, 1, 1, 8, generator=g)

    def score(i, j):
        qi = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return float((qi * kj).sum())

    assert score(3, 1) == pytest.approx(score(13, 11), abs=1e-5)
    assert score(3, 1) != pytest.approx(score(3, 2), abs=1e-3)


def test_activation_checkpointing_gives_same_gradients():
    cfg = tiny_config(dropout=0.0)
    a, b = build_model(cfg, 0).train(), build_model(cfg, 0).train()
    b.gradient_checkpointing = True
    x, y = _ids(cfg), _ids(cfg, seed=11)
    a(x, targets=y).loss.backward()
    b(x, targets=y).loss.backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cpu_and_gpu_outputs_match(variant_config):
    cpu = build_model(variant_config, 0).eval()
    gpu = Transformer(variant_config).eval().cuda()
    gpu.load_state_dict(cpu.state_dict())
    x = _ids(variant_config)
    with torch.no_grad():
        assert torch.allclose(cpu(x).logits, gpu(x.cuda()).logits.cpu(), atol=1e-3, rtol=1e-3)
