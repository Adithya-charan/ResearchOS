"""P8 Gate A: tokenizer extensions (batch API, Indic-aware pretokenizer, manifest, statistics, leakage)."""
import json

import pytest

from model.data.leakage import LeakageError, assert_no_leakage, find_overlaps
from model.tokenizer.bpe import (
    EOS_ID,
    PAD_ID,
    BPETokenizer,
    corpus_fingerprint,
    normalize,
    pretokenize,
    pretokenize_with,
)

TELUGU = "తెలుగు భాష పరిశోధన వ్యవస్థ"
TELUGU_CORPUS = [
    "పరిశోధన వ్యవస్థ తెలుగు భాషలో ఫలితాలను వివరిస్తుంది. ",
    "ఈ పరిశోధన ఫలితాలు ప్రయోగాలపై ఆధారపడి ఉన్నాయి. ",
    "तापमान के साथ चालकता बढ़ती है। अनुसंधान परिणाम स्पष्ट हैं। ",
    "Researchers measured ionic conductivity in 2024 and replication is required. ",
] * 30
ENGLISH_CORPUS = [
    "The quick brown fox jumps over the lazy dog. ",
    "Researchers measured ionic conductivity of solid-state electrolytes in 2024.",
    "Claim C1: conductivity increased with temperature.\nStatus: supported.\n",
] * 40


@pytest.fixture(scope="module")
def tok_v1():
    return BPETokenizer.train(TELUGU_CORPUS, vocab_size=400)


@pytest.fixture(scope="module")
def tok_v2():
    return BPETokenizer.train(TELUGU_CORPUS, vocab_size=400, pretokenizer="v2")


def test_v1_shreds_telugu_and_v2_keeps_words():
    n = normalize(TELUGU)
    assert pretokenize_with("v1", n) == pretokenize(n)
    assert len(pretokenize_with("v1", n)) > 10  # one piece per vowel sign / virama
    assert pretokenize_with("v2", n) == ["తెలుగు", " భాష", " పరిశోధన", " వ్యవస్థ"]
    assert "".join(pretokenize_with("v2", n)) == n


def test_v2_leaves_ascii_pieces_identical_to_v1():
    for text in ["hello world", "snake_case_names 4.56e-7", "tab\tand\nnewlines\n\n", "a, b; c!"]:
        assert pretokenize_with("v2", text) == pretokenize(text)


def test_unknown_pretokenizer_rejected():
    with pytest.raises(ValueError):
        BPETokenizer([], pretokenizer="v9")
    with pytest.raises(ValueError):
        pretokenize_with("nope", "x")


@pytest.mark.parametrize("fixture", ["tok_v1", "tok_v2"])
def test_telugu_and_indic_round_trip(fixture, request):
    tok = request.getfixturevalue(fixture)
    for text in [TELUGU, "తెలుగు English मिश्रित 123", "క్ష్మ్య", "క్‌ష", "", "ఆ" * 500]:
        assert tok.decode(tok.encode(text)) == normalize(text)


def test_v2_compresses_telugu_better_than_v1(tok_v1, tok_v2):
    s1 = tok_v1.vocab_statistics([TELUGU])["sample"]["tokens"]
    s2 = tok_v2.vocab_statistics([TELUGU])["sample"]["tokens"]
    assert s2 < s1


def test_v1_fingerprint_is_unchanged_by_the_new_option():
    # v1 must stay byte-identical to pre-P8 definitions so old checkpoints still match.
    t = BPETokenizer([])
    assert "pretokenizer" not in t._definition()
    t2 = BPETokenizer([], pretokenizer="v2")
    assert t2._definition()["pretokenizer"] == "v2"
    assert t.fingerprint != t2.fingerprint


def test_training_is_deterministic_for_both_pretokenizers():
    for name in ("v1", "v2"):
        a = BPETokenizer.train(TELUGU_CORPUS, vocab_size=330, pretokenizer=name)
        b = BPETokenizer.train(list(TELUGU_CORPUS), vocab_size=330, pretokenizer=name)
        assert a.fingerprint == b.fingerprint
        assert a.training_info == b.training_info


def test_corpus_fingerprint_is_order_and_content_sensitive():
    a = corpus_fingerprint(["x", "y"])
    assert a == corpus_fingerprint(["x", "y"])
    assert a["sha256"] != corpus_fingerprint(["y", "x"])["sha256"]
    assert a["sha256"] != corpus_fingerprint(["xy"])["sha256"]  # boundaries matter
    assert a["num_documents"] == 2
    t = BPETokenizer.train(["abc abc abc", "abc abd"], vocab_size=280)
    assert t.training_info["corpus_sha256"] == corpus_fingerprint(["abc abc abc", "abc abd"])["sha256"]


def test_batch_encode_decode_matches_single(tok_v2):
    texts = [TELUGU, "hello", "", "mixed తెలుగు 12"]
    rows = tok_v2.batch_encode(texts)
    assert rows == [tok_v2.encode(t) for t in texts]
    assert tok_v2.batch_decode(rows) == [normalize(t) for t in texts]


def test_batch_encode_padding_truncation_and_specials(tok_v2):
    rows = tok_v2.batch_encode(["ab", "abcdefghij"], add_bos=True, add_eos=True, max_length=6, pad=True)
    assert all(len(r) == 6 for r in rows)
    assert rows[0][-1] == PAD_ID
    assert rows[1][-1] == EOS_ID  # EOS survives truncation
    ragged = tok_v2.batch_encode(["a", "abcdef"], pad=True)
    assert len(ragged[0]) == len(ragged[1])
    assert tok_v2.batch_encode([]) == []
    with pytest.raises(ValueError):
        tok_v2.batch_encode(["a"], add_bos=True, add_eos=True, max_length=0)


def test_batch_decode_skips_padding(tok_v2):
    rows = tok_v2.batch_encode(["hi", "hello there"], pad=True)
    assert tok_v2.batch_decode(rows) == ["hi", "hello there"]


def test_manifest_stats_and_load_round_trip(tmp_path, tok_v2):
    d = tok_v2.save(tmp_path / "tok", sample_texts=TELUGU_CORPUS[:10])
    for name in ["merges.txt", "vocab.json", "tokenizer_config.json", "tokenizer_version.txt", "manifest.json", "vocab_statistics.json"]:
        assert (d / name).is_file(), name
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == tok_v2.fingerprint
    assert manifest["training_corpus"]["corpus_sha256"] == tok_v2.training_info["corpus_sha256"]
    assert manifest["pretokenizer"] == "v2"
    stats = json.loads((d / "vocab_statistics.json").read_text(encoding="utf-8"))
    assert stats["sample"]["unk_tokens"] == 0
    assert stats["merged_tokens"] == len(tok_v2.merges)
    loaded = BPETokenizer.load(d)
    assert loaded.pretokenizer == "v2"
    assert loaded.fingerprint == tok_v2.fingerprint
    assert loaded.training_info == tok_v2.training_info
    assert loaded.encode(TELUGU) == tok_v2.encode(TELUGU)
    assert BPETokenizer.verify_manifest(d) == []


def test_tampering_is_detected(tmp_path, tok_v2):
    d = tok_v2.save(tmp_path / "tok")
    cfg = d / "tokenizer_config.json"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace('"v2"', '"v1"'), encoding="utf-8")
    assert any("tokenizer_config.json" in p for p in BPETokenizer.verify_manifest(d))
    with pytest.raises(ValueError, match="manifest|fingerprint"):
        BPETokenizer.load(d)
    (d / "manifest.json").unlink()
    assert BPETokenizer.verify_manifest(d) == ["manifest.json is missing"]


def test_pre_p8_directory_without_manifest_still_loads(tmp_path):
    tok = BPETokenizer.train(ENGLISH_CORPUS, vocab_size=300)
    d = tok.save(tmp_path / "old")
    for name in ("manifest.json", "vocab_statistics.json"):
        (d / name).unlink()
    cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg.pop("pretokenizer")
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    loaded = BPETokenizer.load(d)
    assert loaded.fingerprint == tok.fingerprint and loaded.pretokenizer == "v1"


def test_vocab_statistics_without_sample():
    tok = BPETokenizer.train(ENGLISH_CORPUS, vocab_size=300)
    s = tok.vocab_statistics()
    assert "sample" not in s
    assert s["vocab_size"] == 262 + len(tok.merges)
    assert sum(s["merged_token_bytes"]["histogram"].values()) == len(tok.merges)


def test_unknown_bytes_and_special_injection_are_safe(tok_v2):
    assert tok_v2.decode([6 + 0xFF, 6 + 0xFE]) == "��"  # invalid UTF-8 never crashes decoding
    assert tok_v2.encode("<EOS>")[0] != EOS_ID  # control tokens cannot be injected from text
    assert tok_v2.encode("a\ud800b")  # lone surrogate is encoded, not raised


def test_leakage_detector_flags_shared_documents_and_text():
    clean = {"train": [("a", "alpha text"), ("b", "beta text")], "val": [("c", "gamma text")], "test": [("d", "delta text")]}
    assert find_overlaps(clean) == []
    assert_no_leakage(clean)
    dup_id = {"train": [("a", "alpha text")], "val": [("a", "other")]}
    assert any("both train and val" in p for p in find_overlaps(dup_id))
    dup_text = {"train": [("a", "same words here")], "test": [("z", "same words here")]}
    with pytest.raises(LeakageError, match="identical text"):
        assert_no_leakage(dup_text)
