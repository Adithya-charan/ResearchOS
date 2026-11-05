import os
import subprocess
import sys
from pathlib import Path

import pytest

from model.tokenizer.bpe import (
    BASE_VOCAB_SIZE,
    BOS_ID,
    DOC_ID,
    EOS_ID,
    NUM_SPECIAL,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    BPETokenizer,
    normalize,
    pretokenize,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = [
    "The quick brown fox jumps over the lazy dog. ",
    "Researchers measured ionic conductivity of solid-state electrolytes in 2024.",
    "Results: the effect was 12.5 percent (p = 0.03), but replication is required.\n",
    "Claim C1: conductivity increased with temperature.\nStatus: supported.\n",
] * 40

SAMPLES = [
    "",
    "hello world",
    "  leading and trailing  ",
    "tab\tand\nnewlines\n\n\n\nmany",
    "naïve café — ünïcödé",
    "日本語のテキストと English",
    "emoji 🚀🔬 mixed",
    "numbers 1234567890 and 4.56e-7",
    "snake_case_names and __dunder__",
    "é combining mark",
    "control \x00\x01\x7f chars",
    "<EOS> <PAD> literal special-looking text",
]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=400)


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == normalize(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_pretokenizer_is_lossless(text):
    assert "".join(pretokenize(normalize(text))) == normalize(text)


def test_lone_surrogate_does_not_crash(tok):
    assert isinstance(tok.encode("bad \ud800 text"), list)


def test_vocab_layout(tok):
    assert tok.vocab_size == BASE_VOCAB_SIZE + len(tok.merges) <= 400
    assert len(tok.merges) > 50
    assert [SPECIAL_TOKENS.index(t) for t in ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<DOC>", "<SEP>")] == [
        PAD_ID, UNK_ID, BOS_ID, EOS_ID, DOC_ID, 5,
    ]
    for i in range(NUM_SPECIAL, NUM_SPECIAL + 256):
        assert tok.token_bytes(i) == bytes([i - NUM_SPECIAL])


def test_every_encoded_id_is_in_range_and_never_unk(tok):
    for text in SAMPLES:
        ids = tok.encode(text)
        assert all(NUM_SPECIAL <= i < tok.vocab_size for i in ids)
        assert UNK_ID not in ids


def test_compresses_in_domain_text(tok):
    text = "Researchers measured ionic conductivity of solid-state electrolytes"
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_special_tokens_are_data_by_default(tok):
    assert EOS_ID not in tok.encode("<EOS>")  # untrusted text cannot inject control tokens
    assert tok.encode("a <EOS> b", allow_special=True).count(EOS_ID) == 1
    assert tok.encode("x", add_bos=True, add_eos=True) == [BOS_ID, *tok.encode("x"), EOS_ID]
    assert tok.decode([BOS_ID, *tok.encode("hi"), EOS_ID]) == "hi"
    assert tok.decode([BOS_ID, EOS_ID], skip_special=False) == "<BOS><EOS>"


def test_decode_rejects_out_of_range(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])
    with pytest.raises(ValueError):
        tok.decode([-1])


def test_training_is_deterministic_in_process(tok):
    again = BPETokenizer.train(CORPUS, vocab_size=400)
    assert again.merges == tok.merges
    assert again.fingerprint == tok.fingerprint


def test_training_is_independent_of_python_hash_seed():
    code = (
        "from tests.test_tokenizer import CORPUS\n"
        "from model.tokenizer.bpe import BPETokenizer\n"
        "print(BPETokenizer.train(CORPUS, vocab_size=400).fingerprint)\n"
    )
    prints = set()
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        prints.add(r.stdout.strip())
    assert len(prints) == 1


def test_training_stops_when_no_pair_is_frequent_enough():
    tiny = BPETokenizer.train(["ab ab ab"], vocab_size=4096)
    assert tiny.vocab_size < 4096
    rare = BPETokenizer.train(["abcdefg"], vocab_size=4096, min_frequency=2)
    assert rare.merges == []


def test_save_load_round_trip(tok, tmp_path):
    tok.save(tmp_path / "tok")
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer_version.txt"):
        assert (tmp_path / "tok" / name).exists()
    loaded = BPETokenizer.load(tmp_path / "tok")
    assert loaded.fingerprint == tok.fingerprint
    for text in SAMPLES:
        assert loaded.encode(text) == tok.encode(text)


def test_tampered_merges_are_detected(tok, tmp_path):
    tok.save(tmp_path / "tok")
    merges = tmp_path / "tok" / "merges.txt"
    lines = merges.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]  # swapping two merge ranks silently changes token ids
    merges.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        BPETokenizer.load(tmp_path / "tok")


def test_invalid_merge_tables_rejected():
    with pytest.raises(ValueError):
        BPETokenizer([(10_000, 7)])
