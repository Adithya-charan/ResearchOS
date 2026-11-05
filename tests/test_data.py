import json
from pathlib import Path

import numpy as np
import pytest

from model.common import sha256_file
from model.data.cleaning import NearDuplicateIndex, clean_text, exact_hash, quality_check
from model.data.dataset import TokenDataset, write_token_file
from model.data.manifest import DocRecord, load_manifest, manifest_sha256, split_of, verify_manifest, write_manifest
from model.data.pipeline import PrepareConfig, prepare_dataset
from model.data.synthetic import generate_documents
from model.tokenizer.bpe import DOC_ID, EOS_ID, BPETokenizer


def make_corpus(root: Path, n=80, seed=7, license_="CC0-1.0", allowed=("training",)):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for doc_id, text in generate_documents(n, seed=seed):
        p = raw / f"{doc_id}.txt"
        p.write_text(text, encoding="utf-8")
        records.append(DocRecord(doc_id, f"raw/{doc_id}.txt", sha256_file(p), "unit-test", license_, "2026-01-01", list(allowed), p.stat().st_size))
    write_manifest(root / "manifest.jsonl", records)
    return records


# ---------------------------------------------------------------- synthetic corpus
def test_synthetic_corpus_is_deterministic_and_varied():
    a = list(generate_documents(5, seed=1))
    assert a == list(generate_documents(5, seed=1))
    assert a != list(generate_documents(5, seed=2))
    assert len({t for _, t in a}) == 5
    assert all(len(t) > 1500 for _, t in a)


# ---------------------------------------------------------------- manifest
def test_manifest_round_trip_and_hash(tmp_path):
    recs = make_corpus(tmp_path, n=5)
    loaded = load_manifest(tmp_path / "manifest.jsonl")
    assert loaded == recs
    assert manifest_sha256(loaded) == manifest_sha256(list(reversed(loaded)))  # order-independent
    assert verify_manifest(loaded, tmp_path) == []


def test_manifest_detects_tampering_and_missing_metadata(tmp_path):
    recs = make_corpus(tmp_path, n=3)
    (tmp_path / recs[0].path).write_text("changed after the hash was recorded " * 20)
    recs[1].license = ""
    recs[2].path = "raw/missing.txt"
    problems = " | ".join(verify_manifest(recs, tmp_path))
    assert "sha256 mismatch" in problems and "missing license" in problems and "file not found" in problems


def test_split_is_document_level_deterministic_and_proportional():
    ids = [f"doc-{i}" for i in range(4000)]
    splits = [split_of(i, 0.05, 0.05) for i in ids]
    assert splits == [split_of(i, 0.05, 0.05) for i in ids]
    for name in ("val", "test"):
        assert 0.03 < splits.count(name) / len(ids) < 0.07
    assert split_of("same", 0.1, 0.1) == split_of("same", 0.1, 0.1)


# ---------------------------------------------------------------- cleaning
def test_clean_text_normalises():
    out = clean_text("a\r\nb\x00c  \n\n\n\n\ndé\t\n")
    assert out == "a\nbc\n\ndé\n"  # CRLF->LF, control char gone, blank run capped, NFC applied


def test_quality_filters():
    assert quality_check("ok " * 200)[0]
    assert quality_check("short") == (False, "too_short")
    assert quality_check("12345 !!! ### " * 100)[1] == "low_alpha_ratio"
    assert quality_check(("same line\n") * 100 + "x" * 200)[1] == "repeated_lines"


def test_exact_hash_ignores_case_and_spacing():
    assert exact_hash("Hello   World") == exact_hash("hello world")


def test_near_duplicate_detection():
    base = " ".join(f"word{i % 97}" for i in range(400))
    edited = base.replace("word5 ", "other ", 1)
    different = " ".join(f"zzz{i}" for i in range(400))
    idx = NearDuplicateIndex()
    assert idx.check_and_add("a", base) is None
    assert idx.check_and_add("b", edited) == "a"
    assert idx.check_and_add("c", different) is None


# ---------------------------------------------------------------- token dataset
def test_token_dataset_batches(tmp_path):
    write_token_file(tmp_path / "t.bin", np.arange(1000) % 500)
    ds = TokenDataset(tmp_path / "t.bin", seq_len=16)
    x, y = ds.train_batch(seed=1, step=3, micro=0, batch_size=4)
    assert x.shape == y.shape == (4, 16) and x.dtype.is_floating_point is False
    assert np.array_equal(x[:, 1:].numpy(), y[:, :-1].numpy())  # y is x shifted by one
    x2, _ = ds.train_batch(1, 3, 0, 4)
    assert np.array_equal(x.numpy(), x2.numpy())  # same (seed, step, micro) -> same batch
    assert not np.array_equal(x.numpy(), ds.train_batch(1, 4, 0, 4)[0].numpy())
    ev = list(ds.eval_batches(batch_size=4, max_batches=3))
    assert len(ev) == 3 and np.array_equal(ev[0][0].numpy(), list(ds.eval_batches(4, 3))[0][0].numpy())


def test_token_file_rejects_ids_beyond_uint16(tmp_path):
    with pytest.raises(ValueError):
        write_token_file(tmp_path / "x.bin", [70000])


# ---------------------------------------------------------------- pipeline
def run_pipeline(root: Path, **kw):
    cfg = PrepareConfig(
        manifest_path=str(root / "manifest.jsonl"), data_root=str(root), out_dir=str(root / "out"),
        tokenizer_dir=str(root / "tok"), vocab_size=kw.pop("vocab_size", 320), val_frac=0.1, test_frac=0.1, **kw,
    )
    return prepare_dataset(cfg, log=lambda *_: None)


def test_pipeline_end_to_end(tmp_path):
    make_corpus(tmp_path, n=80)
    meta = run_pipeline(tmp_path)
    out = tmp_path / "out"
    assert meta["tokens"]["train"] > meta["tokens"]["val"] > 0
    splits = json.loads((out / "splits.json").read_text())
    assert not (set(splits["train"]) & set(splits["val"]) | set(splits["train"]) & set(splits["test"]) | set(splits["val"]) & set(splits["test"]))
    tok = BPETokenizer.load(tmp_path / "tok")
    assert tok.fingerprint == meta["tokenizer_fingerprint"]
    train = np.fromfile(out / "train.bin", dtype=np.uint16)
    assert train[0] == DOC_ID and train[-1] == EOS_ID and train.max() < tok.vocab_size
    assert (train == DOC_ID).sum() == len(splits["train"]) == (train == EOS_ID).sum()


def test_pipeline_is_reproducible(tmp_path):
    for name in ("a", "b"):
        make_corpus(tmp_path / name, n=60)
        run_pipeline(tmp_path / name)
    for f in ("train.bin", "val.bin", "test.bin"):
        assert sha256_file(tmp_path / "a" / "out" / f) == sha256_file(tmp_path / "b" / "out" / f)
    assert (tmp_path / "a" / "tok" / "tokenizer_version.txt").read_text() == (tmp_path / "b" / "tok" / "tokenizer_version.txt").read_text()


def test_tokenizer_is_trained_on_train_split_only(tmp_path):
    make_corpus(tmp_path, n=80)
    run_pipeline(tmp_path)
    cfg = json.loads((tmp_path / "tok" / "tokenizer_config.json").read_text())
    assert cfg["training"]["trained_on_split"] == "train"
    splits = json.loads((tmp_path / "out" / "splits.json").read_text())
    assert cfg["training"]["num_train_documents"] == len(splits["train"])


def test_pipeline_excludes_documents_whose_licence_forbids_training(tmp_path):
    recs = make_corpus(tmp_path, n=60)
    for r in recs[:20]:
        r.allowed_use = ["search"]  # may be searched/cited, but not trained on
    write_manifest(tmp_path / "manifest.jsonl", recs)
    run_pipeline(tmp_path)
    dropped = json.loads((tmp_path / "out" / "dropped.json").read_text())
    banned = {r.doc_id for r in recs[:20]}
    assert {d for d, why in dropped.items() if why == "license_disallows_training"} == banned
    splits = json.loads((tmp_path / "out" / "splits.json").read_text())
    assert not banned & {d for ids in splits.values() for d in ids}


def test_pipeline_rejects_corpus_with_nothing_trainable(tmp_path):
    make_corpus(tmp_path, n=30, allowed=("search",))
    with pytest.raises(ValueError, match="too small"):
        run_pipeline(tmp_path)


def test_pipeline_refuses_tampered_corpus(tmp_path):
    recs = make_corpus(tmp_path, n=20)
    (tmp_path / recs[0].path).write_text("tampered " * 100)
    with pytest.raises(ValueError, match="manifest verification failed"):
        run_pipeline(tmp_path)


def test_pipeline_drops_exact_and_near_duplicates(tmp_path):
    recs = make_corpus(tmp_path, n=60)
    text = (tmp_path / recs[0].path).read_text()
    for name, body in (("dup-exact", text), ("dup-near", text.replace("the", "THE", 1) + " extra")):
        p = tmp_path / "raw" / f"{name}.txt"
        p.write_text(body, encoding="utf-8")
        recs.append(DocRecord(name, f"raw/{name}.txt", sha256_file(p), "unit-test", "CC0-1.0", "2026-01-01"))
    write_manifest(tmp_path / "manifest.jsonl", recs)
    run_pipeline(tmp_path)
    dropped = json.loads((tmp_path / "out" / "dropped.json").read_text())
    assert "dup-exact" in dropped or "dup-near" in dropped
