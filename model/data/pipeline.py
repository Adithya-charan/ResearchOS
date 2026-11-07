"""Raw corpus -> verified, cleaned, deduplicated, document-split, tokenized binary streams."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from model.common import atomic_write_json
from model.data.cleaning import NearDuplicateIndex, clean_text, exact_hash, quality_check
from model.data.dataset import write_token_file
from model.data.leakage import assert_no_leakage
from model.data.manifest import REQUIRED_USE, load_manifest, manifest_sha256, split_of, verify_manifest
from model.tokenizer.bpe import DOC_ID, EOS_ID, BPETokenizer


@dataclass
class PrepareConfig:
    manifest_path: str
    data_root: str
    out_dir: str
    tokenizer_dir: str
    vocab_size: int = 4096
    val_frac: float = 0.03
    test_frac: float = 0.03
    near_dup_threshold: float = 0.8
    min_chars: int = 200
    pretokenizer: str = "v1"  # "v2" keeps combining marks (Telugu/Indic) inside words; see model/tokenizer/bpe.py


def prepare_dataset(cfg: PrepareConfig, log=print) -> dict:
    records = load_manifest(cfg.manifest_path)
    problems = verify_manifest(records, cfg.data_root)
    if problems:
        raise ValueError("manifest verification failed:\n  " + "\n  ".join(problems[:20]))

    dropped: dict[str, str] = {}
    kept: list[tuple[str, str]] = []  # (doc_id, cleaned text)
    seen_exact: dict[str, str] = {}
    near = NearDuplicateIndex(threshold=cfg.near_dup_threshold)
    for r in sorted(records, key=lambda r: r.doc_id):
        if REQUIRED_USE not in r.allowed_use:
            dropped[r.doc_id] = "license_disallows_training"
            continue
        text = clean_text((Path(cfg.data_root) / r.path).read_text(encoding="utf-8"))
        ok, reason = quality_check(text, min_chars=cfg.min_chars)
        if not ok:
            dropped[r.doc_id] = f"quality:{reason}"
            continue
        h = exact_hash(text)
        if h in seen_exact:
            dropped[r.doc_id] = f"exact_duplicate_of:{seen_exact[h]}"
            continue
        seen_exact[h] = r.doc_id
        dup = near.check_and_add(r.doc_id, text)
        if dup is not None:
            dropped[r.doc_id] = f"near_duplicate_of:{dup}"
            continue
        kept.append((r.doc_id, text))
    log(f"kept {len(kept)}/{len(records)} documents; dropped {len(dropped)}")

    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for doc_id, text in kept:
        splits[split_of(doc_id, cfg.val_frac, cfg.test_frac)].append((doc_id, text))
    if not splits["train"] or not splits["val"]:
        raise ValueError(f"corpus too small for a train/val split: { {k: len(v) for k, v in splits.items()} }")
    assert_no_leakage(splits)  # independent re-check: no document or identical text in two splits
    log("documents per split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # Tokenizer sees ONLY the training split, so validation/test text cannot influence it.
    tok = BPETokenizer.train(
        (t for _, t in splits["train"]),
        vocab_size=cfg.vocab_size,
        progress=lambda i, n: log(f"  bpe merges {i}/{n}"),
        pretokenizer=cfg.pretokenizer,
    )
    tok.save(
        cfg.tokenizer_dir,
        extra_config={
            "trained_on_split": "train",
            "num_train_documents": len(splits["train"]),
            "manifest_sha256": manifest_sha256(records),
            "corpus_sha256": (tok.training_info or {}).get("corpus_sha256"),
        },
        sample_texts=(t for _, t in splits["train"][:50]),
    )
    log(f"tokenizer: vocab_size={tok.vocab_size} fingerprint={tok.fingerprint[:16]}")

    out = Path(cfg.out_dir)
    counts: dict[str, int] = {}
    for name, docs in splits.items():
        ids: list[int] = []
        for _, text in docs:
            ids.append(DOC_ID)
            ids.extend(tok.encode_ordinary(text))
            ids.append(EOS_ID)
        counts[name] = write_token_file(out / f"{name}.bin", ids)
        log(f"  {name}: {len(docs)} docs, {counts[name]:,} tokens")

    # "near_duplicate_of:syn-1" -> "near_duplicate_of", but keep "quality:too_short" whole.
    drop_reasons = Counter(v if v.startswith("quality:") else v.split(":")[0] for v in dropped.values())
    meta = {
        "manifest_sha256": manifest_sha256(records),
        "tokenizer_fingerprint": tok.fingerprint,
        "tokenizer_corpus_sha256": (tok.training_info or {}).get("corpus_sha256"),
        "tokenizer_trained_on": "train split only",
        "vocab_size": tok.vocab_size,
        "tokens": counts,
        "documents": {k: len(v) for k, v in splits.items()},
        "dropped_total": len(dropped),
        "dropped_reasons": dict(drop_reasons),
        "split_salt_policy": "sha256(salt:doc_id) -> document-level split",
        "config": cfg.__dict__,
    }
    atomic_write_json(out / "meta.json", meta)
    atomic_write_json(out / "splits.json", {k: sorted(d for d, _ in v) for k, v in splits.items()})
    atomic_write_json(out / "dropped.json", dropped)
    return meta
