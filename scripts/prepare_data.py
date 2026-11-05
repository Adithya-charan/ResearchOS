"""Verify manifest -> clean -> dedup -> document split -> train tokenizer (train split only) -> tokenize."""
import argparse

import _bootstrap  # noqa: F401
from model.common import repo_root
from model.data.pipeline import PrepareConfig, prepare_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synthetic_v1")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--test-frac", type=float, default=0.03)
    ap.add_argument("--tokenizer-dir", default="artifacts/tokenizer")
    args = ap.parse_args()
    root = repo_root()
    cfg = PrepareConfig(
        manifest_path=str(root / "datasets" / "manifests" / f"{args.name}.jsonl"),
        data_root=str(root / "datasets"),
        out_dir=str(root / "datasets" / "processed" / args.name),
        tokenizer_dir=str(root / args.tokenizer_dir),
        vocab_size=args.vocab_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    meta = prepare_dataset(cfg)
    print("done:", {k: meta[k] for k in ("tokens", "documents", "dropped_total", "vocab_size")})


if __name__ == "__main__":
    main()
