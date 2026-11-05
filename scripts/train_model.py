"""Train the Transformer from random initialisation (or resume / continue one of our own checkpoints)."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from model.architecture import build_model
from model.common import set_seed
from model.data.dataset import TokenDataset
from model.evaluation.metrics import repetition_rate
from model.inference import ResearchOSModel
from model.tokenizer.bpe import DOC_ID, BPETokenizer
from model.training.checkpoint import load_weights, read_meta, resolve_checkpoint
from model.training.lineage import child_lineage, new_lineage
from model.training.run_config import load_run_config
from model.training.trainer import Trainer, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", help="run dir or checkpoint dir to resume exactly (same run)")
    ap.add_argument("--init-from", help="continue training from one of OUR checkpoints on new data (new run)")
    ap.add_argument("--max-steps", type=int, help="override training.max_steps")
    ap.add_argument("--stop-file", help="stop cleanly (after saving a checkpoint) once this file exists; used by the API for pause/cancel")
    args = ap.parse_args()

    # Peek at the config only to find where the tokenizer lives; then resolve vocab_size from it.
    import yaml

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    from model.common import repo_root

    tok_dir = repo_root() / raw.get("data", {}).get("tokenizer_dir", "artifacts/tokenizer")
    tok = BPETokenizer.load(tok_dir)
    rc = load_run_config(args.config, vocab_size_from_tokenizer=tok.vocab_size)
    if args.max_steps:
        rc.train.max_steps = args.max_steps
    if rc.model.vocab_size != tok.vocab_size:
        raise SystemExit(f"model.vocab_size={rc.model.vocab_size} but tokenizer has {tok.vocab_size}")

    proc = rc.resolve(rc.data.processed_dir)
    data_meta = json.loads((proc / "meta.json").read_text())
    if data_meta["tokenizer_fingerprint"] != tok.fingerprint:
        raise SystemExit("processed data was tokenized with a different tokenizer; re-run prepare_data.py")

    run_dir = rc.resolve(rc.data.checkpoint_root) / rc.train.run_name
    set_seed(rc.train.seed)
    train_ds = TokenDataset(proc / "train.bin", rc.model.max_seq_len)
    val_ds = TokenDataset(proc / "val.bin", rc.model.max_seq_len)

    model = build_model(rc.model, rc.train.seed)  # RANDOM INITIALISATION - no pretrained weights anywhere
    if args.resume:
        lineage = read_meta(resolve_checkpoint(args.resume))["lineage"]
    elif args.init_from:
        parent = resolve_checkpoint(args.init_from, prefer="best")
        parent_meta = load_weights(model, parent)
        lineage = child_lineage(parent_meta)
    else:
        lineage = new_lineage(model, rc.train.seed)

    meta_base = {
        "run_name": rc.train.run_name,
        "lineage": lineage,
        "tokenizer": {"fingerprint": tok.fingerprint, "vocab_size": tok.vocab_size},
        "data": {
            "manifest_sha256": data_meta["manifest_sha256"],
            "tokens": data_meta["tokens"],
            "processed_dir": rc.data.processed_dir,
        },
        "description": raw.get("description", "").strip(),
        "disclaimer": "experimental small language model trained from random initialisation on a limited corpus",
    }

    samples_path = run_dir / "samples.jsonl"
    device = resolve_device(rc.train.device)

    def on_eval(step: int, m, dev) -> dict:
        rt = ResearchOSModel(device=str(dev))
        rt.attach(m, tok, dev)
        res = rt.generate([DOC_ID], max_new_tokens=160, temperature=0.8, top_k=40, seed=1234)
        rep = repetition_rate(res.token_ids)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "repetition_3gram": rep, "text": res.text}) + "\n")
        return {"repetition_3gram": rep, "tokens_per_s_generate": res.tokens_per_s}

    stop = (lambda: Path(args.stop_file).exists()) if args.stop_file else None
    trainer = Trainer(model, rc.train, train_ds, val_ds, run_dir, meta_base=meta_base, on_eval=on_eval, should_stop=stop)
    summary = trainer.fit(resume=args.resume)
    print(json.dumps(summary, indent=2))
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
