"""Repeatable evaluation: held-out loss/perplexity on val+test, and generation health metrics."""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import torch
from model.common import atomic_write_json, repo_root
from model.data.dataset import TokenDataset
from model.evaluation.metrics import evaluate_loss, repetition_rate
from model.inference import ResearchOSModel
from model.tokenizer.bpe import DOC_ID
from model.training.lineage import assert_random_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed", default="datasets/processed/synthetic_v1")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = repo_root()
    rt = ResearchOSModel(device=args.device)
    rt.load_checkpoint(args.checkpoint, tokenizer_dir=root / args.tokenizer, prefer="best" if args.best else "latest")
    assert_random_root(rt.meta)  # release criterion: must descend from random init
    cfg = rt.model.cfg
    report = {
        "checkpoint": str(rt.checkpoint_path),
        "step": rt.meta["step"],
        "parameters": rt.model.num_parameters(),
        "lineage": rt.meta["lineage"],
        "tokenizer_fingerprint": rt.tokenizer.fingerprint,
        "device": str(rt.device),
    }
    for split in ("val", "test"):
        ds = TokenDataset(root / args.processed / f"{split}.bin", cfg.max_seq_len)
        report[split] = evaluate_loss(rt.model, ds, batch_size=16, max_batches=None, device=rt.device)

    reps, rates, gen_tokens = [], [], 0
    t0 = time.perf_counter()
    for seed in range(args.samples):
        res = rt.generate([DOC_ID], max_new_tokens=cfg.max_seq_len - 2, temperature=0.8, top_k=40, seed=seed)
        reps.append(repetition_rate(res.token_ids))
        gen_tokens += len(res.token_ids)
        if seed < 2:
            report.setdefault("sample_texts", []).append(res.text)
    elapsed = time.perf_counter() - t0
    report["generation"] = {
        "samples": args.samples,
        "mean_repetition_3gram": sum(reps) / len(reps),
        "tokens_per_s": gen_tokens / elapsed,
    }
    if torch.cuda.is_available():
        report["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    out = args.out or str(root / "artifacts" / "reports" / f"eval_{rt.meta.get('run_name', 'run')}_step{rt.meta['step']}.json")
    atomic_write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "sample_texts"}, indent=2))
    print("report ->", out)


if __name__ == "__main__":
    main()
