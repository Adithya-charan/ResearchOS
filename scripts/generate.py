"""Generate text with a trained RESEARCHOS checkpoint (local runtime, no external services)."""
import argparse
import sys

import _bootstrap  # noqa: F401
from model.common import repo_root
from model.inference import ResearchOSModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir or run dir")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--best", action="store_true", help="use the best-validation checkpoint of a run dir")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    rt = ResearchOSModel(device=args.device)
    rt.load_checkpoint(args.checkpoint, tokenizer_dir=repo_root() / args.tokenizer, prefer="best" if args.best else "latest")
    print(f"[experimental small language model, step {rt.meta['step']}, device {rt.device}]", file=sys.stderr)
    from model.tokenizer.bpe import DOC_ID

    ids = [DOC_ID] + rt.tokenizer.encode(args.prompt)
    kw = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
              repetition_penalty=args.repetition_penalty, seed=args.seed)
    if args.stream:
        print(args.prompt, end="", flush=True)
        buf: list[int] = []
        for t in rt.stream(ids, args.max_new_tokens, **kw):
            buf.append(t)
            piece = rt.tokenizer.decode(buf)
            if not piece.endswith("�"):  # wait until a multi-byte character is complete
                print(piece, end="", flush=True)
                buf = []
        print()
    else:
        res = rt.generate(ids, args.max_new_tokens, **kw)
        print(args.prompt + res.text)
        print(f"\n[{len(res.token_ids)} tokens, {res.tokens_per_s:.0f} tok/s, finish={res.finish_reason}]", file=sys.stderr)


if __name__ == "__main__":
    main()
