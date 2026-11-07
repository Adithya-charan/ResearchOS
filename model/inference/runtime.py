"""Local inference runtime: loads a RESEARCHOS checkpoint and generates text. No external servers."""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from model.architecture.transformer import Transformer
from model.common import atomic_write_json
from model.inference.sampling import DEFAULT_BANNED, sample_next
from model.tokenizer.bpe import EOS_ID, BPETokenizer
from model.training.checkpoint import load_model, resolve_checkpoint


class ContextOverflowError(ValueError):
    """The prompt does not fit in the model's context window."""


class TokenizerMismatchError(RuntimeError):
    """The tokenizer on disk is not the one this checkpoint was trained with."""


@dataclass
class GenerationResult:
    token_ids: list[int]  # newly generated tokens only (stop token / stop sequence excluded)
    finish_reason: str  # eos | stop_sequence | length | context | cancelled
    prompt_tokens: int
    elapsed_s: float = 0.0
    text: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return len(self.token_ids) / self.elapsed_s if self.elapsed_s > 0 else 0.0


class _StopMatcher:
    """Holds back tokens that might be the start of a stop sequence so they are never emitted early."""

    def __init__(self, sequences: Sequence[Sequence[int]]):
        self.seqs = [tuple(s) for s in sequences if len(s)]
        self.pending: list[int] = []

    def push(self, tok: int) -> tuple[list[int], bool]:
        self.pending.append(tok)
        for s in self.seqs:
            if len(self.pending) >= len(s) and tuple(self.pending[-len(s) :]) == s:
                emit = self.pending[: -len(s)]
                self.pending = []
                return emit, True
        hold = 0
        for s in self.seqs:
            for k in range(min(len(s) - 1, len(self.pending)), 0, -1):
                if tuple(self.pending[-k:]) == s[:k]:
                    hold = max(hold, k)
                    break
        cut = len(self.pending) - hold
        emit, self.pending = self.pending[:cut], self.pending[cut:]
        return emit, False

    def flush(self) -> list[int]:
        out, self.pending = self.pending, []
        return out


class ResearchOSModel:
    """Wraps a trained Transformer for generation. Default model = the one RESEARCHOS trained itself."""

    def __init__(self, device: str = "auto"):
        self._requested_device = device
        self.device = torch.device("cpu")
        self.model: Transformer | None = None
        self.tokenizer: BPETokenizer | None = None
        self.meta: dict = {}
        self.checkpoint_path: Path | None = None

    # ------------------------------------------------------------------ loading
    def load_checkpoint(self, checkpoint_path: str | Path, tokenizer_dir: str | Path | None = None, prefer: str = "latest") -> None:
        ckpt = resolve_checkpoint(checkpoint_path, prefer=prefer)
        want = self._requested_device
        device = torch.device("cuda" if want == "auto" and torch.cuda.is_available() else ("cpu" if want == "auto" else want))
        try:
            model, meta = load_model(ckpt, device=device)
        except torch.cuda.OutOfMemoryError:  # CPU fallback
            device = torch.device("cpu")
            torch.cuda.empty_cache()
            model, meta = load_model(ckpt, device=device)
        if tokenizer_dir is not None:
            tok = BPETokenizer.load(tokenizer_dir)
            expected = meta.get("tokenizer", {}).get("fingerprint")
            if expected is not None and tok.fingerprint != expected:
                raise TokenizerMismatchError(
                    f"tokenizer fingerprint {tok.fingerprint[:12]} != checkpoint's {expected[:12]}; "
                    "token ids would not mean what the model learned"
                )
            if tok.vocab_size != model.cfg.vocab_size:
                raise TokenizerMismatchError(f"tokenizer vocab {tok.vocab_size} != model vocab {model.cfg.vocab_size}")
            self.tokenizer = tok
        self.model, self.meta, self.device, self.checkpoint_path = model, meta, device, ckpt

    def attach(self, model: Transformer, tokenizer: BPETokenizer | None = None, device: str | torch.device = "cpu") -> None:
        """Use an in-memory model (tests, notebooks)."""
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    def _require(self) -> Transformer:
        if self.model is None:
            raise RuntimeError("no model loaded; call load_checkpoint() first")
        return self.model

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def forward(self, token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Logits [batch, seq, vocab] for the given token ids."""
        model = self._require()
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if ids.dim() == 1:
            ids = ids[None]
        return model(ids).logits

    # ------------------------------------------------------------------ generation
    def _events(
        self,
        prompt: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int | None,
        stop_token_ids: Sequence[int],
        stop_sequences: Sequence[Sequence[int]],
        use_cache: bool,
        cancel_event: threading.Event | None,
    ) -> Iterator[tuple[list[int], str | None]]:
        """Yield (tokens_to_emit, finish_reason). finish_reason is None until the last event."""
        model = self._require()
        cfg = model.cfg
        prompt = list(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if any(t < 0 or t >= cfg.vocab_size for t in prompt):
            raise ValueError("prompt contains token ids outside the model vocabulary")
        if len(prompt) >= cfg.max_seq_len:
            raise ContextOverflowError(f"prompt has {len(prompt)} tokens; model context is {cfg.max_seq_len}")
        budget = min(max_new_tokens, cfg.max_seq_len - len(prompt))
        out_of_context = budget < max_new_tokens

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed) if seed is not None else gen.seed()
        stop_ids = set(stop_token_ids)
        matcher = _StopMatcher(stop_sequences)
        ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        history = list(prompt)
        past = None

        # NOTE: inference_mode must never be held open across a `yield`: interleaved suspended generators would exit their
        # contexts out of order and leave grad mode disabled in the thread. Each step enters and leaves it before yielding.
        for _ in range(budget):
            if cancel_event is not None and cancel_event.is_set():
                yield matcher.flush(), "cancelled"
                return
            with torch.inference_mode():
                if use_cache:
                    inp = ids if past is None else ids[:, -1:]
                    out = model(inp, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
                else:
                    out = model(ids, logits_to_keep=1)
                tok = sample_next(
                    out.logits[0, -1],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    history=history,
                    banned=DEFAULT_BANNED,
                    generator=gen,
                )
                if tok not in stop_ids:
                    ids = torch.cat([ids, torch.tensor([[tok]], device=self.device)], dim=1)
            if tok in stop_ids:
                yield matcher.flush(), "eos"
                return
            history.append(tok)
            emit, stopped = matcher.push(tok)
            if stopped:
                yield emit, "stop_sequence"
                return
            if emit:
                yield emit, None
        yield matcher.flush(), ("context" if out_of_context else "length")

    def stream(
        self,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        *,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        stop_token_ids: Sequence[int] = (EOS_ID,),
        stop_sequences: Sequence[Sequence[int]] = (),
        use_cache: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[int]:
        """Yield new token ids as they are produced."""
        for emit, _ in self._events(
            prompt_token_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed,
            stop_token_ids, stop_sequences, use_cache, cancel_event,
        ):
            yield from emit

    def stream_events(
        self,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        *,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        stop_token_ids: Sequence[int] = (EOS_ID,),
        stop_sequences: Sequence[Sequence[int]] = (),
        use_cache: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple[list[int], str | None]]:
        """Like `stream`, but yields (new_token_ids, finish_reason); finish_reason is None until the final event."""
        yield from self._events(
            prompt_token_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed,
            stop_token_ids, stop_sequences, use_cache, cancel_event,
        )

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        *,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        stop_token_ids: Sequence[int] = (EOS_ID,),
        stop_sequences: Sequence[Sequence[int]] = (),
        use_cache: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        tokens: list[int] = []
        reason = "length"
        for emit, finish in self._events(
            prompt_token_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed,
            stop_token_ids, stop_sequences, use_cache, cancel_event,
        ):
            tokens.extend(emit)
            if finish is not None:
                reason = finish
        result = GenerationResult(tokens, reason, len(prompt_token_ids), time.perf_counter() - t0)
        if self.tokenizer is not None:
            result.text = self.tokenizer.decode(tokens)
        return result

    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 40,
        *,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seeds: Sequence[int | None] | None = None,
        stop_token_ids: Sequence[int] = (EOS_ID,),
        max_batch_size: int = 16,
        cancel_event: threading.Event | None = None,
    ) -> list[GenerationResult]:
        """Generate for several prompts, batching those of **equal length** into one cached forward pass.

        The model has no attention mask, so unequal-length prompts cannot share a batch without changing results (padding
        would be attended to); they are grouped by length and each group is batched, which keeps every row's output
        identical to `generate` for the same prompt and seed (tested). Stop *sequences* are not supported here, only stop
        tokens. Results come back in input order.
        """
        model = self._require()
        cfg = model.cfg
        n = len(prompts)
        if n == 0:
            return []
        if n > max_batch_size:
            raise ValueError(f"batch of {n} exceeds max_batch_size={max_batch_size}")
        seeds = list(seeds) if seeds is not None else [None] * n
        if len(seeds) != n:
            raise ValueError("seeds must have one entry per prompt")
        for pr in prompts:
            if not len(pr):
                raise ValueError("prompt must contain at least one token")
            if any(t < 0 or t >= cfg.vocab_size for t in pr):
                raise ValueError("prompt contains token ids outside the model vocabulary")
            if len(pr) >= cfg.max_seq_len:
                raise ContextOverflowError(f"prompt has {len(pr)} tokens; model context is {cfg.max_seq_len}")
        results: list[GenerationResult | None] = [None] * n
        groups: dict[int, list[int]] = {}
        for i, pr in enumerate(prompts):
            groups.setdefault(len(pr), []).append(i)
        stop_ids = set(stop_token_ids)
        for length, members in groups.items():
            t0 = time.perf_counter()
            budget = min(max_new_tokens, cfg.max_seq_len - length)
            out_of_context = budget < max_new_tokens
            gens = []
            for i in members:
                g = torch.Generator(device="cpu")
                g.manual_seed(seeds[i]) if seeds[i] is not None else g.seed()
                gens.append(g)
            ids = torch.tensor([list(prompts[i]) for i in members], dtype=torch.long, device=self.device)
            history = [list(prompts[i]) for i in members]
            produced: list[list[int]] = [[] for _ in members]
            reason: list[str | None] = [None] * len(members)
            past = None
            with torch.inference_mode():
                for _ in range(budget):
                    if cancel_event is not None and cancel_event.is_set():
                        for r in range(len(members)):
                            reason[r] = reason[r] or "cancelled"
                        break
                    inp = ids if past is None else ids[:, -1:]
                    out = model(inp, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
                    nxt = []
                    for r in range(len(members)):
                        if reason[r] is not None:
                            nxt.append(EOS_ID)  # finished rows keep being fed a harmless token; their output is discarded
                            continue
                        tok = sample_next(out.logits[r, -1], temperature=temperature, top_k=top_k, top_p=top_p,
                                          repetition_penalty=repetition_penalty, history=history[r],
                                          banned=DEFAULT_BANNED, generator=gens[r])
                        if tok in stop_ids:
                            reason[r] = "eos"
                            nxt.append(EOS_ID)
                            continue
                        produced[r].append(tok)
                        history[r].append(tok)
                        nxt.append(tok)
                    ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=self.device)[:, None]], dim=1)
                    if all(x is not None for x in reason):
                        break
            elapsed = time.perf_counter() - t0
            for r, i in enumerate(members):
                res = GenerationResult(produced[r], reason[r] or ("context" if out_of_context else "length"), length, elapsed)
                if self.tokenizer is not None:
                    res.text = self.tokenizer.decode(produced[r])
                results[i] = res
        return results  # type: ignore[return-value]

    def generate_text(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> GenerationResult:
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer attached")
        from model.tokenizer.bpe import DOC_ID

        ids = [DOC_ID] + self.tokenizer.encode(prompt)
        return self.generate(ids, max_new_tokens, **kwargs)

    # ------------------------------------------------------------------ state
    def save_runtime_state(self, path: str | Path) -> None:
        """Record exactly which weights/tokenizer this runtime is serving (for run reproducibility)."""
        model = self._require()
        atomic_write_json(
            path,
            {
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "model_sha256": self.meta.get("model_sha256"),
                "step": self.meta.get("step"),
                "lineage": self.meta.get("lineage"),
                "tokenizer_fingerprint": self.tokenizer.fingerprint if self.tokenizer else None,
                "device": str(self.device),
                "parameters": model.num_parameters(),
                "model_config": model.cfg.to_dict(),
                "note": "experimental small language model trained from random initialisation",
            },
        )
