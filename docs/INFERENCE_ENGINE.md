# Inference engine

`model/inference/runtime.py` (`ResearchOSModel`), served by `api/model_service.py` (`InferenceService`) and
`/models/{id}/generate[/stream]`. It runs only the project's own checkpoints, on the local device. There is no hosted-model fallback.

## Runtime

* Loads a safetensors checkpoint and the tokenizer **named by that checkpoint's fingerprint**; a mismatch raises
  `TokenizerMismatchError` and the API answers 409.
* **KV cache** (`use_cache=True` by default): the prompt is processed once, then one token per step. Cached and uncached greedy
  generation produce identical tokens (tested). `logits_to_keep=1` avoids computing logits for the whole prompt.
* **Sampling** (`sample_next`): temperature (0 = greedy), top-k, top-p, repetition penalty, optional seed (a seeded request is
  reproducible), banned special tokens, stop tokens and stop sequences.
* **Streaming**: `stream_events` yields `(new_token_ids, finish_reason)`; finish reasons are `eos`, `length`, `context`, `stop`,
  `cancelled`. A cancel event (client gone or timeout) stops generation between tokens.
* **Context**: a prompt at or beyond `max_seq_len` raises `ContextOverflowError` (422); generation stops at the context limit.
* **Batching**: `generate_batch` runs prompts of **equal token length** together in one cached pass and groups other lengths separately.
  The model has no attention mask, so mixing lengths would let padding be attended to; each row's output equals the single-prompt
  result for the same seed (tested). It supports stop tokens but not stop sequences. This is a limited batching feature, not a
  continuous-batching server.
* Models are cached per (checkpoint, tokenizer) key with an LRU of `infer_cache_size`; weights are never re-read per request.

## API limits (settings)

| Setting | Default |
|---|---|
| prompt size | 4,000 characters |
| new tokens | 256 |
| wall-clock per request | 30 s (then `cancelled`) |
| requests per user per minute | 20 |
| concurrent generations | bounded by slots; excess gets 429 |
| concurrent streams | per user and total (`StreamGate`); excess gets 429 |

The request carries a model id and a checkpoint selector (`best` / `latest`), never a path. Every response and stream carries a notice
that the text is an unverified sample, and `done` events include the checkpoint step and weight hash.

## Measured speed

`scripts/bench_model.py` measures cached and uncached generation, time to first token and peak memory per profile and writes
`artifacts/evaluation/model_hardware.json`; the evaluation suite records the same measurement for each evaluated checkpoint. Figures
are specific to the machine that produced them (the file records it) and are not a statement about any other hardware. On the
development GPU a small model is bound by per-token kernel-launch cost, so the KV cache gives little or no speed-up there while it
helps on CPU; the measurements in the file show this rather than assume it.

## What the generated text is

Output from these small models may be fluent in the style of their training text and still be wrong. It is never evidence, never a
citation and never used as a verdict (`docs/AGENT_SYSTEM.md`, "Model integration").
