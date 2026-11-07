# Custom model architecture

`model/architecture/`. A decoder-only Transformer implemented from scratch in PyTorch (tensors, autograd and `nn.Module` only). Weights
are always randomly initialised; no pretrained weight, embedding or tokenizer is loaded, and `lineage` metadata records
`random_init` as the root of every checkpoint (`verify_checkpoint` checks it).

## Configurable components

| Option | Choices |
|---|---|
| Normalisation | `layer_norm` (default), `rmsnorm` |
| Positions | `learned` (default), `rope` (`rope_theta`) |
| MLP | `gelu` (default), `swiglu` |
| Attention | `manual` (explicit softmax) or `sdpa` (PyTorch fused kernel); numerically equal, tested |
| Other | pre-norm blocks, optional bias, dropout, tied input/output embeddings (default), activation checkpointing, KV cache |

Sizes are `vocab_size, max_seq_len, d_model, n_layers, n_heads, d_ff`; `ModelConfig.estimate_params()` gives the parameter count without
building the model.

## Hardware profiles (`model/profiles.py`)

| Profile | Shape (vocab placeholder) | Parameters at that vocab | Intended target |
|---|---|---|---|
| `CPU_DEBUG` | 2 layers, d_model 64, context 64 | 136,960 | any CPU, pipeline debugging and CI smoke runs |
| `CONSUMER_SMALL` | 6 layers, d_model 256, context 256 | 5,853,184 (vocab 4096) | 4-6 GiB GPU or a patient CPU |
| `CONSUMER_MEDIUM` | 8 layers, d_model 512, context 512, RoPE + RMSNorm + SwiGLU | 31,465,984 (vocab 8192) | 6+ GiB bf16 GPU |

`configs/cpu_debug.yaml`, `configs/small.yaml` and `configs/consumer_medium.yaml` are the YAML a training job starts from and are
checked against the profile table by `tests/test_p8_model.py`. The parameter count of a real checkpoint depends on the tokenizer it was
trained with (the operator `small_v1` checkpoint has vocabulary 2,445 and 5,430,528 parameters).

### Planning versus measurement

`model/training/memory.py` gives an **estimate** of training memory (parameters, gradients, optimizer state, activations). It is a planning
figure and is labelled `basis: "estimate"` everywhere. Only `measure_profile()` (real optimizer steps, real peak memory) may be used to
say a profile fits a machine; `scripts/bench_model.py` runs it and records the result in `artifacts/evaluation/model_hardware.json`.
Training refuses to start when the estimate exceeds the free memory budget (`enforce_memory_check`, `max_memory_gb`).

## What this architecture is not

It is a small research model. At a few million parameters trained on a few megabytes of text it models that text's surface patterns; it
has no reliable factual knowledge and no demonstrated reasoning ability. Nothing here has been compared with any other model.
