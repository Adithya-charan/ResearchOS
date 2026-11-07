# Training pipeline

Entry points: `scripts/train_model.py` (CLI), the API training service (`/training/*`, `/modellab/datasets/*`) and the Model Lab UI. All
three drive the same `Trainer` (`model/training/trainer.py`). Training is always **from random initialisation**; `--init-from` only
continues one of this project's own checkpoints on new data and records the lineage.

## 1. Data preparation (`model/data/pipeline.py`)

Raw text -> verified, cleaned, deduplicated, **document-split**, tokenized binary streams:

1. Manifest verification: every file's SHA-256 and `allowed_use` must check out; a document whose license does not allow training is
   dropped (`license_disallows_training`).
2. Cleaning and quality checks (`min_chars`), exact-duplicate and near-duplicate removal (MinHash + LSH over 5-word shingles, threshold 0.8).
3. Split **by document** into train / validation / test (3 % each by default) - never by window, so no text is shared across splits.
4. Leakage gate (`model/data/leakage.py`), an independent re-check of step 2-3: the same document id in two splits, or byte-identical
   cleaned text in two splits, aborts preparation (`LeakageError`). Near-duplicates are removed corpus-wide in step 2 rather than
   re-checked here, so paraphrased or lightly edited copies that survive step 2 are not detected by this gate.
5. Tokenizer fitted on the **training split only** (`docs/TOKENIZER.md`), then each split is written as `train.bin`, `val.bin`, `test.bin`.
6. Records written: `dropped.json` (what was removed and why), `splits.json`, `meta.json` (corpus and tokenizer fingerprints).

**Datasets in the Model Lab.** A user uploads their own text with a license, attribution and an explicit rights confirmation. Nothing is
downloaded, scraped or fetched. Limits (server settings): documents per dataset, characters per document and in total, datasets per user,
tokenizer vocabulary size. Preparation runs in the API process on a background thread; the result is a status (`prepared` / `failed`)
and a hashed manifest.

## 2. Training (`Trainer`)

AdamW with weight-decay groups, linear warm-up then cosine decay to `min_lr_ratio`, gradient accumulation, gradient clipping, bf16
autocast where the device supports it, optional activation checkpointing, early stopping on validation loss, periodic evaluation with
token-weighted validation loss and perplexity, periodic checkpoints (`keep_last`), `max_epochs`, non-finite-loss protection
(`max_nonfinite_steps`), and cooperative stop (`--stop-file`) that checkpoints before exiting.

**Memory gate.** Before the first step the memory estimator (a planning figure) is compared with the free device memory or
`max_memory_gb`; an over-budget configuration is refused up front rather than failing mid-run. `enforce_memory_check=false` disables it.

**Outputs of a run** (`<run_dir>`): `metrics.jsonl` (train and eval records), `samples.jsonl` (generated samples, labelled unverified),
`run.json` (state while running), `summary.json` (final: `completion_state`, `duration_s`, `initial_loss`, `final_loss`, steps, tokens
seen, tokens/second, peak GPU and RAM, epochs) and checkpoints. A failed run records `completion_state: failed` and the error; it is
never reported as trained.

## 3. Checkpoints

`step_XXXXXXX/` containing `model.safetensors`, `meta.json` (model config, train config, tokenizer fingerprint, dataset fingerprint,
step, metrics, precision, lineage `random_init`, SHA-256 of the weights) and optimizer state. Safetensors means loading a checkpoint
never unpickles anything. `best.json` and `latest.json` point at the best-validation and newest checkpoints. Resume restores model,
optimizer, scheduler, RNG and data position exactly (tested).

`verify_checkpoint` re-hashes the weights and checks lineage; the API exposes it per checkpoint, and the Model Lab has a Verify button.

## 4. Training through the API

`POST /training/jobs` starts a **separate process** (`scripts/train_model.py`) for the caller. Bounds: training disabled unless
`RESEARCHOS_TRAINING_ENABLED=true`, one running job platform-wide by default, a per-user job quota, `max_steps` ceiling, only numeric
hyper-parameters from an allow-list can be overridden, the base config comes from the operator's config directory and the request never
carries a path. A user dataset is selected by id and trained with the tokenizer prepared from it. Pause/cancel are cooperative
(checkpoint first); the API restart marks orphaned jobs `interrupted` and they can be resumed.

This is API-supervised subprocess training. It is not the research job queue and it does not distribute work.

## 5. What has and has not been run

* The operator `small_v1` model (CONSUMER_SMALL shape, 3,000 steps, bf16, on the synthetic template corpus) and `tiny_cpu` exist and
  are real runs; see `docs/PHASE_REPORTS.md` for their figures.
* The bundled corpus is synthetic template text. Low loss on it shows the model learned the templates, not scientific language.
* No training on a real licensed corpus has been run. That requires the operator to supply one.
