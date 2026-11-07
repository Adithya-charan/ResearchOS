# Model evaluation

`model/evaluation/suite.py` (`run_evaluation`), `model/evaluation/metrics.py`, driven through `POST /modellab/models/{id}/evaluate` and
the Model Lab "Model evaluation" section. The report is written to `<data>/evaluations/<model>/report.json` (`<data>` = `RESEARCHOS_DATA_DIR`).

## What is measured

| Measurement | How |
|---|---|
| Validation and test loss, perplexity, token accuracy | non-overlapping next-token windows over the model's own `val.bin` / `test.bin` (token-weighted; capped at `eval_max_windows` per split) |
| **Random baseline** | a randomly initialised model of the **same architecture** (fixed seed 12345) scored on exactly the same windows |
| Beats baseline | `validation loss < baseline validation loss` (`beats_random_baseline`) |
| Train overlap / memorisation risk | fraction of evaluation 8-grams that also occur in the training tokens; above 25 % the report sets `memorization_risk` |
| Task probes | seven small sets in `datasets/evaluation/task_sets/` (scientific vocabulary, research terminology, abstract continuation, research-question completion, citation format, structured output, plain English), each scored for loss and accuracy next to the baseline, with its own train-overlap figure |
| Inference speed | cached and uncached tokens/s, ms per token, time to first token, peak GPU memory, cache speed-up |
| Size and identity | parameter count, checkpoint bytes, checkpoint SHA-256 (the report is bound to that checkpoint) |
| Agent-integration stage | the checks in `agent_engine/model_provider.py` (loads, reproducible, structured-output pass rate, invalid-output rejection, instruction evaluation) |
| Hardware | the machine's platform, CPU, RAM and GPU at evaluation time |

The task sets were written for this project (CC0-1.0), contain fictional references, authors and numbers, are 4-8 items each, and are
hash-listed in `MANIFEST.json`. They are smoke probes, not benchmarks.

## How to read a report

* The report always carries its caveats: probe sets are tiny; perplexity is comparable only between models sharing a tokenizer and
  evaluation text; **no comparison with any other model is made**; low loss on text that overlaps training data is memorisation.
* A validation loss below the random baseline shows the model learned *something* about its corpus. It does not show usefulness.
* A model scoring well on its own validation split and **worse than the random baseline on the probes** has learned its training
  distribution and nothing transferable. The suite reports that outcome unchanged.

## Measured results to date (this machine, see the final P8 report for the full table)

Operator `small_v1` (5,430,528 parameters, trained on the synthetic template corpus): validation loss 0.389 / perplexity 1.476 against a
random baseline loss of 7.839; test loss 0.383. The evaluation flags `memorization_risk`: about 81 % of validation 8-grams also occur in
the training data, which is expected for template text. All seven probe sets score *worse* than the random baseline (loss 9-12 against
about 7.8), i.e. the model is confidently wrong outside its template distribution. Agent-integration stage reached: 1 (structured-output
pass rate 0.0 of 8; no instruction-tuning run exists).

## Lifecycle link

VALIDATED requires an evaluation produced **for the current best checkpoint** (matching SHA-256), a finite validation loss below the
baseline, a clean leakage record and verified lineage. See `docs/MODEL_OPERATIONS.md`. An evaluation that predates a new checkpoint is
stale and is ignored.

## Not done

No public benchmark (HellaSwag, MMLU, etc.) was run; nothing here supports a claim about capability on real scientific text.
