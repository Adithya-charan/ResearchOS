# P11 evaluation methodology and results

How the P11 research-quality features were measured, what the numbers mean, and what they do **not** mean.
Reproduce everything with:

```bash
.venv/Scripts/python.exe scripts/eval_retrieval_p11.py   # artifacts/evaluation/retrieval_p11.json
.venv/Scripts/python.exe scripts/eval_p11.py             # artifacts/evaluation/p11_quality.json
```

Both are deterministic: no randomness, no network, no model. Running `eval_p11.py` twice gives identical output apart from wall-clock timings (checked).

## Read this first

* The corpus (`datasets/p11_eval/corpus.json`) is **invented**: 18 short documents (CC0-1.0) on four topics, written for this test. No figure in it is a real measurement and none of it is a real paper.
* Queries, claims and expected labels are **author-written**, and the same author wrote the documents, so wording is shared far more than in real literature. These numbers are an **upper bound** for what a lexical method does on clean text. They are not scientific validation and must not be quoted as the system's accuracy on real papers.
* No real-document evaluation was run. That needs a licensed corpus chosen by the project owner (see `docs/RELEASE_CHECKLIST.md`, "real-corpus evaluation").
* Expected statuses were assigned by meaning before the system ran on them. One label was corrected afterwards (see "A label I got wrong"), and that is disclosed in the dataset itself.

## What is measured

| Area | Dataset | Metric | Where |
|---|---|---|---|
| Retrieval | 45 queries (exact, paraphrase, long, short, phrase, multi-document, vocabulary mismatch, missing topic) | Recall@3/5, Precision@1/3, MRR, duplicate pairs in the top 5 | `research/retrieval_benchmark.py` |
| Evidence | 26 claims (exact, paraphrase, contradictory, insufficient, multi-document) | status accuracy, evidence-support precision/recall, unsupported-claim rate, contradiction precision/recall | `research/p11_evaluation.py::evidence_eval` |
| Citations | every passage cited correctly and with each of 6 injected defects | detection rate per defect, false-alarm rate | `citation_eval` |
| Gaps | author expectations of 2 conflicting pairs and 17 stated limitations | conflict precision/recall, limitation recall, traceability | `gap_eval` |
| End to end | 3 real runs through the API (batteries, soil, federated learning) | completion, 18-section completeness, labelled blocks, citation integrity, retries | `scripts/eval_p11.py` |

## Results (from `artifacts/evaluation/*.json`)

### Retrieval (P11.1): baseline BM25 vs the opt-in advanced ranker

| Setting | Recall@3 | Recall@5 | Precision@1 | Precision@3 | MRR | Duplicate pairs in top 5 |
|---|---|---|---|---|---|---|
| Baseline BM25 (unchanged, still the default) | 0.9537 | 0.9756 | 0.9756 | 0.4797 | 0.9878 | 9 |
| Recommended (normalise + pseudo-relevance feedback + de-duplication) | 0.9659 | 1.0000 | 0.9756 | 0.4878 | 0.9878 | 0 |

What the ablations show (all rows are in `retrieval_p11.json`):

* **Pseudo-relevance feedback** is the only feature that improves recall: vocabulary-mismatch Recall@3 goes from 0.9167 to 1.0 (a query for "swelling" reaching a document that says "volume expansion").
* **De-duplication** removes all duplicate pairs from the top 5 (9 → 0) without changing recall.
* **Phrase bonus, title boost, query normalisation, per-document cap: no measurable change** on this dataset. They are implemented, explainable, off by default or harmless, and were *not* shown to help.
* **Heading boost** lowered Recall@3 slightly (0.9537 → 0.9415). **Prefix variants** raised Recall@5 but lowered MRR and vocabulary-mismatch Recall@3. Neither is in the recommended setting.
* Gains are small because the baseline is already near ceiling on 45 queries over 18 documents. The conclusion is "the advanced ranker does no harm and fixes two specific problems", not "retrieval is much better". The baseline stays the default; `ranker=advanced` is opt-in.

### Evidence quality (P11.2)

| Measure | Value |
|---|---|
| Cases | 26 |
| Status accuracy | 0.6538 (17/26) |
| Exact claims | 8/8 correct |
| Paraphrased claims | 0/5 correct |
| Contradictory claims | 4/6 correct |
| Insufficient-evidence claims | 4/5 correct |
| Multi-document claims | 1/2 correct |
| Evidence-support precision | 0.7778 |
| **Unsupported-claim rate** (labelled SUPPORTED but should not be) | **0.2222** |
| Contradiction detection | precision 1.0, recall 0.6667 |

The nine misses are the honest picture of a lexical verifier:

* **p01–p05 (paraphrases):** the claim shares too few words with the passage, so the claim is not connected to it. This is the documented false-negative class.
* **c04 "Larval settlement was highest on sand" and c06 "Client drift decreased as the number of local epochs grew":** labelled SUPPORTED although the passages say the opposite. These are real **false positives**. c04 matches because "highest" and "sand" both occur in the passage in different roles; c06 contains both a "decreased" and a "grew", so the claim's own direction is ambiguous. They are the reason the unsupported-claim rate is not zero, and `tests/test_p11_evaluation.py` fails if they are quietly dropped.
* **i04 "Biochar improved maize yield in clay soils":** reported as partly supported because the verifier cannot see that the qualifier "clay" is missing from the passages.
* **m02:** a claim split across two documents is not connected, because verification is per passage.

P11.2 improved the verifier where a rule could be written without breaking the labelled set: unit-aware figure conflicts, swapped comparison roles, and null-result-versus-change contradictions. The last was found *by this evaluation*: "no significant difference" against "improved" was being returned as SUPPORTED. Known-limitation set in `datasets/verification_eval`: 0/3 → 2/3 correct, with the other 18 claims unchanged.

### Citation integrity (P11.4)

Injected defects (fabricated id, missing id, wrong document, wrong page, altered quote, changed source text), 44 passages each: **every defect detected (1.0) and each raised exactly its own issue; no valid citation was flagged (false-alarm rate 0.0)**. This is a check that the checker finds these defect types. It says nothing about the passage *supporting* the claim; that is the verifier's job and is reported separately.

### Gap analysis (P11.3)

* Conflicting pairs: 2 of 2 expected found, 2 reported (precision 1.0, recall 1.0). Found: the biochar pair (increased vs decreased yield) and the additive pair ("improved" vs "no significant difference").
* Stated limitations: every document with a limitations section found (17/17).
* Every gap links to a stored passage and passes the traceability validator (`validate_gap_report`); no gap uses novelty wording.
* Single-document themes and missing-comparison hypotheses have no ground truth, so they are listed, not scored.

### End to end (3 real runs through the API)

3/3 runs completed. All 3 reports have the 18 sections in order with none empty; every one of the 287 blocks carries a statement-type label. Two reports had citations, all valid (integrity 1.0); the third (federated learning) found no claims for its question, so it has no citation rate, and that is reported as "no citations" rather than as perfect. No run needed a retry. Three runs on clean input show the pipeline completes; they are not a failure-rate estimate.

## A label I got wrong

Case x05 ("Biochar increased maize yield by fourteen percent…") was first labelled SUPPORTED. The verifier returned PARTIALLY_SUPPORTED with conflicting evidence, because the corpus deliberately contains `soil2`, which reports the opposite. The verifier was right and my label was wrong. I changed the expected status to PARTIALLY_SUPPORTED and wrote that into the case's `note` in `evidence_cases.json`. The accuracy figure above (0.6538) is after that correction; before it, it was 0.6154. No other label was changed after seeing output.

## Limitations of the evaluation

* Invented corpus, shared author, small size (see above).
* Paraphrase and contradiction cases were written to be hard for a lexical method, so accuracy over the whole set understates behaviour on easy text and overstates it on real text with unfamiliar vocabulary.
* Injected citation defects are the defects I thought of.
* The end-to-end part uses three runs; it detects broken plumbing, not rare failures.
* Timings are not part of the pinned results.
