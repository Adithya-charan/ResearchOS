# Verification evaluation set

**Everything here is synthetic.** Seven short documents were invented for this repository (CC0; the facts are fictional) and the claims were written by the project author, who also assigned the expected statuses by meaning. The claims use vocabulary the documents use, because the verifier is lexical. This set checks that the verification *pipeline behaves as designed*; it is **not scientific validation** and says nothing about accuracy on real literature.

Some claims are marked `known_limitation`: cases a lexical checker is expected to get wrong (swapped roles, a paraphrase, 'reduced' vs 'eliminated'). They stay in the set and are reported separately, so accuracy is never reported without them.

## Real-source examples

**None are included.** No real papers were downloaded and none were quoted from memory, because a mis-remembered quotation would be a fabricated citation. To evaluate on real sources, put documents you are licensed to use in `real/docs/` with a `real/corpus.json` and `real/claims.json` in the same format and run `python scripts/eval_verification.py --extra datasets/verification_eval/real`; its results are then reported under a separate heading, labelled as real-source.

Files: `docs/`, `corpus.json` (declared metadata), `claims.json`, `novelty_cases.json`. Run `python scripts/eval_verification.py`.
