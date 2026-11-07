# Tokenizer

`model/tokenizer/bpe.py`. A byte-level BPE tokenizer written from scratch. No third-party tokenizer, vocabulary or merge table is used
or loaded.

## Design

| Property | Value |
|---|---|
| Normalisation | Unicode NFC |
| Pieces | a fixed regex splits text into pieces (letters, single digits, punctuation, whitespace); merges never cross a piece |
| Token ids | `0..5` special (`<PAD> <UNK> <BOS> <EOS> <DOC> <SEP>`), `6..261` the 256 raw bytes, then one id per learned merge |
| Failure mode | none for text: every byte has a token, so `encode` cannot fail and `<UNK>` is never produced from text |
| Determinism | pieces processed in sorted order, ties broken by the lowest `(left_id, right_id)` pair; same corpus gives the same merges |
| Identity | `fingerprint` = SHA-256 over special tokens, normalisation, pattern, pre-tokenizer name (when not v1) and the ordered merges |

Checkpoints store the fingerprint; the inference runtime refuses a checkpoint paired with a different tokenizer
(`TokenizerMismatchError`, HTTP 409 from the API).

### Pre-tokenizers

* **v1** (default; every pre-P8 tokenizer and checkpoint uses it). Unchanged. Its fingerprint definition omits the `pretokenizer` key so
  old fingerprints stay valid.
* **v2** (opt-in, `pretokenizer="v2"`). Identical except that Unicode combining marks (Mn/Mc/Me) and the zero-width joiners are part
  of a word. Under v1 a Telugu or other Indic word is split at every vowel sign, so no merge can span a consonant and its vowel
  sign; v2 lets it. v2 tokenizers have a different fingerprint, by design.

## Training

`BPETokenizer.train(texts, vocab_size, pretokenizer=...)` records a *corpus fingerprint* (order-sensitive SHA-256 over the normalised,
length-prefixed texts, with document and character counts) in `training_info`. The data pipeline fits the tokenizer on the **training
split only**; validation and test text never influence it (`docs/TRAINING_PIPELINE.md`).

## Files written by `save()`

`merges.txt`, `vocab.json`, `tokenizer_config.json`, `tokenizer_version.txt`, `vocab_statistics.json`, and `manifest.json` (written
last) listing the SHA-256 of every other file plus the fingerprint, vocabulary size, merge count, pre-tokenizer, normalisation and
training-corpus record. `BPETokenizer.verify_manifest()` and `load()` detect a missing, partial or edited tokenizer directory.

`vocab_statistics.json`: merged-token byte lengths (min, mean, max, histogram) and, when sample texts are supplied, measured
characters/bytes per token, distinct tokens used, vocabulary coverage and unknown-token count. These are descriptive measurements; none
is a quality claim.

## Batch API

`batch_encode(texts, max_length=None, pad=False)` returns ids and attention masks (padding with `<PAD>`, truncating at `max_length`),
`batch_decode(batches)`. Decoding is lossy only for byte sequences that are not valid UTF-8 (replacement character).

## Limits and honesty

* The bundled tokenizer is fitted on the synthetic template corpus; its vocabulary reflects that text, not real scientific prose.
* Byte-level BPE is robust (no unknown text) but not optimal for any specific language; compression on real text has not been
  measured against alternatives, and no comparison with other tokenizers is claimed.
* A user dataset trained in the Model Lab gets its **own** tokenizer (limited to `dataset_max_vocab`, default 4096), and training on
  that dataset uses that tokenizer, never the operator tokenizer.
