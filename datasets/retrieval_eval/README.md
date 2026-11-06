# Retrieval evaluation set

Twelve short, invented documents (CC0, written for this repository; the facts are fictional), one byte-identical copy to test deduplication, declared metadata in `corpus.json`, and labelled queries in `queries.json`.

Query kinds: `lexical` (shares vocabulary with the answer), `topic` (several relevant documents), `phrase` (quoted phrase), `filtered` (metadata filter), `synonym` (no shared vocabulary: lexical retrieval is **expected to fail** and is reported as such).

Run `python scripts/eval_retrieval.py`. This is a small sanity benchmark, not evidence about performance on real literature.
