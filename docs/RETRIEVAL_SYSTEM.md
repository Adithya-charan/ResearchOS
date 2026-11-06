# Retrieval system

Literature retrieval in RESEARCHOS is **lexical**: an in-process Okapi BM25 index over passages extracted from documents the
user uploaded. It finds documents that contain the query's words. It does **not** understand meaning, and it searches only the
library the project owner uploaded - it makes no claim of coverage of "the literature".

## What is implemented

| Capability | Behaviour |
|---|---|
| Ingestion | PDF, DOCX, TXT, Markdown, HTML; type/size validation, zip-bomb guard, extraction in an isolated subprocess, text normalisation, section detection, passage chunking |
| Duplicate handling | by SHA-256 of the file bytes: a byte-identical upload (any filename) returns the existing document, nothing is stored twice. A file that differs by even one byte is a separate document |
| Declared metadata | title, authors, DOI, source, license, publication date (`YYYY[-MM[-DD]]`), URL (http/https), document type (article, preprint, report, thesis, book, chapter, notes, other). Validated; never guessed from the file; `NULL` when not declared |
| Chunk-level search | BM25 over passages; each hit carries document, page (when the format has pages), section, score |
| Phrase search | `"double quotes"`: the exact contiguous words (no stemming or stopword removal inside the phrase) |
| Document-level search | each document once, ranked by its best passage (an aggregation of passage scores, not a separate model) |
| Metadata search | title, authors, DOI (exact), source, filename, type, date; stemmed, stopwords dropped |
| Combined filters | year range, document type, author substring, license, file type - all applied together; undeclared values never match a filter |
| Citation record | `CorpusHit.citation()`: `document_id, chunk_id, title, excerpt (<=300 chars), page_number, section, score, url, doi, source` |
| Citation resolution | `Corpus.verify_passage(chunk_id)` re-checks the stored passage against its source file; `inspect` returns it |
| Deletion | removes the record, its passages (cascade), the stored original, and rebuilds the in-memory index; citations to it stop resolving |
| Persistence | the index is rebuilt from `corpus.db` on open; a reopened library returns identical results and still deduplicates |
| API | `GET /projects/{project_id}/search?q=&level=chunk|document|metadata&document_type=&author=&year_min=&year_max=`; requires login and project ownership (other users get 404) |

## Integration with the agents

* **Literature Discovery** searches only through the `corpus.search` tool (a `tool_call` event is recorded for each call); every hit is a stored passage.
* **Critic** accepts/rejects hits; **Literature Analysis** reads only documents the critic accepted and extracts *verbatim* sentences with their passage ids.
* **Gap Analyst** and **Hypothesis** cite evidence by passage id. A gap or hypothesis cites evidence only when one exists: gaps built from
  uncovered question terms carry no evidence (they describe what the searched library lacks), and with no stated limitations in the
  accepted documents, hypotheses cite nothing and say so.
* **Verifier** resolves each extracted citation itself (`passage_exists` check); **Reviewer** and **Experiment Design** cite resolvable passages only.
* Unverifiable references are reported as such, never repaired or invented.

## Evaluation (`scripts/eval_retrieval.py`, `datasets/retrieval_eval/`)

12 short invented CC0 documents, one byte-identical copy (deduplication test), declared metadata, 22 labelled queries in five
separately reported categories. Metrics are document-level Precision@K, Recall@K and MRR; chunk precision is also reported.

| Category | n | MRR | P@1 | P@3 | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|---|---|
| lexical (shares words with the answer) | 12 | 1.00 | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| topic (3 relevant documents each) | 3 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 | 1.00 |
| phrase | 2 | 1.00 | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| filtered (metadata filter) | 2 | 1.00 | 1.00 | 0.50 | 0.75 | 1.00 | 1.00 |
| **synonym (no shared vocabulary)** | 3 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| all | 22 | 0.86 | 0.86 | 0.39 | 0.75 | 0.86 | 0.86 |

P@3 is low for single-relevant-document queries by construction (at most 1 of 3 slots can be relevant). Other checks: 13 files
offered, 12 stored (copy recognised); declared metadata equals stored metadata for all 12; 60/60 returned citations resolve and verify.

**Evaluation integrity.** The query set was not tuned to raise scores: the synonym queries were rewritten *down* to genuinely share no
stems with their target documents, and `tests/test_p5_closure.py` asserts this mechanically (analysed stems of each synonym query vs. its
targets' text and authors are disjoint; every non-synonym query does share vocabulary). The evaluation is deterministic: two runs
produce identical metrics (also tested).

**What the numbers mean.** The perfect lexical scores exist because those queries were written to contain words from their targets.
The set is tiny and invented; it is a sanity check on the pipeline, **not** a measure of retrieval quality on real literature.
**The lexical system provides no semantic synonym matching** - the synonym row (0.0) is the measured consequence. No embeddings or
pretrained models are used, by project rule.

## Tests

`tests/test_p5_knowledge.py` (25) and `tests/test_p5_closure.py` (18, incl. parametrised): duplicate handling, deletion and index cleanup
across chunk/document/metadata/phrase search and storage, persistence across reopening, combined filters, chunk-level citation fields,
evaluation reproducibility and category separation, agent integration (tool-call evidence, verbatim statements, resolvable evidence
references from every evidence-producing agent), and authorization (401 unauthenticated, 404 for another user's project, upload and delete
blocked, no cross-project leakage) on every search level.

## Known gaps

* No semantic/synonym/cross-language matching; light English-only stemming; abbreviations and spelling variants not bridged.
* BM25 lives in memory and is rebuilt on open (fine for a modest library); PostgreSQL full-text search is deferred to P7.
* No bundled malware scanner (only a hook, reported `not configured`); no OCR (scanned PDFs are reported, not indexed).
* No external literature sources: only uploaded documents are searched. Nothing here implies coverage of any field.
* IDF depends on the library, so scores are not comparable across libraries; document-level rank is the best passage's score.
* The evaluation set is small and invented; real-corpus evaluation is outstanding and is the user's call.
