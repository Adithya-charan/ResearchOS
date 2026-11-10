# P11 report: research intelligence quality and reliability

Status: **implemented and tested; uncommitted** (nothing has been committed or pushed for P11). Everything below was measured in this session on the
author's Windows machine. Docker, GitHub Actions and real HTTPS were not run, and no real-document evaluation was performed.

## What was added (and what was not changed)

P11 is additive. The BM25 baseline, the 18-section report format, the verification statuses, the API shapes and every earlier test are unchanged. New
behaviour is either opt-in (`ranker=advanced`) or new fields/endpoints.

| Sub-phase | Delivered | Main files |
|---|---|---|
| P11.1 advanced retrieval | Opt-in ranker: query normalisation, phrase bonus, title/heading boost, prefix variants, pseudo-relevance feedback, near-duplicate suppression, per-document cap, explainable score components. Identity configuration reproduces the baseline exactly. Benchmark with ablations. | `research/bm25/advanced.py`, `research/bm25/index.py`, `research/retrieval_benchmark.py`, `research/corpus/corpus.py`, `api/routes/documents.py` |
| P11.2 evidence quality | Evidence records keep source location, original passage text, relevance score, role, reason; irrelevant passages listed by id; unit-aware figure conflicts, swapped comparison roles, null-result-vs-change contradiction. | `research/evidence/verify.py`, `research/evidence/types.py`, `agent_engine/agents/evidence.py`, `api/claims_service.py` |
| P11.3 gap analysis | Corpus-scoped gaps: stated limitations, unanswered questions, conflicting findings, single-document themes, missing-comparison hypotheses, follow-ups. Observed vs hypothesis; every gap linked to passages; traceability validator; no novelty claims. `GET /projects/{id}/gaps`. | `research/gaps.py`, `api/routes/documents.py` |
| P11.4 citation integrity | Existence, document, page, quote and source-text checks per citation; unsupported-claim list; recomputed on every read. | `research/evidence/citations.py`, `api/routes/reports.py` |
| P11.5 report quality | 18 sections kept. Per-block statement-type label, evidence-coverage summary, contradiction/limitation/recommendation counts, citation-integrity summary, Markdown header line. | `agent_engine/reporting/structured.py` |
| P11.6 agent reliability | Audit found bounded retries, per-attempt timeouts, output validation, crash re-queue, dependent skipping and per-task persisted output already in place (and tested). Added one reliability view: attempts, retries, timeouts, failures with root cause, skipped dependents traced to the failure, partial results. `GET /runs/{id}/reliability`, snapshot in each report. | `agent_engine/reliability.py`, `api/routes/runs.py`, `api/routes/reports.py` |
| P11.7 evaluation | Invented CC0 corpus, 45 retrieval queries, 26 evidence cases, citation-defect injection, gap expectations, 3 end-to-end API runs. JSON outputs and methodology. | `datasets/p11_eval/`, `research/p11_evaluation.py`, `scripts/eval_p11.py`, `scripts/eval_retrieval_p11.py`, `docs/EVALUATION_P11.md` |
| P11.8 frontend | Report page: evidence-coverage and citation-integrity panel, per-block statement tag, citation-issue and "no valid citation" markers. Gaps page: library-wide analysis with observed/hypothesis split and linked passages. Agents page: execution-reliability panel. Shared `Table` made keyboard-focusable; light-mode `--warn` colour darkened for AA contrast. | `frontend/src/components/{QualityPanel,LibraryGaps,ReliabilityPanel}.tsx`, `frontend/src/lib/{quality,types}.ts`, `frontend/src/pages/{Reports,Gaps,Agents}.tsx` |
| P11.9 / P11.10 | Tests and docs (this file, `EVALUATION_P11.md`, additions to `RETRIEVAL_SYSTEM.md`, `EVIDENCE_VERIFICATION.md`, `README.md`). | |

## Tests and results (exact, this session)

| Check | Result |
|---|---|
| New P11 backend tests (7 files: retrieval 18, evidence 18, citations 7, gaps 9, API 5, reliability 6, evaluation 6) | 69 passed |
| **Full backend suite** (`pytest tests`, includes the P10 security and isolation tests) | **1060 passed, 1 skipped, 1 warning, exit 0** (344 s). Earlier baseline was 991 passed / 1 skipped, so +69. |
| Frontend type check (`tsc --noEmit`) | clean |
| ESLint (`--max-warnings 0`) | clean |
| Vitest | 39 passed (6 files; 31 before, +8) |
| Production build | succeeded |
| Playwright against the real backend | 3 passed (accessibility scan light+dark on every page, full flow including the new panels, keyboard + mobile scan), 2 skipped (the perf and screenshot specs are gated) |
| CI lint rule set (`ruff --select E9,F63,F7,F82`) | passed |
| Secret scan (gitleaks, working tree) | no findings in tracked, modified or new project files (212 are inside the third-party `.venv`; 5 are in ignored generated files) |

Existing tests changed: `tests/test_p6_evidence.py` (assertions added for integrity, quality and reliability), `tests/test_p8_api.py` (one test now creates its own
project instead of relying on one left by an earlier test; it failed when run alone, passed in the full run). No assertion was weakened.

## Retrieval benchmark comparison (45 queries, invented corpus)

| Setting | Recall@3 | Recall@5 | Precision@1 | MRR | Duplicate pairs in top 5 |
|---|---|---|---|---|---|
| Baseline BM25 (default, unchanged) | 0.9537 | 0.9756 | 0.9756 | 0.9878 | 9 |
| Recommended opt-in (normalise + feedback + de-dup) | 0.9659 | 1.0000 | 0.9756 | 0.9878 | 0 |

Modest by design: the baseline is already near ceiling on this set. Feedback expansion fixed vocabulary-mismatch queries (Recall@3 0.917 → 1.0) and
de-duplication removed duplicates; phrase, title and normalisation features showed no gain; heading boost and variants made some measures worse and are
not recommended. Full table and discussion: [EVALUATION_P11.md](EVALUATION_P11.md).

## Evidence quality results (26 author-labelled claims, invented corpus)

Status accuracy 0.6538 (17/26). Exact 8/8, contradictory 4/6, insufficient 4/5, multi-document 1/2, **paraphrase 0/5**. Evidence-support precision 0.7778;
**unsupported-claim rate 0.2222** (two claims labelled SUPPORTED that the passages contradict); contradiction precision 1.0, recall 0.6667. Citation checker: every
injected defect found, no false alarms. Gap analysis: both expected conflicts found, every stated limitation found, no traceability problems. End to end: 3/3 runs
completed, 3/3 reports with all 18 sections, 287/287 blocks labelled, citation integrity 1.0 on the two reports that had citations (the third found no claims).
One expected label (x05) was wrong and was corrected after the first run; this is disclosed in the dataset and in `EVALUATION_P11.md`.

## Limitations

* All evaluation is on an **invented, small, same-author corpus**; the numbers are an upper bound and not scientific validation. No real documents were used.
* Verification is **lexical**. It misses paraphrases, can label a contradicted claim SUPPORTED when words recur in different roles or a claim is internally ambiguous, does not notice a missing qualifier, and does not join a claim split across documents.
* Gap analysis finds only **stated** limitations and word-pattern themes; "conflicting findings" inherit the verifier's false-alarm risk; "missing comparison" is a hypothesis from word co-occurrence; it says nothing about literature outside the library, and never that anything is new.
* Citation integrity proves a citation is real and accurate, **not** that the passage supports the claim.
* Report "evidence coverage" counts extracted claims, not how much of the field was read.
* Reliability reporting describes existing scheduler behaviour; no new retry logic was added, and failure injection coverage is the existing scheduler tests plus the six new reporting tests.
* The experimental custom Transformer is not research-capable and nothing in P11 uses it.

## Unfinished / not verified

* Docker, Docker Compose, GitHub Actions and real HTTPS were **not run** (no Docker on this machine; `gh` is not authenticated). Unchanged from `RELEASE_CHECKLIST.md`.
* Real-corpus evaluation needs a licensed corpus chosen by the project owner.
* Not done in the UI: a ranker toggle in the library search (the API option exists), and per-claim display of the new evidence-record fields on the Evidence page.
* PyMuPDF (AGPL) licence decision, the missing LICENSE file and Dependabot PR #14 remain open.
* Manual screen-reader testing was not done (automated axe scans only).

## Git status

Nothing from P11 is committed or pushed. Local commit `dfad702` (security fixes) is also not pushed. Staged-but-uncommitted earlier items: README screenshots section, `docs/screenshots/*`,
`artifacts/evaluation/pdf_backend_spike.json`, `scripts/pdf_backend_spike.py`, the `docs/SECURITY.md` spike section. A commit is proposed only on your approval.
