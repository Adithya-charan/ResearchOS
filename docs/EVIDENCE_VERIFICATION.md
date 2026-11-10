# Evidence verification and research intelligence (P6)

This document describes what P6 adds and, just as important, what it does **not** do. The verifier is a
deterministic lexical checker. It does not understand text, it does not prove anything, and a status is a statement
about the stored passages that were examined, never about the world.

## 1. Data model

Persistence lives in the API database (Alembic revision `0003_claims_and_evidence`).

| Table | Purpose |
|---|---|
| `claims` | One row per claim: id, project, run, text, `claim_type`, originating agent and task, `status`, `strength`, `confidence_note`, created/verified timestamps. |
| `claim_evidence` | Passages attached to a claim: role (`supporting` / `contradicting` / `contextual`), directness (`direct` / `indirect`), document id, passage id, page number (only when the stored passage has one), verbatim quote, retrieval method and score, `origin` (`user` or `verification`). User-attached rows survive re-verification; verification rows are replaced. |
| `claim_verifications` | Append-only verification history: status, strength, explanation, checks, round number, verifier version. |
| `novelty_assessments` | Bounded novelty assessments with label, closest documents, overlapping aims, searched queries, limitations. |

The engine (agent) side keeps claims inside task outputs; `persist_run_claims` copies them to the API tables after a
report is generated and is idempotent.

## 2. Verification algorithm (`research/evidence/verify.py`)

`verify_claim(text, claim_type, candidates, searched=...)` is a pure function: same inputs, same output.

1. Each candidate passage is split into sentences. Passages with an invalid citation (cannot be resolved in the
   project library) or an unreliable extraction are listed and **excluded**.
2. Claim and sentence are reduced to stems. Coverage is the share of claim stems present in the sentence. A claim needs
   at least `MIN_CLAIM_STEMS = 3` stems to be checkable.
3. A sentence is evidence at tier DIRECT (coverage ≥ 0.75), INDIRECT (≥ 0.50) or CONTEXT (≥ 0.25).
4. Numbers in the claim must appear in the sentence for direct support.
5. Direction is compared clause by clause (up/down word sets; negation flips it). Opposite direction, a negated
   version of the claim, or different figures for the same quantity (coverage ≥ 0.6, and not when both sentences agree on
   direction) makes the sentence **contradicting**.
6. Roles are combined into a status.

| Status | Rule |
|---|---|
| `SUPPORTED` | at least one direct supporting sentence and no contradicting one |
| `PARTIALLY_SUPPORTED` | only indirect support, or supporting and contradicting evidence together (`conflicting=true`) |
| `CONTRADICTED` | contradicting evidence and no supporting evidence |
| `INSUFFICIENT_EVIDENCE` | passages were searched but none addressed the claim |
| `UNVERIFIED` | not checked: no searchable material, too few stems, or claim type `gap` / `hypothesis` (these are never "supported", their passages are attached as contextual) |

Strength is `none` / `weak` / `moderate` / `strong`, based on the number of distinct supporting documents. It is not a
probability. The retrieval score is recorded for transparency but never influences the status: relevance is not
proof.

Quotes are verbatim sentences from the stored passage. Page numbers come only from stored passages (Markdown has none,
so `page_number` is `null`; nothing is invented).

## 3. Novelty (`research/evidence/novelty.py`)

`assess_novelty` compares a statement with the corpus that was actually searched. Labels are deliberately cautious:

| Label | Meaning |
|---|---|
| Similar prior work identified | best sentence coverage ≥ 0.6 |
| Novelty unclear | partial overlap (≥ 0.3) |
| Potentially novel within searched corpus | nothing close in the searched documents |
| Insufficient literature coverage | fewer than `MIN_CORPUS_DOCUMENTS = 5` documents |

Every assessment lists the queries run, the closest documents, overlapping research aims, differences, unsearched
areas and a disclaimer. It never claims universal novelty and it excludes the statement's own source documents.

## 4. Report (`agent_engine/reporting/structured.py`)

`report["structured"]` has 18 sections: Title, Executive summary, Research question, Scope, Methodology, Literature
landscape, Key findings, Evidence table, Contradictions, Research gaps, Hypotheses, Experiment design, Novelty
assessment, Limitations, Confidence and verification summary, Reviewer findings, Next steps, References. Factual
claims appear with their status, and anything not verified is labelled so. The report content is an immutable snapshot;
citations are enriched with source text and a `resolvable` flag at read time.

## 5. Agent integration

Workflow order: …, Verifier → Evidence review (Supervisor) → Reviewer → Synthesis.

* **Verifier** extracts claims (≤ 24) from Literature Analysis, Discovery, Gap, Hypothesis and Experiment Design
  outputs, verifies them with `corpus.evidence_search` (read-only, k ≤ 10, returns full passage text), and runs novelty on up to
  6 statements. `corpus.search` stays Discovery-only.
* **Supervisor evidence review** reads the merged verification. If claims are `PARTIALLY_SUPPORTED`,
  `INSUFFICIENT_EVIDENCE` or `CONTRADICTED` and rounds remain (`RunLimits.max_evidence_rounds`, default 1), it proposes
  up to 3 follow-up Discovery tasks labelled "evidence follow-up", a re-verification round for the affected claims, and
  the next evidence review. When the budget is exhausted or no new query can be formed it records
  `budget_exhausted` / `no_new_query` and stops. Retries are bounded by the planner, not by agent goodwill.
* `ProposedTask.blocks` lets a proposal hold pending tasks of other agent types (the reviewer and synthesis) until the
  follow-up chain has finished. The engine adds that dependency in the same transaction that creates the tasks; the
  policy field `may_block` restricts who can do this (Supervisor: reviewer, synthesis).
* **Reviewer** challenges unsupported statements: findings `contradicted_claim`, `contradictory_evidence`,
  `unsupported_statement`, `unverifiable_claim`, `partial_support`, `novelty`, `evidence_follow_up`.

## 6. API

All routes require authentication, are scoped to a project the caller owns (another user's project returns 404), and
writes require the CSRF token.

| Method | Path |
|---|---|
| POST / GET | `/projects/{id}/claims` |
| GET | `/projects/{id}/claims/{claim_id}` |
| POST / DELETE | `/projects/{id}/claims/{claim_id}/evidence[/{evidence_id}]` |
| POST | `/projects/{id}/claims/{claim_id}/verify` |
| GET | `/projects/{id}/claims/{claim_id}/verifications` |
| GET | `/projects/{id}/verifications` |
| POST | `/runs/{id}/claims/extract` |
| POST / GET | `/projects/{id}/novelty`, `/projects/{id}/novelty/{assessment_id}` |
| GET | `/reports/{id}/structured`, `/reports/{id}/structured.md` |

Attaching evidence resolves the passage inside the project's library. If it does not resolve, the request fails with
422 and nothing is stored.

## 7. Evaluation (`scripts/eval_verification.py`)

The dataset in `datasets/verification_eval/` is **entirely synthetic**: seven small invented documents, 21 labelled
claims and 4 novelty cases. It exercises the algorithm; it is not validation against real literature, and no real-source
examples ship with it (the README explains how to add a `real/` folder). Results (`artifacts/evaluation/verification_eval.json`):

* accuracy excluding documented known limitations: 18/18
* known-limitation cases: 0/3 (kept in the set on purpose)
* overall: 18/21 = 0.857
* novelty labels: 4/4

The three misses: a figure contradiction where the numbers differ but direction agrees (reported
`PARTIALLY_SUPPORTED`, expected `CONTRADICTED`); swapped roles ("massive colonies bleached earlier than branching
colonies") reported `SUPPORTED` because lexical coverage cannot see argument order; and a heavy paraphrase reported
`INSUFFICIENT_EVIDENCE`. The 100 % figure on the remaining cases says the rules do what they were written to do on data written by
the same author; it must not be read as real-world accuracy.

## 8. Limitations and known gaps

* Lexical matching only: no paraphrase understanding, no argument-order (subject/object) understanding, no scope or
  quantifier reasoning, no coreference.
* Direction and negation use small hand-written word lists; unusual phrasing is missed. "without" and bare "no" are
  deliberately not treated as negation.
* A SUPPORTED status means a stored sentence covers the claim's content words. It is not a proof and not a check of the
  source's own correctness.
* Novelty is relative to the searched corpus only, and is a lexical comparison.
* The evaluation corpus is synthetic and small; thresholds were set with it in view.
* Real-source evaluation is still to do.

## P11.2 evidence quality additions

* **Traceable records.** Each evidence record keeps `source_location` (document, passage, page, section), the original `passage_text`, the claim's `quote`, a
  `relevance_score` (the share of the claim's content stems the sentence covers), the role (supporting / contradicting / contextual), and a plain-language
  `reason`. Passages judged irrelevant are listed by id under `irrelevant_passages` instead of silently disappearing; unresolvable and low-extraction-quality
  passages are kept apart (`invalid_citations`, `unreliable_passages`).
* **New contradiction rules** (each with a false-positive guard and tests): a figure that differs *for the same unit* ("19 percent" vs "40 percent") even
  when another number recurs; comparisons with swapped roles ("A earlier than B" vs "B earlier than A") while treating opposite comparatives that say the same
  thing as agreement; and a null result against a reported change ("no significant difference" vs "improved"), which the evaluation found being returned as
  SUPPORTED.
* **Still wrong, on purpose in the data:** paraphrases with no shared words are missed; a claim whose direction word conflicts with itself ("decreased as … grew") and a claim
  matched on words in different roles ("highest on sand") can come back SUPPORTED; an unseen qualifier ("clay soils") is not noticed; a claim split across two
  documents is not connected. [EVALUATION_P11.md](EVALUATION_P11.md) reports these as the unsupported-claim rate (0.22 on the P11 set).

## P11.4 citation integrity and P11.5 report quality

`GET /reports/{id}/structured` now also returns `citation_integrity`, recomputed on every read against the corpus **as it is now** (`research/evidence/citations.py`):
each citation must name a passage that exists, belong to the cited document, have a matching page, quote text that occurs in the stored passage, and still
match the text embedded in the report; deleted documents show as unresolvable, never as valid. Claims shown as supported with no valid citation are listed as
`unsupported_claims`. It does **not** check that a passage supports its claim; that stays the verification status.

The 18-section structure is unchanged. Each block now has an `epistemic` label (evidence / fact / inference / absence), and `structured.quality` summarises
evidence coverage, contradictions, listed limitations and recommendations, with a reading note that coverage is not a measure of how much of the literature was read.
Run reliability (`GET /runs/{id}/reliability`, also snapshotted in the report as `execution`) states which agents needed retries and which failed and why.
