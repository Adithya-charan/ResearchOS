# Agent system

RESEARCHOS coordinates research with its own orchestrator (no LangChain/LangGraph). Agents are deterministic roles with typed
contracts; an optional, budgeted local model adds untrusted text notes only (never evidence).

## Agents

| Agent type | Role | Reads | Produces (`data`) | Tools | May propose |
|---|---|---|---|---|---|
| `supervisor` | Supervisor | question | terms, queries, stop_criteria (plan); on a failed search, a replan; evidence review (follow-up decision) | none | discovery, critic, verifier, supervisor (may also block reviewer, synthesis) |
| `discovery` | Literature Discovery | query | hits with citations | `corpus.search` | none |
| `critic` | Evidence Critic | discovery output | assessment (accepted/rejected evidence, sufficiency) | `corpus.get_passage` | discovery, critic, and the workflow agents |
| `literature` | Literature Analysis | critic | documents, statements (methods/results/limitations, verbatim) | `corpus.document_passages` | none |
| `gap` | Research Gap | critic, literature | candidate gaps (corpus gaps, never literature gaps) | none | none |
| `hypothesis` | Hypothesis | gap, literature | falsifiable hypotheses with assumptions | none | none |
| `experiment_design` | Experiment Design | hypothesis, literature, gap | reproducible experiment **plans** (no results) | none | none |
| `reviewer` | Critical Reviewer | critic, hypothesis, plans, literature | severity-graded review findings (never "approved") | none | none |
| `verifier` | Verification | literature, gap, hypothesis | claim records, `verification` (status, strength, evidence per claim), `novelty`, `round` | `corpus.get_passage`, `corpus.verify_passage`, `corpus.evidence_search` | none |
| `synthesis` | Synthesis | all of the above | report | none | none |
| `experiment` | Experiment Runner | approved spec | manifest (sandbox, human approval required) | none | none |

`GET /agents` returns each agent's declared spec (`agent_engine/agents/specs.py`): role, responsibilities, input keys, output keys,
timeout/retry ceilings, evaluation criteria, plus its tools and proposal permissions from the policy. A test runs the real workflow and
checks every task's actual input and output against these declarations.

## Orchestration

* Directed acyclic task graph; dependencies may only point backwards, so cycles cannot be created. A task runs only after every
  dependency **succeeded**; a failed dependency skips its dependents (failure propagation).
* Independent tasks run in parallel (`max_concurrency`); dependent ones sequentially. Shared project context is the
  project memory (e.g. the Supervisor's `scope`), read by every agent through the same API.
* Typed communication: every output is validated against `AgentOutput` (pydantic, `extra=forbid`) and its evidence references must
  resolve to stored passages before it is persisted. Agents see only the outputs of their declared dependencies.
* Bounds against runaway loops: `max_tasks`, `max_iterations` (refinement rounds), `max_runtime_s`, `max_retries`, per-task timeout,
  `max_model_tokens`, `max_replans`.
* Duplicate prevention: the planner rejects a proposal identical (agent, title, input, dependencies) to a task that is still pending,
  running or awaiting approval. Finished or failed copies may be repeated by a refinement round.
* Per-agent ceilings (`timeout_s`, `max_retries` in the spec) can only tighten the run's limits.
* **Replanning:** when a *discovery* task fails permanently, the engine adds a Supervisor task (`replan_for=<failed task>`), at most
  `max_replans` times (default 1). The Supervisor retries with a broader query (last term dropped) and a replacement evidence review
  over the searches that did succeed, so the workflow continues. If nothing new can be tried it adds nothing and says so. Only discovery
  failures are replanned today.
* Human approval checkpoints: experiment tasks wait for an approval bound to the exact spec hash.

## Retrieval integration (verified in `tests/test_p5_closure.py`)

Details of the index are in `docs/RETRIEVAL_SYSTEM.md`. Discovery searches only through the `corpus.search` tool (each call is a
`tool_call` event) and every hit is a stored passage. The critic's accepted hits are the only documents Literature Analysis reads, and it
extracts verbatim sentences with passage ids. Gap and Hypothesis cite evidence only when it exists (stated limitations, single-source
terms); otherwise their output says so. The Verifier resolves each extracted citation itself; the Reviewer and Experiment Design cite only
resolvable passages. All of this is lexical: no semantic synonym matching, and "the literature" means the uploaded library only.

## Evidence verification (P6, see `docs/EVIDENCE_VERIFICATION.md`)

Workflow order is now: critic -> literature -> gap -> hypothesis -> experiment design -> **verifier** -> **evidence review
(supervisor)** -> reviewer -> synthesis. The Verifier checks claims with the read-only `corpus.evidence_search` tool (full passage
text; `corpus.search` stays Discovery-only) and assigns SUPPORTED / PARTIALLY_SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE /
UNVERIFIED with a deterministic lexical rule. The Supervisor's evidence review may add at most `max_evidence_rounds` (default 1) rounds
of "evidence follow-up" discovery, re-verification and a further review; it records `budget_exhausted` or `no_new_query` when it stops.
`ProposedTask.blocks` (permitted per agent by `may_block`) makes the reviewer and synthesis tasks wait for such a chain; the engine adds
the dependency in the same transaction that creates the tasks. The Reviewer turns contradicted, unsupported, unverifiable and partially
supported claims, and similar-prior-work novelty results, into findings.

## Model integration (P8, see `docs/MODEL_OPERATIONS.md`)

Every project runs on the **deterministic provider** unless its owner selects the custom Transformer in the Model Lab (`PUT
/modellab/selection`). The selection is resolved when a run starts (`EngineRuntime.provider_resolver`): an eligible model becomes a
`LocalBackend` plus a `RolePolicy`; an ineligible or unavailable one degrades to deterministic and the run records why. Eligibility is
lifecycle-based (production needs DEPLOYED and measured stage >= 3; experimental needs the explicit opt-in).

Role policy: with the custom provider selected, each of the nine roles (supervisor, discovery, literature, gap, hypothesis,
experiment design, reviewer, verifier, synthesis) may attach one short draft note; the scheduler adds it for roles whose own code has no
model hook (`auto_notes`). `ModelGateway.handle(agent_type)` refuses roles outside the policy. The sandbox `experiment` role is never
model-driven.

What a model note can and cannot do: it is stored as an **untrusted `ModelNote`**, labelled as a model draft, and `guard_model_text`
drops any note containing citation-like or source-like text (bracketed numbers, author-year, DOIs, URLs, passage ids). Notes never
enter claim extraction, citation resolution, verification verdicts, novelty assessment or the report's evidence sections; those
remain deterministic. Disabling or losing the model changes nothing about the evidence outputs.

Measured stage today: the bundled models reach stage 1 at most (see `docs/MODEL_EVALUATION.md`). Stages 4 and 5 are not enabled.

## What the agents are not

They are lexical and extractive: BM25 retrieval, cue-word sentence classification, templated hypotheses and plans. Nothing is generated
by the language model except optional, labelled-untrusted notes. The reviewer's contradiction check is word overlap plus direction
words and can raise false alarms; its absence of findings is not an approval. Experiment plans name datasets, baselines and metrics
only when accepted documents mention them; otherwise those fields are listed under `needs_human_input`.

## Test doubles

Tests use scripted agents and a deliberately failing discovery agent (`FlakyDiscovery`) to force failures; they are test code and never
registered in the application.
