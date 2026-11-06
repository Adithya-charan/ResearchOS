"""Declared contract of every agent: role, responsibilities, the input keys it reads, the `data` keys it must produce, memory use,
per-agent timeout/retry ceilings and the criteria it is evaluated against. The scheduler applies the timeout/retry ceilings (they can
only tighten the run's limits); tests run the real workflow and check each agent's actual inputs and outputs against these declarations,
so the table cannot silently drift from the code."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    agent_type: str
    role: str
    responsibilities: tuple[str, ...]
    input_keys: tuple[str, ...]          # keys the agent reads from task.input (required ones)
    output_keys: tuple[str, ...]         # keys it must put in AgentOutput.data
    memory_writes: tuple[str, ...] = ()  # project-memory keys it writes (prefix match)
    timeout_s: float | None = None       # ceiling for one attempt; None = the run's task_timeout_s
    max_retries: int | None = None       # ceiling on retries; None = the run's max_retries
    evaluation_criteria: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


SPECS: dict[str, AgentSpec] = {s.agent_type: s for s in (
    AgentSpec("supervisor", "Supervisor",
              ("receive the research question", "scope it into search facets", "plan discovery and review tasks as a dependency graph",
               "set the stop criteria", "replan when a planned task fails"),
              (), ("terms", "queries", "stop_criteria"), memory_writes=("scope",), timeout_s=30, max_retries=1,
              evaluation_criteria=("every proposed task is accepted by the planner", "plan has at least one discovery task",
                                   "replan produces no duplicate of an existing task")),
    AgentSpec("discovery", "Literature Discovery",
              ("search the local corpus", "return ranked passages with citations and source metadata"),
              ("query", "k"), ("query", "k", "hits", "query_terms"), timeout_s=60,
              evaluation_criteria=("every hit resolves to a stored passage", "hits are ordered by score")),
    AgentSpec("critic", "Evidence Critic",
              ("re-fetch every candidate passage", "check evidence sufficiency and diversity", "request a bounded follow-up search"),
              ("question", "terms"), ("assessment",), memory_writes=("assessment/",),
              evaluation_criteria=("accepted counts recomputed independently", "no unreliable page counted as evidence")),
    AgentSpec("literature", "Literature Analysis",
              ("extract methods, results and stated limitations as verbatim cited sentences", "report per-document metadata"),
              ("terms",), ("documents", "statements", "terms"),
              evaluation_criteria=("every statement is a verbatim sentence of a stored passage", "unreliable pages are skipped")),
    AgentSpec("gap", "Research Gap",
              ("compare findings across documents", "separate stated limitations and uncovered terms from absence of evidence"),
              ("terms",), ("gaps",),
              evaluation_criteria=("no gap is derived when no evidence was accepted", "every gap is labelled a corpus gap, not a literature gap")),
    AgentSpec("hypothesis", "Hypothesis",
              ("draft falsifiable hypotheses", "state assumptions", "state falsification criteria"),
              ("terms",), ("hypotheses",),
              evaluation_criteria=("each hypothesis has assumptions and falsification criteria", "each is labelled untested")),
    AgentSpec("experiment_design", "Experiment Design",
              ("design reproducible experiment plans", "name datasets, baselines and metrics only when the evidence does",
               "leave undefined fields for a human"),
              ("terms",), ("plans",),
              evaluation_criteria=("plans never contain results", "every named dataset/baseline/metric carries a citation",
                                   "at least three repeats planned")),
    AgentSpec("reviewer", "Critical Reviewer",
              ("challenge assumptions", "flag unsupported statements", "flag possible contradictions", "review plan methodology"),
              ("terms",), ("review",),
              evaluation_criteria=("findings carry severities", "never claims approval")),
    AgentSpec("verifier", "Verification",
              ("check every claim against its stored passage and original file", "recompute counts", "track uncertainty"),
              ("terms",), ("claims", "tensions", "counts", "source_files_checked"),
              evaluation_criteria=("a claim is verified only if its quote and numbers are in the cited passage",
                                   "hypotheses and gaps are never labelled verified")),
    AgentSpec("synthesis", "Synthesis",
              ("combine validated outputs into the report", "preserve citations", "document limitations"),
              ("question",), ("report",),
              evaluation_criteria=("report contains only persisted agent outputs", "limitations are always present")),
    AgentSpec("experiment", "Experiment Runner",
              ("run a human-approved experiment spec in the sandbox", "record a reproducibility manifest"),
              (), ("experiment",), timeout_s=600,
              evaluation_criteria=("never runs without a recorded approval for the exact spec",)),
)}


def spec_for(agent_type: str) -> AgentSpec | None:
    return SPECS.get(agent_type)
