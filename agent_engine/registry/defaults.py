"""The default agent set and a one-call engine factory."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agent_engine.agents import (CriticAgent, ExperimentAgent, ExperimentDesignAgent, ReviewerAgent, DiscoveryAgent, GapAnalystAgent, HypothesisAgent, LiteratureAgent,
                                 SupervisorAgent, SynthesisAgent, VerifierAgent)
from agent_engine.model_gateway import ModelGateway, TextBackend
from agent_engine.policies.policy import PolicyEngine, default_policy, full_policy
from agent_engine.registry.registry import AgentRegistry
from agent_engine.scheduler.engine import ResearchEngine
from agent_engine.state.store import Store
from agent_engine.tools.corpus import IndexEvidenceResolver, build_corpus_tools
from agent_engine.tools.registry import ToolRegistry
from research.bm25.index import BM25Index
from research.experiments.sandbox import Sandbox


def default_registry(discovery_k: int = 8, workflow: str = "basic") -> AgentRegistry:
    reg = AgentRegistry()
    agents = [SupervisorAgent(discovery_k, workflow == "full"), DiscoveryAgent(), CriticAgent()]
    if workflow == "full":
        agents += [LiteratureAgent(), GapAnalystAgent(), HypothesisAgent(), ExperimentDesignAgent(), ReviewerAgent(),
                   VerifierAgent(), SynthesisAgent()]
    for agent in agents:
        reg.register(agent)
    return reg


def build_engine(
    store: Store,
    index: BM25Index,
    *,
    backend: TextBackend | None = None,
    discovery_k: int = 8,
    policy: PolicyEngine | None = None,
    workflow: str = "basic",
    experiments: tuple[Sandbox, Path] | None = None,
    catalog: Callable[[str], dict | None] | None = None,
    verifier: Callable[[str], tuple[bool, str]] | None = None,
    **engine_kwargs,
) -> ResearchEngine:
    """Supervisor + Discovery + Critic over a BM25 index (workflow="basic"), or the full research workflow
    (workflow="full": + literature, gap, hypothesis, experiment design, reviewer, verifier, synthesis). `experiments=(sandbox, root_dir)` also registers
    the approval-gated experiment agent. `verifier` re-checks a passage against its
    original file (e.g. Corpus.verify_passage)."""
    if workflow not in ("basic", "full"):
        raise ValueError("workflow must be 'basic' or 'full'")
    registry = default_registry(discovery_k, workflow)
    if experiments is not None:
        registry.register(ExperimentAgent(experiments[0], experiments[1], store))
    tools = ToolRegistry()
    for tool in build_corpus_tools(index, catalog, verifier):
        tools.register(tool)
    return ResearchEngine(
        store, registry, tools,
        policy or (full_policy(experiments=experiments is not None) if workflow == "full" else default_policy()),
        model=ModelGateway(backend, store) if backend is not None else None,
        resolver=IndexEvidenceResolver(index), **engine_kwargs,
    )
