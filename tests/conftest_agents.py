"""Shared helpers for agent-engine tests (imported explicitly; not a pytest plugin)."""
from __future__ import annotations

import asyncio

from agent_engine.agents.base import Agent
from agent_engine.contracts import AgentOutput, RunLimits
from agent_engine.policies.policy import AgentPolicy, PolicyEngine
from agent_engine.registry.registry import AgentRegistry
from agent_engine.scheduler.engine import ResearchEngine
from agent_engine.state.store import Store
from agent_engine.tools.registry import ToolRegistry


def run(coro):
    return asyncio.run(coro)


FAST = dict(retry_backoff_s=0.0, task_timeout_s=5.0, max_runtime_s=20.0)


def limits(**kw) -> RunLimits:
    return RunLimits(**{**FAST, **kw})


class ScriptedAgent(Agent):
    """A test agent whose behaviour is a callable (ctx, task) -> AgentOutput | awaitable."""

    description = "scripted test agent"

    def __init__(self, agent_type: str, fn):
        self.agent_type = agent_type
        self._fn = fn

    async def run(self, ctx, task):
        res = self._fn(ctx, task)
        if asyncio.iscoroutine(res):
            res = await res
        return res


def ok(task, **kw) -> AgentOutput:
    return AgentOutput(task_id=task.id, agent_type=task.agent_type, status="ok", summary=kw.pop("summary", "done"), **kw)


def make_engine(agents: dict, *, policy: PolicyEngine | None = None, store: Store | None = None, **kw) -> tuple[ResearchEngine, Store]:
    store = store or Store()
    reg = AgentRegistry()
    for name, fn in agents.items():
        reg.register(ScriptedAgent(name, fn))
    policy = policy or PolicyEngine({n: AgentPolicy(may_propose=frozenset(agents)) for n in agents})
    engine = ResearchEngine(store, reg, ToolRegistry(), policy, poll_interval=0.01, **kw)
    return engine, store


def start(engine: ResearchEngine, question: str = "how does temperature affect conductivity", **limit_kw) -> str:
    pid = engine.create_project("test project", question)
    return engine.create_run(pid, question, limits(**limit_kw))
