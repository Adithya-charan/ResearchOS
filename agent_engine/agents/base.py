"""Agent base class and the context object an agent receives."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from agent_engine.contracts import AgentOutput, ModelNote, RunLimits, TaskSpec
from agent_engine.memory.manager import BoundMemory
from agent_engine.model_gateway import ModelHandle
from agent_engine.tools.registry import BoundTools


@dataclass
class AgentContext:
    run_id: str
    project_id: str
    question: str
    limits: RunLimits
    tools: BoundTools
    memory: BoundMemory
    model: ModelHandle | None
    _emit: Callable[[str, dict], None]

    def emit(self, event_type: str, payload: dict | None = None) -> None:
        self._emit(event_type, payload or {})

    async def model_note(self, prompt: str, max_new_tokens: int = 48, seed: int = 0) -> ModelNote | None:
        """Optional, budgeted, untrusted. Returns None when no model is attached or the token budget is spent."""
        if self.model is None:
            return None
        return await self.model.note(prompt, max_new_tokens, seed)


class Agent(ABC):
    """A role, not a model: deterministic logic + permitted tools (+ optional untrusted model note)."""

    agent_type: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput: ...
