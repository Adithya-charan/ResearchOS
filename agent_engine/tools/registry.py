"""Tool registry. Every call is permission-checked, argument-validated and logged as events."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agent_engine.errors import NonRetryableError
from agent_engine.policies.policy import PolicyEngine

Emit = Callable[[str, dict], None]


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any], Any]  # receives the validated input model; may be sync or async
    side_effect: str = "read"  # read | write | external - external tools stay off by default


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise NonRetryableError(f"unknown tool '{name}'")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def bind(self, agent_type: str, policy: PolicyEngine, emit: Emit) -> BoundTools:
        return BoundTools(agent_type, self, policy, emit)


def _short(value: Any, limit: int = 160) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "…"
    return value


class BoundTools:
    """The only handle an agent gets to tools: scoped to that agent's policy."""

    def __init__(self, agent_type: str, registry: ToolRegistry, policy: PolicyEngine, emit: Emit):
        self.agent_type, self._registry, self._policy, self._emit = agent_type, registry, policy, emit

    def has(self, name: str) -> bool:
        """True if the tool exists AND this agent is allowed to use it (lets an agent use optional tools gracefully)."""
        return name in self._registry.names() and name in self._policy.policy_for(self.agent_type).allowed_tools

    async def call(self, name: str, **kwargs: Any) -> Any:
        self._policy.check_tool(self.agent_type, name)  # raises PolicyViolation (non-retryable)
        tool = self._registry.get(name)
        try:
            args = tool.input_model(**kwargs)
        except ValidationError as e:
            raise NonRetryableError(f"invalid arguments for {name}: {e.errors()[0]['msg']}") from e
        self._emit("tool_call", {"agent": self.agent_type, "tool": name, "args": {k: _short(v) for k, v in kwargs.items()}})
        result = tool.fn(args)
        if inspect.isawaitable(result):
            result = await result
        self._emit("tool_result", {"agent": self.agent_type, "tool": name, "ok": True})
        return result
