"""Agent registry: maps an agent type name to the agent implementation."""
from __future__ import annotations

from agent_engine.agents.base import Agent
from agent_engine.policies.policy import PolicyEngine


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        if not getattr(agent, "agent_type", ""):
            raise ValueError("agent must define agent_type")
        if agent.agent_type in self._agents:
            raise ValueError(f"agent type '{agent.agent_type}' already registered")
        self._agents[agent.agent_type] = agent

    def get(self, agent_type: str) -> Agent:
        return self._agents[agent_type]

    def __contains__(self, agent_type: str) -> bool:
        return agent_type in self._agents

    def types(self) -> list[str]:
        return sorted(self._agents)

    def describe(self, policy: PolicyEngine) -> list[dict]:
        out = []
        for t in self.types():
            p = policy.policy_for(t)
            out.append({
                "agent_type": t,
                "description": self._agents[t].description,
                "tools": sorted(p.allowed_tools),
                "may_propose": sorted(p.may_propose),
                "requires_approval": policy.requires_approval(t),
            })
        return out
