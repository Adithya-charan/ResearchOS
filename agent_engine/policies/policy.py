"""Permission layer. Deny by default: an agent can use only the tools and propose only the agent types listed for it."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from agent_engine.errors import PolicyViolation


@dataclass(frozen=True)
class AgentPolicy:
    allowed_tools: frozenset[str] = frozenset()
    may_propose: frozenset[str] = frozenset()


class PolicyEngine:
    def __init__(self, policies: Mapping[str, AgentPolicy], approval_required: Iterable[str] = ()):
        self._policies = dict(policies)
        self.approval_required = frozenset(approval_required)

    def policy_for(self, agent_type: str) -> AgentPolicy:
        return self._policies.get(agent_type, AgentPolicy())  # unknown agent -> no permissions

    def check_tool(self, agent_type: str, tool_name: str) -> None:
        if tool_name not in self.policy_for(agent_type).allowed_tools:
            raise PolicyViolation(f"agent '{agent_type}' may not use tool '{tool_name}'")

    def may_propose(self, parent_agent: str, child_agent: str) -> bool:
        return child_agent in self.policy_for(parent_agent).may_propose

    def requires_approval(self, agent_type: str) -> bool:
        """Tasks of these agent types wait for a human decision before they run (e.g. experiments)."""
        return agent_type in self.approval_required


def default_policy(approval_required: Iterable[str] = ("experiment",)) -> PolicyEngine:
    return PolicyEngine(
        {
            "supervisor": AgentPolicy(may_propose=frozenset({"discovery", "critic"})),
            "discovery": AgentPolicy(allowed_tools=frozenset({"corpus.search"})),
            "critic": AgentPolicy(
                allowed_tools=frozenset({"corpus.get_passage"}), may_propose=frozenset({"discovery", "critic"})
            ),
        },
        approval_required=approval_required,
    )


WORKFLOW_AGENTS = frozenset({"literature", "gap", "hypothesis", "verifier", "synthesis"})


def full_policy(approval_required: Iterable[str] = ("experiment",), experiments: bool = False) -> PolicyEngine:
    """The basic policy plus the research workflow. The workflow agents are read-only: none can search, write or propose."""
    return PolicyEngine(
        {
            "supervisor": AgentPolicy(may_propose=frozenset({"discovery", "critic"} | ({"experiment"} if experiments else set()))),
            "discovery": AgentPolicy(allowed_tools=frozenset({"corpus.search"})),
            "critic": AgentPolicy(
                allowed_tools=frozenset({"corpus.get_passage"}),
                may_propose=frozenset({"discovery", "critic"}) | WORKFLOW_AGENTS,
            ),
            "literature": AgentPolicy(allowed_tools=frozenset({"corpus.document_passages"})),
            "gap": AgentPolicy(),
            "hypothesis": AgentPolicy(),
            "verifier": AgentPolicy(allowed_tools=frozenset({"corpus.get_passage", "corpus.verify_passage"})),
            "synthesis": AgentPolicy(),
            "experiment": AgentPolicy(),  # the sandbox is not a tool: the agent calls the runner directly after approval
        },
        approval_required=approval_required,
    )
