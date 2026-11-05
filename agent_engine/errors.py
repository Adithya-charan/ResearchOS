"""Exception hierarchy for the agent engine."""
from __future__ import annotations


class AgentEngineError(Exception):
    """Base class."""


class NonRetryableError(AgentEngineError):
    """The task cannot succeed by trying again (bad input, forbidden action)."""


class PolicyViolation(NonRetryableError):
    """An agent attempted something its policy does not allow (tool, proposal, ...)."""


class OutputValidationError(AgentEngineError):
    """An agent returned output that fails schema or evidence validation. Retryable (bounded)."""


class GraphError(AgentEngineError):
    """The task graph is malformed (cycle, unknown dependency, duplicate id)."""


class RunStateError(AgentEngineError):
    """The requested operation is not valid in the run's current state."""
