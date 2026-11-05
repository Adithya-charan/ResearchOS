"""Planner: creates the initial plan and validates agent-proposed expansions against policy and run limits."""
from __future__ import annotations

from dataclasses import dataclass

from agent_engine.contracts import ProposedTask, RunLimits
from agent_engine.policies.policy import PolicyEngine
from agent_engine.state.store import new_id

ROOT_AGENT = "supervisor"


@dataclass(frozen=True)
class Rejection:
    key: str
    agent_type: str
    reason: str


def initial_plan(question: str) -> list[dict]:
    """A run starts with a single Supervisor task; everything else is proposed by agents and validated here."""
    return [{
        "id": new_id("task"), "agent_type": ROOT_AGENT, "title": "Scope the question and plan the research",
        "input": {"question": question}, "depends_on": [], "iteration": 0, "parent_task_id": None,
    }]


def expand(
    parent: dict,
    proposals: list[ProposedTask],
    existing: list[dict],
    limits: RunLimits,
    policy: PolicyEngine,
    known_agent_types: set[str],
) -> tuple[list[dict], list[Rejection]]:
    """Turn proposals into task rows. A proposal is rejected (never silently trimmed) if it breaks a rule;
    proposals that depend on a rejected sibling are rejected too. Dependencies may only point backwards
    (existing tasks or earlier proposals), so the resulting graph is acyclic by construction."""
    existing_ids = {t["id"] for t in existing}
    key_to_id: dict[str, str] = {}
    rejected_keys: set[str] = set()
    accepted: list[dict] = []
    rejections: list[Rejection] = []

    def reject(p: ProposedTask, reason: str) -> None:
        rejected_keys.add(p.key)
        rejections.append(Rejection(p.key, p.agent_type, reason))

    for p in proposals:
        if p.key in key_to_id or p.key in rejected_keys:
            reject(p, "duplicate_key")
            continue
        if p.agent_type not in known_agent_types:
            reject(p, "unknown_agent_type")
            continue
        if not policy.may_propose(parent["agent_type"], p.agent_type):
            reject(p, f"policy: '{parent['agent_type']}' may not propose '{p.agent_type}'")
            continue
        iteration = parent["iteration"] + (1 if p.followup else 0)
        if iteration > limits.max_iterations:
            reject(p, f"max_iterations ({limits.max_iterations}) reached")
            continue
        if len(existing) + len(accepted) + 1 > limits.max_tasks:
            reject(p, f"max_tasks ({limits.max_tasks}) reached")
            continue
        deps, bad = [], None
        for d in p.depends_on:
            if d in key_to_id:
                deps.append(key_to_id[d])
            elif d in existing_ids:
                deps.append(d)
            elif d in rejected_keys:
                bad = f"depends on rejected proposal '{d}'"
                break
            else:
                bad = f"unknown dependency '{d}'"
                break
        if bad:
            reject(p, bad)
            continue
        task_id = new_id("task")
        key_to_id[p.key] = task_id
        accepted.append({
            "id": task_id, "agent_type": p.agent_type, "title": p.title, "input": p.input,
            "depends_on": list(dict.fromkeys(deps)), "iteration": iteration, "parent_task_id": parent["id"],
            "requires_approval": policy.requires_approval(p.agent_type),
        })
    return accepted, rejections
