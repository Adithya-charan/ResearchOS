"""Run summary assembled from persisted task outputs (never from anything an agent said outside its validated output)."""
from __future__ import annotations

from agent_engine.contracts import AgentOutput
from agent_engine.state.store import Store

LIMITATIONS = [
    "Findings come only from the documents indexed for this run. Failing to find a close match is not proof that an "
    "idea has never been published.",
    "Evidence links point to indexed passages; whether a passage truly supports a claim still needs human review.",
    "Model notes (if any) come from an experimental small language model and are untrusted drafts, not evidence.",
]


def build_summary(store: Store, run_id: str) -> dict:
    run = store.get_run(run_id)
    tasks = store.list_tasks(run_id)
    outputs = [(t, AgentOutput.model_validate(t["output"])) for t in tasks if t["status"] == "succeeded" and t["output"]]

    agents: dict[str, dict[str, int]] = {}
    for t in tasks:
        a = agents.setdefault(t["agent_type"], {"tasks": 0, "succeeded": 0})
        a["tasks"] += 1
        a["succeeded"] += t["status"] == "succeeded"

    assessment = None
    corpus = None
    uncertainties: list[str] = []
    notes = []
    for t, out in outputs:  # task order == creation order, so the last assessment is the most recent one
        if "assessment" in out.data:
            assessment = out.data["assessment"]
        if corpus is None and "corpus" in out.data:
            corpus = out.data["corpus"]
        for u in out.uncertainties:
            if u not in uncertainties:
                uncertainties.append(u)
        notes += [{"task_id": t["id"], "agent_type": t["agent_type"], "text": n.text, "trusted": False} for n in out.model_notes]

    return {
        "question": run["question"],
        "task_counts": store.task_counts(run_id),
        "agents": agents,
        "assessment": assessment,
        "evidence": (assessment or {}).get("accepted_evidence", []),
        "uncertainties": uncertainties,
        "corpus": corpus,
        "model_tokens_used": run["model_tokens_used"],
        "model_notes": notes,
        "failed_tasks": [{"id": t["id"], "agent_type": t["agent_type"], "error": t["error"]} for t in tasks if t["status"] == "failed"],
        "limitations": LIMITATIONS,
    }
