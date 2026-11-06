"""Experiment designer: turns each hypothesis into a *plan* (variables, datasets, baselines, metrics, reproducibility rules).

It designs; it never runs anything and never produces results. Every dataset, baseline or metric it names is a quotation
from an accepted document (with its citation); when the evidence names none, the plan says so and leaves the field for a human
instead of inventing one."""
from __future__ import annotations

import re

from agent_engine.agents.base import Agent, AgentContext
from agent_engine.contracts import AgentOutput, EvidenceReference, TaskSpec
from agent_engine.errors import NonRetryableError

MAX_PLANS = 5
MIN_REPEATS = 3
_DATASET = re.compile(r"\b(data ?sets?|corpus|benchmark|samples?|specimens?|cohort|participants|measurements?)\b", re.I)
_BASELINE = re.compile(r"\b(baseline|control|compared (?:with|to|against)|versus|relative to|reference)\b", re.I)
_METRIC_WORDS = ("accuracy", "precision", "recall", "f1", "perplexity", "loss", "error", "rmse", "mae", "auc", "bleu", "rouge",
                 "throughput", "latency", "yield", "efficiency", "retention", "capacity", "stability", "variance", "bandwidth")
_METRIC = re.compile(r"\b(" + "|".join(_METRIC_WORDS) + r")\b", re.I)
PLAN_LABEL = "Experiment plan - a design proposal. No experiment has been run and no result exists."


def _ref(s: dict) -> dict:
    return {"document_id": s["document_id"], "page_number": s["page_number"], "passage_id": s["passage_id"]}


class ExperimentDesignAgent(Agent):
    agent_type = "experiment_design"
    description = "Designs reproducible experiment plans for the hypotheses (variables, datasets, baselines, metrics); never runs them."

    async def run(self, ctx: AgentContext, task: TaskSpec) -> AgentOutput:
        by = {o.agent_type: o for o in task.dependency_outputs.values()}
        if "hypothesis" not in by or "literature" not in by:
            raise NonRetryableError("experiment_design needs the hypothesis and literature outputs as dependencies")
        statements = by["literature"].data["statements"]
        titles = {d["document_id"]: d["title"] for d in by["literature"].data["documents"]}

        def quoted(pattern: re.Pattern, kinds: tuple[str, ...], limit: int = 3) -> list[dict]:
            found, seen = [], set()
            for s in statements:
                if s["kind"] in kinds and pattern.search(s["text"]) and s["text"] not in seen:
                    seen.add(s["text"])
                    found.append({"quote": s["text"], "document": titles.get(s["document_id"], s["document_id"]), "citation": _ref(s)})
                if len(found) >= limit:
                    break
            return found

        datasets = quoted(_DATASET, ("method", "result"))
        baselines = quoted(_BASELINE, ("method", "result"))
        metric_hits = [s for s in statements if s["kind"] in ("result", "method") and _METRIC.search(s["text"])]
        metrics, seen_m = [], set()
        for s in metric_hits:
            for m in _METRIC.findall(s["text"]):
                if m.lower() not in seen_m:
                    seen_m.add(m.lower())
                    metrics.append({"name": m.lower(), "document": titles.get(s["document_id"], s["document_id"]),
                                    "quote": s["text"], "citation": _ref(s)})
        results = {}
        for s in statements:
            if s["kind"] == "result":
                results.setdefault(s["document_id"], s)

        plans, refs = [], []
        for h in by["hypothesis"].data["hypotheses"][:MAX_PLANS]:
            gap = next((g for g in by["gap"].data["gaps"] if g["id"] == h["gap_id"]), None) if "gap" in by else None
            if gap and gap["kind"] == "unaddressed_term":
                independent = {"name": gap["term"], "levels": "at least three levels, to be chosen by the researcher",
                               "source": "the question term no accepted document addresses"}
            elif gap and gap["kind"] == "stated_limitation":
                independent = {"name": "presence vs. absence of the limiting condition",
                               "levels": "limitation present / removed, everything else fixed",
                               "source": f"stated limitation: \"{gap['quote']}\""}
            else:
                independent = {"name": "to be defined by the researcher", "levels": "", "source": "no gap attached to this hypothesis"}
            outcome = next((results[r["document_id"]] for r in h["evidence"] if r["document_id"] in results), None)
            dependent = ({"name": "the outcome reported in the cited document", "source": f"\"{outcome['text']}\"", "citation": _ref(outcome)}
                         if outcome else {"name": "to be defined by the researcher", "source": "no reported outcome found in the evidence"})
            needs_human = [name for name, val in (("datasets", datasets), ("baselines", baselines), ("metrics", metrics)) if not val]
            if dependent["name"].startswith("to be defined"):
                needs_human.append("dependent variable")
            if independent["name"].startswith("to be defined"):
                needs_human.append("independent variable")
            plans.append({
                "id": f"p{len(plans) + 1}", "hypothesis_id": h["id"], "hypothesis": h["statement"],
                "independent_variable": independent, "dependent_variable": dependent,
                "assumptions": h["assumptions"], "falsification_criteria": h["falsification_criteria"],
                "datasets": datasets, "baselines": baselines, "metrics": metrics,
                "needs_human_input": needs_human,
                "reproducibility": {
                    "repeats_min": MIN_REPEATS, "seed_policy": "fixed, recorded seeds; one seed per repeat",
                    "record": ["configuration", "seeds", "code version", "environment", "input data hashes"],
                    "execution": "only through the approval-gated sandbox (scripts/run_experiment.py); a human approves the exact spec",
                },
                "status": "plan_only", "label": PLAN_LABEL,
            })
            refs += [EvidenceReference(**x["citation"]) for x in datasets + baselines + metrics]
            refs += [EvidenceReference(**r) for r in h["evidence"]]
        uniq = {(r.document_id, r.page_number, r.passage_id): r for r in refs}
        return AgentOutput(
            task_id=task.id, agent_type=self.agent_type, status="ok" if plans else "insufficient_evidence",
            summary=f"{len(plans)} experiment plan(s) drafted; none has been run.",
            findings=[f"{p['id']}: {p['hypothesis'][:140]}" for p in plans], evidence=list(uniq.values()),
            uncertainties=["Plans are templates filled from quoted evidence. Datasets, baselines and metrics are only those the "
                           "accepted documents mention; fields listed under needs_human_input are deliberately empty.",
                           "A plan is not a result: nothing here has been executed or measured."],
            data={"plans": plans},
        )
