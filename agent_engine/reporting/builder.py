"""Assembles the research report from persisted agent outputs. Nothing here is generated: every sentence in the report is a
quoted passage sentence, a template filled from agent data, or a fixed limitation. Only verified claims become findings."""
from __future__ import annotations

from agent_engine.contracts import AgentOutput

NOVELTY_NONE = ("No close match was identified in the indexed corpus searched for this run. "
                "This is not proof that the idea has never been published.")
NOVELTY_FOUND = ("Close matches were found in the indexed corpus searched for this run (listed under closest documents). "
                 "Absence of other matches elsewhere is not established, and these matches do not by themselves show overlap.")
LIMITATIONS = [
    "Everything here comes from the indexed documents only; the report cannot speak for literature outside them.",
    "Findings are verbatim sentences selected by section titles and cue words (heuristics). 'Verified' means the quote and its "
    "numbers are present in the cited stored passage - not that the statement is true or relevant.",
    "Gaps and hypotheses are automatically drafted proposals. They are not findings, have not been tested, and need human judgement.",
    "Possible tensions are flagged by word overlap and direction words and may not be real contradictions.",
    "No experiments were run for this report. Experiments are separate, human-approved sandbox runs (scripts/run_experiment.py).",
]


def _cite(c: dict) -> dict:
    return {"document_id": c["document_id"], "page_number": c["page_number"], "passage_id": c["passage_id"]}


def build_report(question: str, outputs: dict[str, AgentOutput], *, review_status: str = "pending") -> dict:
    """`outputs` maps agent_type -> AgentOutput (critic, literature, gap, hypothesis, verifier)."""
    critic, lit, gap, hyp, ver = (outputs.get(k) for k in ("critic", "literature", "gap", "hypothesis", "verifier"))
    assessment = critic.data["assessment"] if critic else {}
    corpus = critic.data.get("corpus") if critic else None
    documents = lit.data["documents"] if lit else []
    claims = ver.data["claims"] if ver else []
    by_status: dict[str, list[dict]] = {}
    for c in claims:
        by_status.setdefault(c["status"], []).append(c)

    verified = [c for c in by_status.get("verified", []) if c["type"] == "extracted"]
    findings = {k: [{"claim_id": c["id"], "text": c["text"], "citation": _cite(c), "section": c["section"]}
                    for c in verified if c["kind"] == k] for k in ("method", "result", "limitation")}
    titles = {d["document_id"]: d["title"] for d in documents}
    closest = [{**d, "statements": len(d["statement_ids"])} for d in documents]
    accepted = len(assessment.get("accepted", []))
    status = "complete" if assessment.get("sufficient") and verified else "insufficient_evidence"
    years = sorted(d["year"] for d in documents if d.get("year"))

    uncertainties: list[str] = []
    for o in outputs.values():
        for u in o.uncertainties:
            if u not in uncertainties:
                uncertainties.append(u)
    if status == "insufficient_evidence":
        uncertainties.insert(0, "The evidence check did not find enough solid evidence; this report is a partial record, "
                                "not a literature review.")
    return {
        "question": question,
        "status": status,
        "novelty_statement": NOVELTY_NONE if not accepted and not assessment.get("low_quality_matches") else NOVELTY_FOUND,
        "scope": {
            "corpus": corpus, "terms": lit.data["terms"] if lit else [], "years": [years[0], years[-1]] if years else None,
            "accepted_passages": accepted, "accepted_documents": assessment.get("documents", 0),
            "uncovered_terms": assessment.get("uncovered_terms", []),
            "low_quality_matches": len(assessment.get("low_quality_matches", [])),
            "unsearched": "Anything outside the indexed corpus: other databases, unindexed documents, and newer or unpublished work.",
        },
        "closest_documents": closest,
        "findings": findings,
        "needs_review": [{"claim_id": c["id"], "text": c["text"], "reasons": c["reasons"],
                          "citation": _cite(c) if c["passage_id"] else None} for c in by_status.get("needs_review", [])],
        "unsupported_claims": [{"claim_id": c["id"], "text": c["text"], "reasons": c["reasons"]} for c in by_status.get("unsupported", [])],
        "tensions": ver.data["tensions"] if ver else [],
        "candidate_gaps": [{**g, "status": "candidate_gap"} for g in (gap.data["gaps"] if gap else [])],
        "hypotheses": hyp.data["hypotheses"] if hyp else [],
        "claim_counts": ver.data["counts"] if ver else {},
        "source_files_checked": bool(ver and ver.data.get("source_files_checked")),
        "titles": titles,
        "uncertainties": uncertainties,
        "limitations": LIMITATIONS,
        "human_review": {"status": review_status, "note": "No human has reviewed this report."
                         if review_status == "pending" else f"Review status: {review_status}."},
    }
