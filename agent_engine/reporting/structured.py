"""The 18-section research report. Assembled only from persisted agent outputs: every block is a verbatim quotation with its citation, a
template filled from run data, or a fixed statement of limits. There is no free generation, so nothing can be invented.

Every block that states something about the literature either carries at least one citation (document, page where the format has
pages, passage) or says in `status` that it is UNVERIFIED. Proposals (gaps, hypotheses, plans) are labelled as untested proposals."""
from __future__ import annotations

from agent_engine.agents.evidence import merged_verification
from agent_engine.contracts import AgentOutput
from research.evidence.types import STATUSES

SECTION_TITLES = ("Title", "Abstract", "Research question", "Background", "Literature review", "Research gap", "Hypothesis",
                  "Proposed methodology", "Experiment design", "Findings", "Evidence and citations", "Contradicting evidence",
                  "Verification results", "Limitations", "Critical review", "Conclusion", "Future work", "References")
STATIC_LIMITS = [
    "Everything comes from the indexed documents only; the report cannot speak for literature outside them.",
    "Verification is a lexical comparison of claims with stored passages. SUPPORTED means a stored passage states the claim in nearly "
    "the same words; it is not a proof and does not show the source is correct. No claim was checked by an experiment.",
    "Gaps and hypotheses are automatically drafted, untested proposals and are never labelled verified.",
    "Novelty labels apply to the searched corpus only; they are not evidence that an idea is new.",
    "Possible contradictions are found by shared words and opposite direction words and may be false alarms.",
]


def _cite(src: dict | None, quote: str = "") -> dict:
    src = src or {}
    return {"document_id": src.get("document_id"), "page_number": src.get("page_number"), "passage_id": src.get("passage_id"),
            **({"quote": quote} if quote else {})}


#: what a block is, in words a reader can act on (P11.5). Derived from `kind`, never from the text.
EPISTEMIC = {
    "claim": "evidence",         # a statement taken from a stored passage; carries a verification status and a citation
    "derived": "fact",           # computed from run data (counts, metadata, labels); not a claim about the world
    "proposal": "inference",     # a gap, hypothesis or plan drafted by the system; untested, never verified
    "note": "absence",           # nothing to report / not available; states what is missing
}
EPISTEMIC_MEANING = {
    "evidence": "quoted from a stored passage with a verification status (lexical, not proof)",
    "fact": "computed from the run's own data (counts, metadata, labels)",
    "inference": "drafted by the system and untested; never presented as a finding",
    "absence": "records that something was not found or not produced",
}


def _block(text: str, *, kind: str = "note", citations: list[dict] | None = None, claim_key: str | None = None,
           status: str | None = None, **extra) -> dict:
    return {"kind": kind, "text": text, "citations": citations or [], "claim_key": claim_key, "status": status,
            "epistemic": EPISTEMIC.get(kind, "absence"), **extra}


def quality_summary(sections: list[dict], counts: dict, unresolved: list[dict]) -> dict:
    """Evidence coverage and the mix of statement types (P11.5). Pure arithmetic over the report's own blocks; it adds no new claims."""
    mix = dict.fromkeys(EPISTEMIC_MEANING, 0)
    cited_docs: set[str] = set()
    cited_blocks = 0
    for sec in sections:
        for b in sec["blocks"]:
            mix[b.get("epistemic") or EPISTEMIC.get(b["kind"], "absence")] += 1
            cs = [c for c in b["citations"] if c.get("passage_id")]
            cited_blocks += bool(cs)
            cited_docs.update(c["document_id"] for c in cs if c.get("document_id"))
    total = sum(counts.get(s, 0) for s in STATUSES)
    backed = counts.get("SUPPORTED", 0) + counts.get("PARTIALLY_SUPPORTED", 0)
    by_title = {s["title"]: s["blocks"] for s in sections}
    real = lambda blocks: sum(b["kind"] != "note" for b in blocks)  # noqa: E731 - "nothing found" notes are not items
    return {
        "statement_mix": mix, "statement_meaning": EPISTEMIC_MEANING,
        "evidence_coverage": {"claims": total, "by_status": {s: counts.get(s, 0) for s in STATUSES}, "supported_or_partial": backed,
                              "fraction_supported_or_partial": round(backed / total, 4) if total else None,
                              "claims_not_fully_supported": len(unresolved), "documents_cited": len(cited_docs), "blocks_with_citation": cited_blocks},
        "contradictions": counts.get("CONTRADICTED", 0) + real(by_title.get("Contradicting evidence", [])),
        "limitations_listed": real(by_title.get("Limitations", [])),
        "recommendations_listed": real(by_title.get("Future work", [])),
        "assumptions": "The system records no explicit assumptions; where a step depends on one (for example that two phrasings name the same "
                       "thing) it appears as an inference or in the limitations, not as a finding.",
        "reading_note": "Coverage counts claims the agents extracted from the documents. It is not a measure of how much of the literature "
                        "was read, and a high supported fraction says only that stored passages use similar words.",
    }


def build_structured(report: dict, outputs: list[AgentOutput]) -> dict:
    by: dict[str, AgentOutput] = {}
    for o in outputs:
        by[o.agent_type] = o  # latest of each type (verifier passes are merged below)
    merged = merged_verification(outputs)
    claims = {c["key"]: c for c in merged["claims"]}
    titles = report.get("titles", {})
    question = report["question"]
    sc = report["scope"]
    lit, gap, hyp, plans_out, review = (by.get(k) for k in ("literature", "gap", "hypothesis", "experiment_design", "reviewer"))
    counts = merged["counts"]
    cited: dict[str, dict] = {}  # document_id -> passages cited anywhere in the report

    def doc_title(d: str | None) -> str:
        return titles.get(d, d or "unknown")

    def cites_of(key: str) -> list[dict]:
        c = claims.get(key)
        if not c:
            return []
        out = [_cite(e, e["quote"]) for e in c["evidence"] if e["role"] == "supporting" and e["citation_valid"]][:3]
        for x in out:
            cited.setdefault(x["document_id"], {}).setdefault(x["passage_id"], x)
        return out

    sections: list[dict] = []

    def section(i: int, blocks: list[dict]) -> None:
        sections.append({"number": i + 1, "id": SECTION_TITLES[i].lower().replace(" ", "_"), "title": SECTION_TITLES[i], "blocks": blocks})

    section(0, [_block(f"Research report: {question[:140]}", kind="derived")])

    n_claims = sum(counts.values())
    section(1, [_block(
        f"This report addresses the question above using {sc['corpus']['documents'] if sc['corpus'] else 'an unknown number of'} "
        f"indexed document(s). It accepted {sc['accepted_passages']} passage(s) from {sc['accepted_documents']} document(s), checked "
        f"{n_claims} claim(s) ({', '.join(f'{counts[s]} {s}' for s in STATUSES if counts.get(s)) or 'none'}) over "
        f"{merged['rounds']} verification round(s), and lists {len(report['candidate_gaps'])} candidate gap(s) and "
        f"{len(report['hypotheses'])} hypothesis proposal(s), all untested. No experiment was run.", kind="derived")])
    section(2, [_block(question, kind="derived")])

    bg = [_block(f"Search terms: {', '.join(sc['terms']) or 'n/a'}. Documents analysed: "
                 + ("; ".join(f"{d['title']} ({d.get('year') or 'year unknown'})" for d in report["closest_documents"]) or "none")
                 + ".", kind="derived")]
    if sc["uncovered_terms"]:
        bg.append(_block(f"No accepted document addresses: {', '.join(sc['uncovered_terms'])}.", kind="derived"))
    section(3, bg)

    litrev: list[dict] = []
    for kind, label in (("method", "Method"), ("result", "Result"), ("limitation", "Stated limitation")):
        for f in report["findings"][kind]:
            key = f"literature:{f['claim_id']}"
            c = claims.get(key)
            litrev.append(_block(f"{label}: \"{f['text']}\" ({doc_title(f['citation']['document_id'])})", kind="claim",
                                 citations=[_cite(f["citation"], f["text"])], claim_key=key,
                                 status=c["status"] if c else "UNVERIFIED"))
            cited.setdefault(f["citation"]["document_id"], {}).setdefault(f["citation"]["passage_id"], _cite(f["citation"], f["text"]))
    section(4, litrev or [_block("No claim passed the evidence checks, so there is no literature review to show.", kind="note",
                                  status="UNVERIFIED")])

    nov = {n["claim_key"]: n for n in merged["novelty"]}

    def novelty_block(key: str) -> list[dict]:
        n = nov.get(key)
        if not n:
            return []
        near = ", ".join(d.get("title") or d["document_id"] for d in n["closest_documents"][:3]) or "none"
        return [_block(f"Novelty (searched corpus only): {n['label']}. Closest documents: {near}. {n['disclaimer']}", kind="derived",
                       claim_key=key, novelty=n)]

    gaps_out: list[dict] = []
    for g in report["candidate_gaps"]:
        key = f"gap:{g['id']}"
        gaps_out.append(_block(f"{g['statement']} (candidate gap, not a finding)", kind="proposal", claim_key=key,
                               status=claims.get(key, {}).get("status", "UNVERIFIED"),
                               citations=[_cite(e) for e in g["evidence"]]))
        gaps_out += novelty_block(key)
    section(5, gaps_out or [_block("No gap was derived from the accepted evidence.", kind="note", status="UNVERIFIED")])

    hyps: list[dict] = []
    for h in report["hypotheses"]:
        key = f"hypothesis:{h['id']}"
        hyps.append(_block(f"{h['statement']} Falsification: {h['falsification_criteria']}", kind="proposal", claim_key=key,
                           status=claims.get(key, {}).get("status", "UNVERIFIED"), citations=[_cite(e) for e in h["evidence"]],
                           note=h["label"]))
        hyps += novelty_block(key)
    section(6, hyps or [_block("No hypothesis was drafted.", kind="note", status="UNVERIFIED")])

    plans = plans_out.data["plans"] if plans_out else []
    section(7, [_block(f"Plan {p['id']} for {p['hypothesis_id']}: independent variable - {p['independent_variable']['name']}; dependent "
                       f"variable - {p['dependent_variable']['name']}. At least {p['reproducibility']['repeats_min']} repeats with recorded "
                       "seeds. " + ("Needs human input: " + ", ".join(p["needs_human_input"]) + "." if p["needs_human_input"] else ""),
                       kind="proposal", status="UNVERIFIED") for p in plans]
            or [_block("No experiment plan was produced.", kind="note", status="UNVERIFIED")])
    exp: list[dict] = []
    for p in plans:
        for label, items in (("Dataset", p["datasets"]), ("Baseline", p["baselines"]), ("Metric", p["metrics"])):
            for it in items:
                exp.append(_block(f"{label} named in the literature ({p['id']}): \"{it['quote']}\"", kind="claim",
                                  citations=[_cite(it["citation"], it["quote"])], status="SUPPORTED"
                                  if it.get("quote") else "UNVERIFIED"))
                cited.setdefault(it["citation"]["document_id"], {}).setdefault(it["citation"]["passage_id"], _cite(it["citation"], it["quote"]))
    section(8, exp or [_block("The accepted documents name no dataset, baseline or metric; the plans leave them to a human.",
                              kind="note", status="UNVERIFIED")])

    findings = [_block(f"{c['text']}", kind="claim", claim_key=c["key"], status=c["status"],
                       citations=cites_of(c["key"]), strength=c["strength"]["level"])
                for c in merged["claims"] if c["claim_type"] in ("result", "method", "limitation")
                and c["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")]
    section(9, findings or [_block("No claim reached SUPPORTED or PARTIALLY_SUPPORTED.", kind="note", status="UNVERIFIED")])

    ev = [_block(f"{doc_title(d)}: passage `{p}`" + (f", page {c['page_number']}" if c.get("page_number") else " (no page number: the "
                "source format has no pages)"), kind="derived", citations=[c])
          for d, ps in sorted(cited.items()) for p, c in sorted(ps.items())]
    section(10, ev or [_block("No passage is cited in this report.", kind="note")])

    contra: list[dict] = []
    for c in merged["claims"]:
        for e in c["evidence"]:
            if e["role"] == "contradicting":
                contra.append(_block(f"Claim \"{c['text']}\" is challenged by \"{e['quote']}\" ({doc_title(e['document_id'])}): {e['reason']}.",
                                     kind="claim", claim_key=c["key"], status=c["status"], citations=[_cite(e, e["quote"])],
                                     directness=e["directness"]))
    for t in report.get("tensions", []):
        contra.append(_block(f"Possible tension between {t['a']} and {t['b']} (shared terms: {', '.join(t['shared_terms'])}); "
                             "flagged lexically, may not be a real contradiction.", kind="derived", status="UNVERIFIED"))
    section(11, contra or [_block("No contradicting passage was found by the checks that were run (this does not mean none exists).",
                                  kind="note")])

    ver: list[dict] = [_block(f"{s}: {counts.get(s, 0)} claim(s)", kind="derived") for s in STATUSES]
    for c in merged["claims"]:
        ver.append(_block(f"{c['key']} - {c['status']} (evidence strength: {c['strength']['level']}). " + " ".join(c["explanation"][:2]),
                          kind="derived", claim_key=c["key"], status=c["status"]))
    for h in merged["history"]:
        ver.append(_block(f"Verification round {h['round']}: {h['claims_checked']} claim(s) checked; {h['needs_followup']} still needed "
                          "more evidence.", kind="derived"))
    for o in outputs:
        er = o.data.get("evidence_review") if o.agent_type == "supervisor" else None
        if er:
            ver.append(_block(f"Supervisor evidence review (round {er['round']} of at most {er['budget']}): {er['action']}.", kind="derived"))
    section(12, ver)

    section(13, [_block(x, kind="derived") for x in dict.fromkeys([*STATIC_LIMITS, *report["limitations"], *report["uncertainties"][:10]])])

    rv = review.data["review"] if review else None
    section(14, [_block(f"[{f['severity']}] {f['message']}", kind="derived", status=None) for f in (rv["findings"] if rv else [])
                 if f["severity"] != "minor"] + ([_block(rv["disclaimer"], kind="derived")] if rv else [])
            or [_block("No critical review was available.", kind="note")])

    unresolved = merged["needs_followup"]
    section(15, [_block(
        f"The accepted evidence yields {counts.get('SUPPORTED', 0)} supported claim(s), {counts.get('PARTIALLY_SUPPORTED', 0)} partially "
        f"supported, {counts.get('CONTRADICTED', 0)} contradicted, {counts.get('INSUFFICIENT_EVIDENCE', 0)} with insufficient evidence and "
        f"{counts.get('UNVERIFIED', 0)} unverified. Report status: {report['status']}. This is a record of what the searched documents "
        "state, not a conclusion about the world.", kind="derived")]
        + ([_block(f"{len(unresolved)} claim(s) remain not fully supported.", kind="derived")] if unresolved else []))
    fw = [_block(f"Resolve claim {u['key']} ({u['status']}): find or add evidence, or retire the claim.", kind="derived") for u in unresolved]
    fw += [_block(f"Test hypothesis {p['hypothesis_id']} with plan {p['id']} through the approval-gated sandbox.", kind="proposal",
                  status="UNVERIFIED") for p in plans]
    section(16, fw or [_block("No follow-up was identified.", kind="note")])

    refs = []
    for d, ps in sorted(cited.items()):
        meta = next((x for x in report["closest_documents"] if x["document_id"] == d), {})
        refs.append(_block(f"{doc_title(d)} ({meta.get('year') or 'year unknown'}); DOI {meta.get('doi') or 'n/a'}; licence "
                           f"{meta.get('license') or 'n/a'}; document `{d}`; cited passages: {', '.join(sorted(ps))}",
                           kind="derived", citations=list(ps.values())))
    section(17, refs or [_block("No document is cited.", kind="note")])

    return {"title": sections[0]["blocks"][0]["text"], "sections": sections, "claim_counts": counts, "verification_rounds": merged["rounds"],
            "unresolved_claims": unresolved, "quality": quality_summary(sections, counts, unresolved),
            "rules": "Every block is a quotation with its citation, a template filled from run data, or a labelled proposal."}


def render_structured_markdown(s: dict) -> str:
    out = [f"# {s['title']}", ""]
    q = s.get("quality")
    if q:
        cov = q["evidence_coverage"]
        out += [f"> Evidence coverage: {cov['supported_or_partial']} of {cov['claims']} claim(s) supported or partially supported, "
                f"{cov['claims_not_fully_supported']} not fully supported, {q['contradictions']} contradiction(s), "
                f"{cov['documents_cited']} document(s) cited. Statements: "
                + ", ".join(f"{n} {k}" for k, n in q["statement_mix"].items()) + ". " + q["reading_note"], ""]
    for sec in s["sections"][1:]:
        out += [f"## {sec['number']}. {sec['title']}", ""]
        for b in sec["blocks"]:
            tag = f" [{b['status']}]" if b.get("status") else ""
            cites = "".join(f" [{c['document_id']}{', p.' + str(c['page_number']) if c.get('page_number') else ''}; passage `{c['passage_id']}`]"
                            for c in b["citations"][:2] if c.get("passage_id"))
            out.append(f"- {b['text']}{tag}{cites}")
        out.append("")
    return "\n".join(out)
