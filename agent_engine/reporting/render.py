"""Markdown rendering of a report dict. Pure formatting: no content is added that is not in the report."""
from __future__ import annotations


def _cite(c: dict | None, titles: dict) -> str:
    if not c:
        return ""
    page = f", p.{c['page_number']}" if c.get("page_number") else ""
    return f" [{titles.get(c['document_id'], c['document_id'])}{page}; passage `{c['passage_id']}`]"


def render_markdown(r: dict) -> str:
    t, s = r["titles"], r["scope"]
    out = ["# Research report", "", f"**Question:** {r['question']}", "",
           f"**Status:** {r['status']} | **Human review:** {r['human_review']['status']} - {r['human_review']['note']}", "",
           f"> {r['novelty_statement']}", "", "## Scope of the search", ""]
    if s["corpus"]:
        out.append(f"- Indexed corpus: {s['corpus']['documents']} documents, {s['corpus']['passages']} passages")
    out += [f"- Search terms: {', '.join(s['terms']) or 'n/a'}",
            f"- Accepted evidence: {s['accepted_passages']} passage(s) from {s['accepted_documents']} document(s)"]
    if s["years"]:
        out.append(f"- Publication years of analysed documents: {s['years'][0]}-{s['years'][1]}")
    if s["uncovered_terms"]:
        out.append(f"- Question terms with no accepted evidence: {', '.join(s['uncovered_terms'])}")
    if s["low_quality_matches"]:
        out.append(f"- {s['low_quality_matches']} match(es) on unreliably extracted pages were not counted")
    out += [f"- Not searched: {s['unsearched']}", "", "## Closest documents", ""]
    out += [f"- **{d['title']}** ({d.get('year') or 'year unknown'}; DOI {d.get('doi') or 'n/a'}; licence {d.get('license') or 'n/a'}) "
            f"- {d['statements']} statement(s) extracted from {d['passages_read']} reliable passage(s)" for d in r["closest_documents"]] \
        or ["_None: no accepted documents._"]
    out += ["", "## Verified findings (quoted from the documents)", ""]
    any_f = False
    for kind, label in (("method", "Methods"), ("result", "Results"), ("limitation", "Stated limitations")):
        if r["findings"][kind]:
            any_f = True
            out.append(f"### {label}")
            out += [f"- {f['claim_id']}: \"{f['text']}\"{_cite(f['citation'], t)}" for f in r["findings"][kind]]
            out.append("")
    if not any_f:
        out += ["_No claim passed verification._", ""]
    if r["needs_review"]:
        out += ["## Needs human review", ""]
        out += [f"- {n['claim_id']}: \"{n['text']}\"{_cite(n['citation'], t)} - {'; '.join(n['reasons'])}" for n in r["needs_review"]]
        out.append("")
    out += ["## Unsupported claims", ""]
    out += [f"- {u['claim_id']}: \"{u['text']}\" - {'; '.join(u['reasons'])}" for u in r["unsupported_claims"]] or ["_None._"]
    out += ["", "## Candidate gaps (not findings)", ""]
    out += [f"- {g['id']} ({g['kind']}): {g['statement']}{_cite(g['evidence'][0], t) if g['evidence'] else ''}" for g in r["candidate_gaps"]] \
        or ["_None identified._"]
    out += ["", "## Hypotheses (untested proposals, not findings)", ""]
    for h in r["hypotheses"]:
        out += [f"### {h['id']} [{h['status']}]", h["statement"], "", f"- Assumptions: {' '.join(h['assumptions'])}",
                f"- Falsification criteria: {h['falsification_criteria']}", f"- Derived from gap {h['gap_id']}", f"- _{h['label']}_", ""]
    if not r["hypotheses"]:
        out += ["_None drafted._", ""]
    out += ["## Uncertainty", ""] + [f"- {u}" for u in r["uncertainties"]]
    out += ["", "## Limitations", ""] + [f"- {u}" for u in r["limitations"]]
    return "\n".join(out) + "\n"
