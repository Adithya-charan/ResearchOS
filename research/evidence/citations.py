"""Citation integrity (P11.4): does every citation in a structured report point at something real, and say what the report says it says?

For each citation the check asks, in order: is there a passage id; does the passage exist in the corpus; does it belong to the cited
document; does the cited page match the page the passage came from; does the quoted text occur in the stored passage; has the stored text
changed since the report was made. For each claim block it asks whether any valid citation backs it.

What this does NOT check: whether the passage *supports* the claim. That is the verification status's job (and is lexical); a citation can
be perfectly valid and still not support its claim. The two answers are reported separately on purpose.

`resolve(passage_id)` returns the stored passage (an object with `document_id`, `page_number`, `text`) or None. The check never reads a
network resource, so a citation to anything that is not a stored corpus passage is, by definition, unresolvable."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

SCOPE = ("Checks that each citation resolves to a stored passage of the cited document, that the page matches, and that the quoted text occurs "
         "in that passage. It does not check that the passage supports the claim; that is the verification status, which is lexical and not proof.")

ISSUES = {
    "no_passage_id": "the citation names no passage",
    "passage_missing": "the cited passage does not exist in the corpus (deleted, never stored, or a fabricated identifier)",
    "document_mismatch": "the passage belongs to a different document than the one cited",
    "page_mismatch": "the cited page differs from the page the passage came from",
    "quote_not_in_passage": "the quoted text does not occur in the stored passage",
    "source_changed": "the stored passage text no longer matches the text embedded in the report",
}
WARNINGS = {"page_omitted": "the passage has a page number the citation does not give"}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[‘’“”]", "'", text or "").split()).lower()


def _quote_core(quote: str) -> str:
    q = _norm(quote).strip(" .\"'")
    return q[:-3].strip() if q.endswith("...") else q


def check_citation(c: dict, resolve: Callable[[str], Any]) -> tuple[list[str], list[str]]:
    """(issues, warnings) for one citation dict (`document_id`, `passage_id`, `page_number`, optional `quote`, optional `source_text`)."""
    pid = c.get("passage_id")
    if not pid:
        return ["no_passage_id"], []
    p = resolve(pid)
    if p is None:
        return ["passage_missing"], []
    issues, warnings = [], []
    if c.get("document_id") != p.document_id:
        issues.append("document_mismatch")
    if c.get("page_number") is not None and p.page_number != c.get("page_number"):
        issues.append("page_mismatch")
    elif c.get("page_number") is None and p.page_number is not None:
        warnings.append("page_omitted")
    if c.get("quote") and _quote_core(c["quote"]) and _quote_core(c["quote"]) not in _norm(p.text):
        issues.append("quote_not_in_passage")
    if c.get("source_text") and _norm(c["source_text"]) != _norm(p.text):
        issues.append("source_changed")
    return issues, warnings


def check_report_citations(structured: dict, resolve: Callable[[str], Any]) -> dict:
    """Integrity summary for one structured report (see module docstring)."""
    issues: list[dict] = []
    warnings: list[dict] = []
    total = valid = 0
    unsupported: list[dict] = []
    claim_blocks = claims_backed = 0
    cited_docs: set[str] = set()
    for sec in structured.get("sections", []):
        for i, b in enumerate(sec.get("blocks", [])):
            valid_here = 0
            for c in b.get("citations") or []:
                total += 1
                iss, warn = check_citation(c, resolve)
                where = {"section": sec["number"], "block": i, "claim_key": b.get("claim_key"), "passage_id": c.get("passage_id"),
                         "document_id": c.get("document_id")}
                for x in iss:
                    issues.append({**where, "issue": x, "detail": ISSUES[x]})
                for x in warn:
                    warnings.append({**where, "issue": x, "detail": WARNINGS[x]})
                if not iss:
                    valid += 1
                    valid_here += 1
                    if c.get("document_id"):
                        cited_docs.add(c["document_id"])
            if b.get("kind") == "claim":
                claim_blocks += 1
                if valid_here:
                    claims_backed += 1
                elif b.get("status") in ("SUPPORTED", "PARTIALLY_SUPPORTED") or not b.get("status"):
                    unsupported.append({"section": sec["number"], "block": i, "claim_key": b.get("claim_key"), "status": b.get("status"),
                                        "text": (b.get("text") or "")[:160],
                                        "reason": "claimed as supported (or unlabelled) but no valid citation backs it"})
                elif not (b.get("citations")):
                    pass  # an explicitly UNVERIFIED / INSUFFICIENT / CONTRADICTED claim with no citation is honest, not unsupported
    by_issue = Counter(x["issue"] for x in issues)
    return {
        "citations_checked": total, "citations_valid": valid, "citations_with_issues": total - valid,
        "integrity_rate": round(valid / total, 4) if total else None,
        "documents_cited": len(cited_docs), "claim_blocks": claim_blocks, "claim_blocks_with_a_valid_citation": claims_backed,
        "unsupported_claims": unsupported, "issues": issues, "warnings": warnings, "by_issue": dict(by_issue),
        "ok": not issues and not unsupported, "scope": SCOPE,
    }
