"""P11.4 citation integrity: every way a citation can be wrong is detected, valid ones pass, and unsupported claims are flagged."""
from dataclasses import dataclass

from research.evidence.citations import check_citation, check_report_citations

TEXT = "Capacity fade after twelve months was 19 percent at 45 degrees. Bleaching began once heating weeks passed four."


@dataclass
class P:
    document_id: str
    text: str
    page_number: int | None = None


STORE = {"d1:p0": P("d1", TEXT, 3), "d2:p0": P("d2", "Unrelated text about soil.", None)}


def resolve(pid):
    return STORE.get(pid)


def cite(**kw):
    base = {"document_id": "d1", "passage_id": "d1:p0", "page_number": 3, "quote": "Capacity fade after twelve months was 19 percent"}
    base.update(kw)
    return base


def test_a_correct_citation_has_no_issues():
    assert check_citation(cite(source_text=TEXT), resolve) == ([], [])


def test_each_failure_mode_is_detected_separately():
    assert check_citation(cite(passage_id=None), resolve)[0] == ["no_passage_id"]
    assert check_citation(cite(passage_id="d9:p9"), resolve)[0] == ["passage_missing"]  # fabricated identifier
    assert check_citation(cite(document_id="d2"), resolve)[0] == ["document_mismatch"]
    assert check_citation(cite(page_number=9), resolve)[0] == ["page_mismatch"]
    assert check_citation(cite(quote="Capacity fade was 90 percent"), resolve)[0] == ["quote_not_in_passage"]
    assert check_citation(cite(source_text="An older version of the passage"), resolve)[0] == ["source_changed"]


def test_quote_matching_ignores_case_whitespace_quotes_and_ellipsis():
    assert check_citation(cite(quote="  capacity  FADE after twelve months was 19 percent..."), resolve)[0] == []
    assert check_citation(cite(quote=""), resolve)[0] == []  # no quote claimed, nothing to contradict


def test_an_omitted_page_is_a_warning_not_an_error():
    assert check_citation(cite(page_number=None), resolve) == ([], ["page_omitted"])


def block(kind="claim", status="SUPPORTED", citations=(), key="c1"):
    return {"kind": kind, "text": "A claim", "claim_key": key, "status": status, "citations": list(citations)}


def report(*blocks):
    return {"sections": [{"number": 1, "blocks": list(blocks)}]}


def test_report_summary_counts_valid_and_invalid_and_flags_unsupported_claims():
    r = report(block(citations=[cite()], key="ok"),
               block(citations=[cite(passage_id="zzz")], key="fabricated"),
               block(citations=[], key="uncited"),
               block(status="UNVERIFIED", citations=[], key="honest"),
               block(kind="text", status=None, citations=[], key="prose"))
    s = check_report_citations(r, resolve)
    assert (s["citations_checked"], s["citations_valid"], s["citations_with_issues"], s["integrity_rate"]) == (2, 1, 1, 0.5)
    assert s["claim_blocks"] == 4 and s["claim_blocks_with_a_valid_citation"] == 1
    assert {u["claim_key"] for u in s["unsupported_claims"]} == {"fabricated", "uncited"}  # the UNVERIFIED claim is honest, not unsupported
    assert s["by_issue"] == {"passage_missing": 1} and s["ok"] is False and "does not check" in s["scope"]


def test_a_clean_report_is_ok_and_an_empty_one_has_no_rate():
    assert check_report_citations(report(block(citations=[cite()])), resolve)["ok"] is True
    e = check_report_citations({"sections": []}, resolve)
    assert e["integrity_rate"] is None and e["ok"] is True


def test_with_no_corpus_everything_is_unresolvable_not_silently_valid():
    s = check_report_citations(report(block(citations=[cite()])), lambda pid: None)
    assert s["citations_valid"] == 0 and s["by_issue"] == {"passage_missing": 1}
