"""P11.3 corpus-scoped gap analysis: every gap links to stored passages, observed and hypothesised gaps are told apart, nothing claims novelty,
and degenerate libraries produce an honest empty answer."""
import pytest

from research.bm25.index import BM25Index, passages_from_text
from research.gaps import analyze_corpus_gaps, validate_gap_report
from research.retrieval_benchmark import load


@pytest.fixture(scope="module")
def world():
    idx, titles, _, _ = load()
    return idx, titles, analyze_corpus_gaps(idx, titles)


def of(report, kind):
    return [g for g in report["gaps"] if g["type"] == kind]


def test_every_gap_is_traceable_and_none_claims_novelty(world):
    idx, _, rep = world
    assert rep["gaps"] and validate_gap_report(rep, idx) == []
    assert all(g["basis"] in ("observed", "hypothesis") and g["evidence"] and g["documents"] for g in rep["gaps"])
    assert "outside the library" in rep["scope"] and "none says an idea is new" in rep["scope"]


def test_stated_limitations_are_quoted_from_their_document(world):
    idx, _, rep = world
    lim = [g for g in rep["gaps"] if g["type"] in ("methodological_limitation", "unanswered_question")]
    assert len(lim) >= 10
    by_doc = {g["documents"][0] for g in lim}
    assert {"bat1", "bat2", "coral2", "fl2"} <= by_doc
    g = next(g for g in lim if g["documents"] == ["bat2"])
    assert g["type"] == "unanswered_question" and "Coating thickness was not varied" in g["statement"] and g["basis"] == "observed"
    assert g["evidence"][0]["quote"] in " ".join(idx.get(g["evidence"][0]["passage_id"]).text.split())


def test_opposing_findings_are_reported_with_both_passages_and_a_caveat(world):
    idx, _, rep = world
    pairs = {frozenset(g["documents"]) for g in of(rep, "conflicting_findings")}
    assert frozenset({"soil1", "soil2"}) in pairs  # increased vs decreased maize yield
    assert frozenset({"bat4", "bat5"}) in pairs  # 'improved retention' vs 'no significant difference'
    for g in of(rep, "conflicting_findings"):
        assert len(g["evidence"]) == 2 and g["evidence"][0]["document_id"] != g["evidence"][1]["document_id"]
        assert "false alarm" in g["caveat"] and g["verifier_reason"]


def test_hypotheses_are_labelled_and_say_they_are_not_knowledge_about_the_literature(world):
    _, _, rep = world
    hyp = [g for g in rep["gaps"] if g["basis"] == "hypothesis"]
    assert hyp and all(g["type"] == "missing_comparison" and "not known from this library" in g["statement"] and g["caveat"] for g in hyp)
    assert rep["follow_ups"] and all(f["basis"] == "hypothesis" for f in rep["follow_ups"])
    ids = {g["id"] for g in rep["gaps"]}
    assert all(f["for_gap"] in ids for f in rep["follow_ups"])  # each follow-up names the gap that motivates it


def test_unrelated_fields_do_not_produce_cross_domain_missing_comparisons(world):
    _, titles, rep = world
    domain = {d: d.rstrip("0123456789") for d in titles}
    domain["dup1"] = "bat"
    for g in of(rep, "missing_comparison"):
        assert len({domain[d] for d in g["documents"]}) == 1, g["statement"]


def test_themes_are_title_words_and_generic_words_are_excluded(world):
    _, _, rep = world
    freq = {t["theme"] for t in rep["themes"]["frequent"]}
    assert {"silicon", "capacity", "soil"} <= freq and not ({"after", "percent", "lower", "cycles"} & freq)
    assert all(t["document_count"] >= 2 for t in rep["themes"]["frequent"])


def test_the_report_states_its_limits(world):
    _, _, rep = world
    assert any("synonyms" in x for x in rep["limitations"]) and any("false alarms" in x for x in rep["limitations"])
    assert rep["counts"]["observed"] + rep["counts"]["hypothesis"] == len(rep["gaps"])


def test_an_empty_or_tiny_library_gives_an_honest_answer_not_invented_gaps():
    empty = analyze_corpus_gaps(BM25Index(), {})
    assert empty["gaps"] == [] and "nothing to analyse" in empty["note"]
    idx = BM25Index()
    idx.add_passages(passages_from_text("only", "Abstract\n\nGraphene membranes filter salt water efficiently.\n\n1. Limitations\n\nOnly one membrane was tested."))
    one = analyze_corpus_gaps(idx, {"only": "Graphene membranes"})
    assert [g["type"] for g in one["gaps"]] == ["unanswered_question"] and validate_gap_report(one, idx) == []  # no theme/conflict claims from one document
    assert one["themes"]["frequent"] == []


def test_the_validator_catches_untraceable_gaps_and_novelty_claims(world):
    idx, _, rep = world
    bad = {"gaps": [{"id": "x1", "type": "missing_comparison", "basis": "hypothesis", "statement": "Nobody has compared A and B; this is a novel idea.",
                     "evidence": []},
                    {"id": "x2", "type": "methodological_limitation", "basis": "observed", "statement": "s",
                     "evidence": [{"document_id": "bat1", "passage_id": "zzz:p9", "quote": "q"}]},
                    {"id": "x3", "type": "methodological_limitation", "basis": "guess", "statement": "s",
                     "evidence": [{"document_id": "bat1", "passage_id": rep["gaps"][0]["evidence"][0]["passage_id"], "quote": "q"}]}]}
    problems = " | ".join(validate_gap_report(bad, idx))
    assert "no linked passage" in problems and "claims novelty" in problems and "does not resolve" in problems and "unknown basis" in problems
