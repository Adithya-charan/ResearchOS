"""P11.2 evidence quality: unit-aware figure conflicts, swapped comparisons, traceable evidence records, irrelevant-passage reporting, and
the false positives and false negatives these rules could introduce. All lexical: a status is an evidence-strength label, not proof."""
import pytest

from research.evidence import Candidate, verify_claim

DOC = ("Capacity fade after twelve months was 19 percent at 45 degrees. Bleaching began once accumulated heating weeks passed four. "
       "Branching colonies bleached earlier than massive colonies.")


def cand(text, pid="d1:p000", doc="d1", **kw):
    return Candidate(passage_id=pid, document_id=doc, text=text, resolved=True, **kw)


def status(claim, text, **kw):
    return verify_claim(claim, "result", [cand(text, **kw)], searched=True)


# ---- figures are compared per unit -----------------------------------------------------------------------------------------------------
def test_a_different_figure_for_the_same_unit_contradicts_even_when_another_number_recurs():
    r = status("Capacity fade after twelve months was 40 percent at 45 degrees", DOC)
    assert r["status"] == "CONTRADICTED"
    why = r["evidence"][0]["reason"]
    assert "19" in why and "40 percent" in why and "same unit" in why  # the explanation names both figures


def test_the_stated_figure_is_supported_and_figures_in_other_units_do_not_conflict():
    assert status("Capacity fade after twelve months was 19 percent at 45 degrees", DOC)["status"] == "SUPPORTED"
    r = status("Capacity fade after twelve months was 19 percent at 25 degrees", DOC)  # 45 degrees is stated, 25 degrees is not: a conflict on degrees
    assert r["status"] == "CONTRADICTED"
    assert status("Capacity fade was 19 percent", "Capacity fade was 19 percent after 12 months at room temperature.")["status"] == "SUPPORTED"


def test_a_figure_with_an_unrecognised_unit_is_not_treated_as_a_conflict():
    r = status("The trial enrolled 30 cells", "The trial enrolled 50 cells in total.")  # 'cells' is not a recognised unit: stays on the old rule
    assert r["status"] in ("CONTRADICTED", "PARTIALLY_SUPPORTED", "SUPPORTED")  # never crashes, never invents a unit conflict
    assert "same unit" not in " ".join(e["reason"] for e in r["evidence"])


# ---- swapped comparisons ---------------------------------------------------------------------------------------------------------------
def test_a_comparison_with_the_roles_swapped_is_contradicted():
    r = status("Massive colonies bleached earlier than branching colonies", DOC)
    assert r["status"] == "CONTRADICTED" and "roles swapped" in r["evidence"][0]["reason"]


def test_the_comparison_in_the_documents_order_is_supported():
    assert status("Branching colonies bleached earlier than massive colonies", DOC)["status"] == "SUPPORTED"


def test_opposite_comparatives_that_say_the_same_thing_are_not_called_contradictions():
    text = "Slow charging cells lasted longer than fast charging cells in every test."
    # 'A longer than B' vs the reversed pair with the opposite comparative ('B shorter than A') is one statement, so it must not be CONTRADICTED
    r = status("Fast charging cells lasted shorter than slow charging cells", text)
    assert r["status"] != "CONTRADICTED"


# ---- null result against a reported change ----------------------------------------------------------------------------------------------
NULL = "The additive gave no significant difference in capacity retention after three hundred fast charge cycles."
CHANGE = "The additive improved capacity retention after three hundred fast charge cycles."


def test_a_null_result_and_a_reported_change_on_the_same_subject_contradict_each_other():
    r = status(NULL, CHANGE)  # this used to come back SUPPORTED: neither sentence had a matching negation word
    assert r["status"] == "CONTRADICTED" and "no effect or difference" in r["evidence"][0]["reason"]
    assert status(CHANGE, NULL)["evidence"][0]["role"] == "contradicting"  # the other way round is at least flagged (indirect: not a verdict)


def test_two_null_results_or_two_changes_still_support_each_other_and_a_negated_change_is_not_a_change():
    assert status(NULL, NULL)["status"] == "SUPPORTED"
    assert status(CHANGE, CHANGE)["status"] == "SUPPORTED"
    # "did not improve" is itself a null result, so it must not be read as a change that contradicts a null claim
    assert status(NULL, "The additive did not improve capacity retention after three hundred fast charge cycles.")["status"] != "CONTRADICTED"


# ---- traceable evidence records ---------------------------------------------------------------------------------------------------------
def test_every_evidence_record_keeps_its_source_original_text_and_relevance():
    full = "Abstract text here. " + DOC
    r = verify_claim("Branching colonies bleached earlier than massive colonies", "result",
                     [cand(full, pid="doc-9:p004", doc="doc-9", page_number=7, section="2. Results", retrieval_method="bm25_search",
                           retrieval_score=3.2)], searched=True)
    e = r["evidence"][0]
    assert e["source_location"] == {"document_id": "doc-9", "passage_id": "doc-9:p004", "page_number": 7, "section": "2. Results"}
    assert e["passage_text"] == " ".join(full.split())  # the original passage, not just the quoted sentence
    assert e["quote"] in e["passage_text"] and 0 < e["relevance_score"] <= 1 and e["relevance_score"] == e["coverage"]
    assert e["role"] and e["reason"] and e["citation_valid"] is True and e["retrieval_score"] == 3.2


def test_irrelevant_passages_are_recorded_by_id_not_silently_dropped():
    r = verify_claim("Capacity fade after twelve months was 19 percent at 45 degrees", "result",
                     [cand(DOC, pid="a:p0", doc="a"), cand("The committee met on Tuesday to discuss the budget.", pid="b:p3", doc="b")],
                     searched=True)
    assert [e["passage_id"] for e in r["evidence"]] == ["a:p0"]
    irr = r["irrelevant_passages"]
    assert [x["passage_id"] for x in irr] == ["b:p3"] and irr[0]["role"] == "irrelevant" and irr[0]["relevance_score"] < 0.25
    assert irr[0]["passage_text"].startswith("The committee") and r["checks"]["dropped_irrelevant"] == 1


def test_unresolvable_and_unreliable_citations_are_kept_separate_from_evidence():
    r = verify_claim("Capacity fade after twelve months was 19 percent", "result",
                     [Candidate("x:p1", "x", text="", resolved=False), cand(DOC, quality="low", pid="y:p1", doc="y")], searched=True)
    assert r["evidence"] == [] and r["status"] == "UNVERIFIED"
    assert r["invalid_citations"][0]["passage_id"] == "x:p1" and r["unreliable_passages"][0]["passage_id"] == "y:p1"


# ---- false positives and false negatives (the limits stay visible) -----------------------------------------------------------------------
def test_conflicting_sources_are_both_kept_and_reported_as_conflicting():
    r = verify_claim("The additive improved capacity retention after three hundred fast charge cycles", "result",
                     [cand("The additive improved capacity retention after three hundred fast charge cycles.", pid="a:p0", doc="a"),
                      cand("The additive did not improve capacity retention after three hundred fast charge cycles.", pid="b:p0", doc="b")],
                     searched=True)
    assert r["status"] == "PARTIALLY_SUPPORTED" and r["conflicting"] is True
    assert {e["role"] for e in r["evidence"]} == {"supporting", "contradicting"}


def test_a_paraphrase_without_shared_vocabulary_is_a_documented_false_negative():
    r = verify_claim("Heat stress made reef coral lose its colour sooner when branched", "result",
                     [cand("Bleaching began once accumulated heating weeks passed four. Branching colonies bleached earlier than massive colonies.")],
                     searched=True)
    assert r["status"] != "SUPPORTED"  # lexical matching cannot connect them; this limit is stated in the report scope note
    assert "lexical" in r["scope_note"].lower() and "not proofs" in r["scope_note"].lower()


def test_negation_about_another_subject_is_not_a_contradiction():
    r = status("Cells with the additive kept 91 percent of their capacity", "Cells without the additive did not keep their capacity as well. "
                                                                           "Cells with the additive kept 91 percent of their capacity after 300 cycles.")
    assert r["status"] == "SUPPORTED"


@pytest.mark.parametrize("claim", ["The treatment worked", "Yes", ""])
def test_claims_with_too_few_content_words_are_unverified_not_guessed(claim):
    assert status(claim, DOC)["status"] == "UNVERIFIED"


def test_the_labelled_set_stays_at_the_recorded_accuracy(tmp_path):
    from pathlib import Path  # noqa: PLC0415

    from research.verification_eval import run_verification_eval  # noqa: PLC0415

    rep = run_verification_eval(Path(__file__).resolve().parents[1] / "datasets" / "verification_eval", tmp_path)["synthetic"]
    assert rep["accuracy_excluding_known_limitations"]["accuracy"] == 1.0
    known = rep["accuracy_on_known_limitations"]
    assert (known["correct"], known["claims"]) == (2, 3)  # figures per unit and swapped roles are fixed; the paraphrase case remains wrong
    wrong = [r["id"] for r in rep["claims"] if not r["correct"]]
    assert wrong == ["c17"]
