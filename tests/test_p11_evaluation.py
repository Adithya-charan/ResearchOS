"""P11.7: the evaluation suite is deterministic and its recorded numbers are pinned, so a change that makes a feature better or worse shows up
as a failing test rather than a silent drift. The numbers describe an invented corpus (see research/p11_evaluation.LIMITATIONS)."""
from research.p11_evaluation import LIMITATIONS, citation_eval, evidence_eval, gap_eval, run_offline
from research.retrieval_benchmark import load


def test_the_offline_suite_is_deterministic():
    a, b = run_offline(), run_offline()
    assert a == b


def test_evidence_results_are_pinned_and_the_misses_are_the_documented_ones():
    idx, *_ = load()
    e = evidence_eval(idx)
    assert e["cases"] == 26 and e["status_accuracy"] == 0.6538
    assert e["by_kind"]["exact"]["accuracy"] == 1.0 and e["by_kind"]["paraphrase"]["accuracy"] == 0.0  # lexical: paraphrases are not connected
    assert {w["id"] for w in e["wrong"]} == {"p01", "p02", "p03", "p04", "p05", "c04", "c06", "i04", "m02"}
    assert e["contradiction_detection"]["precision"] == 1.0 and e["contradiction_detection"]["recall"] == 0.6667
    assert e["evidence_support_precision"] == 0.7778 and e["unsupported_claim_rate"] == 0.2222
    assert e["conflicting_sources_surfaced"] == 0.5  # m01 (additive: two papers disagree) is surfaced as contested; m02 (a claim split over two papers) is not


def test_the_two_false_supports_are_in_the_open():
    """c04 ('highest on sand') and c06 ('decreased as ... grew') are labelled SUPPORTED by the lexical verifier although the passages say the
    opposite. They stay in the results as the unsupported-claim rate; this test fails if someone hides them by dropping the cases."""
    idx, *_ = load()
    wrong = {w["id"]: w for w in evidence_eval(idx)["wrong"]}
    assert wrong["c04"]["predicted"] == "SUPPORTED" and wrong["c06"]["predicted"] == "SUPPORTED"
    assert evidence_eval(idx)["unsupported_claim_rate"] > 0


def test_every_injected_citation_defect_is_found_and_valid_citations_pass():
    idx, *_ = load()
    c = citation_eval(idx)
    assert c["citation_correctness"] == {"valid_citations_accepted": 1.0, "defective_citations_flagged": 1.0, "lowest_defect_detection_rate": 1.0}
    assert all(v["flagged_only_that_issue"] == 1.0 for k, v in c["by_case"].items() if k != "valid")  # each defect raises exactly its own issue


def test_gap_results_are_pinned():
    idx, titles, *_ = load()
    g = gap_eval(idx, titles)
    assert g["conflict_detection"] == {"expected_pairs": 2, "reported_pairs": 2, "precision": 1.0, "recall": 1.0}
    assert g["stated_limitation_recall"] == 1.0 and g["traceability_problems"] == [] and g["every_gap_linked_to_a_passage"] and g["hypotheses_labelled"]


def test_the_suite_states_that_it_is_synthetic():
    r = run_offline()
    assert r["synthetic"] is True and r["limitations"] == LIMITATIONS and any("INVENTED" in x for x in LIMITATIONS)
    assert r["retrieval"]["recommended_opt_in"]["duplicate_pairs_in_top5"] == 0 < r["retrieval"]["baseline_bm25"]["duplicate_pairs_in_top5"]
