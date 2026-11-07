"""P6: claims, evidence matching, verification statuses, novelty assessment, the 18-section report, agent integration, supervisor
follow-up with bounded retries, the claim/novelty/report APIs and project authorization."""
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from agent_engine.contracts import AgentOutput, RunLimits, RunStatus
from agent_engine.registry.defaults import build_engine
from agent_engine.reporting import build_report, render_structured_markdown
from agent_engine.reporting.structured import SECTION_TITLES
from agent_engine.state.store import Store
from api import migrate
from research.corpus import Corpus
from research.evidence import STATUSES, Candidate, NoveltyLabel, assess_novelty, verify_claim
from research.verification_eval import run_verification_eval
from tests.conftest_agents import run
from tests.test_api_corpus import make_app, new_project, sign_in, upload, wait_done

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "verification_eval"
DECLARED = json.loads((DATASET / "corpus.json").read_text(encoding="utf-8"))


def cand(text, pid="d1:p1", doc="d1", **kw):
    kw.setdefault("resolved", True)
    return Candidate(passage_id=pid, document_id=doc, text=text, **kw)


PASSAGE = ("Silicon anode swelling reached 18 percent after two hundred cycles, and capacity fell by 22 percent over the same period. "
           "Cells with a carbon coating swelled less than uncoated cells.")


def norm(s):
    return " ".join(s.split())


# ================================================================== the five statuses
def test_a_claim_stated_in_nearly_the_same_words_with_matching_numbers_is_supported():
    v = verify_claim("Silicon anode swelling reached 18 percent after two hundred cycles", "result", [cand(PASSAGE)], searched=True)
    assert v["status"] == "SUPPORTED" and v["strength"]["level"] == "moderate" and v["strength"]["independent_documents"] == 1
    e = v["evidence"][0]
    assert e["role"] == "supporting" and e["directness"] == "direct" and e["numbers_match"] is True and e["citation_valid"]
    assert v["explanation"] and "not proofs" in v["scope_note"]


def test_two_independent_documents_make_the_evidence_strong_but_it_is_still_not_proof():
    v = verify_claim("Silicon anode swelling reached 18 percent after two hundred cycles", "result",
                     [cand(PASSAGE), cand("Silicon anode swelling reached 18 percent after two hundred cycles in a second study.", "d2:p1", "d2")],
                     searched=True)
    assert v["status"] == "SUPPORTED" and v["strength"]["level"] == "strong" and v["strength"]["independent_documents"] == 2
    assert "not a probability" in v["strength"]["note"]


def test_a_claim_only_partly_covered_is_partially_supported_and_says_why():
    claim = "Silicon anode swelling reached 18 percent and the cells were stored in a humid chamber at forty degrees"
    v = verify_claim(claim, "result", [cand(PASSAGE)], searched=True)
    assert v["status"] == "PARTIALLY_SUPPORTED" and v["strength"]["level"] == "weak"
    assert v["evidence"][0]["directness"] == "indirect" and any("indirect" in r for r in v["explanation"])


def test_numbers_that_do_not_match_prevent_a_direct_support():
    v = verify_claim("Silicon anode swelling reached 18 percent after two hundred cycles and capacity fell by 40 percent", "result",
                     [cand(PASSAGE)], searched=True)
    assert v["status"] != "SUPPORTED"
    assert v["evidence"][0]["numbers_match"] is False


@pytest.mark.parametrize("claim,passage,reason", [
    ("Water holding capacity fell by 25 percent at five percent biochar",
     "Water holding capacity rose by 12 percent at two percent biochar and by 25 percent at five percent biochar.", "opposite direction"),
    ("The benefit did not disappear when the soil was fertilised heavily",
     "The benefit disappeared when the soil was fertilised heavily.", "negat"),
    ("Capacity fade after twelve months was 40 percent at 50 degrees",
     "Capacity fade after twelve months was 3 percent at 10 degrees and 19 percent at 45 degrees.", "different figure"),
])
def test_a_passage_that_says_the_opposite_makes_the_claim_contradicted(claim, passage, reason):
    v = verify_claim(claim, "result", [cand(passage)], searched=True)
    assert v["status"] == "CONTRADICTED", v["explanation"]
    e = v["evidence"][0]
    assert e["role"] == "contradicting" and reason in e["reason"]
    assert e["quote"] in norm(passage)


def test_a_comparison_baseline_is_not_mistaken_for_a_negation():
    # 'without it' names the control group; it must not turn a supported claim into a contradicted one
    p = "Cells with the additive kept 91 percent of their capacity after three hundred fast charge cycles, against 74 percent without it."
    v = verify_claim("Cells with the additive kept 91 percent of their capacity after three hundred fast charge cycles", "result",
                     [cand(p)], searched=True)
    assert v["status"] == "SUPPORTED"


def test_direction_is_judged_on_the_clause_about_the_claim_not_the_whole_sentence():
    p = "Client drift grew with the number of local epochs, and test accuracy fell from 81 percent at one epoch to 63 percent at twenty."
    v = verify_claim("Test accuracy rose from 81 percent at one epoch to 63 percent at twenty", "result", [cand(p)], searched=True)
    assert v["status"] == "CONTRADICTED"


def test_insufficient_evidence_when_a_search_found_nothing_relevant():
    assert verify_claim("Zebra migration routes follow seasonal rainfall in the savanna", "result", [], searched=True)["status"] == \
        "INSUFFICIENT_EVIDENCE"
    v = verify_claim("Zebra migration routes follow seasonal rainfall in the savanna", "result", [cand(PASSAGE)], searched=True)
    assert v["status"] == "INSUFFICIENT_EVIDENCE" and v["checks"]["dropped_irrelevant"] == 1 and v["evidence"] == []


def test_unverified_when_no_check_was_possible():
    claim = "Silicon anode swelling reached 18 percent after two hundred cycles"
    assert verify_claim(claim, "result", [], searched=False)["status"] == "UNVERIFIED"  # nothing attached, nothing searched
    assert verify_claim("Cells fade", "result", [cand(PASSAGE)], searched=True)["status"] == "UNVERIFIED"  # too few words to compare
    for t in ("gap", "hypothesis"):
        v = verify_claim(claim, t, [cand(PASSAGE)], searched=False)
        assert v["status"] == "UNVERIFIED" and v["evidence"][0]["role"] == "contextual"  # cited passages are context, never support


def test_invalid_citations_are_reported_and_never_count_as_evidence():
    claim = "Silicon anode swelling reached 18 percent after two hundred cycles"
    only_bad = verify_claim(claim, "result", [Candidate("x:p9", "x", resolved=False)], searched=False)
    assert only_bad["status"] == "UNVERIFIED" and only_bad["invalid_citations"][0]["citation_valid"] is False
    mixed = verify_claim(claim, "result", [Candidate("x:p9", "x", resolved=False), cand(PASSAGE)], searched=True)
    assert mixed["status"] == "SUPPORTED" and len(mixed["invalid_citations"]) == 1  # the real passage decides; the bad one is listed
    assert any("could not be resolved" in r for r in mixed["explanation"])


def test_unreliably_extracted_pages_are_not_used_as_evidence():
    v = verify_claim("Silicon anode swelling reached 18 percent after two hundred cycles", "result", [cand(PASSAGE, quality="low")], searched=True)
    assert v["status"] == "UNVERIFIED" and v["unreliable_passages"] and v["evidence"] == []


def test_contradictory_passages_are_both_kept_as_conflicting_evidence():
    claim = "Cells with the additive retained more capacity than cells without it after fast charging"
    v = verify_claim(claim, "result", [
        cand("Cells with the additive retained more capacity than cells without it after fast charging.", "a:p1", "a"),
        cand("In this replication cells with the additive retained less capacity than cells without it after fast charging.", "b:p1", "b")],
        searched=True)
    assert v["status"] == "PARTIALLY_SUPPORTED" and v["conflicting"] is True
    assert {e["role"] for e in v["evidence"]} == {"supporting", "contradicting"}
    assert any("conflicting" in r for r in v["explanation"])


def test_only_a_contradicting_passage_gives_contradicted_not_unverified():
    v = verify_claim("Cells with the additive retained more capacity than cells without it after fast charging", "result",
                     [cand("In this replication cells with the additive retained less capacity than cells without it after fast charging.")],
                     searched=True)
    assert v["status"] == "CONTRADICTED" and not v["conflicting"]


def test_retrieval_scores_never_decide_a_status():
    claim = "Silicon anode swelling reached 18 percent after two hundred cycles"
    texts = [PASSAGE, "Zebra migration routes follow seasonal rainfall in the savanna."]
    low = verify_claim(claim, "result", [cand(texts[0], retrieval_score=0.001), cand(texts[1], "d:p2", retrieval_score=0.001)], searched=True)
    high = verify_claim(claim, "result", [cand(texts[0], retrieval_score=999.0), cand(texts[1], "d:p2", retrieval_score=999.0)], searched=True)
    flip = verify_claim(claim, "result", [cand(texts[0], retrieval_score=0.001), cand(texts[1], "d:p2", retrieval_score=999.0)], searched=True)
    assert low["status"] == high["status"] == flip["status"] == "SUPPORTED"
    assert [e["role"] for e in low["evidence"]] == [e["role"] for e in high["evidence"]] == [e["role"] for e in flip["evidence"]]
    assert high["evidence"][0]["retrieval_score"] == 999.0  # recorded for the audit trail only
    # a top-ranked irrelevant passage stays irrelevant
    assert verify_claim(claim, "result", [cand(texts[1], retrieval_score=999.0)], searched=True)["status"] == "INSUFFICIENT_EVIDENCE"


def test_page_numbers_and_quotes_come_only_from_the_passage_never_invented():
    v = verify_claim("Silicon anode swelling reached 18 percent after two hundred cycles", "result", [cand(PASSAGE, page_number=None)], searched=True)
    assert v["evidence"][0]["page_number"] is None  # a format without pages has no page number: none is made up
    w = verify_claim("Silicon anode swelling reached 18 percent after two hundred cycles", "result", [cand(PASSAGE, page_number=7)], searched=True)
    assert w["evidence"][0]["page_number"] == 7
    for e in (*v["evidence"], *w["evidence"]):
        assert e["quote"] in norm(PASSAGE)  # verbatim


# ================================================================== novelty
BIG = {"documents": 12, "passages": 80}


def test_novelty_labels_are_cautious_and_scoped_to_the_searched_corpus():
    similar = assess_novelty("Silicon anode swelling reached 18 percent after two hundred cycles", passages=[cand(PASSAGE)], corpus=BIG,
                             queries=["q"], doc_info={"d1": {"title": "Anode study", "year": 2021, "doi": None}})
    assert similar["label"] == NoveltyLabel.SIMILAR_PRIOR_WORK.value
    assert similar["similar_existing_research"][0]["title"] == "Anode study" and similar["closest_documents"]
    novel = assess_novelty("Perovskite films degrade under ultraviolet light in humid air", passages=[cand(PASSAGE)], corpus=BIG, queries=["q"])
    assert novel["label"] == NoveltyLabel.POTENTIALLY_NOVEL.value
    tiny = assess_novelty("Perovskite films degrade under ultraviolet light in humid air", passages=[cand(PASSAGE)],
                          corpus={"documents": 2, "passages": 6}, queries=["q"])
    assert tiny["label"] == NoveltyLabel.INSUFFICIENT_COVERAGE.value  # nothing found in a tiny corpus says nothing
    assert assess_novelty("Cells fade", passages=[], corpus=BIG, queries=["q"])["label"] == NoveltyLabel.UNCLEAR.value
    assert assess_novelty("Perovskite films degrade under ultraviolet light", passages=[], corpus=BIG, queries=[])["label"] == \
        NoveltyLabel.INSUFFICIENT_COVERAGE.value  # no search was run


def test_every_novelty_assessment_states_its_limits_and_never_claims_universal_novelty():
    for passages, corpus in (([cand(PASSAGE)], BIG), ([], BIG), ([], {"documents": 1, "passages": 1})):
        a = assess_novelty("Perovskite films degrade under ultraviolet light in humid air", passages=passages, corpus=corpus, queries=["q"])
        assert {"similar_existing_research", "closest_documents", "overlapping_research_questions", "differences_from_closest",
                "search_coverage", "unsearched", "limitations", "label"} <= set(a)
        assert a["unsearched"] and len(a["limitations"]) >= 3 and a["search_coverage"]["corpus_documents"] == corpus["documents"]
        blob = json.dumps(a).lower()
        assert "no semantic" in blob and "not establish" in blob
        for negation in ("does not establish that the idea is new or that it is not", "not evidence that the idea is new"):
            blob = blob.replace(negation, "")  # the only places the words appear are denials
        for banned in ("is new", "never been", "proven novel", "universally", "globally novel", "first to", "is novel"):
            assert banned not in blob, banned
        assert a["label"] in {n.value for n in NoveltyLabel}


def test_overlapping_research_questions_are_aim_sentences_from_the_closest_documents():
    p = "We study silicon anode swelling and capacity fade in thirty pouch cells. Other text about unrelated matters follows."
    a = assess_novelty("Silicon anode swelling and capacity fade in pouch cells", passages=[cand(p)], corpus=BIG, queries=["q"])
    assert a["overlapping_research_questions"] and a["overlapping_research_questions"][0]["quote"].startswith("We study")


# ================================================================== evaluation
def test_the_verification_evaluation_is_reproducible_and_does_not_hide_its_failures(tmp_path):
    a = run_verification_eval(DATASET, tmp_path / "a")
    b = run_verification_eval(DATASET, tmp_path / "b")
    assert a == b
    s = a["synthetic"]
    assert s["dataset"]["documents"] == 7 and s["dataset"]["claims"] >= 20
    assert set(s["confusion_matrix"]) == set(STATUSES)
    for st in STATUSES:  # every status is exercised by a labelled claim
        assert any(r["expected"] == st for r in s["claims"]), st
    assert s["accuracy_excluding_known_limitations"]["accuracy"] >= 0.9
    hard = [r for r in s["claims"] if r["known_limitation"]]
    assert hard and any(not r["correct"] for r in hard)  # the cases a lexical check gets wrong stay in the set, and are wrong
    assert "real_source" not in a and a["labels"].startswith("synthetic")
    assert any("not scientific validation" in x.lower() for x in a["limitations"])
    readme = (DATASET / "README.md").read_text(encoding="utf-8").lower()
    assert "everything here is synthetic" in readme and "none are included" in readme


# ================================================================== agents: claims, evidence, novelty, report, supervisor
CONFLICT_DOCS = {
    "k1_additive.md": ("Additive study one", "Results",
                       "Cells with the additive retained more capacity than cells without it, 91 percent against 74 percent after fast charging. "
                       "However, only cells cycled at 25 degrees were examined, so the result may not hold at other temperatures."),
    "k2_additive_replication.md": ("Additive replication", "Results",
                                   "In this replication cells with the additive retained less capacity than cells without it, 62 percent against 80 percent "
                                   "after fast charging."),
    "k3_additive_third.md": ("Additive study three", "Results",
                             "Cells with the additive retained more capacity than cells without it, 88 percent against 70 percent after fast charging."),
    "k4_coral.md": ("Coral bleaching", "Results", "Bleaching began once accumulated heating weeks passed four on the fringing reef."),
    "k5_soil.md": ("Soil water", "Results", "Water holding capacity rose by 25 percent at five percent biochar in sandy soil."),
}
QUESTION = "Do cells with the additive retain more capacity than cells without it after fast charging?"


def conflict_corpus(root: Path) -> Corpus:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    c = Corpus(root / "lib", isolate=False)
    for name, (title, section, body) in CONFLICT_DOCS.items():
        f = src / name
        f.write_text(f"# {title}\n\n{section}\n\n{body}\n", encoding="utf-8")
        c.ingest(f, metadata={"title": title, "authors": ["A. Test"], "publication_date": "2022-01-01", "document_type": "article"},
                 license="CC0-1.0")
    return c


def run_workflow(c: Corpus, question=QUESTION, **limits):
    def catalog(doc_id):
        d = c.get_document(doc_id)
        return {"title": d.title, "year": d.year, "doi": d.doi, "license": d.license}

    def verifier(pid):
        v = c.verify_passage(pid)
        return v.ok, v.message

    store = Store()
    engine = build_engine(store, c.build_index(), workflow="full", catalog=catalog, verifier=verifier, poll_interval=0.01)
    rid = engine.create_run(engine.create_project("p6", question), question,
                            RunLimits(retry_backoff_s=0, task_timeout_s=20, max_runtime_s=120, **limits))
    result = run(engine.run(rid))
    return store, rid, result


@pytest.fixture(scope="module")
def conflict_run(tmp_path_factory):
    c = conflict_corpus(tmp_path_factory.mktemp("p6"))
    store, rid, result = run_workflow(c)
    yield c, store, rid, result
    c.close()


def tasks_of(store, rid, agent):
    return [t for t in store.list_tasks(rid) if t["agent_type"] == agent]


def report_of(store, rid):
    return AgentOutput.model_validate(tasks_of(store, rid, "synthesis")[0]["output"]).data["report"]


def test_claims_carry_their_originating_agent_task_status_strength_and_explanation(conflict_run):
    c, store, rid, result = conflict_run
    assert result.status == RunStatus.COMPLETED
    claims = report_of(store, rid)["verification"]["claims"]
    assert claims
    task_ids = {t["id"]: t["agent_type"] for t in store.list_tasks(rid) if t["status"] == "succeeded"}
    for cl in claims:
        assert cl["status"] in STATUSES and cl["explanation"] and cl["strength"]["level"] in ("none", "weak", "moderate", "strong")
        assert cl["origin_agent"] in ("literature", "gap", "hypothesis")
        assert task_ids[cl["origin_task_id"]] == cl["origin_agent"]  # traceable to the task that made it
        assert cl["text"] and cl["claim_type"] and cl["key"].startswith(cl["origin_agent"] + ":")
    kinds = {cl["claim_type"] for cl in claims}
    assert "result" in kinds and kinds & {"gap", "hypothesis"}


def test_every_evidence_item_resolves_to_a_stored_passage_with_a_verbatim_quote(conflict_run):
    c, store, rid, _ = conflict_run
    seen = 0
    for cl in report_of(store, rid)["verification"]["claims"]:
        for e in cl["evidence"]:
            p = c.index.get(e["passage_id"])
            assert p is not None and p.document_id == e["document_id"] and c.verify_passage(e["passage_id"]).ok
            assert e["page_number"] == p.page_number  # None for Markdown: never invented
            assert e["quote"] in norm(p.text) and e["role"] in ("supporting", "contradicting", "contextual")
            assert e["retrieval_method"] in ("source_passage", "bm25_search", "followup_search")
            seen += 1
    assert seen


def test_hypotheses_and_gaps_are_never_verified_and_get_a_novelty_assessment(conflict_run):
    _, store, rid, _ = conflict_run
    v = report_of(store, rid)["verification"]
    for cl in v["claims"]:
        if cl["claim_type"] in ("gap", "hypothesis"):
            assert cl["status"] == "UNVERIFIED"
    assert v["novelty"]
    for n in v["novelty"]:
        assert n["label"] in {x.value for x in NoveltyLabel} and n["limitations"] and n["unsearched"]
        assert n["claim_key"] in {cl["key"] for cl in v["claims"]}


def test_contradicting_results_across_documents_are_represented_and_challenged_by_the_reviewer(conflict_run):
    _, store, rid, _ = conflict_run
    report = report_of(store, rid)
    claims = report["verification"]["claims"]
    results = [cl for cl in claims if cl["claim_type"] == "result"]
    assert results
    assert any(e["role"] == "contradicting" for cl in results for e in cl["evidence"])  # contradiction found by the verifier itself
    assert any(cl["status"] in ("CONTRADICTED", "PARTIALLY_SUPPORTED") and cl["conflicting"] or cl["status"] == "CONTRADICTED" for cl in results)
    review = AgentOutput.model_validate(tasks_of(store, rid, "reviewer")[0]["output"]).data["review"]
    cats = {f["category"] for f in review["findings"]}
    assert cats & {"contradicted_claim", "contradictory_evidence"}, cats  # the Critical Reviewer challenged them
    assert "Contradicting evidence" in {s["title"] for s in report["structured"]["sections"]}
    contra = next(s for s in report["structured"]["sections"] if s["title"] == "Contradicting evidence")
    assert any(b["kind"] == "claim" and b["citations"] for b in contra["blocks"])


def test_the_structured_report_has_the_18_sections_and_every_claim_block_cites_a_resolvable_passage(conflict_run):
    c, store, rid, _ = conflict_run
    s = report_of(store, rid)["structured"]
    assert [x["title"] for x in s["sections"]] == list(SECTION_TITLES) and len(SECTION_TITLES) == 18
    assert [x["number"] for x in s["sections"]] == list(range(1, 19))
    checked = 0
    for sec in s["sections"]:
        assert sec["blocks"], sec["title"]
        for b in sec["blocks"]:
            if b["kind"] == "claim":
                assert b["citations"], (sec["title"], b["text"][:60])  # a factual statement is cited ...
            if b["kind"] == "proposal":
                assert b["status"] == "UNVERIFIED"  # ... or explicitly marked unverified
            for cit in b["citations"]:
                if cit.get("passage_id"):
                    p = c.index.get(cit["passage_id"])
                    assert p is not None and p.document_id == cit["document_id"] and c.verify_passage(cit["passage_id"]).ok
                    if cit.get("quote"):
                        assert cit["quote"] in norm(p.text)
                    checked += 1
    assert checked > 5
    md = render_structured_markdown(s)
    assert all(f"{n}. {t}" in md for n, t in enumerate(SECTION_TITLES[1:], 2))
    refs = next(x for x in s["sections"] if x["title"] == "References")
    assert all(b["citations"] for b in refs["blocks"] if b["kind"] == "derived")


def test_report_generation_is_reproducible(conflict_run, tmp_path):
    c, store, rid, _ = conflict_run
    first = report_of(store, rid)
    synth = tasks_of(store, rid, "synthesis")[0]  # the same inputs, in the order the synthesis agent received them
    tasks = {t["id"]: t for t in store.list_tasks(rid)}
    outs = [AgentOutput.model_validate(tasks[d]["output"]) for d in synth["depends_on"]]
    by = {o.agent_type: o for o in outs}
    again = build_report(first["question"], by, all_outputs=outs)
    assert json.dumps(again["structured"], sort_keys=True) == json.dumps(first["structured"], sort_keys=True)  # same inputs, same report
    store2, rid2, _ = run_workflow(c)  # a second, independent run over the same library
    second = report_of(store2, rid2)

    def without_run_ids(x):  # task ids are random per run (they only trace a claim to its task); everything else must match
        if isinstance(x, dict):
            return {k: without_run_ids(v) for k, v in x.items() if k not in ("origin_task_id", "task_id")}
        return [without_run_ids(v) for v in x] if isinstance(x, list) else x

    assert without_run_ids(second["structured"]["sections"]) == without_run_ids(first["structured"]["sections"])
    assert second["verification"]["counts"] == first["verification"]["counts"]


def test_the_supervisor_requests_follow_up_retrieval_with_a_bounded_budget(conflict_run):
    c, store, rid, result = conflict_run  # default max_evidence_rounds = 1
    sups = [t for t in tasks_of(store, rid, "supervisor") if t["input"].get("evidence_review")]
    assert len(sups) == 2 and [t["input"]["evidence_round"] for t in sups] == [0, 1]
    first = AgentOutput.model_validate(sups[0]["output"]).data["evidence_review"]
    last = AgentOutput.model_validate(sups[1]["output"]).data["evidence_review"]
    assert first["action"] == "follow_up" and first["unresolved"]
    assert last["action"] in ("budget_exhausted", "none", "no_new_query")
    reverify = [t for t in tasks_of(store, rid, "verifier") if t["input"].get("round") == 2]
    followups = [t for t in tasks_of(store, rid, "discovery") if t["input"].get("label") == "evidence follow-up"]
    assert len(reverify) == 1 and followups and all(t["status"] == "succeeded" for t in [*reverify, *followups])
    synth = tasks_of(store, rid, "synthesis")[0]
    assert reverify[0]["id"] in synth["depends_on"] and synth["started_at"] >= reverify[0]["completed_at"]  # the report waited
    assert any(e["type"] == "task_created" and e["payload"].get("agent_type") == "verifier" for e in store.list_events(rid))
    hist = report_of(store, rid)["verification"]["history"]
    assert [h["round"] for h in hist] == [1, 2]


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_the_follow_up_budget_is_never_exceeded(tmp_path, budget):
    c = conflict_corpus(tmp_path)
    store, rid, result = run_workflow(c, max_evidence_rounds=budget, max_tasks=60)
    assert result.status == RunStatus.COMPLETED
    reverify = [t for t in tasks_of(store, rid, "verifier") if t["input"].get("round", 1) > 1]
    assert len(reverify) <= budget
    if budget == 0:
        assert not reverify and not [t for t in tasks_of(store, rid, "discovery") if t["input"].get("label") == "evidence follow-up"]
    last = [t for t in tasks_of(store, rid, "supervisor") if t["input"].get("evidence_review")][-1]
    out = AgentOutput.model_validate(last["output"])
    assert out.data["evidence_review"]["action"] in ("budget_exhausted", "none", "no_new_query")
    assert not out.proposed_tasks  # the last review proposes nothing more
    assert report_of(store, rid)["structured"]["unresolved_claims"] is not None
    c.close()


def test_the_followup_proposal_can_only_block_what_its_policy_allows():
    from agent_engine.contracts import ProposedTask
    from agent_engine.planner.planner import expand
    from agent_engine.policies.policy import full_policy
    parent = {"id": "t0", "agent_type": "discovery", "iteration": 0}
    rows, rej = expand(parent, [ProposedTask(key="x", agent_type="discovery", title="t", blocks=["synthesis"])], [], RunLimits(),
                       full_policy(), {"discovery", "synthesis"})
    assert rows == [] and rej  # discovery may not even propose; and may_block is checked separately below
    sup = {"id": "t1", "agent_type": "supervisor", "iteration": 0}
    ok, rej = expand(sup, [ProposedTask(key="v", agent_type="verifier", title="t", blocks=["synthesis"])], [], RunLimits(),
                     full_policy(), {"verifier", "synthesis"})
    assert len(ok) == 1 and ok[0]["blocks"] == ["synthesis"]
    bad, rej = expand(sup, [ProposedTask(key="v", agent_type="verifier", title="t", blocks=["literature"])], [], RunLimits(),
                      full_policy(), {"verifier", "literature"})
    assert bad == [] and "may not block" in rej[0].reason


def test_a_run_over_a_corpus_without_conflicts_asks_for_no_follow_up(tmp_path):
    c = Corpus(tmp_path / "lib", isolate=False)
    for f in sorted((DATASET / "docs").glob("v[4-7]*.md")) + [DATASET / "docs" / "v3_bleaching.md"]:
        c.ingest(f, metadata=DECLARED[f.name], license="CC0-1.0")
    store, rid, result = run_workflow(c, question="How does client drift affect accuracy as local epochs grow?")
    assert result.status == RunStatus.COMPLETED
    sups = [t for t in tasks_of(store, rid, "supervisor") if t["input"].get("evidence_review")]
    assert len(sups) == 1 and AgentOutput.model_validate(sups[0]["output"]).data["evidence_review"]["action"] == "none"
    assert not [t for t in tasks_of(store, rid, "verifier") if t["input"].get("round", 1) > 1]
    c.close()


# ================================================================== API
def md(name):
    return (DATASET / "docs" / name).read_bytes()


def put_docs(c, h, pid, names):
    for n in names:
        assert upload(c, h, pid, n, md(n)).status_code == 201


def passage_id(c, pid, doc_substring):
    for d in c.get(f"/projects/{pid}/documents").json():
        if doc_substring in d["filename"]:
            return c.get(f"/projects/{pid}/documents/{d['id']}/passages").json()[0]["passage_id"]
    raise AssertionError(doc_substring)


@pytest.fixture()
def api(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        put_docs(c, h, pid, ["v1_additive.md", "v2_additive_replication.md", "v3_bleaching.md", "v6_biochar.md", "v7_ageing.md"])
        yield c, h, pid


def test_a_claim_can_be_created_cited_verified_listed_and_its_history_read(api):
    c, h, pid = api
    pa = passage_id(c, pid, "v6")
    r = c.post(f"/projects/{pid}/claims", headers=h, json={"text": "Water holding capacity rose by 25 percent at five percent biochar",
                                                            "claim_type": "result", "evidence": [{"passage_id": pa}]})
    assert r.status_code == 201, r.text
    cl = r.json()
    assert cl["status"] == "UNVERIFIED" and cl["source"] == "user" and cl["origin_agent"] == "user" and cl["id"].startswith("clm_")
    ev = cl["evidence"][0]  # page, quote and document come from the stored passage
    stored = c.get(f"/projects/{pid}/documents/{ev['document_id']}/passages").json()[0]
    assert ev["quote"] in norm(stored["text"]) and ev["page_number"] == stored.get("page_number") and ev["origin"] == "user"
    v = c.post(f"/projects/{pid}/claims/{cl['id']}/verify", headers=h).json()
    assert v["status"] == "SUPPORTED" and v["strength"]["level"] in ("moderate", "strong") and v["explanation"]
    assert any(e["role"] == "supporting" and e["retrieval_method"] in ("attached", "bm25_search") for e in v["evidence"])
    c.post(f"/projects/{pid}/claims/{cl['id']}/verify", headers=h)
    hist = c.get(f"/projects/{pid}/claims/{cl['id']}/verifications").json()
    assert [x["round"] for x in hist] == [1, 2] and hist[0]["status"] == "SUPPORTED" and hist[0]["verifier"].startswith("deterministic")
    listing = c.get(f"/projects/{pid}/claims", params={"status": "SUPPORTED"}).json()
    assert listing["total"] == 1 and listing["claims"][0]["id"] == cl["id"]
    assert c.get(f"/projects/{pid}/claims", params={"status": "CONTRADICTED"}).json()["total"] == 0
    assert c.get(f"/projects/{pid}/claims", params={"status": "bogus"}).status_code == 422
    assert c.get(f"/projects/{pid}/verifications").json()[0]["claim_id"] == cl["id"]
    assert c.get(f"/projects/{pid}/claims/{cl['id']}").json()["verifications"] == 2


def test_the_api_assigns_contradicted_insufficient_and_unverified(api):
    c, h, pid = api

    def status_of(text, ctype="result"):
        cl = c.post(f"/projects/{pid}/claims", headers=h, json={"text": text, "claim_type": ctype}).json()
        return c.post(f"/projects/{pid}/claims/{cl['id']}/verify", headers=h).json()

    contradicted = status_of("Water holding capacity fell by 25 percent at five percent biochar")
    assert contradicted["status"] == "CONTRADICTED" and any(e["role"] == "contradicting" for e in contradicted["evidence"])
    assert status_of("Zebra migration routes follow seasonal rainfall in the savanna")["status"] == "INSUFFICIENT_EVIDENCE"
    assert status_of("Varying biochar content changes soil water retention", "hypothesis")["status"] == "UNVERIFIED"


def test_evidence_that_does_not_exist_is_refused_and_nothing_is_stored(api):
    c, h, pid = api
    r = c.post(f"/projects/{pid}/claims", headers=h, json={"text": "Water holding capacity rose by 25 percent at five percent biochar",
                                                            "claim_type": "result", "evidence": [{"passage_id": "doc-nope:p001"}]})
    assert r.status_code == 422 and "does not exist" in r.json()["detail"]
    assert c.get(f"/projects/{pid}/claims").json()["total"] == 0  # no half-cited claim was left behind
    cl = c.post(f"/projects/{pid}/claims", headers=h, json={"text": "Water holding capacity rose by 25 percent", "claim_type": "result"}).json()
    assert c.post(f"/projects/{pid}/claims/{cl['id']}/evidence", headers=h, json={"passage_id": "doc-nope:p001"}).status_code == 422
    ok = c.post(f"/projects/{pid}/claims/{cl['id']}/evidence", headers=h, json={"passage_id": passage_id(c, pid, "v6")})
    assert ok.status_code == 201 and ok.json()["role"] == "contextual"  # attaching is not endorsing: the verifier decides the role
    assert c.delete(f"/projects/{pid}/claims/{cl['id']}/evidence/{ok.json()['id']}", headers=h).status_code == 204
    assert c.delete(f"/projects/{pid}/claims/{cl['id']}/evidence/999999", headers=h).status_code == 404


def test_invalid_input_is_rejected(api):
    c, h, pid = api
    bad = [{"text": "short", "claim_type": "result"}, {"text": "A long enough claim text here", "claim_type": "nonsense"},
           {"text": "x" * 2001, "claim_type": "result"}]
    for body in bad:
        assert c.post(f"/projects/{pid}/claims", headers=h, json=body).status_code == 422
    assert c.post(f"/projects/{pid}/novelty", headers=h, json={"statement": "tiny"}).status_code == 422


def test_novelty_endpoint_states_limits_and_is_stored(api):
    c, h, pid = api
    r = c.post(f"/projects/{pid}/novelty", headers=h, json={"statement": "Perovskite films degrade under ultraviolet light in humid air"})
    assert r.status_code == 201, r.text
    n = r.json()
    a = n["assessment"]
    assert n["id"].startswith("nov_") and n["label"] == a["label"] and a["label"] in {x.value for x in NoveltyLabel}
    assert a["limitations"] and a["unsearched"] and "not establish" in a["disclaimer"]
    assert c.get(f"/projects/{pid}/novelty/{n['id']}").json()["assessment"]["statement"].startswith("Perovskite")
    assert [x["id"] for x in c.get(f"/projects/{pid}/novelty").json()] == [n["id"]]
    same = c.post(f"/projects/{pid}/novelty", headers=h, json={"statement": "Water holding capacity rose by 25 percent at five percent biochar"}).json()
    assert same["label"] == NoveltyLabel.SIMILAR_PRIOR_WORK.value and same["assessment"]["similar_existing_research"]


def test_a_project_without_documents_cannot_verify_or_assess(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        cl = c.post(f"/projects/{pid}/claims", headers=h, json={"text": "Water holding capacity rose by 25 percent", "claim_type": "result"}).json()
        assert c.post(f"/projects/{pid}/claims/{cl['id']}/verify", headers=h).status_code == 409
        assert c.post(f"/projects/{pid}/novelty", headers=h, json={"statement": "Some statement about soil water"}).status_code == 409


ROUTES = [
    ("post", "/projects/{p}/claims", {"text": "Water holding capacity rose by 25 percent", "claim_type": "result"}),
    ("get", "/projects/{p}/claims", None), ("get", "/projects/{p}/claims/{c}", None),
    ("post", "/projects/{p}/claims/{c}/evidence", {"passage_id": "x"}), ("post", "/projects/{p}/claims/{c}/verify", None),
    ("get", "/projects/{p}/claims/{c}/verifications", None), ("get", "/projects/{p}/verifications", None),
    ("post", "/projects/{p}/novelty", {"statement": "Some statement about soil water"}), ("get", "/projects/{p}/novelty", None),
    ("delete", "/projects/{p}/claims/{c}/evidence/1", None),
]


def test_every_claim_endpoint_requires_login_csrf_and_ownership(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        alice = sign_in(c, "alice@example.org")
        pid = new_project(c, alice)
        put_docs(c, alice, pid, ["v6_biochar.md"])
        cl = c.post(f"/projects/{pid}/claims", headers=alice, json={"text": "Water holding capacity rose by 25 percent", "claim_type": "result"}).json()
        mine = {"p": pid, "c": cl["id"]}

        def call(method, path, body, headers):
            return getattr(c, method)(path.format(**mine), headers=headers, **({"json": body} if body is not None else {}))

        c.post("/auth/logout", headers=alice)
        for method, path, body in ROUTES:
            assert call(method, path, body, {}).status_code == 401, (method, path)  # not signed in
        mallory = sign_in(c, "mallory@example.org")
        for method, path, body in ROUTES:
            r = call(method, path, body, mallory)
            assert r.status_code == 404, (method, path, r.status_code)  # someone else's project looks missing
            assert "water" not in r.text.lower()
        own = new_project(c, mallory)
        put_docs(c, mallory, own, ["v7_ageing.md"])
        assert c.get(f"/projects/{own}/claims/{cl['id']}").status_code == 404  # her project, Alice's claim id
        assert c.post(f"/projects/{own}/claims/{cl['id']}/verify", headers=mallory).status_code == 404
        assert c.get(f"/projects/{own}/claims").json()["total"] == 0
        assert c.post(f"/projects/{own}/claims", headers={}, json={"text": "Capacity fade after twelve months", "claim_type": "result"}).status_code == 403  # no CSRF
        # Alice's claim is unchanged and her data was never visible
        c.post("/auth/logout", headers=mallory)
        c.post("/auth/login", json={"email": "alice@example.org", "password": "correct horse battery 42"})


def test_run_report_claims_and_structured_view_over_the_api(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        for i, (name, (title, section, body)) in enumerate(CONFLICT_DOCS.items()):
            data = f"# {title}\n\n{section}\n\n{body}\n".encode()
            assert upload(c, h, pid, name, data).status_code == 201
        rid = c.post(f"/projects/{pid}/runs", json={"question": QUESTION}, headers=h).json()["id"]
        assert wait_done(c, rid, h)["status"] == "completed"
        rep = c.post(f"/runs/{rid}/reports", headers=h)
        assert rep.status_code == 201, rep.text
        view = rep.json()
        sections = view["content"]["structured"]["sections"]
        assert [s["title"] for s in sections] == list(SECTION_TITLES)
        passages = {p["passage_id"]: p for d in c.get(f"/projects/{pid}/documents").json()
                    for p in c.get(f"/projects/{pid}/documents/{d['id']}/passages").json()}
        n = 0
        for s in sections:
            for b in s["blocks"]:
                for cit in b["citations"]:
                    if cit.get("passage_id"):
                        assert cit["resolvable"] is True and cit["passage_id"] in passages  # resolves in the uploaded library
                        assert cit["source_text"] and (not cit.get("quote") or cit["quote"] in cit["source_text"])
                        n += 1
        assert n > 5
        st = c.get(f"/reports/{view['id']}/structured").json()
        assert st["content_sha256"] == view["meta"]["content_sha256"] and len(st["structured"]["sections"]) == 18
        md_resp = c.get(f"/reports/{view['id']}/structured.md")
        assert md_resp.status_code == 200 and "## 13. Verification results" in md_resp.text
        # version history is the existing mechanism; the generated content stays immutable
        assert [v["version"] for v in c.get(f"/reports/{view['id']}/versions").json()] == [1]
        # claims were persisted from the run
        claims = c.get(f"/projects/{pid}/claims", params={"run_id": rid}).json()
        assert claims["total"] > 0
        assert all(x["origin_agent"] in ("literature", "gap", "hypothesis") and x["origin_task_id"] and x["status"] in STATUSES
                   for x in claims["claims"])
        detail = c.get(f"/projects/{pid}/claims/{claims['claims'][0]['id']}").json()
        assert detail["explanation"] and detail["verifications"] >= 1
        again = c.post(f"/runs/{rid}/claims/extract", headers=h).json()
        assert again["created"] == 0 and again["claims"] == claims["total"]  # idempotent
        assert c.get(f"/projects/{pid}/claims", params={"run_id": rid}).json()["total"] == claims["total"]
        assert c.get(f"/projects/{pid}/novelty").json()  # the run's novelty assessments were stored too
        # another user sees none of it
        c.post("/auth/logout", headers=h)
        sign_in(c, "mallory@example.org")
        assert c.get(f"/reports/{view['id']}/structured").status_code == 404
        assert c.get(f"/reports/{view['id']}/structured.md").status_code == 404
        assert c.get(f"/projects/{pid}/claims").status_code == 404
        mh = {"X-CSRF-Token": c.get("/auth/me").json().get("csrf_token", "")} if c.get("/auth/me").status_code == 200 else {}
        assert c.post(f"/runs/{rid}/claims/extract", headers=mh).status_code in (403, 404)


def test_extracting_claims_from_a_run_that_has_not_finished_or_reported_is_refused(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        put_docs(c, h, pid, ["v6_biochar.md"])
        rid = c.post(f"/projects/{pid}/runs", json={"start": False, "question": QUESTION}, headers=h).json()["id"]
        assert c.post(f"/runs/{rid}/claims/extract", headers=h).status_code == 409


# ================================================================== migration
def test_migration_0003_adds_the_claim_tables_without_touching_existing_data(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path / 'a.db'}")
    assert migrate.upgrade(e, "0002") == "0002"
    with e.begin() as conn:
        conn.execute(text("INSERT INTO users (id,email,password_hash,display_name,created_at,is_active) "
                          "VALUES ('usr_1','a@b.c','x','', '2026-01-01 00:00:00', 1)"))
    assert migrate.upgrade(e, "0003") == "0003"  # revision 0003 itself (later revisions are tested with their own phase)
    names = set(inspect(e).get_table_names())
    assert {"claims", "claim_evidence", "claim_verifications", "novelty_assessments", "users"} <= names
    with e.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar() == 1
    assert migrate.upgrade(e) == migrate.head_revision()
    assert migrate.schema_diff(e) == []  # the models and the migrations agree
    migrate.downgrade(e, "0002")
    assert "claims" not in set(inspect(e).get_table_names())
    assert migrate.upgrade(e, "0003") == "0003"
