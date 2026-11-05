"""Phase 8: the research workflow - literature analysis, gaps, hypotheses, verification, synthesis, report export.
Exit condition: an end-to-end report with linked evidence and explicit uncertainty."""
import json

import pytest

pytest.importorskip("pymupdf")

from agent_engine.contracts import AgentOutput, RunLimits, RunStatus  # noqa: E402
from agent_engine.policies.policy import full_policy  # noqa: E402
from agent_engine.registry.defaults import build_engine  # noqa: E402
from agent_engine.reporting import export_report, render_markdown, report_from_store  # noqa: E402
from agent_engine.reporting.builder import NOVELTY_NONE  # noqa: E402
from agent_engine.agents.textutil import classify, polarity, sentences  # noqa: E402
from agent_engine.agents.verifier import VerifierAgent  # noqa: E402
from agent_engine.state.store import Store  # noqa: E402
from research.bm25.index import BM25Index, Passage  # noqa: E402
from research.corpus import Corpus  # noqa: E402
from tests.conftest_agents import run  # noqa: E402
from tests.corpus_fixtures import build_library  # noqa: E402

QUESTION = "What percent accuracy was retained?"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    lib = build_library(tmp_path_factory.mktemp("pdfs"))
    c = Corpus(tmp_path_factory.mktemp("corpus"), isolate=False)
    for path in lib.values():
        assert c.ingest(path).status == "ingested"
    yield c
    c.close()


def make(corpus, *, verifier=None, workflow="full", index=None, **kw):
    def catalog(doc_id):
        d = corpus.get_document(doc_id)
        return {"title": d.title, "year": d.year, "doi": d.doi, "license": d.license}

    def default_verifier(pid):
        v = corpus.verify_passage(pid)
        return v.ok, v.message

    store = Store()
    engine = build_engine(store, index or corpus.build_index(), workflow=workflow, catalog=catalog,
                          verifier=verifier or default_verifier, poll_interval=0.01, **kw)
    return store, engine


def go(corpus, question=QUESTION, **kw):
    store, engine = make(corpus, **kw)
    rid = engine.create_run(engine.create_project("wf", question), question,
                            RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60))
    result = run(engine.run(rid))
    return store, rid, result


@pytest.fixture(scope="module")
def full_run(corpus):
    store, rid, result = go(corpus)
    return store, rid, result, report_from_store(store, rid)


def outputs(store, rid):
    return {t["agent_type"]: AgentOutput.model_validate(t["output"]) for t in store.list_tasks(rid) if t["output"]}


# ================================================================== text helpers
def test_sentence_classification_uses_section_and_cues():
    assert classify("Short cells retained ninety percent of their efficiency.", "3. Results") == "result"
    assert classify("We caution that the metric does not measure usefulness.", "Discussion") == "limitation"
    assert classify("Each client clips its update to a fixed norm before sending it.", "1. Method") == "method"
    assert classify("This sentence says nothing in particular about anything at all.", "Intro") is None
    assert len(sentences("Too short. This one has enough words in it to count as a sentence.")) == 1
    assert polarity("accuracy increased") == 1 and polarity("accuracy decreased") == -1
    assert polarity("accuracy increased then decreased") == 0


# ================================================================== end to end
def test_full_workflow_runs_every_stage_and_produces_a_report(full_run):
    store, rid, result, report = full_run
    assert result.status == RunStatus.COMPLETED and result.summary["failed_tasks"] == []
    agents = result.summary["agents"]
    for a in ("literature", "gap", "hypothesis", "verifier", "synthesis"):
        assert agents[a]["succeeded"] == 1
    assert report and report["status"] == "complete"
    assert report["human_review"]["status"] == "pending"


def test_report_findings_are_verbatim_quotes_with_checkable_citations(full_run, corpus):
    _, _, _, report = full_run
    index = corpus.build_index()
    findings = [f for fs in report["findings"].values() for f in fs]
    assert len(findings) >= 6
    for f in findings:
        c = f["citation"]
        p = index.get(c["passage_id"])
        assert p is not None and p.document_id == c["document_id"] and p.page_number == c["page_number"]
        assert " ".join(f["text"].split()) in " ".join(p.text.split())  # a quote, not a paraphrase
        assert corpus.verify_passage(c["passage_id"]).ok
    # numbers in findings appear in the cited passages (nothing invented)
    assert any("ninety percent" in f["text"] for f in findings)


def test_every_claim_has_a_status_and_non_findings_are_never_called_verified(full_run):
    store, rid, _, report = full_run
    claims = outputs(store, rid)["verifier"].data["claims"]
    assert {c["status"] for c in claims} <= {"verified", "unsupported", "needs_review", "hypothesis", "candidate_gap"}
    assert all(c["status"] == "hypothesis" for c in claims if c["type"] == "hypothesis")
    assert all(c["status"] == "candidate_gap" for c in claims if c["type"] == "gap")
    verified_ids = {f["claim_id"] for fs in report["findings"].values() for f in fs}
    assert verified_ids == {c["id"] for c in claims if c["status"] == "verified" and c["type"] == "extracted"}


def test_hypotheses_are_labelled_falsifiable_and_traceable_to_gaps(full_run):
    _, _, _, report = full_run
    gap_ids = {g["id"] for g in report["candidate_gaps"]}
    assert report["hypotheses"]
    for h in report["hypotheses"]:
        assert h["status"] == "proposed" and "not a finding" in h["label"]
        assert h["assumptions"] and h["falsification_criteria"] and h["gap_id"] in gap_ids
    assert all(g["status"] == "candidate_gap" for g in report["candidate_gaps"])


def test_report_states_scope_uncertainty_limits_and_the_bounded_novelty_claim(full_run):
    _, _, _, report = full_run
    assert report["scope"]["corpus"]["documents"] == 5
    assert report["scope"]["unsearched"] and report["uncertainties"] and report["limitations"]
    assert "Absence of other matches elsewhere is not established" in report["novelty_statement"]  # matches found: bounded
    assert any("No experiments were run for this report" in line for line in report["limitations"])
    assert report["source_files_checked"] is True


def test_markdown_links_every_finding_to_its_page_and_passage(full_run):
    _, _, _, report = full_run
    md = render_markdown(report)
    for section in ("## Scope of the search", "## Verified findings", "## Unsupported claims",
                    "## Candidate gaps (not findings)", "## Hypotheses (untested proposals, not findings)",
                    "## Uncertainty", "## Limitations"):
        assert section in md
    for fs in report["findings"].values():
        for f in fs:
            assert f["citation"]["passage_id"] in md and f"p.{f['citation']['page_number']}" in md
    assert "Human review:** pending" in md


def test_export_writes_json_and_markdown_that_match_the_persisted_report(full_run, tmp_path):
    store, rid, _, report = full_run
    paths = export_report(report_from_store(store, rid), tmp_path / "out")
    assert json.loads(paths["json"].read_text(encoding="utf-8")) == report
    assert paths["markdown"].read_text(encoding="utf-8") == render_markdown(report)


# ================================================================== honesty under failure
def test_a_passage_that_no_longer_matches_its_original_is_labelled_unsupported(corpus):
    bad = {}

    def verifier(pid):
        if not bad:
            bad["pid"] = pid  # fail the first passage the verifier checks
        return (pid != bad["pid"]), ("stored file changed" if pid == bad["pid"] else "ok")

    store, rid, result = go(corpus, verifier=verifier)
    report = report_from_store(store, rid)
    assert result.status == RunStatus.COMPLETED
    assert report["unsupported_claims"]
    unsupported = {u["claim_id"] for u in report["unsupported_claims"]}
    shown = {f["claim_id"] for fs in report["findings"].values() for f in fs}
    assert not unsupported & shown  # unsupported claims never appear as findings
    assert any("stored file changed" in " ".join(u["reasons"]) for u in report["unsupported_claims"])
    assert "## Unsupported claims" in render_markdown(report) and "stored file changed" in render_markdown(report)
    assert outputs(store, rid)["verifier"].status == "partial"


def test_no_evidence_gives_an_honest_empty_report(corpus):
    store, rid, result = go(corpus, question="quantum chromodynamics lattice gauge")
    report = report_from_store(store, rid)
    assert result.status == RunStatus.COMPLETED
    assert report["status"] == "insufficient_evidence"
    assert report["novelty_statement"] == NOVELTY_NONE
    assert not any(report["findings"].values()) and report["candidate_gaps"] == [] and report["hypotheses"] == []
    assert any("did not find enough solid evidence" in u for u in report["uncertainties"])


def _index(passages):
    idx = BM25Index()
    idx.add_passages(passages)
    return idx


def _fillers():
    topics = ["Granite quarries supply building stone across the northern valleys and harbours of the region.",
              "Medieval monasteries copied manuscripts by candlelight during the long winter months of the year.",
              "Migrating geese follow river valleys southward every autumn before the first frost arrives."]
    return [Passage(passage_id=f"f{i}:p001", document_id=f"f{i}", text=t, page_number=1, section="Intro", quality="ok")
            for i, t in enumerate(topics)]


def test_unreliable_pages_are_not_analysed_or_reported_as_findings(tmp_path):
    good = [Passage(passage_id=f"d{i}:p001", document_id=f"d{i}", text=t, page_number=1, section="3. Results", quality="ok")
            for i, t in enumerate(["Cells retained ninety percent of their initial efficiency after two thousand hours of ageing.",
                                   "Cells retained eighty percent of their initial efficiency after one thousand hours of ageing.",
                                   "Cells retained seventy percent of their initial efficiency after five hundred hours of ageing."])]
    bad = Passage(passage_id="dx:p001", document_id="dx", page_number=1, section="3. Results", quality="low",
                  text="Cells retained ninety five percent of their initial efficiency after ten thousand hours of ageing.")
    index = _index(good + [bad] + _fillers())  # unrelated passages keep BM25 idf positive in a tiny corpus
    store = Store()
    engine = build_engine(store, index, workflow="full", poll_interval=0.01)
    rid = engine.create_run(engine.create_project("lq", "cells retained efficiency"), "cells retained efficiency",
                            RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60))
    run(engine.run(rid))
    report = report_from_store(store, rid)
    texts = " ".join(f["text"] for fs in report["findings"].values() for f in fs)
    assert "ninety five" not in texts and report["scope"]["low_quality_matches"] == 1
    assert report["source_files_checked"] is False
    assert any("not re-checked" in u for u in report["uncertainties"])


def test_opposite_results_in_different_documents_are_flagged_for_review_not_resolved(tmp_path):
    texts = ["Cell efficiency increased by ten percent after the heat treatment of the contact layer in our trial.",
             "Cell efficiency decreased by ten percent after the heat treatment of the contact layer in our trial.",
             "Cell efficiency was measured at ten percent after the heat treatment of the contact layer in the trial."]
    index = _index([Passage(passage_id=f"d{i}:p001", document_id=f"d{i}", text=t, page_number=1, section="Results",
                            quality="ok") for i, t in enumerate(texts)] + _fillers())
    store = Store()
    engine = build_engine(store, index, workflow="full", poll_interval=0.01)
    q = "cell efficiency heat treatment contact layer"
    rid = engine.create_run(engine.create_project("t", q), q, RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60))
    run(engine.run(rid))
    report = report_from_store(store, rid)
    assert report["tensions"] and report["needs_review"]
    assert any("possible tension" in " ".join(n["reasons"]) for n in report["needs_review"])
    assert any("tension" in u for u in report["uncertainties"])


# ================================================================== engine behaviour
def test_basic_workflow_is_unchanged(corpus):
    store, rid, result = go(corpus, workflow="basic")
    assert result.status == RunStatus.COMPLETED
    assert {t["agent_type"] for t in store.list_tasks(rid)} == {"supervisor", "discovery", "critic"}
    assert report_from_store(store, rid) is None


def test_workflow_agents_are_read_only_and_policy_bound():
    p = full_policy()
    for a in ("literature", "gap", "hypothesis", "verifier", "synthesis"):
        assert p.policy_for(a).may_propose == frozenset()
    assert p.policy_for("gap").allowed_tools == frozenset() and p.policy_for("synthesis").allowed_tools == frozenset()
    assert p.may_propose("critic", "synthesis") and not p.may_propose("supervisor", "synthesis")
    assert "corpus.search" not in p.policy_for("verifier").allowed_tools
    with pytest.raises(ValueError):
        build_engine(Store(), BM25Index(), workflow="bogus")


def test_task_limit_stops_the_workflow_without_a_report(corpus):
    store, rid, result = go(corpus, question=QUESTION)
    tight = Store()
    engine = build_engine(tight, corpus.build_index(), workflow="full", poll_interval=0.01)
    rid2 = engine.create_run(engine.create_project("lim", QUESTION), QUESTION,
                             RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60, max_tasks=7))
    run(engine.run(rid2))
    assert {t["agent_type"] for t in tight.list_tasks(rid2)} >= {"supervisor", "critic"}
    assert report_from_store(tight, rid2) is None  # the chain was rejected as a whole; nothing half-built is reported


def test_verifier_requires_the_critic_assessment():
    from agent_engine.contracts import TaskSpec
    from agent_engine.errors import NonRetryableError
    from agent_engine.tools.registry import ToolRegistry
    from agent_engine.agents.base import AgentContext  # noqa: F401
    task = TaskSpec(id="t", run_id="r", agent_type="verifier", title="v", status="running", depends_on=[], input={},
                    dependency_outputs={}, iteration=0, attempt=1)
    with pytest.raises(NonRetryableError):
        run(VerifierAgent().run(None, task))
