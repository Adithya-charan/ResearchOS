"""Phase 7 x Phase 6: the agents answer questions over the user's own documents, cite page-level evidence that can be
verified against the original files, and treat text from badly-extracted pages as visible-but-not-trusted."""
import pytest

pytest.importorskip("pymupdf")

from agent_engine.contracts import AgentOutput, RunLimits, RunStatus  # noqa: E402
from agent_engine.registry.defaults import build_engine  # noqa: E402
from agent_engine.state.store import Store  # noqa: E402
from agent_engine.tools.corpus import IndexEvidenceResolver  # noqa: E402
from research.bm25.index import BM25Index, Passage  # noqa: E402
from research.corpus import Corpus, SearchFilters  # noqa: E402
from tests.conftest_agents import run  # noqa: E402
from tests.corpus_fixtures import build_library  # noqa: E402
from tests.pdfgen import make_pdf  # noqa: E402

NOVELTY = "No close match was identified in the indexed corpus searched for this run"


def go(index, question):
    store = Store()
    engine = build_engine(store, index, poll_interval=0.01)
    pid = engine.create_project("corpus agents", question)
    rid = engine.create_run(pid, question, RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60))
    result = run(engine.run(rid))
    outputs = [(t["agent_type"], AgentOutput.model_validate(t["output"])) for t in store.list_tasks(rid) if t["output"]]
    critic = [o for a, o in outputs if a == "critic"][-1]  # the last review is the one that counts
    return result, critic, outputs


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("pdfs"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, library):
    c = Corpus(tmp_path_factory.mktemp("corpus"), isolate=False)
    for path in library.values():
        assert c.ingest(path).status == "ingested"
    yield c
    c.close()


def titles(corpus, evidence):
    return {corpus.get_document(r["document_id"]).title for r in evidence}


# ================================================================== the exit condition, end to end
def test_a_question_over_pdfs_returns_page_level_evidence_that_checks_out(corpus, library):
    question = "What percent accuracy was retained?"
    result, critic, outputs = go(corpus.build_index(), question)

    assert result.status == RunStatus.COMPLETED
    assessment = critic.data["assessment"]
    assert assessment["sufficient"] is True and assessment["documents"] >= 3
    evidence = result.summary["evidence"]
    assert len(evidence) >= 3
    assert {"Sparse Attention for Long Documents", "Differentially Private Federated Training",
            "Stability of Perovskite Solar Cells"} <= titles(corpus, evidence)

    for ref in evidence:
        passage = corpus.store.get_passage(ref["passage_id"])[0]
        assert ref["page_number"] == passage.page_number and ref["page_number"] in (1, 2, 3)  # page-level references
        v = corpus.verify_passage(ref["passage_id"])  # ...that survive a check against the original file
        assert v.ok, (ref, v)
    # everything an agent cited resolves to a real indexed passage
    resolver = IndexEvidenceResolver(corpus.index)
    assert all(resolver.check(r) is None for _, o in outputs for r in o.evidence)


def test_thin_evidence_is_reported_as_insufficient_with_the_reason(corpus):
    result, critic, _ = go(corpus.build_index(), "Do cells degrade after thousands of hours?")
    a = critic.data["assessment"]
    assert a["sufficient"] is False and a["documents"] == 2 and len(a["accepted"]) >= 3  # enough passages, too few documents
    assert a["reasons"] == ["evidence from only 2 document(s); need 3"]
    assert titles(corpus, a["accepted_evidence"]) == {"Solid-State Electrolytes for Lithium Batteries",
                                                      "Stability of Perovskite Solar Cells"}
    assert any("insufficient" in u for u in critic.uncertainties)


def test_a_question_the_documents_cannot_answer_gets_the_bounded_negative_statement(corpus):
    result, critic, _ = go(corpus.build_index(), "quantum entanglement between neutron stars")
    assert critic.data["assessment"]["sufficient"] is False and result.summary["evidence"] == []
    assert any(NOVELTY in u for u in critic.uncertainties)
    assert any("not proof that the idea has never been published" in u for u in critic.uncertainties)


def test_an_agent_run_can_be_scoped_to_part_of_the_library(corpus):
    only_old = corpus.build_index(SearchFilters(year_max=2019))
    assert {corpus.get_document(d).year for d in {p.document_id for p in only_old._passages}} == {2018, 2019}
    result, _, _ = go(only_old, "What percent accuracy was retained?")
    assert {r["document_id"] for r in result.summary["evidence"]} <= {p.document_id for p in only_old._passages}
    assert result.summary["corpus"]["documents"] == 2  # the run reports exactly what it searched


def test_the_filtered_index_holds_the_same_passages_as_the_live_one(corpus):
    full = corpus.build_index()
    assert full.num_passages == corpus.index.num_passages and full.num_documents == corpus.index.num_documents
    q = "ionic conductivity of sulfide electrolytes"
    assert [h.passage.passage_id for h in full.search(q, k=5)] == [h.passage.passage_id for h in corpus.index.search(q, k=5)]
    assert corpus.build_index(SearchFilters(year_min=3000)).num_passages == 0


# ================================================================== low-quality extraction
BASE = [
    Passage("d1:p000", "d1", "Increasing temperature raises the ionic conductivity of solid state electrolytes because ion "
            "hopping becomes thermally activated.", 1, "Abstract"),
    Passage("d2:p000", "d2", "Sulfide solid electrolytes reach high ionic conductivity near room temperature, exceeding oxide "
            "ceramics.", 2, "2. Results"),
    Passage("d3:p000", "d3", "Arrhenius analysis of ionic conductivity against temperature gives the activation energy for "
            "lithium transport in solid state electrolyte films.", 1),
]
QUESTION = "How does temperature affect ionic conductivity in solid state electrolytes?"


def low(doc, page):
    return Passage(f"{doc}:p000", doc, "Temperature dependence of ionic conductivity in solid state electrolytes was measured "
                   "for every sample in the series.", page, "3. Data", "low")


def index_of(*passages):
    idx = BM25Index()
    idx.add_passages(passages)
    return idx


def test_passages_from_unreliable_pages_are_shown_but_never_counted_as_evidence():
    result, critic, outputs = go(index_of(*BASE, low("l1", 4), low("l2", 7)), QUESTION)
    a = critic.data["assessment"]

    assert a["sufficient"] is True and a["documents"] == 3                       # the three solid documents decide
    assert {r["document_id"] for r in a["accepted_evidence"]} == {"d1", "d2", "d3"}
    assert {m["document_id"]: m["page_number"] for m in a["low_quality_matches"]} == {"l1": 4, "l2": 7}
    assert all(m["quality"] == "low" for m in a["low_quality_matches"]) and all(m["quality"] == "ok" for m in a["accepted"])
    note = next(u for u in critic.uncertainties if "unreliable" in u)
    assert "l1 p.4" in note and "l2 p.7" in note and "not counted as evidence" in note
    assert result.summary["evidence"] == a["accepted_evidence"]

    discovery = [o for agent, o in outputs if agent == "discovery"]
    flagged = {h["passage_id"] for o in discovery for h in o.data["hits"] if h["quality"] == "low"}
    assert flagged == {"l1:p000", "l2:p000"}
    assert any("unreliable" in u for o in discovery for u in o.uncertainties)
    assert any("(unreliable extraction)" in f for o in discovery for f in o.findings)


def test_badly_extracted_text_cannot_make_thin_evidence_look_sufficient():
    # Three documents match perfectly - but all three only on pages whose extraction is unreliable.
    result, critic, _ = go(index_of(low("l1", 1), low("l2", 2), low("l3", 3), BASE[-1]), QUESTION)
    a = critic.data["assessment"]

    assert a["sufficient"] is False and a["documents"] == 1
    assert {r["document_id"] for r in a["accepted_evidence"]} == {"d3"}
    assert {m["document_id"] for m in a["low_quality_matches"]} == {"l1", "l2", "l3"}
    assert any("only 1 accepted passage(s)" in r for r in a["reasons"])


def test_when_every_match_is_unreliable_the_negative_statement_is_not_made():
    result, critic, _ = go(index_of(low("l1", 1), low("l2", 2)), QUESTION)
    assert result.summary["evidence"] == []
    assert not any(NOVELTY in u for u in critic.uncertainties)  # a close match WAS found - just not a trustworthy one
    assert any("unreliable" in u and "not counted as evidence" in u for u in critic.uncertainties)


def test_extraction_quality_flows_from_the_pdf_all_the_way_to_the_critic(tmp_path):
    table = "conductivity " + "12.5 13.1 14.8 15.2 16.9 17.4 18.0 19.3 20.1 21.7 " * 9
    pdf = make_pdf(tmp_path / "mixed.pdf", [[("p", "Measured values follow in the next table of the supplement.")], [("p", table)]])
    with Corpus(tmp_path / "c", isolate=False) as c:
        rep = c.ingest(pdf)
        assert rep.pages_low_quality == [2]
        _, critic, _ = go(c.build_index(), "conductivity")
        a = critic.data["assessment"]
        assert [m["page_number"] for m in a["low_quality_matches"]] == [2] and a["accepted_evidence"] == []
        assert any("unreliable" in u for u in critic.uncertainties)
        assert c.verify_passage(a["low_quality_matches"][0]["passage_id"]).ok  # shown with its flag, still traceable
