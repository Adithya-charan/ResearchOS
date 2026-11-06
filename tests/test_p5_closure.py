"""P5 closure checks: deletion and index cleanup, persistence across restarts, combined filters, duplicate handling, evaluation integrity
and reproducibility, agent integration with the retrieval system, and project authorization on every search level."""
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_engine.contracts import AgentOutput, RunLimits, RunStatus
from agent_engine.registry.defaults import build_engine
from agent_engine.state.store import Store
from research.bm25.analysis import analyze
from research.corpus import Corpus, SearchFilters
from research.retrieval_eval import run_evaluation
from tests.conftest_agents import run
from tests.test_api_corpus import make_app, new_project, sign_in, upload

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "retrieval_eval"
DECLARED = json.loads((DATASET / "corpus.json").read_text(encoding="utf-8"))
QUERIES = json.loads((DATASET / "queries.json").read_text(encoding="utf-8"))


def load(root: Path, names=None) -> Corpus:
    c = Corpus(root, isolate=False)
    for f in sorted((DATASET / "docs").glob("*.md")):
        if names is None or f.name in names:
            c.ingest(f, metadata=DECLARED.get(f.name), source="eval", license="CC0-1.0")
    return c


@pytest.fixture()
def lib(tmp_path):
    c = load(tmp_path / "lib")
    yield c
    c.close()


def by_name(c, filename):
    return next(d for d in c.list_documents() if d.filename == filename)


def result_ids(c, query, **kw):
    return {h.passage.document_id for h in c.search(query, k=20, **kw)}


# ------------------------------------------------------------------ duplicates
def test_duplicate_handling_by_content_not_by_name(lib, tmp_path):
    assert lib.stats()["documents"] == 12  # the byte-identical copy was not stored a second time
    original = by_name(lib, "b1_silicon_anode.md")
    again = lib.ingest(DATASET / "docs" / "b1_silicon_anode.md")
    renamed = tmp_path / "renamed_copy.md"
    shutil.copyfile(DATASET / "docs" / "b1_silicon_anode.md", renamed)
    assert (again.status, again.document_id) == ("duplicate", original.id)
    assert lib.ingest(renamed).document_id == original.id and lib.stats()["documents"] == 12
    edited = tmp_path / "edited.md"
    edited.write_text((DATASET / "docs" / "b1_silicon_anode.md").read_text(encoding="utf-8") + "\nAn extra closing sentence about swelling.\n",
                      encoding="utf-8")
    assert lib.ingest(edited).status == "ingested" and lib.stats()["documents"] == 13  # different bytes = a different document
    assert len({h.passage.passage_id for h in lib.search("silicon anode swelling", k=50)}) == len(lib.search("silicon anode swelling", k=50))


# ------------------------------------------------------------------ deletion and index cleanup
def test_deleting_a_document_removes_it_from_every_kind_of_search_and_from_storage(lib):
    doc = by_name(lib, "c1_bleaching.md")
    before = lib.index.num_passages
    n_passages = len(lib.store.document_passages(doc.id))
    assert doc.id in result_ids(lib, "bleaching degree heating weeks") and lib.search_metadata("Marlowe")
    stored_file = lib.root / doc.stored_path
    assert stored_file.exists()
    pid = lib.store.document_passages(doc.id)[0].passage_id

    assert lib.remove_document(doc.id) is True
    assert lib.remove_document(doc.id) is False  # idempotent: already gone

    assert doc.id not in result_ids(lib, "bleaching degree heating weeks")
    assert all(r["document"]["id"] != doc.id for r in lib.search_documents("bleaching heating weeks", k=20))
    assert all(r["document"]["id"] != doc.id for r in lib.search_metadata("Thermal bleaching thresholds fringing reef"))
    assert lib.search('"heating weeks"') == []  # phrase search no longer finds it
    assert lib.index.num_passages == before - n_passages and not lib.index.has_document(doc.id)
    assert lib.get_document(doc.id) is None and lib.store.get_passage(pid) is None  # passages cascade
    assert not stored_file.exists()  # the stored original is gone too
    assert lib.inspect(pid) is None and lib.verify_passage(pid).ok is False  # a citation to it no longer resolves
    # and the same file can be ingested again afterwards, as a fresh document
    assert lib.ingest(DATASET / "docs" / "c1_bleaching.md", metadata=DECLARED["c1_bleaching.md"]).status == "ingested"
    assert doc.id in result_ids(lib, "bleaching degree heating weeks")


# ------------------------------------------------------------------ persistence after restart
def test_a_reopened_library_returns_identical_results_and_still_verifies_citations(tmp_path):
    root = tmp_path / "lib"
    first = load(root)
    queries = ["silicon anode swelling", '"client drift"', "lithium cells capacity", "biochar water holding capacity"]
    snapshot = {q: [(h.passage.passage_id, round(h.score, 6)) for h in first.search(q, k=10)] for q in queries}
    docs_before = [(d.id, d.title, d.authors, d.publication_date, d.url, d.document_type, d.year) for d in first.list_documents()]
    metadata_before = first.search_metadata("Brandt")
    doc_level_before = [r["document"]["id"] for r in first.search_documents("lithium cells capacity", k=5)]
    first.close()

    second = Corpus(root, isolate=False)  # a new process opening the same files
    try:
        assert {q: [(h.passage.passage_id, round(h.score, 6)) for h in second.search(q, k=10)] for q in queries} == snapshot
        assert [(d.id, d.title, d.authors, d.publication_date, d.url, d.document_type, d.year) for d in second.list_documents()] == docs_before
        assert second.search_metadata("Brandt") == metadata_before
        assert [r["document"]["id"] for r in second.search_documents("lithium cells capacity", k=5)] == doc_level_before
        for hit in second.search("silicon anode swelling", k=5):
            assert second.verify_passage(hit.passage.passage_id).ok
        assert second.ingest(DATASET / "docs" / "b1_silicon_anode.md").status == "duplicate"  # dedup survives restart
    finally:
        second.close()


# ------------------------------------------------------------------ filters and levels
def test_combined_metadata_filters_are_all_applied_together(lib):
    f = SearchFilters(document_types=frozenset({"article"}), author="Brandt", year_min=2021)
    assert {lib.get_document(i).filename for i in result_ids(lib, "lithium cells capacity", filters=f)} == {"b1_silicon_anode.md"}
    assert result_ids(lib, "lithium cells capacity", filters=SearchFilters(document_types=frozenset({"article"}), author="Brandt", year_min=2022)) == set()
    assert result_ids(lib, "lithium cells capacity", filters=SearchFilters(document_types=frozenset({"report"}), author="Okafor")) == set()
    both = lib.search_documents("lithium cells capacity", filters=f)
    assert [r["document"]["title"] for r in both] == ["Silicon anode swelling in lithium cells"]
    meta = lib.search_metadata("Brandt", filters=SearchFilters(document_types=frozenset({"report"})))
    assert [r["document"]["title"] for r in meta] == ["Calendar ageing of stored lithium cells"]
    phrase = lib.search('"capacity fell"', filters=SearchFilters(author="Okafor"))
    assert phrase and all(h.document.filename == "b1_silicon_anode.md" for h in phrase)


def test_chunk_level_hits_belong_to_their_documents_and_carry_page_and_section(lib):
    for hit in lib.search("client drift local epochs", k=5):
        c = hit.citation()
        assert lib.get_document(c["document_id"]) is not None and lib.store.get_passage(c["chunk_id"]) is not None
        assert c["chunk_id"].startswith(c["document_id"]) and c["title"] == hit.document.title and c["section"] == hit.passage.section
        assert c["page_number"] == hit.passage.page_number  # None for page-less formats such as Markdown: not invented


# ------------------------------------------------------------------ evaluation integrity
def test_the_evaluation_is_reproducible_run_to_run(tmp_path):
    a = run_evaluation(DATASET, tmp_path / "a")
    b = run_evaluation(DATASET, tmp_path / "b")
    for key in ("overall", "by_kind", "per_query", "deduplication", "metadata", "citations", "dataset"):
        assert a[key] == b[key], key


def test_synonym_queries_share_no_stems_with_their_target_documents_and_other_kinds_do():
    for case in QUERIES:
        target_stems = set()
        for fname in case["relevant"]:
            text = (DATASET / "docs" / fname).read_text(encoding="utf-8")
            target_stems |= set(analyze(text)) | set(analyze(" ".join(DECLARED[fname]["authors"])))
        overlap = set(analyze(case["query"].replace('"', " "))) & target_stems
        if case["kind"] == "synonym":
            assert overlap == set(), f"{case['id']} shares stems {overlap}: it would not be a vocabulary-mismatch query"
        else:
            assert overlap, f"{case['id']} ({case['kind']}) shares no vocabulary with its target"


def test_all_five_categories_are_reported_separately_with_the_requested_metrics(tmp_path):
    rep = run_evaluation(DATASET, tmp_path / "e")
    assert set(rep["by_kind"]) == {"lexical", "topic", "phrase", "filtered", "synonym"}
    for kind, agg in rep["by_kind"].items():
        for metric in ("mrr", "precision@1", "precision@3", "precision@5", "recall@1", "recall@3", "recall@5"):
            assert metric in agg, (kind, metric)
    assert rep["by_kind"]["topic"]["recall@1"] == pytest.approx(1 / 3, abs=1e-3)  # three relevant documents: rank 1 holds one (rounded to 4 d.p.)
    assert rep["by_kind"]["topic"]["precision@3"] == 1.0
    assert sum(a["queries"] for a in rep["by_kind"].values()) == rep["overall"]["queries"] == 22
    assert any("not meaning" in line for line in rep["limitations"])


# ------------------------------------------------------------------ integration with the agents
@pytest.fixture(scope="module")
def agent_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("agentlib")
    c = load(root / "lib")
    # the evaluation documents state no limitations, so add one document that does: gap and hypothesis cite evidence only when it exists
    extra = root / "drift_limits.md"
    extra.write_text("# Limits of client drift studies\n\nAbstract\n\nClient drift under many local epochs is studied here in simulation.\n\n"
                     "Limitations\n\nThe study of client drift was limited to simulated clients and may not hold for real federated deployments. "
                     "Local epochs above twenty were not examined.\n", encoding="utf-8")
    assert c.ingest(extra, metadata={"title": "Limits of client drift studies", "authors": ["Q. Test"], "publication_date": "2024-01-01",
                                     "document_type": "article"}).status == "ingested"

    def catalog(doc_id):
        d = c.get_document(doc_id)
        return {"title": d.title, "year": d.year, "doi": d.doi, "license": d.license}

    def verifier(pid):
        v = c.verify_passage(pid)
        return v.ok, v.message

    store = Store()
    engine = build_engine(store, c.build_index(), workflow="full", catalog=catalog, verifier=verifier, poll_interval=0.01)
    q = "How does client drift affect federated accuracy as local epochs grow?"
    rid = engine.create_run(engine.create_project("p5", q), q, RunLimits(retry_backoff_s=0, task_timeout_s=10, max_runtime_s=60))
    result = run(engine.run(rid))
    yield c, store, rid, result
    c.close()


def test_discovery_searches_through_the_retrieval_system(agent_run):
    c, store, rid, result = agent_run
    assert result.status == RunStatus.COMPLETED
    calls = [e for e in store.list_events(rid) if e["type"] == "tool_call" and e["payload"].get("tool") == "corpus.search"]
    assert calls and all(e["payload"]["agent"] == "discovery" for e in calls)
    disc = [t for t in store.list_tasks(rid) if t["agent_type"] == "discovery"]
    hits = [h for t in disc for h in t["output"]["data"]["hits"]]
    assert hits and all(c.store.get_passage(h["passage_id"]) is not None for h in hits)  # every hit is a stored passage
    top = {h["document_id"] for h in hits}
    assert c.get_document(next(iter(top))) is not None and by_name(c, "f1_client_drift.md").id in top


def test_every_agents_evidence_resolves_to_a_real_passage_that_verifies(agent_run):
    c, store, rid, _ = agent_run
    checked = {}
    for t in store.list_tasks(rid):
        if t["status"] != "succeeded":
            continue
        out = AgentOutput.model_validate(t["output"])
        for ref in out.evidence:
            assert ref.passage_id and c.get_document(ref.document_id) is not None, (t["agent_type"], ref)
            assert c.verify_passage(ref.passage_id).ok, (t["agent_type"], ref)
            checked.setdefault(t["agent_type"], 0)
            checked[t["agent_type"]] += 1
    # the stages that must be evidence-backed all cited something
    assert {"discovery", "critic", "literature", "gap", "hypothesis", "verifier", "synthesis"} <= set(checked)


def test_literature_statements_gap_and_hypothesis_evidence_trace_to_retrieved_passages(agent_run):
    c, store, rid, _ = agent_run
    tasks = {t["agent_type"]: t for t in store.list_tasks(rid) if t["status"] == "succeeded"}
    retrieved = {a["passage_id"] for a in tasks["critic"]["output"]["data"]["assessment"]["accepted"]}
    docs_retrieved = {a["document_id"] for a in tasks["critic"]["output"]["data"]["assessment"]["accepted"]}
    for s in tasks["literature"]["output"]["data"]["statements"]:
        assert s["document_id"] in docs_retrieved  # analysis reads only documents the retrieval step accepted
        p = c.store.get_passage(s["passage_id"])[0]
        assert " ".join(s["text"].split()) in " ".join(p.text.split())  # a verbatim quotation of the stored passage
    lit_passages = {s["passage_id"] for s in tasks["literature"]["output"]["data"]["statements"]}
    for g in tasks["gap"]["output"]["data"]["gaps"]:
        for ref in g["evidence"]:
            assert ref["passage_id"] in lit_passages | retrieved
    for h in tasks["hypothesis"]["output"]["data"]["hypotheses"]:
        for ref in h["evidence"]:
            assert ref["passage_id"] in lit_passages | retrieved
    claims = tasks["verifier"]["output"]["data"]["claims"]
    extracted = [cl for cl in claims if cl["type"] == "extracted"]
    assert extracted and all(cl["checks"]["passage_exists"] for cl in extracted)  # the verifier resolved each citation itself


def test_the_reviewer_works_from_resolvable_evidence(agent_run):
    c, store, rid, _ = agent_run
    reviewer = next(t for t in store.list_tasks(rid) if t["agent_type"] == "reviewer")
    assert reviewer["status"] == "succeeded"
    plans = next(t for t in store.list_tasks(rid) if t["agent_type"] == "experiment_design")["output"]["data"]["plans"]
    for p in plans:
        for item in p["datasets"] + p["baselines"] + p["metrics"]:
            assert c.verify_passage(item["citation"]["passage_id"]).ok  # a plan that cites evidence cites something that exists


# ------------------------------------------------------------------ authorization on every search level
@pytest.mark.parametrize("params", [{"q": "silicon anode"}, {"q": "silicon anode", "level": "document"},
                                    {"q": "Okafor", "level": "metadata"}, {"q": '"silicon anode"'},
                                    {"q": "silicon", "document_type": "article", "author": "Okafor"}])
def test_search_enforces_login_and_project_ownership_at_every_level(tmp_path, params):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        alice = sign_in(c, "alice@example.org")
        pid = new_project(c, alice)
        assert upload(c, alice, pid, "b1.md", (DATASET / "docs" / "b1_silicon_anode.md").read_bytes()).status_code == 201
        assert c.get(f"/projects/{pid}/search", params=params).status_code == 200
        c.post("/auth/logout", headers=alice)
        assert c.get(f"/projects/{pid}/search", params=params).status_code == 401  # not signed in
        mallory = sign_in(c, "mallory@example.org")
        r = c.get(f"/projects/{pid}/search", params=params)
        assert r.status_code == 404 and "silicon" not in r.text.lower()  # someone else's project looks missing
        own = new_project(c, mallory)
        assert upload(c, mallory, own, "s1.md", (DATASET / "docs" / "s1_mycorrhizae.md").read_bytes()).status_code == 201
        mine = c.get(f"/projects/{own}/search", params={"q": "silicon anode swelling", "level": "chunk"}).json()
        assert mine["hits"] == []  # her library does not contain Alice's documents


def test_upload_and_document_reads_enforce_ownership_too(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        alice = sign_in(c, "alice@example.org")
        pid = new_project(c, alice)
        doc = upload(c, alice, pid, "b1.md", (DATASET / "docs" / "b1_silicon_anode.md").read_bytes(), document_type="article").json()["document"]
        c.post("/auth/logout", headers=alice)
        mallory = sign_in(c, "mallory@example.org")
        assert c.get(f"/projects/{pid}/documents/{doc['id']}").status_code == 404
        assert c.get(f"/projects/{pid}/documents").status_code == 404
        assert upload(c, mallory, pid, "x.md", b"# T\n\nAn attempt to add a document to someone else's project, long enough to index.").status_code == 404
        assert c.delete(f"/projects/{pid}/documents/{doc['id']}", headers=mallory).status_code == 404
