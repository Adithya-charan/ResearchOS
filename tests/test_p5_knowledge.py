"""P5: the knowledge system - declared metadata (never invented), phrase / document-level / metadata search, filters, citation records,
the corpus schema upgrade, and the retrieval evaluation harness. Documents are the invented set in datasets/retrieval_eval."""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research.bm25.index import BM25Index, Passage, contains_phrase, split_phrases
from research.corpus import Corpus, SearchFilters
from research.corpus import store as corpus_store
from research.retrieval_eval import precision_at_k, recall_at_k, reciprocal_rank, run_evaluation
from tests.test_api_corpus import make_app, new_project, sign_in, upload

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "retrieval_eval"


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory):
    return run_evaluation(DATASET, tmp_path_factory.mktemp("eval") / "corpus")


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    import json
    declared = json.loads((DATASET / "corpus.json").read_text(encoding="utf-8"))
    c = Corpus(tmp_path_factory.mktemp("lib"), isolate=False)
    for f in sorted((DATASET / "docs").glob("*.md")):
        c.ingest(f, metadata=declared.get(f.name), source="eval", license="CC0-1.0")
    yield c
    c.close()


# ------------------------------------------------------------------ metrics
def test_ranking_metrics_match_hand_computed_values():
    ranked, rel = ["a", "x", "b", "y", "c"], {"a", "b", "z"}
    assert precision_at_k(ranked, rel, 1) == 1.0
    assert precision_at_k(ranked, rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, rel, 5) == pytest.approx(2 / 5)
    assert recall_at_k(ranked, rel, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, rel, 1) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "b"], rel) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], rel) == 0.0
    assert precision_at_k(["a"], {"a"}, 3) == pytest.approx(1 / 3)  # missing slots count as misses
    assert recall_at_k([], set(), 3) == 0.0


# ------------------------------------------------------------------ the evaluation harness
def test_lexical_topic_phrase_and_filtered_queries_are_all_found_on_this_set(evaluation):
    for kind in ("lexical", "topic", "phrase", "filtered"):
        a = evaluation["by_kind"][kind]
        assert a["mrr"] == 1.0 and a["recall@5"] == 1.0, kind
    assert evaluation["by_kind"]["lexical"]["precision@1"] == 1.0


def test_vocabulary_mismatch_is_a_measured_failure_not_hidden(evaluation):
    syn = evaluation["by_kind"]["synonym"]
    assert syn["queries"] == 3 and syn["mrr"] == 0.0 and syn["recall@5"] == 0.0
    assert evaluation["overall"]["mrr"] < evaluation["by_kind"]["lexical"]["mrr"]
    assert any("not meaning" in text for text in evaluation["limitations"])


def test_the_byte_identical_copy_is_deduplicated(evaluation):
    d = evaluation["deduplication"]
    assert d["files_offered"] == 13 and d["documents_stored"] == 12
    assert d["duplicates_detected"] == ["copy_of_b1.md"] and d["duplicate_points_at"] == {"copy_of_b1.md": "b1_silicon_anode.md"}
    assert not d["failed_or_rejected"]


def test_stored_metadata_equals_what_was_declared(evaluation):
    assert evaluation["metadata"]["documents_checked"] == 12 and evaluation["metadata"]["mismatches"] == []


def test_every_returned_result_cites_a_real_verifiable_passage(evaluation):
    c = evaluation["citations"]
    assert c["results_checked"] > 40 and c["resolved"] == c["results_checked"] and c["unresolved"] == []


def test_a_wrong_relevance_label_would_be_scored_as_a_miss(tmp_path, evaluation):
    """The harness must be able to fail: point a query at the wrong document and its metrics drop to zero."""
    import json
    import shutil
    d = tmp_path / "set"
    shutil.copytree(DATASET, d)
    qs = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    qs[0]["relevant"] = ["s3_legume_rotation.md"]
    (d / "queries.json").write_text(json.dumps(qs), encoding="utf-8")
    bad = run_evaluation(d, tmp_path / "corpus")
    assert bad["per_query"][0]["reciprocal_rank"] == 0.0 and bad["overall"]["mrr"] < evaluation["overall"]["mrr"]


# ------------------------------------------------------------------ phrase search
def test_phrase_words_must_be_adjacent_and_in_order():
    assert contains_phrase("Client drift grows.", "client drift")
    assert not contains_phrase("The client shows drift.", "client drift")
    assert not contains_phrase("drift client", "client drift")
    assert contains_phrase("A state-of-charge limit", "state of charge")  # punctuation is ignored
    assert split_phrases('"solid state" battery "fast charge"') == (["solid state", "fast charge"], "  battery  ")


def test_phrase_queries_filter_in_the_index():
    idx = BM25Index()
    idx.add_passages([Passage("d1:p0", "d1", "client drift grows with epochs"),
                      Passage("d2:p0", "d2", "the client reported drift in its data"),
                      Passage("d3:p0", "d3", "unrelated text about soil")])
    assert [h.passage.passage_id for h in idx.search("client drift")] == ["d1:p0", "d2:p0"]  # plain: any order, any distance
    quoted = idx.search('"client drift"')
    assert [h.passage.passage_id for h in quoted] == ["d1:p0"] and quoted[0].matched_phrases == ("client drift",)
    assert idx.search('"drift client"') == []


# ------------------------------------------------------------------ document level, metadata search, filters
def test_document_level_results_list_each_document_once_best_first(library):
    res = library.search_documents("lithium cells capacity", k=5)
    ids = [r["document"]["id"] for r in res]
    assert len(ids) == len(set(ids)) and [r["score"] for r in res] == sorted((r["score"] for r in res), reverse=True)
    titles = {r["document"]["title"] for r in res[:3]}
    assert {"Silicon anode swelling in lithium cells", "Calendar ageing of stored lithium cells",
            "Electrolyte additives for fast charging"} == titles
    assert all(r["citations"] for r in res) and all(c["document_id"] == r["document"]["id"] for r in res for c in r["citations"])


def test_metadata_search_matches_titles_authors_and_exact_dois_only(library):
    assert [r["document"]["title"] for r in library.search_metadata("Okafor")] == ["Silicon anode swelling in lithium cells"]
    by_title = library.search_metadata("federated averaging")
    assert by_title[0]["document"]["title"] == "Client drift in federated averaging" and by_title[0]["matched_fields"] == ["title"]
    assert library.search_metadata("zebra") == []
    assert library.search_metadata("text that appears only in the body: hyphae") == []  # body text is not metadata


def test_type_and_author_filters_exclude_everything_else(library):
    reports = library.search_metadata("soil biochar", filters=SearchFilters(document_types=frozenset({"report"})))
    assert {r["document"]["title"] for r in reports} == {"Biochar and soil water retention"}
    brandt = library.search("capacity cells", k=10, filters=SearchFilters(author="brandt"))
    assert {h.document.title for h in brandt} == {"Silicon anode swelling in lithium cells", "Calendar ageing of stored lithium cells"}
    assert library.search("capacity", filters=SearchFilters(document_types=frozenset({"thesis"}))) == []


def test_a_citation_carries_only_stored_facts(library):
    hit = library.search("silicon anode swelling", k=1)[0]
    c = hit.citation()
    assert set(c) == {"document_id", "chunk_id", "title", "excerpt", "page_number", "section", "score", "url", "doi", "source"}
    assert c["url"] == "https://example.org/retrieval-eval/b1_silicon_anode" and c["doi"] is None  # no DOI declared: None, not invented
    assert " ".join(hit.passage.text.split()).startswith(c["excerpt"].rstrip("."))
    assert len(c["excerpt"]) <= 303


# ------------------------------------------------------------------ declared metadata is validated and never invented
def test_undeclared_metadata_stays_empty(tmp_path):
    f = tmp_path / "plain.txt"
    f.write_text("Cells were cycled at a fixed rate and the capacity was recorded after every hundred cycles in the lab.", encoding="utf-8")
    with Corpus(tmp_path / "c", isolate=False) as c:
        assert c.ingest(f).status == "ingested"
        d = c.list_documents()[0]
        assert d.publication_date is None and d.url is None and d.document_type is None


@pytest.mark.parametrize("field,value", [("publication_date", "2021-02-30"), ("publication_date", "21-03-2020"),
                                         ("publication_date", "1500-01-01"), ("url", "javascript:alert(1)"),
                                         ("url", "ftp://example.org/x"), ("url", "https://exa mple.org"),
                                         ("document_type", "blogpost")])
def test_invalid_declared_metadata_is_rejected(tmp_path, field, value):
    f = tmp_path / "doc.txt"
    f.write_text("Cells were cycled at a fixed rate and the capacity was recorded after every hundred cycles in the lab.", encoding="utf-8")
    with Corpus(tmp_path / "c", isolate=False) as c:
        rep = c.ingest(f, metadata={field: value})
        assert rep.status == "rejected" and rep.code == "invalid_metadata" and c.list_documents() == []


def test_a_declared_publication_date_sets_the_year_and_labels_its_origin(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Cells were cycled at a fixed rate and the capacity was recorded after every hundred cycles in the lab.", encoding="utf-8")
    with Corpus(tmp_path / "c", isolate=False) as c:
        c.ingest(f, metadata={"publication_date": "2021-03"})
        d = c.list_documents()[0]
        assert (d.publication_date, d.year, d.year_source) == ("2021-03", 2021, "user")


# ------------------------------------------------------------------ corpus schema upgrade
def test_a_library_created_before_the_new_columns_is_upgraded_in_place(tmp_path):
    db = tmp_path / "corpus.db"
    c = sqlite3.connect(db)
    old_schema = corpus_store.SCHEMA.replace(",\n    publication_date TEXT, url TEXT, document_type TEXT", "")
    assert old_schema != corpus_store.SCHEMA
    c.executescript(old_schema)
    c.execute("INSERT INTO schema_info VALUES (1)")
    c.execute("INSERT INTO documents VALUES ('doc-old','h1','old.txt','txt',10,'Old','filename','[]',2019,'user',NULL,1,'x','ok','{}','[]',"
              "NULL,NULL,'[\"search\"]',1,'2026-01-01T00:00:00','orig/old.txt')")
    c.commit()
    c.close()
    s = corpus_store.CorpusStore(db)
    d = s.get_document("doc-old")
    assert d.title == "Old" and d.year == 2019 and d.publication_date is None and d.url is None and d.document_type is None
    assert s._one("SELECT version FROM schema_info")[0] == corpus_store.SCHEMA_VERSION
    s.close()
    corpus_store.CorpusStore(db).close()  # reopening is a no-op


def test_a_library_written_by_a_newer_version_is_refused(tmp_path):
    db = tmp_path / "corpus.db"
    corpus_store.CorpusStore(db).close()
    c = sqlite3.connect(db)
    c.execute("UPDATE schema_info SET version=99")
    c.commit()
    c.close()
    with pytest.raises(RuntimeError, match="newer version"):
        corpus_store.CorpusStore(db)


# ------------------------------------------------------------------ the HTTP surface
def test_upload_declared_metadata_and_search_at_every_level(tmp_path):
    with TestClient(make_app(tmp_path), base_url="http://testserver") as c:
        h = sign_in(c)
        pid = new_project(c, h)
        for f in ("b1_silicon_anode.md", "c1_bleaching.md"):
            body = (DATASET / "docs" / f).read_bytes()
            r = upload(c, h, pid, f, body, title="Declared title " + f[:2], authors="A. Okafor; L. Brandt", publication_date="2021-03-14",
                       url="https://example.org/x/" + f[:2], document_type="article", source="test")
            assert r.status_code == 201, r.text
        doc = r.json()["document"]
        assert (doc["publication_date"], doc["url"], doc["document_type"], doc["ingestion_status"]) == \
            ("2021-03-14", "https://example.org/x/c1", "article", "ingested")
        assert doc["authors"] == ["A. Okafor", "L. Brandt"] and doc["file_hash"] and doc["created_at"] and doc["doi"] is None
        chunk = c.get(f"/projects/{pid}/search", params={"q": "silicon anode swelling"}).json()
        assert chunk["level"] == "chunk" and chunk["hits"][0]["citation"]["url"] == "https://example.org/x/b1"
        assert set(chunk["hits"][0]["citation"]) >= {"document_id", "chunk_id", "title", "excerpt", "page_number", "score", "url"}
        phrase = c.get(f"/projects/{pid}/search", params={"q": '"heating weeks"'}).json()
        assert [x["document_id"] for x in phrase["hits"]] and all(x["matched_phrases"] == ["heating weeks"] for x in phrase["hits"])
        assert c.get(f"/projects/{pid}/search", params={"q": '"weeks heating"'}).json()["hits"] == []
        docs = c.get(f"/projects/{pid}/search", params={"q": "bleaching silicon", "level": "document"}).json()
        assert docs["level"] == "document" and len({x["document"]["id"] for x in docs["hits"]}) == 2
        meta = c.get(f"/projects/{pid}/search", params={"q": "Okafor", "level": "metadata"}).json()
        assert len(meta["hits"]) == 2 and meta["hits"][0]["matched_fields"] == ["authors"]
        assert c.get(f"/projects/{pid}/search", params={"q": "bleaching", "document_type": "report"}).json()["hits"] == []
        assert upload(c, h, pid, "x.md", b"# T\n\nSome text about nothing in particular that is long enough to index.",
                      publication_date="2021-13-40").status_code == 422
