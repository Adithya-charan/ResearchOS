"""Phase 7 exit condition: users can ask questions about their own documents and inspect the supporting passages.

Real PDFs go in; searchable, page-level, verifiable evidence comes out. Covers ingestion records, retrieval accuracy,
persistence, metadata filters and source permissions, honest handling of files that cannot be read (scans, encrypted,
corrupt), and citation verification - including what happens when the stored data or the stored original is tampered with.
"""
import hashlib
import re
import sqlite3

import pytest

pymupdf = pytest.importorskip("pymupdf")

from research.bm25.analysis import stem  # noqa: E402
from research.corpus import Corpus, SearchFilters  # noqa: E402
from research.corpus.models import Block, ExtractedDocument, Page  # noqa: E402
from research.corpus.segment import squash  # noqa: E402
from research.corpus.store import sha256_text  # noqa: E402
from tests.corpus_fixtures import PAPERS, build_library  # noqa: E402
from tests.pdfgen import make_pdf  # noqa: E402

LICENSES = {"electrolytes": "CC-BY-4.0", "sparse": "CC-BY-4.0", "coral": "CC0-1.0", "federated": "proprietary",
            "perovskite": "CC-BY-4.0"}
YEARS = {"electrolytes": 2021, "sparse": 2019, "coral": 2020, "federated": 2022, "perovskite": 2018}
QUERIES = {
    "electrolytes": "ionic conductivity of sulfide glass-ceramic electrolytes",
    "sparse": "block-sparse attention for long documents on a consumer GPU",
    "coral": "coral bleaching mortality and marine heat stress",
    "federated": "differential privacy noise and clipping in federated learning",
    "perovskite": "moisture damp heat encapsulation of perovskite cells",
}


def ingest_library(corpus: Corpus, library: dict) -> dict:
    ids = {}
    for key, path in library.items():
        report = corpus.ingest(path, source="invented test fixtures", license=LICENSES[key])
        assert report.status == "ingested", report
        ids[key] = report.document_id
    return ids


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("pdfs"))


@pytest.fixture(scope="module")
def shared(tmp_path_factory, library):
    """One ingested corpus for the read-only tests (anything that changes data uses `fresh`)."""
    corpus = Corpus(tmp_path_factory.mktemp("shared-corpus"), isolate=False)
    yield corpus, ingest_library(corpus, library)
    corpus.close()


@pytest.fixture
def fresh(tmp_path, library):
    corpus = Corpus(tmp_path / "corpus", isolate=False)
    yield corpus, ingest_library(corpus, library)
    corpus.close()


# ================================================================== what is stored
def test_ingestion_stores_everything_needed_to_trust_a_passage_later(shared, library):
    corpus, ids = shared
    path = library["electrolytes"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rec = corpus.get_document(ids["electrolytes"])

    assert rec.id == f"doc-{digest[:12]}" and rec.file_hash == digest and rec.size_bytes == path.stat().st_size
    assert (rec.file_type, rec.page_count, rec.filename) == ("pdf", 3, "electrolytes.pdf")
    assert rec.extraction_method.startswith("pymupdf-") and rec.extraction_status == "ok" and rec.quality["status"] == "ok"
    assert (rec.source, rec.license, rec.allowed_use) == ("invented test fixtures", "CC-BY-4.0", ["search", "quote"])
    assert rec.stored_path == f"files/{digest[:2]}/{digest}.pdf"
    assert (corpus.root / rec.stored_path).read_bytes() == path.read_bytes()  # a private, byte-identical copy

    # every passage row carries: document id, page, section, text, checksum, extraction method, quality
    rows = corpus.store._conn.execute("SELECT * FROM passages WHERE document_id=? ORDER BY ordinal", (rec.id,)).fetchall()
    assert len(rows) >= 6
    for i, row in enumerate(rows):
        assert row["passage_id"] == f"{rec.id}:p{i:03d}" and row["ordinal"] == i
        assert 1 <= row["page_number"] <= 3 and row["text"].strip() and row["quality"] == "ok"
        assert row["text_sha256"] == sha256_text(row["text"]) and row["extraction_method"] == rec.extraction_method


def test_metadata_fields_say_where_they_came_from(shared):
    corpus, ids = shared
    got = {k: corpus.get_document(v) for k, v in ids.items()}
    assert {k: (d.year, d.year_source) for k, d in got.items()} == {
        "electrolytes": (2021, "file_created_date"), "sparse": (2019, "first_page_text"), "coral": (2020, "file_created_date"),
        "federated": (2022, "first_page_text"), "perovskite": (2018, "file_created_date")}
    assert {k: d.title for k, d in got.items()} == {k: v["title"] for k, v in PAPERS.items()}
    assert got["coral"].title_source == "file_metadata"  # declared by the file
    assert all(d.title_source == "first_page_largest_text" for k, d in got.items() if k != "coral")  # read off the page
    assert got["electrolytes"].doi == "10.5555/sse.2021.014" and all(d.doi is None for k, d in got.items() if k != "electrolytes")
    assert got["coral"].authors == ["Lena Park", "Ahmed Rahimi", "Sofia Duarte"] and got["sparse"].authors == []


def test_every_passage_really_is_on_its_cited_page_of_the_original_pdf(shared, library):
    """Ground truth independent of the pipeline: open each PDF directly with PyMuPDF."""
    corpus, ids = shared
    for key, doc_id in ids.items():
        with pymupdf.open(str(library[key])) as pdf:
            pages = {i + 1: squash(pdf[i].get_text()) for i in range(pdf.page_count)}
        passages = corpus.store.document_passages(doc_id)
        assert passages
        for p in passages:
            assert squash(p.text) in pages[p.page_number], (key, p.passage_id, p.page_number)


def test_running_headers_page_numbers_and_hyphenation_are_cleaned(shared):
    corpus, ids = shared
    for doc_id in ids.values():
        text = " ".join(p.text for p in corpus.store.document_passages(doc_id))
        assert "Journal of Invented Results" not in text
    electro = corpus.store.document_passages(ids["electrolytes"])
    glove = next(p for p in electro if "glove box" in p.text)
    assert "sulfide electrolyte was handled in an argon glove box" in glove.text and "electro-" not in glove.text


def test_sections_and_pages_follow_the_document_structure(shared):
    corpus, ids = shared
    by_section = {}
    for p in corpus.store.document_passages(ids["electrolytes"]):
        by_section.setdefault(p.section, set()).add(p.page_number)
    assert by_section["Abstract"] == {1} and by_section["1. Introduction"] == {1}
    assert by_section["2. Methods"] == {2} and by_section["3. Results"] == {2} and by_section["4. Conclusion"] == {3}
    assert by_section["Solid-State Electrolytes for Lithium Batteries"] == {1}  # the title block's own text (authors, DOI)


# ================================================================== asking questions
@pytest.mark.parametrize("key", list(QUERIES))
def test_a_question_finds_the_right_document_first(shared, key):
    corpus, ids = shared
    hits = corpus.search(QUERIES[key], k=5)
    assert hits and hits[0].passage.document_id == ids[key]
    assert hits[0].document.title == PAPERS[key]["title"] and hits[0].matched_terms


def test_hits_cite_the_exact_page_and_section(shared):
    corpus, ids = shared
    top = corpus.search("short circuits when the stack pressure fell below two megapascals", k=1)[0].passage
    assert (top.document_id, top.page_number, top.section) == (ids["electrolytes"], 2, "3. Results")
    top = corpus.search("global tokens act as a memory that summarises distant blocks", k=1)[0].passage
    assert (top.document_id, top.page_number, top.section) == (ids["sparse"], 3, "4. Discussion")


def test_a_question_with_no_support_gets_no_hits(shared):
    corpus, _ = shared
    assert corpus.search("quantum entanglement between neutron stars", k=5) == []
    assert corpus.search("the of and", k=5) == []  # stop words only


def test_search_is_deterministic_and_ranked(shared):
    corpus, _ = shared
    a = corpus.search("cells hours temperature percent", k=10)
    b = corpus.search("cells hours temperature percent", k=10)
    assert [(h.passage.passage_id, h.score) for h in a] == [(h.passage.passage_id, h.score) for h in b]
    assert [h.score for h in a] == sorted((h.score for h in a), reverse=True)


# ================================================================== persistence
def test_the_corpus_survives_a_restart_and_answers_identically(tmp_path, library):
    root = tmp_path / "corpus"
    first = Corpus(root, isolate=False)
    ids = ingest_library(first, library)
    before = {k: [(h.passage.passage_id, h.score) for h in first.search(q, k=8)] for k, q in QUERIES.items()}
    stats, docs = first.stats(), first.list_documents()
    first.close()

    again = Corpus(root, isolate=False)  # the index is rebuilt from the stored passages
    assert again.stats() == stats and again.list_documents() == docs
    assert {k: [(h.passage.passage_id, h.score) for h in again.search(q, k=8)] for k, q in QUERIES.items()} == before
    assert all(again.verify_passage(p.passage_id).ok for p in again.store.document_passages(ids["coral"]))
    again.close()


# ================================================================== duplicates
def test_the_same_file_is_never_ingested_twice(fresh, library, tmp_path):
    corpus, ids = fresh
    before = corpus.stats()
    again = corpus.ingest(library["coral"])
    assert (again.status, again.document_id) == ("duplicate", ids["coral"]) and again.ok
    copy = tmp_path / "renamed-copy.pdf"
    copy.write_bytes(library["coral"].read_bytes())
    assert corpus.ingest(copy).status == "duplicate"  # identity is the content hash, not the file name
    assert corpus.stats() == before


def test_force_replaces_a_document_and_updates_its_metadata(fresh, library):
    corpus, ids = fresh
    before = corpus.stats()
    report = corpus.ingest(library["coral"], license="CC-BY-NC-4.0", source="re-checked", force=True)
    assert (report.status, report.document_id) == ("ingested", ids["coral"])
    rec = corpus.get_document(ids["coral"])
    assert (rec.license, rec.source) == ("CC-BY-NC-4.0", "re-checked")
    assert corpus.stats() == before  # replaced, not doubled
    assert corpus.search(QUERIES["coral"], k=1)[0].document.license == "CC-BY-NC-4.0"
    assert len(corpus.index._passages) == before["passages"]


# ================================================================== filters and permissions
BROAD = "cells hours temperature percent training accuracy mortality"


def hit_docs(corpus, **filters):
    return {h.document.id for h in corpus.search(BROAD, k=100, filters=SearchFilters(**filters))}


def test_metadata_filters(fresh):
    corpus, ids = fresh
    everything = hit_docs(corpus)
    assert everything == set(ids.values())
    assert hit_docs(corpus, year_min=2020) == {ids[k] for k in ("electrolytes", "coral", "federated")}
    assert hit_docs(corpus, year_max=2019) == {ids[k] for k in ("sparse", "perovskite")}
    assert hit_docs(corpus, year_min=2019, year_max=2020) == {ids[k] for k in ("sparse", "coral")}
    assert hit_docs(corpus, licenses=frozenset({"CC0-1.0"})) == {ids["coral"]}
    assert hit_docs(corpus, licenses=frozenset({"CC0-1.0", "proprietary"})) == {ids["coral"], ids["federated"]}
    assert hit_docs(corpus, file_types=frozenset({"pdf"})) == everything and hit_docs(corpus, file_types=frozenset({"docx"})) == set()
    assert hit_docs(corpus, document_ids=frozenset({ids["sparse"]})) == {ids["sparse"]}
    assert hit_docs(corpus, document_ids=frozenset()) == set()
    assert hit_docs(corpus, year_min=2020, licenses=frozenset({"CC-BY-4.0"})) == {ids["electrolytes"]}  # filters combine (AND)
    assert hit_docs(corpus, year_min=2030) == set()


def test_documents_with_an_unknown_year_never_match_a_year_filter(fresh, tmp_path):
    corpus, ids = fresh
    note = tmp_path / "note.txt"
    note.write_text("Abstract\n\nMortality of cells and temperature records have no date attached to them at all.\n")
    nid = corpus.ingest(note).document_id
    assert corpus.get_document(nid).year is None
    assert nid in hit_docs(corpus) and nid not in hit_docs(corpus, year_min=1900) and nid not in hit_docs(corpus, year_max=2100)
    assert hit_docs(corpus, file_types=frozenset({"txt"})) == {nid}


def test_source_permissions_decide_what_is_searchable(fresh, tmp_path):
    corpus, ids = fresh
    private = tmp_path / "private.txt"
    private.write_text("Abstract\n\nConfidential perovskite encapsulation results from an unpublished industrial collaboration.\n")
    rep = corpus.ingest(private, allowed_use=("quote",))
    assert rep.status == "ingested"
    rec = corpus.get_document(rep.document_id)
    assert rec.allowed_use == ["quote"] and not rec.searchable
    assert corpus.stats()["documents"] == 6 and corpus.stats()["searchable_documents"] == 5
    assert all(h.document.id != rec.id for h in corpus.search("confidential perovskite encapsulation unpublished", k=20))
    assert rec.id not in {p.document_id for p in corpus.build_index()._passages}
    with pytest.raises(ValueError):
        corpus.ingest(private, allowed_use=("search", "publish"))


def test_permissions_survive_a_restart(tmp_path):
    root = tmp_path / "c"
    note = tmp_path / "private.txt"
    note.write_text("Abstract\n\nConfidential unpublished industrial collaboration data about solid electrolytes.\n")
    with Corpus(root, isolate=False) as c:
        c.ingest(note, allowed_use=("quote",))
    with Corpus(root, isolate=False) as c:
        assert c.index.num_passages == 0 and c.search("confidential unpublished collaboration electrolytes") == []
        assert c.stats()["documents"] == 1


# ================================================================== files that cannot be read
def test_encrypted_pdf_is_rejected_and_leaves_nothing_behind(tmp_path):
    with Corpus(tmp_path / "c", isolate=False) as c:
        locked = make_pdf(tmp_path / "locked.pdf", [[("p", "secret contents that nobody may read without the key")]], password="pw")
        rep = c.ingest(locked)
        assert (rep.status, rep.code) == ("rejected", "encrypted") and "password" in rep.message
        assert c.stats()["documents"] == 0 and not [f for f in c.files_dir.rglob("*") if f.is_file()]


def test_corrupt_files_and_wrong_types_are_rejected_with_reasons(tmp_path):
    with Corpus(tmp_path / "c", isolate=False, max_bytes=5000) as c:
        def attempt(name, data):
            p = tmp_path / name
            p.write_bytes(data)
            return c.ingest(p)

        assert attempt("broken.pdf", b"%PDF-1.7\n" + bytes(range(256)) * 4).code == "corrupt"
        assert attempt("fake.pdf", b"plain text pretending to be a pdf").code == "type_mismatch"
        assert attempt("empty.pdf", b"").code == "empty"
        assert attempt("big.pdf", b"%PDF-1.7\n" + b"0" * 6000).code == "too_large"
        assert attempt("program.exe", b"MZ\x90\x00").code == "unsupported_type"
        assert c.ingest(tmp_path / "does-not-exist.pdf").code == "corrupt"
        assert c.stats()["documents"] == 0 and not [f for f in c.files_dir.rglob("*") if f.is_file()]


def test_a_scanned_pdf_is_reported_honestly_not_indexed_as_empty(tmp_path):
    scan = [("scan", [("p", "pixels only: this page is a picture of text and has no text layer at all")])] * 2
    with Corpus(tmp_path / "c", isolate=False) as c:
        rep = c.ingest(make_pdf(tmp_path / "scan.pdf", scan))
        assert (rep.status, rep.code) == ("rejected", "no_text") and rep.pages_without_text == [1, 2]
        assert "OCR is not available" in rep.message
        assert c.stats()["documents"] == 0 and not [f for f in c.files_dir.rglob("*") if f.is_file()]


def test_a_mixed_pdf_keeps_what_is_readable_and_flags_the_rest(tmp_path):
    table = ("conductivity " + "12.5 13.1 14.8 15.2 16.9 17.4 18.0 19.3 20.1 21.7 " * 9)
    pages = [[("h1", "Overview"), ("p", "Sulfide electrolyte conductivity improves with annealing temperature in these glass "
                                          "ceramics, and the samples stayed stable during extended cycling tests.")],
             [("p", table)],
             ("scan", [("p", "a scanned appendix page that is only an image without any text layer")]),
             "blank",
             [("p", "Closing remarks on pressure dependence of the interfacial resistance in pressed pellet cells.")]]
    with Corpus(tmp_path / "c", isolate=False) as c:
        rep = c.ingest(make_pdf(tmp_path / "mixed.pdf", pages))
        assert rep.status == "ingested" and rep.pages == 5 and rep.quality_status == "partial"
        assert rep.pages_without_text == [3] and rep.pages_low_quality == [2]
        assert any("pages 3 have no text layer" in w and "OCR is not available" in w for w in rep.warnings)
        assert any("pages 2 have unreliable" in w for w in rep.warnings)
        assert not any("blank" in w for w in rep.warnings)  # an empty page is not a defect

        rec = c.get_document(rep.document_id)
        assert rec.extraction_status == "partial" and rec.quality["pages_blank"] == [4]
        quality_by_page = {p.page_number: p.quality for p in c.store.document_passages(rec.id)}
        assert quality_by_page == {1: "ok", 2: "low", 5: "ok"}  # nothing from the scan or the blank page
        assert c.stats()["low_quality_passages"] == 1

        hits = c.search("conductivity", k=10)
        assert {h.passage.quality for h in hits} == {"ok", "low"}  # visible, and labelled...
        assert {h.passage.quality for h in c.search("conductivity", k=10, filters=SearchFilters(exclude_low_quality=True))} == {"ok"}
        assert {p.quality for p in c.build_index(SearchFilters(exclude_low_quality=True))._passages} == {"ok"}
        assert c.verify_passage(next(h.passage.passage_id for h in hits if h.passage.quality == "low")).ok  # ...but verifiable


def test_a_file_that_is_not_readable_text_is_not_silently_empty(tmp_path):
    with Corpus(tmp_path / "c", isolate=False) as c:
        blank = tmp_path / "blank.pdf"
        make_pdf(blank, ["blank", "blank"])
        rep = c.ingest(blank)
        assert (rep.status, rep.code) == ("rejected", "no_text") and "blank" in rep.message


def test_ingest_many_walks_folders_and_skips_unsupported_files(tmp_path, library):
    folder = tmp_path / "inbox"
    (folder / "nested").mkdir(parents=True)
    (folder / "a.pdf").write_bytes(library["coral"].read_bytes())
    (folder / "nested" / "b.pdf").write_bytes(library["sparse"].read_bytes())
    (folder / "notes.md").write_text("# Soil notes\n\nBiochar changed soil organic carbon over three seasons of trials.\n")
    (folder / "photo.png").write_bytes(b"\x89PNG\r\n")
    with Corpus(tmp_path / "c", isolate=False) as c:
        reports = c.ingest_many([folder, tmp_path / "stray.xyz"])
        got = {r.path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]: r.status for r in reports}
        assert got == {"a.pdf": "ingested", "b.pdf": "ingested", "notes.md": "ingested", "stray.xyz": "rejected"}
        assert c.stats()["documents"] == 3


def test_metadata_supplied_by_the_user_wins_and_is_labelled(tmp_path, library):
    with Corpus(tmp_path / "c", isolate=False) as c:
        rep = c.ingest(library["sparse"], metadata={"title": "My Title", "year": 2024, "doi": "10.1000/Mine.1",
                                                    "authors": ["P. Natarajan"]})
        rec = c.get_document(rep.document_id)
        assert (rec.title, rec.title_source, rec.year, rec.year_source) == ("My Title", "user", 2024, "user")
        assert (rec.doi, rec.authors) == ("10.1000/mine.1", ["P. Natarajan"])
        bad = c.ingest(library["coral"], metadata={"year": 1200})
        assert (bad.status, bad.code) == ("rejected", "invalid_metadata")
        assert c.stats()["documents"] == 1 and len([f for f in c.files_dir.rglob("*") if f.is_file()]) == 1


def test_default_ingestion_parses_in_a_separate_process(tmp_path, library):
    with Corpus(tmp_path / "c") as c:  # isolate=True is the default
        rep = c.ingest(library["coral"])
        assert rep.status == "ingested" and rep.passages >= 4
        assert c.search(QUERIES["coral"], k=1)[0].document.id == rep.document_id


def test_a_passage_that_cannot_be_matched_to_its_page_is_not_stored(monkeypatch, tmp_path):
    from research.corpus import corpus as corpus_mod

    good = "Sulfide electrolytes conduct lithium ions at room temperature and resist dendrite growth under pressure."
    page_one = Page(1, [Block(good, size=10.5, bbox=(72, 200, 500, 212))], good, 842.0)
    # page two's blocks do not appear in its own reference text: such a passage could never be verified
    page_two = Page(2, [Block("Invented sentence that the page text never contained anywhere at all.", size=10.5,
                              bbox=(72, 200, 500, 212))], "completely different reference text for this page", 842.0)
    monkeypatch.setattr(corpus_mod, "extract_file", lambda *a, **k: ExtractedDocument("pdf", "test", [page_one, page_two]))
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.7\n fake")
    with Corpus(tmp_path / "c", isolate=False) as c:
        rep = c.ingest(src)
        assert rep.status == "ingested" and rep.passages == 1
        assert any("could not be matched back to their page text" in w for w in rep.warnings)
        assert [p.page_number for p in c.store.document_passages(rep.document_id)] == [1]


# ================================================================== removing documents
def test_removing_a_document_removes_it_everywhere(fresh):
    corpus, ids = fresh
    pid = corpus.search(QUERIES["coral"], k=1)[0].passage.passage_id
    stored = corpus.root / corpus.get_document(ids["coral"]).stored_path
    assert stored.is_file() and corpus.remove_document(ids["coral"]) is True

    assert corpus.get_document(ids["coral"]) is None and not stored.exists()
    assert all(h.document.id != ids["coral"] for h in corpus.search(QUERIES["coral"], k=20))
    assert corpus.stats() == {"documents": 4, "searchable_documents": 4, "passages": corpus.index.num_passages,
                              "low_quality_passages": 0, "indexed_documents": 4, "indexed_passages": corpus.index.num_passages}
    v = corpus.verify_passage(pid)
    assert not v.ok and not v.checks["passage_exists"]  # past citations of it stop verifying
    assert corpus.remove_document(ids["coral"]) is False and corpus.remove_document("doc-nope") is False


def test_a_removed_document_can_be_added_again(fresh, library):
    corpus, ids = fresh
    corpus.remove_document(ids["sparse"])
    again = corpus.ingest(library["sparse"])
    assert again.status == "ingested" and again.document_id == ids["sparse"]
    assert corpus.search(QUERIES["sparse"], k=1)[0].document.id == ids["sparse"]


def test_a_failed_save_leaves_no_half_written_document(fresh):
    corpus, ids = fresh
    rec = corpus.get_document(ids["coral"])
    p = corpus.store.document_passages(rec.id)[0]
    ghost = type(rec)(**{**rec.to_dict(), "id": "doc-ghost", "file_hash": "f" * 64})
    dup = [p.__class__("doc-ghost:p000", "doc-ghost", "one two three four five", 1), p.__class__("doc-ghost:p000", "doc-ghost",
                                                                                                "six seven eight nine ten", 1)]
    with pytest.raises(sqlite3.IntegrityError):
        corpus.store.save_document(ghost, dup, "test")
    assert corpus.get_document("doc-ghost") is None  # the whole save was rolled back


# ================================================================== verification and inspection
def test_every_stored_passage_verifies_against_its_original(shared):
    corpus, ids = shared
    total = 0
    for doc_id in ids.values():
        for p in corpus.store.document_passages(doc_id):
            v = corpus.verify_passage(p.passage_id)
            assert v.ok and all(v.checks.values()), (p.passage_id, v)
            total += 1
    assert total >= 25
    assert v.message == "passage text found on the cited page of the unmodified original"


def test_verification_keeps_working_after_the_users_own_copy_is_deleted(tmp_path, library):
    mine = tmp_path / "mine.pdf"
    mine.write_bytes(library["coral"].read_bytes())
    with Corpus(tmp_path / "c", isolate=False) as c:
        rep = c.ingest(mine)
        mine.unlink()
        assert all(c.verify_passage(p.passage_id).ok for p in c.store.document_passages(rep.document_id))


def test_unknown_passages_do_not_verify(shared):
    corpus, ids = shared
    v = corpus.verify_passage(f"{ids['coral']}:p999")
    assert not v.ok and v.checks["passage_exists"] is False and "no passage" in v.message
    assert not corpus.verify_passage("not-even-an-id").ok


def first_passage(corpus, doc_id):
    return corpus.store.document_passages(doc_id)[2]  # a body passage


def test_tampering_with_the_stored_original_is_detected(fresh):
    corpus, ids = fresh
    p = first_passage(corpus, ids["electrolytes"])
    stored = corpus.root / corpus.get_document(ids["electrolytes"]).stored_path
    stored.write_bytes(stored.read_bytes() + b"\n%% tampered after ingestion\n")
    v = corpus.verify_passage(p.passage_id)
    assert not v.ok and v.checks["source_file_unchanged"] is False and "modified" in v.message
    assert corpus.verify_passage(first_passage(corpus, ids["coral"]).passage_id).ok  # other documents are unaffected


def test_a_missing_original_is_detected(fresh):
    corpus, ids = fresh
    (corpus.root / corpus.get_document(ids["perovskite"]).stored_path).unlink()
    v = corpus.verify_passage(first_passage(corpus, ids["perovskite"]).passage_id)
    assert not v.ok and v.checks["source_file_present"] is False and "missing" in v.message


def set_passage(corpus, passage_id, **cols):
    sets = ", ".join(f"{k}=?" for k in cols)
    corpus.store._conn.execute(f"UPDATE passages SET {sets} WHERE passage_id=?", (*cols.values(), passage_id))


def test_text_edited_in_the_database_is_detected_by_its_checksum(fresh):
    corpus, ids = fresh
    p = first_passage(corpus, ids["sparse"])
    set_passage(corpus, p.passage_id, text=p.text + " It also cures cancer.")
    v = corpus.verify_passage(p.passage_id)
    assert not v.ok and v.checks["stored_text_intact"] is False and "checksum" in v.message


def test_even_a_consistently_forged_passage_is_not_on_the_original_page(fresh):
    corpus, ids = fresh
    p = first_passage(corpus, ids["sparse"])
    forged = "Sparse attention was proven to cure every disease in all clinical trials."
    set_passage(corpus, p.passage_id, text=forged, text_sha256=sha256_text(forged))  # the attacker fixes the checksum too
    v = corpus.verify_passage(p.passage_id)
    assert not v.ok and v.checks["stored_text_intact"] is True and v.checks["text_found_on_page"] is False


def test_a_wrong_or_nonexistent_page_citation_is_detected(fresh):
    corpus, ids = fresh
    p = first_passage(corpus, ids["electrolytes"])
    other_page = 1 if p.page_number != 1 else 3
    set_passage(corpus, p.passage_id, page_number=other_page)
    v = corpus.verify_passage(p.passage_id)
    assert not v.ok and v.checks["page_exists"] is True and v.checks["text_found_on_page"] is False
    set_passage(corpus, p.passage_id, page_number=99)
    v = corpus.verify_passage(p.passage_id)
    assert not v.ok and v.checks["page_exists"] is False and "page 99" in v.message


def test_inspect_shows_the_passage_in_context_and_checks_it(shared):
    corpus, ids = shared
    pid = corpus.search("activation energy Arrhenius conductivity", k=1)[0].passage.passage_id
    ins = corpus.inspect(pid, context=1)
    siblings = corpus.store.document_passages(ids["electrolytes"])
    i = [p.passage_id for p in siblings].index(pid)

    assert ins.passage.passage_id == pid and ins.ordinal == i and ins.document.id == ids["electrolytes"]
    assert [p.passage_id for p in ins.before] == [p.passage_id for p in siblings[max(0, i - 1):i]]
    assert [p.passage_id for p in ins.after] == [p.passage_id for p in siblings[i + 1:i + 2]]
    assert ins.text_sha256 == sha256_text(ins.passage.text) and ins.extraction_method.startswith("pymupdf-")
    assert ins.verification.ok
    assert corpus.inspect(pid, context=0, verify=False).verification is None
    assert corpus.inspect("doc-none:p000") is None


def test_inspect_of_a_tampered_document_reports_the_failure_instead_of_raising(fresh):
    corpus, ids = fresh
    stored = corpus.root / corpus.get_document(ids["coral"]).stored_path
    stored.write_bytes(b"%PDF-1.7 replaced")
    ins = corpus.inspect(first_passage(corpus, ids["coral"]).passage_id)
    assert ins is not None and not ins.verification.ok


# ================================================================== comparing two papers
def test_compare_two_documents(shared):
    corpus, ids = shared
    r = corpus.compare(ids["electrolytes"], ids["perovskite"])
    assert r["a"]["title"] == PAPERS["electrolytes"]["title"] and r["b"]["id"] == ids["perovskite"]
    assert r["a"]["year"] == 2021 and r["b"]["year_source"] == "file_created_date" and r["a"]["pages"] == 3

    only_a, only_b, shared_terms = r["only_in_a"], r["only_in_b"], r["shared_terms"]
    assert any(t.startswith("electrolyt") for t in only_a) and any(t.startswith("perovskit") for t in only_b)
    assert shared_terms
    # check against the documents' own words (whole words, so "avoid" does not count as "void")
    words = {k: set(re.findall(r"[a-z]+", " ".join(p.text for p in corpus.store.document_passages(ids[k])).lower()))
             for k in ("electrolytes", "perovskite")}
    wa, wb = words["electrolytes"], words["perovskite"]
    assert all(t in wa and t not in wb for t in only_a) and all(t in wb and t not in wa for t in only_b)
    sa, sb = {stem(w) for w in wa}, {stem(w) for w in wb}
    assert all(stem(t) in sa and stem(t) in sb for t in shared_terms)
    assert 0 < r["similarity"]["tfidf_cosine"] < 0.5 and 0 < r["similarity"]["vocabulary_jaccard"] < 0.5
    assert "lexical" in r["note"] and "not of meaning" in r["note"]  # the limitation travels with the result

    swapped = corpus.compare(ids["perovskite"], ids["electrolytes"])
    assert swapped["similarity"] == r["similarity"] and swapped["only_in_a"] == only_b


def test_a_document_compared_with_itself_is_identical(shared):
    corpus, ids = shared
    r = corpus.compare(ids["coral"], ids["coral"])
    assert r["similarity"] == {"tfidf_cosine": 1.0, "vocabulary_jaccard": 1.0} and r["only_in_a"] == r["only_in_b"] == []


def test_compare_can_show_each_documents_best_passages_for_a_question(shared):
    corpus, ids = shared
    r = corpus.compare(ids["electrolytes"], ids["perovskite"], query="degrees Celsius processing", top_passages=2)
    assert set(r["best_passages"]) == {ids["electrolytes"], ids["perovskite"]}
    for doc_id, passages in r["best_passages"].items():
        assert 1 <= len(passages) <= 2
        assert all(corpus.store.get_passage(p["passage_id"])[0].document_id == doc_id for p in passages)
    with pytest.raises(KeyError):
        corpus.compare(ids["coral"], "doc-unknown")
