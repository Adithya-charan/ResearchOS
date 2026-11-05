"""The runnable Phase 7 product: scripts/manage_corpus.py (ingest / list / search / inspect / verify / compare / remove / stats)
and scripts/run_research.py --library (agents answering questions over the user's own documents)."""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import manage_corpus  # noqa: E402
import run_research  # noqa: E402
from research.corpus import Corpus  # noqa: E402
from tests.corpus_fixtures import PAPERS, build_library  # noqa: E402
from tests.pdfgen import make_pdf  # noqa: E402


@pytest.fixture(scope="module")
def pdf_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pdfs")
    build_library(folder)
    return folder


@pytest.fixture(scope="module")
def library_root(tmp_path_factory, pdf_dir):
    """The five-paper library, ingested once through the CLI itself."""
    r = tmp_path_factory.mktemp("library") / "corpus"
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli(r, "ingest", str(pdf_dir), "--license", "CC-BY-4.0", "--source", "invented fixtures") == 0
    return r


@pytest.fixture
def root(library_root, tmp_path):
    """A private copy of the library for each test, so tests that edit or tamper with it cannot affect one another."""
    dest = tmp_path / "corpus"
    shutil.copytree(library_root, dest)
    return dest


def cli(root, *argv) -> int:
    return manage_corpus.main(["--root", str(root), *argv])


def listing(root, capsys) -> list[dict]:
    assert cli(root, "list", "--json") == 0
    return json.loads(capsys.readouterr().out)


def by_title(docs):
    return {d["title"]: d for d in docs}


# ================================================================== ingest / list
def test_ingesting_a_folder_reports_each_file_and_the_totals(tmp_path, pdf_dir, capsys):
    assert cli(tmp_path / "c", "ingest", str(pdf_dir), "--license", "CC-BY-4.0") == 0
    out = capsys.readouterr().out
    assert out.count("[ingested]") == 5 and "5 ingested, 0 duplicate, 0 rejected, 0 failed" in out
    assert "Corpus now: 5 document(s)" in out

    assert cli(tmp_path / "c", "ingest", str(pdf_dir)) == 0  # running it again is harmless
    again = capsys.readouterr().out
    assert again.count("[duplicate]") == 5 and "0 ingested, 5 duplicate" in again


def test_list_shows_metadata_with_provenance_and_quality(root, capsys):
    assert cli(root, "list") == 0
    out = capsys.readouterr().out
    for paper in PAPERS.values():
        assert paper["title"] in out
    assert "year 2021 [file_created_date]" in out and "year 2019 [first_page_text]" in out
    assert "quality ok" in out and "license CC-BY-4.0" in out and "pdf, 3 page(s)" in out

    docs = listing(root, capsys)
    assert len(docs) == 5 and all(d["passages"] >= 4 and d["source"] == "invented fixtures" for d in docs)


def test_an_empty_or_missing_corpus_is_explained_not_created(tmp_path, capsys):
    missing = tmp_path / "nowhere"
    for command in (["list"], ["search", "anything"], ["stats"], ["inspect", "doc-x:p000"], ["verify", "doc-x:p000"]):
        assert cli(missing, *command) == 2
    assert "no corpus at" in capsys.readouterr().err and not missing.exists()


def test_problem_files_are_each_explained_and_the_exit_code_says_something_failed(tmp_path, capsys):
    bad = tmp_path / "inbox"
    bad.mkdir()
    make_pdf(bad / "locked.pdf", [[("p", "secret text that nobody may read without the password")]], password="pw")
    make_pdf(bad / "scan.pdf", [("scan", [("p", "pixels only no text layer on this page at all in the file")])])
    (bad / "broken.pdf").write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 4)
    make_pdf(bad / "good.pdf", [[("h1", "Reef notes"), ("p", "Coral colonies recovered slowly after the bleaching event of that summer.")]])

    assert cli(tmp_path / "c", "ingest", str(bad)) == 1
    out = capsys.readouterr().out
    assert "[rejected:encrypted] locked.pdf" in out and "[rejected:corrupt] broken.pdf" in out
    assert "[rejected:no_text] scan.pdf" in out and "OCR is not available" in out
    assert "[ingested]  good.pdf" in out and "1 ingested, 0 duplicate, 2 rejected" not in out
    assert "1 ingested, 0 duplicate, 3 rejected, 0 failed" in out


def test_ingest_warns_about_unreadable_pages_in_a_partly_readable_pdf(tmp_path, capsys):
    pages = [[("p", "Opening page with perfectly readable text about sulfide electrolytes in lithium cells.")],
             ("scan", [("p", "scanned appendix without a text layer holding extra tables of results")])]
    pdf = make_pdf(tmp_path / "partial.pdf", pages)
    assert cli(tmp_path / "c", "ingest", str(pdf)) == 0
    out = capsys.readouterr().out
    assert "[ingested]" in out and "! pages 2 have no text layer (scanned?) and were NOT indexed; OCR is not available" in out
    assert cli(tmp_path / "c", "list") == 0
    assert "quality partial (1 without text layer)" in capsys.readouterr().out


def test_metadata_overrides_apply_to_exactly_one_document(tmp_path, pdf_dir, capsys):
    one = pdf_dir / "sparse.pdf"
    assert cli(tmp_path / "c", "ingest", str(one), str(pdf_dir / "coral.pdf"), "--title", "X") == 2
    assert cli(tmp_path / "c", "ingest", str(pdf_dir), "--year", "2020") == 2  # a folder is many documents
    assert "describe one document" in capsys.readouterr().err

    assert cli(tmp_path / "c", "ingest", str(one), "--title", "My Title", "--authors", "P. Natarajan; T. Berg", "--year", "2024",
               "--doi", "10.1000/Mine.1") == 0
    capsys.readouterr()
    (doc,) = listing(tmp_path / "c", capsys)
    assert (doc["title"], doc["title_source"], doc["year"], doc["year_source"]) == ("My Title", "user", 2024, "user")
    assert doc["authors"] == ["P. Natarajan", "T. Berg"] and doc["doi"] == "10.1000/mine.1"

    assert cli(tmp_path / "d", "ingest", str(one), "--year", "1200") == 1
    assert "[rejected:invalid_metadata]" in capsys.readouterr().out


def test_documents_not_licensed_for_search_are_stored_but_not_searchable(tmp_path, pdf_dir, capsys):
    assert cli(tmp_path / "c", "ingest", str(pdf_dir / "coral.pdf"), "--allowed-use", "quote") == 0
    assert "stored but not searchable" in capsys.readouterr().out
    assert cli(tmp_path / "c", "list") == 0 and "NOT SEARCHABLE" in capsys.readouterr().out
    assert cli(tmp_path / "c", "search", "coral bleaching mortality") == 0
    assert "No passage matched" in capsys.readouterr().out
    assert cli(tmp_path / "d", "ingest", str(pdf_dir / "coral.pdf"), "--allowed-use", "search,publish") == 2


# ================================================================== search / inspect / verify
def first_hit(root, capsys, query, *extra):
    assert cli(root, "search", query, "--json", *extra) == 0
    return json.loads(capsys.readouterr().out)


def test_search_prints_title_page_section_and_a_snippet(root, capsys):
    assert cli(root, "search", "ionic conductivity of sulfide glass-ceramic electrolytes", "-k", "3") == 0
    out = capsys.readouterr().out
    assert out.startswith("1. [") and "Solid-State Electrolytes for Lithium Batteries (2021), p.1, Abstract" in out
    assert "matched:" in out and "twelve millisiemens" in out and "Lexical BM25 only" in out


def test_search_filters_from_the_command_line(root, capsys):
    broad = "cells hours temperature percent training accuracy mortality"
    titles = lambda extra: {h["title"] for h in first_hit(root, capsys, broad, "-k", "100", *extra)}  # noqa: E731
    assert len(titles([])) == 5
    assert titles(["--year-min", "2022"]) == {PAPERS["federated"]["title"]}
    assert titles(["--year-max", "2018"]) == {PAPERS["perovskite"]["title"]}
    assert titles(["--type", "pdf", "--license", "CC-BY-4.0", "--year-min", "2021"]) == {
        PAPERS["electrolytes"]["title"], PAPERS["federated"]["title"]}  # filters combine
    assert titles(["--license", "CC0-1.0"]) == set() and titles(["--type", "docx"]) == set()
    doc_id = by_title(listing(root, capsys))[PAPERS["coral"]["title"]]["id"]
    assert titles(["--document", doc_id]) == {PAPERS["coral"]["title"]}
    assert cli(root, "search", "quantum entanglement neutron stars") == 0
    assert "No passage matched" in capsys.readouterr().out


def test_inspect_shows_context_and_a_fresh_verification(root, capsys):
    hit = first_hit(root, capsys, "short circuits when stack pressure fell below two megapascals")[0]
    assert (hit["page"], hit["section"]) == (2, "3. Results")

    assert cli(root, "inspect", hit["passage_id"]) == 0
    out = capsys.readouterr().out
    assert PAPERS["electrolytes"]["title"] in out and "(title from: first_page_largest_text)" in out
    assert "10.5555/sse.2021.014" in out and "year: 2021 [file_created_date]" in out
    assert "page 2, section 3. Results" in out and hit["text"] in out
    assert "=> VERIFIED" in out and "text found on page" in out and "extracted with pymupdf-" in out

    assert cli(root, "inspect", hit["passage_id"], "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["passage"]["page"] == 2 and data["verification"]["ok"] is True and data["document"]["license"] == "CC-BY-4.0"
    assert cli(root, "inspect", "doc-000000000000:p000") == 2
    assert "no passage" in capsys.readouterr().err


def test_verify_checks_passages_and_whole_documents(root, capsys):
    doc = by_title(listing(root, capsys))[PAPERS["sparse"]["title"]]
    pid = first_hit(root, capsys, "block sparse attention memory", "--document", doc["id"])[0]["passage_id"]
    assert cli(root, "verify", pid) == 0 and "1 of 1 passage(s) verified" in capsys.readouterr().out
    assert cli(root, "verify", "--document", doc["id"]) == 0
    assert f"{doc['passages']} of {doc['passages']} passage(s) verified" in capsys.readouterr().out
    assert cli(root, "verify") == 2 and cli(root, "verify", "--document", "doc-nope") == 2
    capsys.readouterr()

    assert cli(root, "verify", "doc-0000:p000") == 1 and "NOT VERIFIED - no passage" in capsys.readouterr().out


def test_verify_and_inspect_fail_loudly_when_the_stored_original_changes(root, capsys):
    doc = by_title(listing(root, capsys))[PAPERS["coral"]["title"]]
    pid = first_hit(root, capsys, "coral bleaching mortality heat stress")[0]["passage_id"]
    stored = root / doc["stored_path"]
    stored.write_bytes(stored.read_bytes() + b"\n%% edited\n")

    assert cli(root, "verify", pid) == 1
    assert "has been modified since it was ingested" in capsys.readouterr().out
    assert cli(root, "inspect", pid) == 1
    out = capsys.readouterr().out
    assert "FAILED  source file unchanged" in out and "NOT VERIFIED" in out


def test_search_marks_passages_from_unreliable_pages(tmp_path, capsys):
    table = "conductivity " + "12.5 13.1 14.8 15.2 16.9 17.4 18.0 19.3 20.1 21.7 " * 9
    pdf = make_pdf(tmp_path / "mixed.pdf", [[("p", "Measured values follow in the next table of the supplement.")], [("p", table)]])
    assert cli(tmp_path / "c", "ingest", str(pdf)) == 0
    capsys.readouterr()
    assert cli(tmp_path / "c", "search", "conductivity") == 0
    assert "[unreliable text extraction]" in capsys.readouterr().out
    assert cli(tmp_path / "c", "search", "conductivity", "--exclude-low-quality") == 0
    assert "No passage matched" in capsys.readouterr().out
    assert cli(tmp_path / "c", "stats") == 0 and "1 from pages with unreliable extraction" in capsys.readouterr().out


# ================================================================== compare / remove / stats
def test_compare_prints_both_documents_and_their_vocabulary(root, capsys):
    docs = by_title(listing(root, capsys))
    a, b = docs[PAPERS["electrolytes"]["title"]]["id"], docs[PAPERS["perovskite"]["title"]]["id"]
    assert cli(root, "compare", a, b, "--query", "degrees Celsius") == 0
    out = capsys.readouterr().out
    assert f"A: {a}" in out and f"B: {b}" in out and "vocabulary overlap (Jaccard)" in out
    assert "only in A:" in out and "electrolytes" in out and "perovskite" in out
    assert f"Best passages in {a}" in out and f"Best passages in {b}" in out and "lexical comparison only" in out

    assert cli(root, "compare", a, b, "--json") == 0
    assert json.loads(capsys.readouterr().out)["a"]["id"] == a
    assert cli(root, "compare", a, "doc-unknown") == 2


def test_remove_needs_confirmation_and_then_removes_everything(root, capsys):
    doc = by_title(listing(root, capsys))[PAPERS["federated"]["title"]]
    assert cli(root, "remove", doc["id"]) == 2
    assert "--yes" in capsys.readouterr().err and len(listing(root, capsys)) == 5

    assert cli(root, "remove", doc["id"], "--yes") == 0
    assert f"removed {doc['id']}" in capsys.readouterr().out
    assert len(listing(root, capsys)) == 4 and not (root / doc["stored_path"]).exists()
    assert all(h["document_id"] != doc["id"] for h in first_hit(root, capsys, "differential privacy noise clipping", "-k", "20"))
    assert cli(root, "remove", doc["id"], "--yes") == 2  # already gone


def test_stats(root, capsys):
    assert cli(root, "stats") == 0
    out = capsys.readouterr().out
    assert "5 document(s) (5 searchable)" in out and "0 from pages with unreliable extraction" in out
    assert cli(root, "stats", "--json") == 0
    s = json.loads(capsys.readouterr().out)
    assert s["documents"] == 5 and s["indexed_passages"] == s["passages"] > 20


# ================================================================== agents over the library
QUESTION = "What percent accuracy was retained?"


def research(tmp_path, root, *extra):
    out = tmp_path / "summary.json"
    code = run_research.main(["--library", str(root), "--question", QUESTION, "--db", str(tmp_path / "runs.db"),
                              "--json", str(out), *extra])
    return code, json.loads(out.read_text(encoding="utf-8")) if out.exists() else None


def test_research_run_over_the_library_cites_documents_pages_and_verifies_them(tmp_path, root, capsys):
    code, data = research(tmp_path, root)
    out = capsys.readouterr().out

    assert code == 0 and "Library " in out and "5 document(s)" in out and "Evidence check: SUFFICIENT" in out
    assert "synthetic template corpus" not in out
    assert out.count("[verified against original]") >= 3 and "NOT VERIFIED" not in out
    evidence_lines = [line for line in out.split("Evidence check:")[1].split("Uncertainties:")[0].splitlines()
                      if line.startswith("  - ")]
    assert len(evidence_lines) >= 3
    assert all(any(line.startswith(f"  - {p['title']} (") for p in PAPERS.values()) for line in evidence_lines)  # titles, not ids
    assert "Limitations:" in out and "not proof" in out

    evidence_ids = [r["passage_id"] for r in data["summary"]["evidence"]]
    assert set(data["citation_checks"]) == set(evidence_ids) and all(v["ok"] for v in data["citation_checks"].values())


def test_a_tampered_library_is_caught_when_the_run_checks_its_citations(tmp_path, root, capsys):
    for d in listing(root, capsys):
        stored = root / d["stored_path"]
        stored.write_bytes(stored.read_bytes() + b"\n%% changed\n")
    code, data = research(tmp_path, root)
    out = capsys.readouterr().out
    assert code == 0  # the run itself completes...
    assert "[NOT VERIFIED:" in out and "modified" in out  # ...but the evidence says it cannot be trusted
    assert data["citation_checks"] and not any(v["ok"] for v in data["citation_checks"].values())


def test_a_run_can_be_scoped_by_year_and_licence(tmp_path, root, capsys):
    code, data = research(tmp_path, root, "--year-max", "2019")
    out = capsys.readouterr().out
    assert code == 0 and "2 document(s)" in out and data["summary"]["corpus"]["documents"] == 2
    cited = {r["document_id"] for r in data["summary"]["evidence"]}
    assert cited <= {d["id"] for d in listing(root, capsys) if d["year"] <= 2019}
    assert "INSUFFICIENT" in out  # two documents cannot satisfy a three-document standard, and the run says so

    code, data = research(tmp_path, root, "--license", "CC-BY-4.0", "--exclude-low-quality")
    assert code == 0 and data["summary"]["corpus"]["documents"] == 5


def test_a_run_over_the_library_gives_the_bounded_negative_result(tmp_path, root, capsys):
    out_json = tmp_path / "s.json"
    code = run_research.main(["--library", str(root), "--question", "quantum entanglement between neutron stars",
                              "--db", str(tmp_path / "r.db"), "--json", str(out_json)])
    out = capsys.readouterr().out
    assert code == 0 and "Evidence check: INSUFFICIENT" in out
    assert "No close match was identified in the indexed corpus searched for this run" in out
    assert "This is not proof that the idea has never been published." in out


def test_library_problems_are_reported_not_crashed_on(tmp_path, root, capsys):
    db = str(tmp_path / "x.db")
    assert run_research.main(["--library", str(tmp_path / "missing"), "--db", db]) == 2
    assert "no document library" in capsys.readouterr().err
    assert run_research.main(["--library", str(root), "--year-min", "3000", "--db", db]) == 2
    assert "no searchable passages" in capsys.readouterr().err
    assert Corpus(root).stats()["documents"] == 5  # a failed run leaves the library untouched
