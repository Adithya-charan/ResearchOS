"""Phase 7: reading files. Real PDFs (written by PyMuPDF), real DOCX/HTML/Markdown/text files, and hostile or broken ones:
every failure must surface as an ExtractionError with a stable code, never a crash and never silently-empty text."""
import json
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from research.corpus.extract import extract_file, extract_in_process, run_isolated  # noqa: E402
from research.corpus.extract_pdf import extract_pdf  # noqa: E402
from research.corpus.extract_text import extract_html, extract_md, extract_txt  # noqa: E402
from research.corpus.models import ExtractionError  # noqa: E402
from research.corpus.validate import detect_file_type  # noqa: E402
from tests.corpus_fixtures import write_paper  # noqa: E402
from tests.pdfgen import make_docx, make_pdf  # noqa: E402


def code_of(fn, *a, **kw) -> str:
    with pytest.raises(ExtractionError) as e:
        fn(*a, **kw)
    return e.value.code


# ================================================================== PDF
def test_pdf_extraction_keeps_pages_text_and_font_information(tmp_path):
    doc = extract_pdf(write_paper(tmp_path / "e.pdf", "electrolytes"))

    assert doc.file_type == "pdf" and doc.method.startswith("pymupdf-")
    assert [p.number for p in doc.pages] == [1, 2, 3]
    assert all(p.height == pytest.approx(842) for p in doc.pages)
    first = doc.pages[0]
    # raw_text is the untouched reference text (running header included); cleaning happens later, on blocks
    assert "ionic conductivity" in first.raw_text and "Journal of Invented Results" in first.raw_text
    title = first.blocks[0]
    assert title.text == "Solid-State Electrolytes for Lithium Batteries" and title.bold and title.size == pytest.approx(20, abs=0.5)
    body = [b for b in first.blocks if b.text.startswith("Solid electrolytes could")][0]
    assert not body.bold and body.size == pytest.approx(10.5, abs=0.5)
    assert doc.pages[1].blocks[0].bbox is not None  # positions are kept: header/footer detection needs them


def test_pdf_metadata_is_only_what_the_file_declares(tmp_path):
    coral = extract_pdf(write_paper(tmp_path / "c.pdf", "coral"))
    assert coral.metadata["title"] == "Coral Bleaching Under Marine Heatwaves" and "Sofia Duarte" in coral.metadata["author"]
    bare = extract_pdf(write_paper(tmp_path / "s.pdf", "sparse"))
    assert "title" not in bare.metadata and "author" not in bare.metadata  # empty fields are not invented or kept


def test_two_columns_are_read_one_column_after_the_other(tmp_path):
    left = [("h1", "Left Column"), ("p", "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi")]
    right = [("h1", "Right Column"), ("p", "one two three four five six seven eight nine ten eleven twelve thirteen fourteen")]
    path = make_pdf(tmp_path / "cols.pdf", [[("columns", [left, right])]])
    text = extract_pdf(path).pages[0].raw_text
    assert text.index("alpha") < text.index("pi") < text.index("Right Column") < text.index("one two")
    blocks = [b.text for b in extract_pdf(path).pages[0].blocks]
    assert blocks.index("Left Column") < blocks.index(next(b for b in blocks if b.startswith("alpha"))) < blocks.index("Right Column")


def test_a_scanned_page_has_no_text_layer_and_is_marked_as_holding_an_image(tmp_path):
    path = make_pdf(tmp_path / "scan.pdf", [[("p", "readable page text for contrast with the scan next to it")],
                                             ("scan", [("p", "pixels only no text layer here whatsoever")])])
    doc = extract_pdf(path)
    assert doc.pages[0].raw_text.strip() and not doc.pages[0].has_images
    assert doc.pages[1].raw_text.strip() == "" and doc.pages[1].has_images and doc.pages[1].blocks == []


def test_encrypted_pdf_is_refused_not_cracked(tmp_path):
    path = make_pdf(tmp_path / "locked.pdf", [[("p", "secret text that must not be read without the password")]], password="hunter2")
    assert code_of(extract_pdf, path) == "encrypted"


@pytest.mark.parametrize("payload", [b"%PDF-1.7\n" + bytes(range(256)) * 20, b"%PDF-1.4\n1 0 obj\n<<\n"])
def test_damaged_pdf_is_a_clean_error(tmp_path, payload):
    path = tmp_path / "broken.pdf"
    path.write_bytes(payload)
    assert code_of(extract_pdf, path) == "corrupt"


def test_page_limit_is_enforced_before_reading_pages(tmp_path):
    path = write_paper(tmp_path / "e.pdf", "electrolytes")
    assert code_of(extract_pdf, path, max_pages=2) == "too_many_pages"


# ================================================================== validation
def test_validation_rejects_by_extension_size_and_content(tmp_path):
    def f(name: str, data: bytes) -> Path:
        p = tmp_path / name
        p.write_bytes(data)
        return p

    assert code_of(detect_file_type, f("run.exe", b"MZ\x90\x00")) == "unsupported_type"
    assert code_of(detect_file_type, f("noext", b"hello")) == "unsupported_type"
    assert code_of(detect_file_type, f("empty.txt", b"")) == "empty"
    assert code_of(detect_file_type, f("big.txt", b"x" * 2000), max_bytes=1000) == "too_large"
    assert code_of(detect_file_type, f("fake.pdf", b"just some text, not a pdf")) == "type_mismatch"
    assert code_of(detect_file_type, f("sneaky.txt", b"%PDF-1.7 actually a pdf")) == "type_mismatch"
    assert code_of(detect_file_type, f("binary.md", b"# title\x00\x01\x02binary")) == "type_mismatch"
    assert code_of(detect_file_type, f("fake.docx", b"not a zip at all")) == "type_mismatch"
    assert code_of(detect_file_type, tmp_path / "missing.pdf") == "corrupt"
    assert code_of(detect_file_type, tmp_path) == "corrupt"  # a directory is not a file
    assert detect_file_type(f("ok.TXT", b"fine")) == "txt" and detect_file_type(f("p.markdown", b"# x")) == "md"


def test_a_zip_that_is_not_a_word_document_is_refused(tmp_path):
    path = tmp_path / "archive.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("hello.txt", "not word")
    assert code_of(detect_file_type, path) == "type_mismatch"


def test_zip_bombs_are_refused_before_anything_is_decompressed(tmp_path):
    path = make_docx(tmp_path / "bomb.docx", [("", "innocent text")], extra_member=("word/media/blob.bin", bytes(30_000_000)))
    assert code_of(detect_file_type, path) == "unsafe"
    assert code_of(extract_file, path, isolate=False) == "unsafe"


# ================================================================== DOCX / TXT / MD / HTML
def test_docx_text_headings_and_core_properties(tmp_path):
    path = make_docx(tmp_path / "memo.docx", [("Title", "Soil Carbon Memo"), ("Heading1", "Findings"),
                                              ("", "Biochar raised soil organic carbon by twelve percent in two seasons."),
                                              ("", "Rainfall variability confounded the second season.")],
                     title="Soil Carbon Memo", author="R. Alvarez", created="2023-05-02T10:00:00Z")
    doc = extract_file(path, isolate=False)
    assert doc.file_type == "docx" and [p.number for p in doc.pages] == [None]
    assert [(b.text, b.heading_level) for b in doc.pages[0].blocks] == [
        ("Soil Carbon Memo", 1), ("Findings", 1), ("Biochar raised soil organic carbon by twelve percent in two seasons.", 0),
        ("Rainfall variability confounded the second season.", 0)]
    assert doc.metadata == {"title": "Soil Carbon Memo", "author": "R. Alvarez", "date": "2023-05-02T10:00:00Z"}


def test_docx_with_a_dtd_is_refused(tmp_path):
    path = tmp_path / "xxe.docx"
    xml = (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaaaaaaaa">]><w:document xmlns:w="http://schemas.openxmlformats.org/'
           b'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>&a;</w:t></w:r></w:p></w:body></w:document>')
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", xml)
    assert code_of(extract_file, path, isolate=False) == "unsafe"


def test_docx_with_no_text_is_an_error_not_an_empty_document(tmp_path):
    assert code_of(extract_file, make_docx(tmp_path / "blank.docx", []), isolate=False) == "no_text"


def test_plain_text_headings_and_encoding_fallback(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_bytes("Abstract\n\nBatteries store energy.\n\n1. Introduction\n\nCafé data was collected.".encode("cp1252"))
    doc = extract_txt(p)
    assert [(b.text, b.heading_level) for b in doc.pages[0].blocks] == [
        ("Abstract", 1), ("Batteries store energy.", 0), ("1. Introduction", 1), ("Café data was collected.", 0)]
    assert any("Windows-1252" in w for w in doc.warnings)  # the guess is disclosed


def test_markdown_structure_front_matter_and_markup_removal(tmp_path):
    p = tmp_path / "paper.md"
    p.write_text("---\ntitle: \"Notes on Grain Boundaries\"\nauthor: J. Okafor\nyear: 2020\n---\n# Grain Boundaries\n\n"
                 "Intro with a [link](http://example.com) and **bold** text.\n\n## Details\n\n```\ncode line one\n```\n\n"
                 "Final paragraph here.\n", encoding="utf-8")
    doc = extract_md(p)
    assert doc.metadata == {"title": "Notes on Grain Boundaries", "author": "J. Okafor", "year": "2020"}
    got = [(b.text, b.heading_level) for b in doc.pages[0].blocks]
    assert got == [("Grain Boundaries", 1), ("Intro with a link and bold text.", 0), ("Details", 2), ("code line one", 0),
                   ("Final paragraph here.", 0)]
    assert "http://example.com" not in doc.pages[0].raw_text and "**" not in doc.pages[0].raw_text


def test_html_skips_scripts_and_reads_headings_and_citation_meta(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><head><title>Fallback</title><meta name='citation_title' content='Reef Recovery'>"
                 "<meta name='citation_author' content='A. Lee'><meta name='citation_publication_date' content='2019/03/01'>"
                 "<style>p{color:red}</style></head><body><script>alert('ignore me')</script>"
                 "<h1>Reef Recovery</h1><p>Corals regrow slowly after bleaching.</p><nav>skip? no, text</nav>"
                 "<h2>Method</h2><p>We counted colonies.</p></body></html>", encoding="utf-8")
    doc = extract_html(p)
    assert doc.metadata == {"title": "Reef Recovery", "author": "A. Lee", "date": "2019/03/01"}
    texts = [(b.text, b.heading_level) for b in doc.pages[0].blocks]
    assert ("Reef Recovery", 1) in texts and ("Method", 2) in texts and ("We counted colonies.", 0) in texts
    assert "ignore me" not in doc.pages[0].raw_text and "color:red" not in doc.pages[0].raw_text


def test_text_files_with_nothing_in_them_are_errors(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n\n  \n", encoding="utf-8")
    assert code_of(extract_file, p, isolate=False) == "no_text"


# ================================================================== process isolation
def test_isolated_extraction_gives_exactly_the_same_result_as_in_process(tmp_path):
    path = write_paper(tmp_path / "e.pdf", "electrolytes")
    assert extract_file(path, isolate=True).to_dict() == extract_in_process(path, "pdf").to_dict()


def test_errors_cross_the_process_boundary_with_their_codes(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 20)
    assert code_of(extract_file, bad, isolate=True) == "corrupt"
    locked = make_pdf(tmp_path / "locked.pdf", [[("p", "secret text that must not be read")]], password="pw")
    assert code_of(extract_file, locked, isolate=True) == "encrypted"
    assert code_of(extract_file, write_paper(tmp_path / "e.pdf", "electrolytes"), isolate=True, max_pages=1) == "too_many_pages"


def test_a_parser_that_hangs_is_killed(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    assert code_of(run_isolated, cmd, 1.0) == "timeout"


def test_a_parser_that_crashes_hard_does_not_take_the_caller_down(tmp_path):
    cmd = [sys.executable, "-c", "import os; os._exit(3)"]
    assert code_of(run_isolated, cmd, 20.0) == "worker_failed"


@pytest.mark.parametrize("reply", [b"this is not json", b"\xff\xfe\x00", b"[1, 2, 3]", b'{"nothing": 1}',
                                   b'{"document": {"nonsense": true}}', b'{"error": {"code": "corrupt"}}', b'{"error": 5}'])
def test_anything_odd_coming_back_from_the_worker_is_a_clean_failure(monkeypatch, tmp_path, reply):
    from research.corpus import extract as ex

    monkeypatch.setattr(ex, "run_isolated", lambda cmd, timeout_s: reply)
    assert code_of(ex.extract_isolated, tmp_path / "x.pdf", "pdf", 10, 5.0) == "worker_failed"


def test_a_handled_error_reported_by_the_worker_keeps_its_code(monkeypatch, tmp_path):
    from research.corpus import extract as ex

    reply = json.dumps({"error": {"code": "encrypted", "message": "locked"}}).encode()
    monkeypatch.setattr(ex, "run_isolated", lambda cmd, timeout_s: reply)
    with pytest.raises(ExtractionError) as e:
        ex.extract_isolated(tmp_path / "x.pdf", "pdf", 10, 5.0)
    assert (e.value.code, e.value.message) == ("encrypted", "locked")
