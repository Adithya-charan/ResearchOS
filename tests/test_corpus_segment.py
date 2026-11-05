"""Phase 7: from extracted pages to evidence-grade passages - quality scoring, header/footer removal, heading detection,
page-bounded segmentation and metadata provenance. Hand-built documents keep every expectation exact; the end-to-end
behaviour on real PDFs is covered in test_corpus.py."""
import datetime

import pytest

from research.corpus.metadata import extract_metadata
from research.corpus.models import Block, ExtractedDocument, Page
from research.corpus.quality import assess_text, summarize
from research.corpus.segment import (build_passages, detect_headings, normalize, repair_hyphenation, squash,
                                     strip_margin_furniture)

PARA = ("Sulfide electrolytes conduct lithium ions at room temperature. Their grain boundaries limit the usable current "
        "density in practice. Pressure keeps the interface intact during cycling.")


def blk(text, size=10.5, bold=False, y0=300.0, level=0):
    return Block(text, size=size, bold=bold, heading_level=level, bbox=(72.0, y0, 500.0, y0 + 12.0))


def page(n, blocks, raw=None, images=False, height=842.0):
    return Page(n, blocks, raw if raw is not None else "\n".join(b.text for b in blocks), height, images)


def doc_of(*pages, file_type="pdf", metadata=None) -> ExtractedDocument:
    return ExtractedDocument(file_type, "test", list(pages), dict(metadata or {}), [])


def ok_q(n):
    return assess_text(PARA, False)


# ================================================================== text helpers
def test_normalize_folds_ligatures_and_whitespace():
    assert normalize("ﬁnal  ﬂow\n\t of  ions") == "final flow of ions"


@pytest.mark.parametrize("raw,fixed", [
    ("elec-\ntrolyte", "electrolyte"),                     # broken across lines
    ("elec‐\ntrolyte", "electrolyte"),                # Unicode hyphen
    ("sol­id", "solid"),                              # soft hyphen
    ("well-known result", "well-known result"),            # a real hyphen is left alone
    ("section 3-\n4 of the text", "section 3-\n4 of the text"),   # digits are not word fragments
    ("co-\nOperate", "co-\nOperate"),                      # a capital after the break: probably a real compound
])
def test_hyphenation_repair_only_joins_word_fragments(raw, fixed):
    assert repair_hyphenation(raw) == fixed


def test_squash_is_the_comparison_form_used_for_verification():
    assert squash("Elec-tro lyte, 2021!") == squash("ELECTROLYTE 2021") == "electrolyte2021"
    assert squash("ﬁnal") == squash("final")


# ================================================================== quality
def test_clean_prose_is_ok():
    q = assess_text(PARA)
    assert (q.status, q.issues) == ("ok", ()) and q.alpha_ratio > 0.8


def test_blank_versus_scanned_pages():
    assert assess_text("  \n ", has_images=False).status == "blank"
    scanned = assess_text("Fig. 1", has_images=True)
    assert scanned.status == "no_text" and "OCR was not performed" in scanned.issues[0]


@pytest.mark.parametrize("text,fragment", [
    (PARA + " �������", "garbled"),
    (PARA + " ", "private-use"),
    ("1 2 3 4 5 6 7 8 9 0 + - = " * 6, "non-letter"),
    ("T h e e l e c t r o l y t e c o n d u c t s l i t h i u m i o n s a n d b l o c k s d e n d r i t e s " * 2,
     "separated by spaces"),
    (" ".join(["supercalifragilisticexpialidocious"] * 25), "run together"),
])
def test_each_kind_of_bad_extraction_is_detected(text, fragment):
    q = assess_text(text)
    assert q.status == "low" and any(fragment in issue for issue in q.issues), q


def test_document_quality_summary():
    q_ok, q_low = assess_text(PARA), assess_text("1 2 3 4 5 6 7 8 9 0 + - = " * 6)
    q_none, q_blank = assess_text("x", has_images=True), assess_text("")

    assert summarize([(1, q_ok), (2, q_ok)])["status"] == "ok"
    s = summarize([(1, q_ok), (2, q_low), (3, q_none), (4, q_blank)])
    assert s["status"] == "partial" and s["pages_low"] == [2] and s["pages_no_text"] == [3] and s["pages_blank"] == [4]
    assert s["pages_ok"] == 1 and s["pages"] == 4 and any(i.startswith("page 2:") for i in s["issues"])
    assert summarize([(1, q_none), (2, q_none)])["status"] == "no_text"
    assert summarize([(1, q_blank)])["status"] == "no_text"  # nothing at all is not "ok"
    assert summarize([(1, q_ok), (2, q_blank)])["status"] == "ok"  # a blank page is not a defect


# ================================================================== running headers and footers
def furniture_doc(n_pages=4):
    pages = []
    for i in range(1, n_pages + 1):
        pages.append(page(i, [
            blk("Journal of Invented Results", size=9, y0=30),
            blk(PARA + f" Page body number {i}.", y0=300),
            blk(str(i), size=9, y0=800),
        ]))
    return doc_of(*pages)


def test_repeated_headers_and_page_numbers_are_removed_but_body_text_stays():
    doc = furniture_doc(4)
    assert strip_margin_furniture(doc) == 8
    for p in doc.pages:
        assert [b.text for b in p.blocks] == [PARA + f" Page body number {p.number}."]


def test_headers_whose_numbers_change_are_still_recognised():
    pages = [page(i, [blk(f"Smith et al. 2021, vol {i}", size=9, y0=30), blk(PARA, y0=300)]) for i in (1, 2, 3)]
    doc = doc_of(*pages)
    assert strip_margin_furniture(doc) == 3 and all(len(p.blocks) == 1 for p in doc.pages)


def test_a_one_off_margin_note_and_repeated_body_text_are_not_furniture():
    pages = [page(i, [blk("Received 3 May", size=9, y0=30) if i == 1 else blk("Different top line " + "x" * i, size=9, y0=30),
                      blk("Table 1", y0=400), blk(PARA, y0=300)]) for i in (1, 2, 3, 4)]
    doc = doc_of(*pages)
    assert strip_margin_furniture(doc) == 0
    assert all(any(b.text == "Table 1" for b in p.blocks) for p in doc.pages)  # repeated, but mid-page


def test_two_pages_are_enough_to_see_a_running_header():
    doc = doc_of(*[page(i, [blk("Confidential draft", size=9, y0=30), blk(PARA)]) for i in (1, 2)])
    assert strip_margin_furniture(doc) == 2


def test_on_a_single_page_only_a_bare_page_number_can_be_dropped():
    doc = doc_of(page(1, [blk("Journal of Invented Results", size=9, y0=30), blk(PARA), blk("Page 1 of 1", size=9, y0=800)]))
    assert strip_margin_furniture(doc) == 1
    assert [b.text for b in doc.pages[0].blocks] == ["Journal of Invented Results", PARA]


def test_formats_without_geometry_are_left_alone():
    doc = doc_of(Page(None, [Block("1"), Block(PARA)], "1\n" + PARA), file_type="txt")
    assert strip_margin_furniture(doc) == 0 and len(doc.pages[0].blocks) == 2


# ================================================================== headings
def test_headings_are_found_from_size_boldness_and_numbering():
    body = [blk(PARA, y0=200 + 60 * i) for i in range(6)]
    doc = doc_of(page(1, [blk("A Study of Grain Boundaries", size=20, bold=True, y0=100),
                          blk("Abstract", size=14, bold=True, y0=140),
                          blk("Bold lead-in", size=10.5, bold=True, y0=170),
                          blk("3.2 Experimental setup", size=10.5, y0=180),
                          blk("This bold sentence is long enough to be body text, not a heading, and ends with a period.",
                              size=10.5, bold=True, y0=190),
                          blk("1 2 3 " * 30, size=20, y0=195), *body]))
    detect_headings(doc)

    def level(prefix):
        (b,) = [b for b in doc.pages[0].blocks if b.text.startswith(prefix)]
        return b.heading_level

    assert level("A Study of Grain") == 1 and level("Abstract") == 2
    assert level("Bold lead-in") == 3 and level("3.2 Experimental") == 3
    assert level("This bold sentence") == 0                            # too long and ends in a full stop
    assert level("1 2 3 1 2 3") == 0                                   # large, but far too many words
    assert all(b.heading_level == 0 for b in doc.pages[0].blocks if b.text == PARA)


def test_headings_the_extractor_already_knows_are_not_reclassified():
    doc = doc_of(Page(None, [Block("Intro", heading_level=2), Block(PARA)], ""), file_type="md")
    detect_headings(doc)
    assert [b.heading_level for b in doc.pages[0].blocks] == [2, 0]


# ================================================================== passages
def test_passages_never_cross_a_page_boundary_and_know_their_page():
    p1 = page(1, [blk("1. Methods", size=14, bold=True, y0=100, level=2), blk(PARA, y0=150),
                  blk("The paragraph continues onto the next page because the text did not fit here at all.", y0=250)])
    p2 = page(2, [blk("and this is the remainder of that paragraph on the following page of the document.", y0=100),
                  blk("2. Results", size=14, bold=True, y0=160, level=2), blk(PARA, y0=200)])
    doc = doc_of(p1, p2)
    got = build_passages("doc-x", doc, {1: ok_q(1), 2: ok_q(2)})

    assert [(p.page_number, p.section) for p in got] == [(1, "1. Methods"), (2, "1. Methods"), (2, "2. Results")]
    for passage in got:
        source = next(pg for pg in doc.pages if pg.number == passage.page_number)
        assert squash(passage.text) in squash(source.raw_text)
    assert [p.passage_id for p in got] == ["doc-x:p000", "doc-x:p001", "doc-x:p002"]


def test_small_adjacent_fragments_are_merged_and_a_lone_label_is_dropped():
    bullets = [blk(t, y0=200 + 20 * i) for i, t in enumerate(
        ["Lower cost per kilowatt hour", "Higher energy density overall", "Simpler cell packaging design"])]
    got = build_passages("d", doc_of(page(1, bullets), page(2, [blk("Fig. 2")])), {1: ok_q(1), 2: ok_q(2)})
    assert [(p.page_number, p.text) for p in got] == [
        (1, "Lower cost per kilowatt hour Higher energy density overall Simpler cell packaging design")]


def test_long_paragraphs_are_split_at_sentence_boundaries():
    sentence = "Ion hopping is thermally activated in this family of glassy sulfide conductors."  # 12 words
    long_block = blk(" ".join([sentence] * 30))  # 360 words
    got = build_passages("d", doc_of(page(1, [long_block])), {1: ok_q(1)}, max_words=100)
    assert len(got) == 4
    assert all(len(p.text.split()) <= 100 and p.text.endswith(".") for p in got)
    assert " ".join(p.text for p in got) == " ".join([sentence] * 30)  # nothing lost, nothing duplicated


def test_pages_without_a_text_layer_produce_no_passages_and_low_pages_are_flagged():
    pages = [page(1, [blk(PARA)]), page(2, [], raw="", images=True), page(3, [blk("1 2 3 4 5 6 7 8 9 0 " * 10)]),
             page(4, [blk(PARA + " last page.")])]
    quality = {1: assess_text(PARA), 2: assess_text("", True), 3: assess_text("1 2 3 4 5 6 7 8 9 0 " * 10),
               4: assess_text(PARA)}
    got = build_passages("d", doc_of(*pages), quality)
    assert [(p.page_number, p.quality) for p in got] == [(1, "ok"), (3, "low"), (4, "ok")]


def test_hyphenation_is_repaired_in_pdfs_only():
    text = "The electro-\nlyte remained stable during extended cycling at high current density."
    pdf = build_passages("d", doc_of(page(1, [blk(text)])), {1: ok_q(1)})
    txt = build_passages("d", doc_of(page(None, [blk(text)]), file_type="txt"), {None: ok_q(None)})
    assert "electrolyte remained" in pdf[0].text and "electro- lyte" in txt[0].text


def test_segmentation_is_deterministic():
    doc = doc_of(page(1, [blk(PARA), blk(PARA + " Second.", y0=400)]), page(2, [blk(PARA + " Third.")]))
    q = {1: ok_q(1), 2: ok_q(2)}
    assert build_passages("d", doc, q) == build_passages("d", doc, q)


# ================================================================== metadata provenance
def first_page(text, **meta):
    return doc_of(page(1, [blk(text)], raw=text), metadata=meta)


def test_title_sources_in_order_of_trust():
    declared = first_page("ignored body", title="Declared Title Here")
    assert extract_metadata(declared, "f.pdf")["title"] == "Declared Title Here"
    assert extract_metadata(declared, "f.pdf")["title_source"] == "file_metadata"

    for junk in ("Untitled", "Microsoft Word - draft.docx", "paper.pdf", "ab"):
        d = doc_of(page(1, [blk("Heading From Page", size=20, bold=True, level=1), blk(PARA)]), metadata={"title": junk})
        got = extract_metadata(d, "f.pdf")
        assert (got["title"], got["title_source"]) == ("Heading From Page", "first_page_largest_text"), junk

    nothing = extract_metadata(first_page("just a body paragraph with no heading at all"), "my_great-paper.pdf")
    assert (nothing["title"], nothing["title_source"]) == ("my great paper", "filename")


def test_title_from_text_formats():
    txt = doc_of(page(None, [blk("x")], raw="Title: Notes on Dendrites\n\nbody"), file_type="txt")
    assert extract_metadata(txt, "n.txt")["title_source"] == "first_page_text"
    md = doc_of(Page(None, [Block("Doc Title", heading_level=1), Block("## sub", heading_level=2)], ""), file_type="md")
    got = extract_metadata(md, "n.md")
    assert (got["title"], got["title_source"]) == ("Doc Title", "first_heading")


@pytest.mark.parametrize("text,meta,year,source", [
    ("Copyright 2017 The Authors. All rights reserved.", {}, 2017, "first_page_text"),
    ("Published March 2019 in a journal", {}, 2019, "first_page_text"),
    ("Received: accepted 2015", {}, 2015, "first_page_text"),
    ("Some text with no context", {"creationDate": "D:20200215083000"}, 2020, "file_created_date"),
    ("Some text with no context", {"date": "2023-05-02T10:00:00Z"}, 2023, "file_date"),
    ("Published 2018", {"creationDate": "D:20240101000000"}, 2018, "first_page_text"),   # the page beats the file stamp
    ("Since 1998 researchers have asked this.", {}, None, None),                          # a year in prose is not a date
    ("Published 1850", {}, None, None),                                                    # implausible
    (f"Published {datetime.date.today().year + 5}", {}, None, None),                       # in the future
])
def test_publication_year_and_where_it_came_from(text, meta, year, source):
    got = extract_metadata(first_page(text, **meta), "f.pdf")
    assert (got["year"], got["year_source"]) == (year, source)


def test_doi_is_found_normalised_and_never_looked_up():
    assert extract_metadata(first_page("see doi:10.1234/ABC.5."), "f.pdf")["doi"] == "10.1234/abc.5"
    assert extract_metadata(first_page("x", subject="https://doi.org/10.5555/xyz-9)"), "f.pdf")["doi"] == "10.5555/xyz-9"
    assert extract_metadata(first_page("no identifier here"), "f.pdf")["doi"] is None
    far = doc_of(page(1, [blk("a")], raw="a"), page(2, [blk("b")], raw="b"), page(3, [blk("c")], raw="DOI 10.9999/late.1"))
    assert extract_metadata(far, "f.pdf")["doi"] is None  # only the first two pages are searched


def test_authors_come_from_the_file_only():
    assert extract_metadata(first_page("x", author="A. Smith; B. Jones & C. Lee and D. Wu"), "f.pdf")["authors"] == [
        "A. Smith", "B. Jones", "C. Lee", "D. Wu"]
    assert extract_metadata(first_page("By Someone Famous on the page"), "f.pdf")["authors"] == []


def test_user_overrides_win_and_are_labelled_and_validated():
    base = first_page("Published 2018", title="Declared Title")
    got = extract_metadata(base, "f.pdf", {"title": "My Title", "year": 2020, "doi": "10.1000/Mine.1",
                                           "authors": ["A. B.", "C. D."]})
    assert (got["title"], got["title_source"], got["year"], got["year_source"]) == ("My Title", "user", 2020, "user")
    assert got["doi"] == "10.1000/mine.1" and got["authors"] == ["A. B.", "C. D."]
    assert extract_metadata(base, "f.pdf", {"title": "", "year": None})["title"] == "Declared Title"  # blanks are ignored
    for bad in ({"year": 1500}, {"doi": "not-a-doi"}, {"colour": "red"}):
        with pytest.raises(ValueError):
            extract_metadata(base, "f.pdf", bad)
