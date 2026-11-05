"""Lexical analysis and BM25 retrieval (the Discovery agent's search backend)."""
import pytest

from research.bm25.analysis import STOPWORDS, analyze, key_terms, phrase_chunks, stem, term_surface_forms
from research.bm25.index import BM25Index, Passage, passages_from_text


# ---------------------------------------------------------------- analysis
def test_stem_conflates_inflections():
    assert stem("electrolytes") == stem("electrolyte")
    assert len({stem(w) for w in ("increase", "increases", "increased")}) == 1
    assert stem("studies") == "study"
    assert stem("class") == "class"  # 'ss' is not a plural


def test_analyze_drops_stopwords_and_lowercases():
    assert analyze("The Effect of TEMPERATURE") == [stem("effect"), stem("temperature")]
    assert analyze("") == []
    assert analyze("the of and") == []
    assert all(w not in STOPWORDS for w in ("effect", "temperature"))


def test_term_surface_forms_are_unique_ordered_and_limited():
    pairs = term_surface_forms("Electrolytes and electrolyte: ionic conductivity of ionic solids")
    stems = [s for s, _ in pairs]
    assert stems == list(dict.fromkeys(stems))  # unique
    assert stems[0] == stem("electrolytes") and pairs[0][1] == "electrolytes"  # first surface form wins
    assert [w for _, w in pairs][1:3] == ["ionic", "conductivity"]
    assert len(term_surface_forms("alpha beta gamma delta epsilon", limit=3)) == 3


def test_surface_forms_reanalyse_to_their_own_stem():
    """Queries are built from surface words: re-analysing each one must give back exactly the stem the Critic matches."""
    assert stem(stem("ratings")) != stem("ratings")  # the light stemmer is not idempotent, hence surface words
    pairs = term_surface_forms("increased ratings of electrolytes and moving ions")
    assert ("rating", "ratings") in pairs
    for s, w in pairs:
        assert analyze(w) == [s]


def test_key_terms_matches_term_surface_forms():
    q = "How does temperature affect ionic conductivity?"
    assert key_terms(q) == [s for s, _ in term_surface_forms(q)]


def test_phrase_chunks_split_at_stopwords():
    assert phrase_chunks("effect of temperature on conductivity") == [["effect"], ["temperature"], ["conductivity"]]
    assert phrase_chunks("how does temperature affect ionic conductivity") == [["temperature", "affect", "ionic", "conductivity"]]
    assert phrase_chunks("the of") == []


# ---------------------------------------------------------------- index
def small_index() -> BM25Index:
    idx = BM25Index()
    idx.add_passages([
        Passage("a:p000", "a", "Lithium electrolytes show high ionic conductivity at elevated temperature.", 1, "Abstract"),
        Passage("b:p000", "b", "Sulfide electrolytes are brittle.", 2),
        Passage("c:p000", "c", "A study of unrelated gardening practices and soil moisture."),
    ])
    return idx


def test_ranking_prefers_the_relevant_passage():
    hits = small_index().search("ionic conductivity of electrolytes", k=5)
    assert [h.passage.passage_id for h in hits] == ["a:p000", "b:p000"]
    assert hits[0].score > hits[1].score > 0
    assert set(hits[0].matched_terms) == {stem("ionic"), stem("conductivity"), stem("electrolytes")}


def test_stemming_connects_query_and_document_forms():
    assert {h.passage.passage_id for h in small_index().search("electrolyte")} == {"a:p000", "b:p000"}


def test_no_match_and_empty_index_return_nothing():
    assert small_index().search("quantum chromodynamics") == []
    assert BM25Index().search("anything") == []
    assert small_index().search("the of and") == []  # only stopwords


def test_k_limits_results():
    assert len(small_index().search("electrolytes", k=1)) == 1


def test_document_filter():
    hits = small_index().search("electrolytes", k=5, document_ids={"b"})
    assert [h.passage.document_id for h in hits] == ["b"]


def test_ties_are_broken_deterministically_by_passage_id():
    idx = BM25Index()
    idx.add_passages([Passage(f"d:p{i:03d}", "d", "identical text about batteries") for i in (3, 1, 2, 0)])
    ids = [h.passage.passage_id for h in idx.search("batteries")]
    assert ids == ["d:p000", "d:p001", "d:p002", "d:p003"]
    assert ids == [h.passage.passage_id for h in idx.search("batteries")]  # repeatable


def test_rarer_terms_weigh_more():
    idx = small_index()
    assert idx.idf("conductivity") > idx.idf("electrolyt")  # appears in 1 passage vs 2
    assert idx.idf("never-seen") > idx.idf("electrolyt")  # unseen term: maximal idf, still finite and positive
    assert idx.idf("electrolyt") > 0


def test_length_normalisation_favours_the_focused_passage():
    idx = BM25Index()
    idx.add_passages([
        Passage("s:p000", "s", "lithium conductivity"),
        Passage("l:p000", "l", "lithium conductivity " + " ".join(f"filler{i}" for i in range(60))),
    ])
    hits = idx.search("lithium conductivity")
    assert [h.passage.passage_id for h in hits] == ["s:p000", "l:p000"]


def test_statistics_and_lookup():
    idx = small_index()
    assert (idx.num_documents, idx.num_passages) == (3, 3)
    assert idx.get("b:p000").page_number == 2
    assert idx.get("nope") is None
    assert idx.has_document("a") and not idx.has_document("zzz")


def test_duplicate_passage_ids_are_rejected():
    idx = small_index()
    with pytest.raises(ValueError, match="duplicate"):
        idx.add_passages([Passage("a:p000", "a", "again")])


# ---------------------------------------------------------------- passage splitting
def test_paragraphs_become_passages_with_stable_ids():
    ps = passages_from_text("doc", "First paragraph here.\n\nSecond paragraph here.\n\n\n  \nThird.")
    assert [p.passage_id for p in ps] == ["doc:p000", "doc:p001", "doc:p002"]
    assert ps[1].text == "Second paragraph here."
    assert all(p.document_id == "doc" for p in ps)


def test_headings_set_the_section_and_are_not_passages():
    text = "Abstract\n\nWe study things.\n\n2. Results\n\nThings were found.\n\nConclusion\n\nDone."
    ps = passages_from_text("doc", text)
    assert [(p.section, p.text) for p in ps] == [("Abstract", "We study things."), ("2. Results", "Things were found."),
                                                 ("Conclusion", "Done.")]


def test_long_paragraphs_are_cut_at_sentence_boundaries_without_losing_text():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(30)]
    text = " ".join(sentences)
    ps = passages_from_text("doc", text, max_words=50)
    assert len(ps) > 1
    assert all(len(p.text.split()) <= 50 for p in ps)
    assert all(p.text.rstrip().endswith(".") for p in ps)  # never cut mid-sentence
    assert " ".join(p.text for p in ps).split() == text.split()


def test_page_number_is_propagated_and_empty_text_is_empty():
    assert passages_from_text("doc", "Some text.", page_number=7)[0].page_number == 7
    assert passages_from_text("doc", "") == []
    assert passages_from_text("doc", "\n\n  \n") == []
