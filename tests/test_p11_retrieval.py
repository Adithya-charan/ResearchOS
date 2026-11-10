"""P11.1 advanced retrieval: the baseline is untouched and recoverable, every feature is explainable and deterministic, and the recommended
setting is justified by (and pinned to) the benchmark on the invented p11 dataset."""
import pytest

from research.bm25.advanced import BASELINE, RECOMMENDED, AdvancedConfig, AdvancedSearcher, normalize_query
from research.retrieval_benchmark import evaluate, load, run


@pytest.fixture(scope="module")
def world():
    idx, titles, dup_of, queries = load()
    return idx, titles, dup_of, queries, AdvancedSearcher(idx, titles)


def docs(hits):
    return list(dict.fromkeys(h.passage.document_id for h in hits))


def test_identity_configuration_reproduces_the_baseline_exactly(world):
    idx, _, _, queries, adv = world
    for q in queries:
        base = idx.search(q["query"], k=20)
        ours = adv.search(q["query"], k=20, config=BASELINE)
        assert [(h.passage.passage_id, round(h.score, 9)) for h in base] == [(h.passage.passage_id, round(h.score, 9)) for h in ours], q["id"]


def test_scores_are_explained_and_the_components_add_up(world):
    _, _, _, _, adv = world
    cfg = AdvancedConfig(normalize=True, phrase_bonus=0.5, title_boost=1.0, heading_boost=0.5, prf_terms=3)
    for h in adv.search("silicon electrode swelling", k=8, config=cfg):
        assert h.score == pytest.approx(sum(h.components.values()), abs=1e-4)
        assert h.components.get("bm25", 0) == pytest.approx(sum(h.term_contributions.values()), abs=1e-4)
        assert set(h.components) <= {"bm25", "expansion", "phrase", "title", "heading"}


def test_results_are_deterministic(world):
    _, _, _, _, adv = world
    a = [(h.passage.passage_id, h.score) for h in adv.search("sparse gradients", k=10, config=RECOMMENDED)]
    b = [(h.passage.passage_id, h.score) for h in adv.search("sparse gradients", k=10, config=RECOMMENDED)]
    assert a == b and a


@pytest.mark.parametrize("raw, want", [
    ("Top-K  Sparsification!", "top k sparsification"),
    ("non_iid/data", "non iid data"),
    ("  ÉLECTRODE swelling ", "électrode swelling"),
    ('"capacity retention" additive', 'additive "capacity retention"'),
    ("", ""),
])
def test_query_normalisation(raw, want):
    assert normalize_query(raw) == want


def test_exact_short_and_long_queries_find_the_right_document(world):
    _, _, _, _, adv = world
    for query, doc in [("degree heating weeks bleaching threshold", "coral1"), ("epsilon", "fl3"), ("biochar", "soil1"),
                       ("I would like to know whether coral larvae prefer to settle on rubble compared with bare rock or sand", "coral2")]:
        assert doc in docs(adv.search(query, k=5, config=RECOMMENDED)), query


def test_a_query_about_something_else_gets_only_weak_one_word_leads(world):
    """Lexical retrieval cannot say 'nothing relevant': an off-topic query that shares one common word ('error') still returns that passage.
    What the ranker must do is keep such leads visibly weak: they match a single query term, never the several an on-topic hit matches."""
    _, _, _, _, adv = world
    weak = adv.search("quantum error correction surface code threshold", k=5, config=RECOMMENDED)
    assert all(len(h.term_contributions) <= 1 for h in weak)
    strong = adv.search("silicon anode volume expansion during cycling", k=1, config=RECOMMENDED)[0]
    assert len(strong.term_contributions) >= 4
    assert adv.search("", k=5) == [] and adv.search("the of and", k=5) == []


def test_phrase_queries_remain_hard_filters(world):
    _, _, _, _, adv = world
    hits = adv.search('"local epochs" accuracy', k=10, config=RECOMMENDED)
    assert hits and all("local epochs" in h.passage.text.lower() for h in hits)


def test_feedback_expansion_finds_a_document_that_uses_other_words_and_says_so(world):
    """'swelling' is in bat2 only ('volume expansion' in bat1). The baseline cannot reach bat1; expansion terms from the top passage can, and the
    hit records which terms were added so a reader can see why it appeared."""
    idx, _, _, _, adv = world
    assert "bat1" not in docs(idx.search("swelling", k=20))
    hits = adv.search("swelling", k=20, config=AdvancedConfig(prf_terms=5))
    bat1 = [h for h in hits if h.passage.document_id == "bat1"]
    assert bat1 and bat1[0].expansion_terms and "expansion" in bat1[0].components
    assert all(h.components.get("bm25", 0) == 0 for h in bat1)  # it matched only through expansion


def test_variant_expansion_bridges_morphology(world):
    _, _, _, _, adv = world
    hits = adv.search("sparse gradients", k=20, config=AdvancedConfig(variants=True))
    assert "fl2" in docs(hits)  # 'sparse' ~ 'sparsification' via the shared prefix


def test_near_duplicates_are_suppressed_and_recorded(world):
    idx, _, _, _, adv = world
    q = "silicon anode volume expansion during cycling"
    plain = adv.search(q, k=30, config=BASELINE)
    deduped = adv.search(q, k=30, config=AdvancedConfig(normalize=False, suppress_duplicates=True))
    assert {"bat1", "dup1"} <= set(docs(plain))
    assert any(h.duplicates for h in deduped)
    kept_ids = {h.passage.passage_id for h in deduped}
    suppressed = {d for h in deduped for d in h.duplicates}
    assert suppressed and not (suppressed & kept_ids)
    ranked = [d for d, _ in adv.rank_documents(q, k=5, config=AdvancedConfig(normalize=False, suppress_duplicates=True))]
    assert "bat1" in ranked and "dup1" not in ranked  # the copy is dropped as a duplicate document, the original stays


def test_distinct_documents_that_merely_share_words_are_not_called_duplicates(world):
    _, _, _, _, adv = world
    ranked = [d for d, _ in adv.rank_documents("additive fast charge capacity retention", k=5, config=AdvancedConfig(suppress_duplicates=True))]
    assert {"bat4", "bat5"} <= set(ranked)  # opposing findings on the same topic are both kept


def test_per_document_cap(world):
    _, _, _, _, adv = world
    hits = adv.search("silicon", k=20, config=AdvancedConfig(normalize=False, max_per_document=1))
    ids = [h.passage.document_id for h in hits]
    assert len(ids) == len(set(ids)) and len(ids) >= 3


def test_the_recommended_setting_does_not_lose_to_the_baseline_and_removes_duplicates(world):
    idx, titles, dup_of, queries, _ = world
    base = evaluate(None, idx, titles, dup_of, queries)
    rec = evaluate(RECOMMENDED, idx, titles, dup_of, queries)
    b, r = base["overall"], rec["overall"]
    assert r["mrr"] >= b["mrr"] - 1e-9 and r["recall@5"] >= b["recall@5"] and r["recall@3"] >= b["recall@3"] and r["precision@1"] >= b["precision@1"]
    assert r["duplicate_pairs_in_top5"] == 0 < b["duplicate_pairs_in_top5"]
    vm_b, vm_r = base["by_kind"]["vocab_mismatch"], rec["by_kind"]["vocab_mismatch"]
    assert vm_r["recall@3"] > vm_b["recall@3"]  # the case expansion exists for


def test_the_benchmark_report_is_complete_and_honest():
    rep = run()
    assert rep["dataset"]["synthetic"] is True and rep["dataset"]["queries"] >= 40
    assert {"exact", "paraphrase", "long", "short", "phrase", "multi", "vocab_mismatch"} <= set(rep["dataset"]["kinds"] + ["multi"])
    assert "baseline BM25 (original search_documents)" in rep["results"] and "identity (all off)" in rep["results"]
    assert rep["results"]["identity (all off)"]["overall"] == rep["results"]["baseline BM25 (original search_documents)"]["overall"]
    assert any("invented" in x for x in rep["limitations"]) and any("lexical" in x for x in rep["limitations"])
    for res in rep["results"].values():
        assert res["no_relevant_document_queries"]["queries"] >= 3
