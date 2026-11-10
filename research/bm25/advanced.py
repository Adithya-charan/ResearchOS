"""Advanced lexical retrieval (P11.1): optional, explainable improvements layered over the unchanged BM25 baseline.

Everything is deterministic, local and lexical (no embeddings, no model): query normalisation, adjacent-term phrase bonus, title and heading
boosting, expansion from the corpus's own vocabulary (orthographic/morphological variants and pseudo-relevance feedback from the top
passages), near-duplicate suppression and a per-document cap. Each feature is switched by `AdvancedConfig`; with every switch off the result
is the baseline ranking (a test pins that). A feature is only worth enabling if `scripts/eval_retrieval_p11.py` shows it helps; see
docs/RETRIEVAL_SYSTEM.md. None of this is semantic search: a paraphrase that shares no words and no corpus co-occurrence is still missed.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from research.bm25.analysis import STOPWORDS, analyze, raw_tokens, stem
from research.bm25.index import BM25Index, Passage, contains_phrase, split_phrases

_QUOTED = re.compile(r'"[^"]*"')


@dataclass(frozen=True)
class AdvancedConfig:
    normalize: bool = True              # NFKC, case, hyphen/underscore/slash to space, punctuation stripped (quoted phrases kept)
    phrase_bonus: float = 0.0           # added per query bigram found contiguously, scaled by the bigram's mean idf
    title_boost: float = 0.0            # share of the query's idf mass found in the document title, times this value
    heading_boost: float = 0.0          # same, for the passage's section heading
    variants: bool = False              # add corpus terms sharing a long prefix with a query term (expan-sion / expan-d)
    variant_weight: float = 0.5
    prf_terms: int = 0                  # number of feedback terms taken from the top passages (0 = off)
    prf_docs: int = 3
    prf_weight: float = 0.3
    suppress_duplicates: bool = False
    duplicate_threshold: float = 0.8    # overlap of 5-word shingles at which a lower-ranked passage counts as a near-copy
    max_per_document: int | None = None  # diversity: at most this many passages from one document


BASELINE = AdvancedConfig(normalize=False)
# The setting the P11 benchmark supports (artifacts/evaluation/retrieval_p11.json): feedback expansion lifted Recall@5 0.976 -> 1.000 and P@3
# 0.480 -> 0.488 with MRR unchanged, and duplicate suppression removed all duplicate pairs from the top 5 at no cost in relevance. Title/heading/phrase
# boosts and variant expansion showed no gain (or a small loss) on that set, so they stay off. It is opt-in: the BM25 baseline remains the default.
RECOMMENDED = AdvancedConfig(normalize=True, prf_terms=5, suppress_duplicates=True)


@dataclass(frozen=True)
class ExplainedHit:
    passage: Passage
    score: float
    components: dict[str, float]              # bm25 | expansion | phrase | title | heading  (their sum is `score`)
    term_contributions: dict[str, float]      # query stem -> BM25 contribution (original query terms only)
    expansion_terms: tuple[str, ...] = ()     # stems added by expansion that matched this passage
    phrases: tuple[str, ...] = ()             # adjacent query terms found contiguously
    duplicates: tuple[str, ...] = ()          # ids of lower-ranked passages suppressed as near-copies of this one


def normalize_query(query: str) -> str:
    """Canonical form of a query: quoted phrases are kept verbatim (lower-cased), everything else is NFKC-folded and cleaned."""
    quoted = _QUOTED.findall(query)
    rest = _QUOTED.sub(" ", query)
    rest = unicodedata.normalize("NFKC", rest).lower()
    rest = re.sub(r"[-_/\\]+", " ", rest)
    rest = re.sub(r"[^\w\s]", " ", rest)
    parts = [" ".join(rest.split())] + [q.lower() for q in quoted]
    return " ".join(p for p in parts if p)


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = raw_tokens(text)
    if len(w) < n:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _overlap(a: set, b: set) -> float:
    """Overlap coefficient |A and B| / min(|A|,|B|): a truncated or lightly edited copy scores near 1 even when it is much shorter than the
    original (plain Jaccard would not), and a very short passage cannot be 'contained' by accident (needs at least 3 shingles)."""
    if min(len(a), len(b)) < 3:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class AdvancedSearcher:
    def __init__(self, index: BM25Index, titles: dict[str, str] | None = None):
        self.index = index
        self.titles = {d: set(analyze(t)) for d, t in (titles or {}).items()}
        self._vocab: list[str] | None = None

    # ------------------------------------------------------------------ expansion
    def _vocabulary(self) -> list[str]:
        if self._vocab is None:
            self._vocab = self.index.vocabulary()
        return self._vocab

    def _variants(self, term: str) -> list[str]:
        """Corpus terms that look like morphological/orthographic variants of `term`: a shared prefix of at least 5 characters that
        covers 80 % of the shorter word ('expansion'/'expand', 'swelling'/'swell'). Terms that are too common (in over 30 % of
        passages) are never used, so a variant cannot flood the ranking."""
        if len(term) < 5:
            return []
        cap = 0.3 * max(self.index.num_passages, 1)
        out = []
        for v in self._vocabulary():
            if v == term or len(v) < 4 or self.index.doc_freq(v) > cap:
                continue
            common = 0
            for a, b in zip(term, v):
                if a != b:
                    break
                common += 1
            if common >= 5 and common >= 0.8 * min(len(term), len(v)):
                out.append(v)
        return out

    def _feedback_terms(self, base_scores: dict[int, dict[str, float]], query_terms: set[str], n: int, docs: int) -> list[str]:
        """Pseudo-relevance feedback: terms that are frequent in the top-ranked passages and rare in the corpus (tf x idf), excluding
        the query's own terms. Deterministic: ties are broken alphabetically."""
        if not base_scores or n <= 0:
            return []
        top = sorted(base_scores, key=lambda i: (-sum(base_scores[i].values()), self.index.passage_at(i).passage_id))[:docs]
        weight: Counter = Counter()
        for idx in top:
            for term, tf in Counter(analyze(self.index.passage_at(idx).text)).items():
                if term not in query_terms and len(term) >= 3:
                    weight[term] += tf * self.index.idf(term)
        ranked = sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))
        return [t for t, _ in ranked[:n]]

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 10, config: AdvancedConfig = RECOMMENDED,
               document_ids: set[str] | None = None) -> list[ExplainedHit]:
        idx_ = self.index
        if not idx_.num_passages:
            return []
        q = normalize_query(query) if config.normalize else query
        phrases, _ = split_phrases(q)
        q_terms = list(dict.fromkeys(analyze(re.sub(r'"', " ", q))))
        if not q_terms:
            return []
        base = idx_.score_stems({t: 1.0 for t in q_terms}, document_ids)

        weights: dict[str, float] = {}
        if config.variants:
            for t in q_terms:
                for v in self._variants(t):
                    if v not in q_terms:
                        weights[v] = max(weights.get(v, 0.0), config.variant_weight)
        if config.prf_terms:
            for t in self._feedback_terms(base, set(q_terms), config.prf_terms, config.prf_docs):
                weights.setdefault(t, config.prf_weight)
        expansion = idx_.score_stems(weights, document_ids) if weights else {}

        # adjacent non-stopword query terms, in order; a bigram counts as matched when its words are contiguous in the passage
        raw = [t for t in raw_tokens(re.sub(r'"', " ", q)) if t not in STOPWORDS]
        bigrams = list(dict.fromkeys(zip(raw, raw[1:])))
        q_idf = {t: idx_.idf(t) for t in q_terms}
        q_mass = sum(q_idf.values()) or 1.0

        candidates = set(base) | set(expansion)
        scored: list[tuple[float, str, int, dict, dict, tuple, tuple]] = []
        for i in candidates:
            p = idx_.passage_at(i)
            if phrases and not all(contains_phrase(p.text, ph) for ph in phrases):
                continue  # quoted phrases stay hard filters, as in the baseline
            comp = {"bm25": sum(base.get(i, {}).values())}
            ex = expansion.get(i, {})
            comp["expansion"] = sum(ex.values())
            matched_bi: list[str] = []
            if config.phrase_bonus and bigrams:
                for a, b in bigrams:
                    phrase = f"{a} {b}"
                    if contains_phrase(p.text, phrase):
                        matched_bi.append(phrase)
                        comp["phrase"] = comp.get("phrase", 0.0) + config.phrase_bonus * (idx_.idf(stem(a)) + idx_.idf(stem(b))) / 2
            if config.title_boost and p.document_id in self.titles:
                share = sum(q_idf[t] for t in q_terms if t in self.titles[p.document_id]) / q_mass
                if share:
                    comp["title"] = config.title_boost * share
            if config.heading_boost and p.section:
                sec = set(analyze(p.section))
                share = sum(q_idf[t] for t in q_terms if t in sec) / q_mass
                if share:
                    comp["heading"] = config.heading_boost * share
            if not any(comp.values()):
                continue
            scored.append((sum(comp.values()), p.passage_id, i, comp, base.get(i, {}), tuple(sorted(ex)), tuple(matched_bi)))
        scored.sort(key=lambda r: (-r[0], r[1]))

        out: list[ExplainedHit] = []
        kept_shingles: list[tuple[int, set]] = []  # (position in `out`, shingles)
        duplicates: dict[int, list[str]] = {}
        per_doc: Counter = Counter()
        for score, pid, i, comp, contrib, ex_terms, bi in scored:
            p = idx_.passage_at(i)
            if config.suppress_duplicates:
                sh = _shingles(p.text)
                dup_of = next((pos for pos, other in kept_shingles if _overlap(sh, other) >= config.duplicate_threshold), None)
                if dup_of is not None:
                    duplicates.setdefault(dup_of, []).append(pid)
                    continue
            if config.max_per_document and per_doc[p.document_id] >= config.max_per_document:
                continue
            per_doc[p.document_id] += 1
            if config.suppress_duplicates:
                kept_shingles.append((len(out), sh))
            out.append(ExplainedHit(p, score, {k_: round(v, 6) for k_, v in comp.items() if v}, {t: round(v, 6) for t, v in contrib.items()},
                                    ex_terms, tuple(bi) + tuple(phrases)))
            if len(out) >= k:
                break
        return [ExplainedHit(h.passage, h.score, h.components, h.term_contributions, h.expansion_terms, h.phrases, tuple(duplicates.get(n, ())))
                for n, h in enumerate(out)]

    def rank_documents(self, query: str, k: int = 10, config: AdvancedConfig = RECOMMENDED) -> list[tuple[str, float]]:
        """Documents ranked by their best passage (ties by document id), from a deep passage list. With duplicate suppression on, a
        document whose passages were suppressed as near-copies of a document that ranks at least as high is dropped as a duplicate document."""
        hits = self.search(query, k=max(self.index.num_passages, 1), config=config)
        best: dict[str, float] = {}
        for h in hits:
            best.setdefault(h.passage.document_id, h.score)
        if config.suppress_duplicates:
            for h in hits:
                for dup_id in h.duplicates:
                    dup_doc = self.index.get(dup_id).document_id
                    owner = h.passage.document_id
                    if dup_doc != owner and dup_doc in best and owner in best and best[owner] >= best[dup_doc]:
                        best.pop(dup_doc, None)
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
