"""Corpus-scoped research gap analysis (P11.3).

Everything here describes the documents that are IN the library. A "gap" is something this set of documents does not do, says it does not do,
or disagrees about; it is never a claim about the literature at large and never a claim that an idea is novel. Each gap is labelled by its basis:

  observed    a quotation or a computed fact about stored passages (stated limitation, unanswered question, single-document theme,
              conflicting findings found by the claim verifier). Every observed gap links to the documents and passages it comes from.
  hypothesis  a suggestion derived from word co-occurrence (a comparison no document makes, a follow-up). It is a prompt for a person, linked
              to the passages that motivated it, and says so.

No model is used: themes are stems that recur across documents, conflicts come from the same lexical verifier as claim checking (so they carry
its false-positive risk), limitations are sentences in a limitations section or with limitation wording."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations

from research.bm25.analysis import analyze, raw_tokens
from research.bm25.index import BM25Index
from research.evidence import Candidate, verify_claim
from research.evidence.verify import _sentences

SCOPE = ("These gaps describe only the documents in this library. 'Observed' means quoted or computed from stored passages; 'hypothesis' means a "
         "suggestion from word patterns that a person must judge. Neither says anything about the literature outside the library, and none "
         "says an idea is new.")
LIMITATIONS = [
    "Themes are recurring word stems, not concepts: synonyms count as different themes and a shared word can mean different things.",
    "Conflicts come from lexical claim verification and can be false alarms (shared words, opposite direction words); both passages are shown so a person can judge.",
    "A document that does not mention a limitation has not necessarily avoided it; only stated limitations are found.",
    "Missing comparisons are hypotheses from word co-occurrence; two themes never appearing together is weak evidence that nobody compared them.",
    "Small libraries give unstable theme counts: with few documents, 'frequent' and 'single-document' mean little.",
]

LIMITATION_SECTION = re.compile(r"limitation|threat|caveat|weakness", re.I)
LIMITATION_WORDING = re.compile(r"\b(?:limitation|limited to|only (?:one|a single|\w+ (?:was|were))|single (?:manufacturer|site|cell format|cohort)|"
                                r"not (?:varied|separated|controlled|measured|tested|evaluated|determined|explored)|was not|were not|"
                                r"may not (?:reflect|generali[sz]e)|cannot (?:be )?generali[sz]ed?)\b", re.I)
UNANSWERED = re.compile(r"\b(?:unknown|unclear|not (?:varied|separated|controlled|determined|explored|tested|evaluated)|remains?|"
                        r"no (?:comparison|\w+ comparison)|only (?:one|a single)|was not varied|were not varied|for a subset)\b", re.I)
MAX_THEMES, MAX_CONFLICT_CHECKS, MAX_EACH = 12, 600, 8
#: words that appear in titles of any field and say nothing about topic; they would link unrelated documents
GENERIC = frozenset(s for w in (
    "after before during between within without with against across over under lower higher shows show showed free store stored effect effects "
    "study studies using based new more less large small high low result results method methods analysis impact role case towards toward "
    "improved improves reduces reduced raises increase increases decrease decreases cycles cycle rate rates level levels time long short").split()
    for s in analyze(w))
MAX_QUOTE = 240


def _quote(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= MAX_QUOTE else s[:MAX_QUOTE - 3].rstrip() + "..."


def _ref(p, quote: str = "") -> dict:
    return {"document_id": p.document_id, "passage_id": p.passage_id, "page_number": p.page_number, "quote": _quote(quote or p.text)}


def _themes(index: BM25Index, titles: dict[str, str]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """stem -> documents using it (title or text), and stem -> the most common surface word."""
    docs: dict[str, set[str]] = defaultdict(set)
    surface: dict[str, Counter] = defaultdict(Counter)
    for d, title in titles.items():
        texts = [title, *(p.text for p in index.document_passages(d))]
        for t in texts:
            for tok in raw_tokens(t):
                s = analyze(tok)
                if s and len(s[0]) >= 4 and not s[0].isdigit():
                    docs[s[0]].add(d)
                    surface[s[0]][tok.lower()] += 1
    return docs, {s: c.most_common(1)[0][0] for s, c in surface.items()}


def _sentence_passages(index: BM25Index, doc: str):
    for p in index.document_passages(doc):
        for s in _sentences(p.text):
            yield p, s


def analyze_corpus_gaps(index: BM25Index, titles: dict[str, str]) -> dict:
    """Gap report for every document in `titles` (document id -> title) that is in `index`."""
    titles = {d: t for d, t in titles.items() if index.has_document(d)}
    n = len(titles)
    out: dict = {"scope": SCOPE, "limitations": LIMITATIONS, "documents": n, "gaps": [], "themes": {"frequent": [], "single_document": []},
                 "follow_ups": []}
    if n == 0:
        out["note"] = "The library has no indexed documents, so there is nothing to analyse."
        out["counts"] = {"observed": 0, "hypothesis": 0, "by_type": {}}
        return out
    gaps: list[dict] = out["gaps"]

    def add(kind: str, basis: str, statement: str, evidence: list[dict], **extra) -> str:
        gid = f"gap{len(gaps) + 1}"
        gaps.append({"id": gid, "type": kind, "basis": basis, "statement": statement, "evidence": evidence,
                     "documents": sorted({e["document_id"] for e in evidence}), **extra})
        return gid

    # ---- themes ----------------------------------------------------------------------------------------------------------------------
    stem_docs, surface = _themes(index, titles)
    title_stems = {s for t in titles.values() for tok in raw_tokens(t) for s in analyze(tok) if len(s) >= 4 and s not in GENERIC}
    ceiling = max(2, int(0.7 * n))  # a word in nearly every document names the whole corpus, not a theme
    # only title words count as themes: a title states what a document is about, whereas body words like 'percent' or 'after' are everywhere
    freq = sorted(((s, d) for s, d in stem_docs.items() if s in title_stems and 2 <= len(d) <= ceiling),
                  key=lambda x: (-len(x[1]), x[0]))[:MAX_THEMES]
    out["themes"]["frequent"] = [{"theme": surface[s], "stem": s, "documents": sorted(d), "document_count": len(d)} for s, d in freq]
    single = sorted(((s, next(iter(d))) for s, d in stem_docs.items() if len(d) == 1 and s in title_stems), key=lambda x: (x[1], x[0]))
    out["themes"]["single_document"] = [{"theme": surface[s], "stem": s, "document": d, "title": titles[d]} for s, d in single[:MAX_THEMES * 2]]
    docs_stems = {d: {s for s, ds in stem_docs.items() if d in ds} for d in titles}
    if n >= 3:  # a theme is only "uncorroborated" relative to related documents; an unrelated document corroborates nothing about it
        by_doc: dict[str, list[str]] = defaultdict(list)
        for s, d in single:
            by_doc[d].append(surface[s])
        ranked = []
        for d, words in by_doc.items():
            related = [o for o in titles if o != d and len(docs_stems[d] & docs_stems[o] & title_stems) >= 1 and len(docs_stems[d] & docs_stems[o]) >= 6]
            if related:
                ranked.append((-len(related), d, words, related))
        for _, d, words, related in sorted(ranked)[:MAX_EACH]:
            p0 = index.document_passages(d)[0]
            add("single_document_theme", "observed",
                f"Among {len(related)} related document(s), only \"{titles[d]}\" uses {', '.join(repr(w) for w in words[:3])}; "
                "the related documents do not corroborate it within this library.", [_ref(p0, titles[d])], themes=words[:3], related_documents=sorted(related))

    # ---- stated limitations and unanswered questions ---------------------------------------------------------------------------------
    for d in titles:
        seen: set[str] = set()
        for p, s in _sentence_passages(index, d):
            in_section = bool(LIMITATION_SECTION.search(p.section or ""))
            if s in seen or not (in_section or LIMITATION_WORDING.search(s)):
                continue
            seen.add(s)
            unanswered = bool(UNANSWERED.search(s))
            add("unanswered_question" if unanswered else "methodological_limitation", "observed",
                f"{titles[d]} states: \"{_quote(s)}\"" + (" This leaves a question open that later work could address." if unanswered else ""),
                [_ref(p, s)])

    # ---- conflicting findings (same verifier as claim checking) ----------------------------------------------------------------------
    checks = 0
    seen_pairs: set[frozenset] = set()
    for a, b in [x for pair in combinations(sorted(titles), 2) for x in (pair, pair[::-1])]:  # both directions: A against B and B against A
        if len(docs_stems[a] & docs_stems[b]) < 4:
            continue
        cand_b = [Candidate(passage_id=p.passage_id, document_id=b, text=p.text, resolved=True, page_number=p.page_number, section=p.section)
                  for p in index.document_passages(b)]
        for p, s in _sentence_passages(index, a):
            if checks >= MAX_CONFLICT_CHECKS or len(s.split()) < 6 or p.section.lower().startswith("abstract"):
                continue
            checks += 1
            r = verify_claim(s, "result", cand_b, searched=True)
            if r["status"] != "CONTRADICTED":
                continue
            contra = next((e for e in r["evidence"] if e["role"] == "contradicting"), None)
            key = frozenset({p.passage_id, contra["passage_id"]}) if contra else None
            if contra is None or key in seen_pairs:
                continue
            seen_pairs.add(key)
            q = index.get(contra["passage_id"])
            add("conflicting_findings", "observed",
                f"{titles[a]} and {titles[b]} appear to disagree. {titles[a]}: \"{_quote(s)}\" {titles[b]}: \"{_quote(contra['quote'])}\" "
                f"(flagged by the lexical verifier: {contra['reason']}).",
                [_ref(p, s), _ref(q, contra["quote"])], verifier_reason=contra["reason"],
                caveat="Lexical detection can be a false alarm (different populations, conditions or measures); read both passages.")

    # ---- missing comparisons (hypothesis) --------------------------------------------------------------------------------------------
    pair_count = 0
    frequent = [s for s, _ in freq]
    for (s1, d1), (s2, d2) in combinations(freq, 2):
        if pair_count >= 4 or d1 & d2:
            continue  # they co-occur somewhere: a document already relates them
        # context = themes found in documents of BOTH groups; unrelated fields share only generic words, so they have none
        common = [t for t in frequent if t not in (s1, s2) and (stem_docs[t] & d1) and (stem_docs[t] & d2)]
        if len(common) < 3 or len(d1 | d2) > 0.5 * n:
            continue  # little shared context, or the two groups together are most of the library: not a specific missing comparison
        e1 = index.document_passages(sorted(d1)[0])[0]
        e2 = index.document_passages(sorted(d2)[0])[0]
        add("missing_comparison", "hypothesis",
            f"'{surface[s1]}' ({', '.join(sorted(d1))}) and '{surface[s2]}' ({', '.join(sorted(d2))}) are never discussed in the same document, "
            f"although both groups also discuss {', '.join(repr(surface[t]) for t in common[:3])}. Whether anyone compared them is not known from this library.",
            [_ref(e1), _ref(e2)], themes=[surface[s1], surface[s2]],
            caveat="Word co-occurrence only; the two themes may be unrelated.")
        pair_count += 1

    # ---- follow-ups (hypothesis), each tied to the gap that motivates it -------------------------------------------------------------
    for g in gaps:
        if g["type"] == "conflicting_findings":
            fu = "Reconcile the two findings: compare populations, conditions and measures, or add a document that tests both."
        elif g["type"] == "unanswered_question":
            fu = "Design a study that varies the factor the document says it left fixed or unmeasured."
        elif g["type"] == "missing_comparison":
            fu = "Check whether a comparison exists outside the library; if not, it is a candidate for a direct comparison."
        else:
            continue
        out["follow_ups"].append({"for_gap": g["id"], "basis": "hypothesis", "suggestion": fu, "documents": g["documents"]})

    by_type = Counter(g["type"] for g in gaps)
    out["counts"] = {"observed": sum(g["basis"] == "observed" for g in gaps), "hypothesis": sum(g["basis"] == "hypothesis" for g in gaps),
                     "by_type": dict(by_type), "conflict_checks_run": checks}
    return out


def validate_gap_report(report: dict, index: BM25Index) -> list[str]:
    """Problems with a gap report: any gap without evidence, a link to a passage that does not exist, a quote not in its passage, or a
    claim of novelty. An empty list means every gap is traceable."""
    problems: list[str] = []
    for g in report["gaps"]:
        if g["basis"] not in ("observed", "hypothesis"):
            problems.append(f"{g['id']}: unknown basis {g['basis']!r}")
        if not g["evidence"]:
            problems.append(f"{g['id']}: no linked passage")
        if re.search(r"\b(?:novel|never been|first to|no one has)\b", g["statement"], re.I):
            problems.append(f"{g['id']}: claims novelty")
        for e in g["evidence"]:
            p = index.get(e["passage_id"])
            if p is None or p.document_id != e["document_id"]:
                problems.append(f"{g['id']}: passage {e['passage_id']} does not resolve to {e['document_id']}")
            elif g["basis"] == "observed" and g["type"] in ("methodological_limitation", "unanswered_question", "conflicting_findings") \
                    and e["quote"].rstrip(".").removesuffix("...") and " ".join(e["quote"].rstrip(".").removesuffix("...").split()).lower() not in " ".join(p.text.split()).lower():
                problems.append(f"{g['id']}: quote is not in passage {e['passage_id']}")
    return problems
