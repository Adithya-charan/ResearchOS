"""Claim verification against stored passages.

What this decides: whether the cited text, read sentence by sentence, *addresses* the claim (shares its content words and its
numbers), and whether any sentence about the same thing points the other way (opposite direction words, a negation, or a different
figure). What it does not decide: whether the claim is true. A SUPPORTED claim has a stored passage that states it in nearly the
same words; it is not a proof, and the source itself may be wrong. Retrieval scores are recorded for the audit trail but are never
an input to the status - a highly ranked passage is relevant, not correct.
"""
from __future__ import annotations

import re

from research.bm25.analysis import analyze
from research.bm25.index import _SENTENCE_RE
from research.evidence.types import UNCHECKABLE_TYPES, Candidate, ClaimStatus, EvidenceRole

DIRECT_COVERAGE = 0.75     # share of the claim's content stems found in one sentence
INDIRECT_COVERAGE = 0.50   # at least half: the sentence covers part of the claim
CONTEXT_COVERAGE = 0.25    # below this a passage is not evidence of any kind and is dropped
MIN_CLAIM_STEMS = 3
MAX_QUOTE = 400
MAX_PASSAGE = 4000   # the stored original passage is kept with each evidence record (passages are ~110 words, so this never truncates in practice)

_NUM = re.compile(r"\d+(?:\.\d+)?")
# Negation of the statement itself. "without" and a bare "no" are left out on purpose: "cells without the additive" names a
# comparison baseline and "no. 5" is not a negation, so both would raise false contradictions.
_NEG = re.compile(r"\b(?:not|never|cannot|neither|nor|fail(?:s|ed)? to|no (?:significant|measurable|clear|effect|evidence|difference|"
                  r"change|improvement|increase|decrease|benefit))\b|n't\b", re.I)
# "no significant difference", "did not improve": a null result. Compared with a sentence that reports a change on the same subject it is a
# contradiction even though neither side has a matching negation word (the old rule needed both directions to be unclear).
_NULL = re.compile(r"\bno (?:(?:statistically |significant |measurable |clear |meaningful )*)?(?:difference|effect|benefit|improvement|change)\b"
                   r"|\b(?:did|does|do) not (?:improve|change|differ|help|reduce|increase)\b|\bwithout (?:any )?(?:benefit|improvement)\b", re.I)
_CLAUSE = re.compile(r",\s*(?:and|but|while|whereas)\b|;|\bwhile\b|\bwhereas\b|\bbut\b", re.I)
_UP = {"increase", "increased", "increases", "rise", "rose", "higher", "improve", "improved", "improves", "faster", "more",
       "gain", "raises", "raised", "longer", "above", "grew", "grow", "grows"}
_DOWN = {"decrease", "decreased", "decreases", "reduce", "reduced", "reduces", "lower", "less", "slower", "decline", "declined",
         "drop", "dropped", "fell", "fall", "falls", "below", "degrade", "degraded", "loss", "lost", "shrank"}

SCOPE_NOTE = ("Statuses come from a lexical comparison of the claim with stored passages. They are evidence-strength labels, not "
              "proofs and not probabilities of truth; a human must judge anything that matters.")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_RE.split(_norm(text))]
    return [s for s in parts if len(s.split()) >= 3] or ([_norm(text)[:MAX_QUOTE]] if _norm(text) else [])


def _direction(sentence: str) -> int:
    """+1 / -1 for an increase / decrease statement (flipped by a negation), 0 when unclear or mixed."""
    words = set(re.findall(r"[a-z]+", sentence.lower()))
    up, down = bool(words & _UP), bool(words & _DOWN)
    d = 1 if up and not down else -1 if down and not up else 0
    return -d if d and _NEG.search(sentence) else d


def _clause_direction(sentence: str, claim_stems: set[str]) -> int:
    """Direction of the clause of `sentence` that overlaps the claim most: a sentence such as 'drift grew ..., and accuracy fell ...'
    has two directions, and only the clause about the claim's subject should be compared with the claim."""
    clauses = [c for c in _CLAUSE.split(sentence) if c and c.strip()]
    if len(clauses) <= 1:
        return _direction(sentence)
    best = max(clauses, key=lambda c: len(claim_stems & set(analyze(c))))
    return _direction(best)


def _numbers(text: str) -> set[str]:
    return set(_NUM.findall(text))


# A figure is compared with another only when both carry the same recognised unit ("40 percent" vs "19 percent"), so a number that
# merely recurs for a different quantity ("45 degrees" in both) cannot hide a conflicting figure, and a different unit never conflicts.
_UNITS = {"percent": "percent", "%": "percent", "degree": "degree", "degrees": "degree", "°c": "degree", "k": "kelvin", "kelvin": "kelvin",
          "volt": "volt", "volts": "volt", "v": "volt", "hour": "hour", "hours": "hour", "day": "day", "days": "day", "week": "week",
          "weeks": "week", "month": "month", "months": "month", "year": "year", "years": "year", "mg": "mg", "g": "g", "kg": "kg",
          "mm": "mm", "cm": "cm", "nm": "nm", "ms": "ms", "ml": "ml", "points": "point", "point": "point"}
_PAIR = re.compile(r"(\d+(?:\.\d+)?)\s*(%|°\s?c|[A-Za-z]+)")


def _figure_pairs(text: str) -> set[tuple[str, str]]:
    out = set()
    for num, unit in _PAIR.findall(text):
        u = _UNITS.get(unit.lower().replace(" ", ""))
        if u:
            out.add((num, u))
    return out


def _unit_mismatch(claim_pairs: set[tuple[str, str]], sentence: str) -> tuple[str, str] | None:
    """A (claim figure, sentence figure) pair that has the same unit but a different value, provided the sentence does not also state the
    claim's figure for that unit. None when nothing conflicts."""
    sent = _figure_pairs(sentence)
    for num, unit in sorted(claim_pairs):
        same_unit = {n for n, u in sent if u == unit}
        if same_unit and num not in same_unit:
            return (f"{num} {unit}", f"{', '.join(sorted(same_unit))} {unit}")
    return None


# comparative word -> (dimension, sign): 'longer' and 'shorter' are one dimension with opposite signs
_COMPARATIVES = {"earlier": ("time", 1), "later": ("time", -1), "faster": ("speed", 1), "slower": ("speed", -1), "longer": ("length", 1),
                 "shorter": ("length", -1), "higher": ("level", 1), "lower": ("level", -1), "more": ("amount", 1), "less": ("amount", -1),
                 "fewer": ("amount", -1), "larger": ("size", 1), "bigger": ("size", 1), "smaller": ("size", -1), "better": ("quality", 1),
                 "worse": ("quality", -1), "greater": ("degree", 1), "stronger": ("strength", 1), "weaker": ("strength", -1)}


def _comparative(text: str) -> tuple[str, int] | None:
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _COMPARATIVES:
            return _COMPARATIVES[w]
    return None


def _swapped_comparison(claim: str, sentence: str) -> bool:
    """True when both state 'A <predicate> than B' about the same two things but with A and B exchanged ('massive colonies bleached earlier
    than branching colonies' against 'branching colonies bleached earlier than massive colonies'). A reversed pair with the opposite
    comparative ('A longer than B' against 'B shorter than A') is one statement written two ways, so it does not count."""
    cl, sl = re.split(r"\bthan\b", claim, maxsplit=1, flags=re.I), re.split(r"\bthan\b", sentence, maxsplit=1, flags=re.I)
    if len(cl) != 2 or len(sl) != 2:
        return False
    cl_l, cl_r, sl_l, sl_r = (set(analyze(x)) for x in (cl[0], cl[1], sl[0], sl[1]))
    moved_l = (cl_l & sl_r) - sl_l  # words the claim puts before 'than' that the sentence puts after it
    moved_r = (cl_r & sl_l) - sl_r
    if not moved_l or not moved_r:
        return False
    c_claim, c_sent = _comparative(cl[0]), _comparative(sl[0])
    if c_claim and c_sent and c_claim[0] == c_sent[0] and c_claim[1] != c_sent[1]:
        return False  # opposite comparatives on swapped items: they agree
    d_claim, d_sent = _direction(cl[0]), _direction(sl[0])
    return not (d_claim and d_sent and d_claim != d_sent)


def _null_vs_change(claim_text: str, claim_dir: int, sentence: str, sdir: int) -> bool:
    """A null result on one side ('no significant difference') against an un-negated change on the other ('improved')."""
    c_null, s_null = bool(_NULL.search(claim_text)), bool(_NULL.search(sentence))
    if c_null == s_null:
        return False  # both null or neither: other rules decide
    changed_dir = sdir if c_null else claim_dir
    changed_text = sentence if c_null else claim_text
    return bool(changed_dir) and not _NEG.search(changed_text)


def _score_sentence(claim_stems: set[str], claim_nums: set[str], claim_dir: int, claim_neg: bool, sentence: str, claim_text: str = "") -> dict:
    stems = set(analyze(sentence))
    cov = len(claim_stems & stems) / len(claim_stems)
    nums = _numbers(sentence)
    nums_ok = None if not claim_nums else claim_nums <= nums
    sdir = _clause_direction(sentence, claim_stems)
    contradicts = False
    reason = ""
    if cov >= INDIRECT_COVERAGE:
        if claim_dir and sdir and claim_dir * sdir == -1:
            contradicts, reason = True, "the sentence reports the opposite direction of change on the same subject"
        elif _null_vs_change(claim_text, claim_dir, sentence, sdir):
            contradicts, reason = True, "one side reports no effect or difference on the subject while the other reports a change"
        elif cov >= DIRECT_COVERAGE and not claim_dir and not sdir and claim_neg != bool(_NEG.search(sentence)):
            contradicts, reason = True, "the sentence negates (or, if the claim is negated, affirms) the same statement"
        elif claim_nums and nums and not (claim_nums & nums) and cov >= 0.6 and not (claim_dir and sdir and claim_dir == sdir):
            # different figures for the same quantity - unless both sentences agree on the direction of change, in which case
            # they are two measurements of the same finding, not a contradiction (the numbers just do not match)
            contradicts, reason = True, f"the sentence gives different figure(s) ({', '.join(sorted(nums))}) for the same subject"
    if not contradicts and cov >= 0.6:
        pairs = _figure_pairs(claim_text)
        mm = _unit_mismatch(pairs, sentence) if pairs else None
        if mm is not None:
            contradicts, reason = True, f"the sentence gives {mm[1]} where the claim says {mm[0]} (same unit, different figure)"
        elif cov >= DIRECT_COVERAGE and _swapped_comparison(claim_text, sentence):
            contradicts, reason = True, "the sentence compares the same two things the other way round (roles swapped)"
    return {"coverage": round(cov, 4), "numbers_match": nums_ok, "contradicts": contradicts, "reason": reason,
            "matched_terms": sorted(claim_stems & stems), "sentence": sentence}


def _best(claim_stems, claim_nums, claim_dir, claim_neg, text: str, claim_text: str = "") -> dict | None:
    scored = [_score_sentence(claim_stems, claim_nums, claim_dir, claim_neg, s, claim_text) for s in _sentences(text)]
    if not scored:
        return None
    # a contradicting sentence about the same subject is more informative than a merely-overlapping one
    return max(scored, key=lambda r: (r["coverage"] >= INDIRECT_COVERAGE and r["contradicts"], r["coverage"],
                                      r["numbers_match"] is True))


def _classify(r: dict) -> tuple[EvidenceRole, str | None]:
    if r["coverage"] < CONTEXT_COVERAGE:
        return EvidenceRole.CONTEXTUAL, None  # filtered out by the caller
    if r["contradicts"]:
        return EvidenceRole.CONTRADICTING, "direct" if r["coverage"] >= DIRECT_COVERAGE else "indirect"
    if r["coverage"] >= DIRECT_COVERAGE and r["numbers_match"] is not False:
        return EvidenceRole.SUPPORTING, "direct"
    if r["coverage"] >= INDIRECT_COVERAGE:
        return EvidenceRole.SUPPORTING, "indirect"
    return EvidenceRole.CONTEXTUAL, None


def verify_claim(text: str, claim_type: str, candidates: list[Candidate], *, searched: bool = False) -> dict:
    """Verify one claim against candidate passages. `searched` says whether a retrieval step was actually run for this claim
    (so that finding nothing is INSUFFICIENT_EVIDENCE rather than UNVERIFIED)."""
    claim_stems = set(analyze(text))
    claim_nums, claim_neg = _numbers(text), bool(_NEG.search(text))
    claim_dir = _direction(text)
    evidence: list[dict] = []
    invalid: list[dict] = []
    unreliable: list[dict] = []
    irrelevant: list[dict] = []
    dropped = 0

    def base(c: Candidate) -> dict:
        # every record keeps where the text came from and the stored original, so a reader can check the quote against its source
        return {"passage_id": c.passage_id, "document_id": c.document_id, "page_number": c.page_number,
                "source_location": {"document_id": c.document_id, "passage_id": c.passage_id, "page_number": c.page_number,
                                    "section": c.section or None},
                "passage_text": _norm(c.text)[:MAX_PASSAGE] if c.text else "",
                "retrieval_method": c.retrieval_method, "retrieval_score": c.retrieval_score}

    for c in candidates:
        if c.resolved is False or not c.text:
            invalid.append({**base(c), "citation_valid": False,
                            "reason": "the cited passage could not be resolved in the corpus" if c.resolved is False
                            else "no passage text was available to read"})
            continue
        if c.quality != "ok":
            unreliable.append({**base(c), "citation_valid": True, "reason": "the page text was extracted unreliably; not used as evidence"})
            continue
        if claim_type in UNCHECKABLE_TYPES or len(claim_stems) < MIN_CLAIM_STEMS:
            evidence.append({**base(c), "citation_valid": True, "role": EvidenceRole.CONTEXTUAL.value, "directness": None,
                             "coverage": None, "relevance_score": None, "numbers_match": None, "matched_terms": [],
                             "quote": _norm(c.text)[:MAX_QUOTE], "reason": "context only; this claim cannot be checked by reading documents"})
            continue
        r = _best(claim_stems, claim_nums, claim_dir, claim_neg, c.text, text)
        if r is None or r["coverage"] < CONTEXT_COVERAGE:
            dropped += 1
            irrelevant.append({**base(c), "citation_valid": True, "role": "irrelevant", "relevance_score": r["coverage"] if r else 0.0,
                               "reason": f"shares too little with the claim (term coverage below {CONTEXT_COVERAGE}) to count as evidence of any kind"})
            continue
        role, directness = _classify(r)
        evidence.append({**base(c), "citation_valid": True, "role": role.value, "directness": directness, "coverage": r["coverage"],
                         "relevance_score": r["coverage"],
                         "numbers_match": r["numbers_match"], "matched_terms": r["matched_terms"], "quote": r["sentence"],
                         "reason": r["reason"] or (
                             "states the claim in nearly the same words" if directness == "direct" and role == EvidenceRole.SUPPORTING
                             else "covers only part of the claim" if role == EvidenceRole.SUPPORTING
                             else "overlaps in topic only")})

    def pick(role: EvidenceRole, directness: str | None = None) -> list[dict]:
        return [e for e in evidence if e["role"] == role.value and (directness is None or e["directness"] == directness)]

    sup_d, sup_i = pick(EvidenceRole.SUPPORTING, "direct"), pick(EvidenceRole.SUPPORTING, "indirect")
    con_d, con_i = pick(EvidenceRole.CONTRADICTING, "direct"), pick(EvidenceRole.CONTRADICTING, "indirect")
    reasons: list[str] = []
    conflicting = False

    if claim_type in UNCHECKABLE_TYPES:
        status = ClaimStatus.UNVERIFIED
        reasons.append(f"a {claim_type} is a proposal, not a statement a document can confirm; cited passages are listed as context only")
    elif len(claim_stems) < MIN_CLAIM_STEMS:
        status = ClaimStatus.UNVERIFIED
        reasons.append("the claim has too few content words to compare against text")
    elif not evidence and (invalid or unreliable) and not dropped:
        status = ClaimStatus.UNVERIFIED
        reasons.append("every piece of attached evidence was unusable (unresolvable citation or unreliable extraction)")
    elif not evidence and not searched and not dropped and not candidates:
        status = ClaimStatus.UNVERIFIED
        reasons.append("no evidence was attached and no search was run for this claim")
    elif con_d and not sup_d:
        status = ClaimStatus.CONTRADICTED
        reasons.append("a stored passage directly contradicts the claim and nothing directly supports it")
    elif sup_d and (con_d or con_i):
        status, conflicting = ClaimStatus.PARTIALLY_SUPPORTED, True
        reasons.append("conflicting evidence: one passage supports the claim and another contradicts it; both are kept")
    elif sup_d:
        status = ClaimStatus.SUPPORTED
        reasons.append("at least one stored passage states the claim in nearly the same words, and no passage contradicts it")
    elif con_d:
        status = ClaimStatus.CONTRADICTED
    elif sup_i:
        status = ClaimStatus.PARTIALLY_SUPPORTED
        reasons.append("only indirect evidence: passages cover part of the claim or its numbers do not match")
        if con_i:
            conflicting = True
            reasons.append("an indirect passage points the other way")
    else:
        status = ClaimStatus.INSUFFICIENT_EVIDENCE
        reasons.append("no passage directly or indirectly addresses the claim" if evidence or dropped or searched
                       else "no evidence was attached")
        if con_i:
            reasons.append("one indirect passage may point the other way (possible contradiction, not established)")

    if invalid:
        reasons.append(f"{len(invalid)} citation(s) could not be resolved and were not counted")
    if unreliable:
        reasons.append(f"{len(unreliable)} passage(s) came from unreliably extracted pages and were not counted")
    if dropped:
        reasons.append(f"{dropped} retrieved passage(s) shared too little with the claim to count as evidence of any kind")

    docs_direct = {e["document_id"] for e in (sup_d if status != ClaimStatus.CONTRADICTED else con_d)}
    if status in (ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED):
        level = "strong" if len(docs_direct) >= 2 else "moderate"
    elif status == ClaimStatus.PARTIALLY_SUPPORTED:
        level = "weak"
    else:
        level = "none"
    strength = {"level": level, "independent_documents": len(docs_direct),
                "basis": "direct passages in distinct documents" if docs_direct else "no direct passage",
                "note": "an evidence-strength label, not a probability that the claim is true"}
    return {
        "status": status.value, "strength": strength, "explanation": reasons, "conflicting": conflicting,
        "evidence": evidence, "invalid_citations": invalid, "unreliable_passages": unreliable, "irrelevant_passages": irrelevant,
        "checks": {"claim_terms": sorted(claim_stems), "claim_numbers": sorted(claim_nums), "searched": searched,
                   "candidates": len(candidates), "dropped_irrelevant": dropped},
        "scope_note": SCOPE_NOTE,
    }
