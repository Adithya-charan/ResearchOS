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

_NUM = re.compile(r"\d+(?:\.\d+)?")
# Negation of the statement itself. "without" and a bare "no" are left out on purpose: "cells without the additive" names a
# comparison baseline and "no. 5" is not a negation, so both would raise false contradictions.
_NEG = re.compile(r"\b(?:not|never|cannot|neither|nor|fail(?:s|ed)? to|no (?:significant|measurable|clear|effect|evidence|difference|"
                  r"change|improvement|increase|decrease|benefit))\b|n't\b", re.I)
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


def _score_sentence(claim_stems: set[str], claim_nums: set[str], claim_dir: int, claim_neg: bool, sentence: str) -> dict:
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
        elif cov >= DIRECT_COVERAGE and not claim_dir and not sdir and claim_neg != bool(_NEG.search(sentence)):
            contradicts, reason = True, "the sentence negates (or, if the claim is negated, affirms) the same statement"
        elif claim_nums and nums and not (claim_nums & nums) and cov >= 0.6 and not (claim_dir and sdir and claim_dir == sdir):
            # different figures for the same quantity - unless both sentences agree on the direction of change, in which case
            # they are two measurements of the same finding, not a contradiction (the numbers just do not match)
            contradicts, reason = True, f"the sentence gives different figure(s) ({', '.join(sorted(nums))}) for the same subject"
    return {"coverage": round(cov, 4), "numbers_match": nums_ok, "contradicts": contradicts, "reason": reason,
            "matched_terms": sorted(claim_stems & stems), "sentence": sentence}


def _best(claim_stems, claim_nums, claim_dir, claim_neg, text: str) -> dict | None:
    scored = [_score_sentence(claim_stems, claim_nums, claim_dir, claim_neg, s) for s in _sentences(text)]
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
    dropped = 0

    def base(c: Candidate) -> dict:
        return {"passage_id": c.passage_id, "document_id": c.document_id, "page_number": c.page_number,
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
                             "coverage": None, "numbers_match": None, "matched_terms": [],
                             "quote": _norm(c.text)[:MAX_QUOTE], "reason": "context only; this claim cannot be checked by reading documents"})
            continue
        r = _best(claim_stems, claim_nums, claim_dir, claim_neg, c.text)
        if r is None or r["coverage"] < CONTEXT_COVERAGE:
            dropped += 1
            continue
        role, directness = _classify(r)
        evidence.append({**base(c), "citation_valid": True, "role": role.value, "directness": directness, "coverage": r["coverage"],
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
        "evidence": evidence, "invalid_citations": invalid, "unreliable_passages": unreliable,
        "checks": {"claim_terms": sorted(claim_stems), "claim_numbers": sorted(claim_nums), "searched": searched,
                   "candidates": len(candidates), "dropped_irrelevant": dropped},
        "scope_note": SCOPE_NOTE,
    }
