from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIED = "UNVERIFIED"


STATUSES = tuple(s.value for s in ClaimStatus)


class EvidenceRole(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    CONTEXTUAL = "contextual"


class NoveltyLabel(str, Enum):
    POTENTIALLY_NOVEL = "Potentially novel within searched corpus"
    SIMILAR_PRIOR_WORK = "Similar prior work identified"
    UNCLEAR = "Novelty unclear"
    INSUFFICIENT_COVERAGE = "Insufficient literature coverage"


CLAIM_TYPES = ("result", "method", "limitation", "background", "finding", "gap", "hypothesis", "other")
UNCHECKABLE_TYPES = ("gap", "hypothesis")  # proposals about the future or about the corpus: reading documents cannot verify them


@dataclass
class Candidate:
    """A passage offered as evidence for or against a claim. `resolved` is None until the passage was looked up in the corpus;
    only text that was actually read from a stored passage is ever used - the caller supplies it, this module never invents any."""
    passage_id: str
    document_id: str
    text: str = ""
    page_number: int | None = None
    quality: str = "ok"
    resolved: bool | None = None
    retrieval_method: str = "attached"   # bm25_search | followup_search | source_passage | attached
    retrieval_score: float | None = None  # recorded for the audit trail; never used to decide a status
