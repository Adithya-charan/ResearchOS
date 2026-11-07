"""Evidence verification and bounded novelty assessment. Deterministic and lexical: nothing here uses a model, and nothing here
proves a natural-language claim true. See docs/EVIDENCE_VERIFICATION.md."""
from research.evidence.novelty import assess_novelty  # noqa: F401
from research.evidence.types import (CLAIM_TYPES, STATUSES, Candidate, ClaimStatus, EvidenceRole,  # noqa: F401
                                     NoveltyLabel)
from research.evidence.verify import verify_claim  # noqa: F401
