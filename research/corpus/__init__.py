"""Document ingestion and persistent retrieval (blueprint Phase 7).

The public names are imported lazily (PEP 562). The isolated extraction subprocess starts with `python -m research.corpus.worker`, which
imports this package first; loading the whole retrieval stack there cost ~280 ms per uploaded document for code the parser never uses
(measured in docs/PERFORMANCE.md). `from research.corpus import Corpus` works exactly as before."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Corpus": "research.corpus.corpus", "CorpusHit": "research.corpus.corpus", "Inspection": "research.corpus.corpus",
    "SearchFilters": "research.corpus.corpus", "Verification": "research.corpus.corpus",
    "DocumentRecord": "research.corpus.models", "ExtractionError": "research.corpus.models", "IngestReport": "research.corpus.models",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_EXPORTS])
