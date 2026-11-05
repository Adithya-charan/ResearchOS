"""Read-only corpus tools backed by the BM25 index, plus the evidence resolver that stops fabricated citations."""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from agent_engine.contracts import EvidenceReference
from agent_engine.tools.registry import Tool
from research.bm25.index import BM25Index

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
SNIPPET_CHARS = 240


def sanitize_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Passage text is untrusted data: strip control characters and cap the length before agents see it."""
    text = " ".join(_CONTROL_RE.sub(" ", text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=8, ge=1, le=50)
    document_ids: list[str] | None = None


class HitModel(BaseModel):
    passage_id: str
    document_id: str
    page_number: int | None
    section: str
    score: float
    snippet: str
    matched_terms: list[str]
    quality: str = "ok"  # extraction quality of the passage's source page: ok | low


class SearchResult(BaseModel):
    query_terms: list[str]
    hits: list[HitModel]
    corpus_documents: int
    corpus_passages: int


class GetPassageArgs(BaseModel):
    passage_id: str = Field(min_length=1)


class PassageResult(BaseModel):
    found: bool
    passage_id: str
    document_id: str | None = None
    page_number: int | None = None
    section: str = ""
    text: str = ""
    quality: str = "ok"


class DocumentPassagesArgs(BaseModel):
    document_id: str = Field(min_length=1)
    limit: int = Field(default=200, ge=1, le=500)


class DocPassage(BaseModel):
    passage_id: str
    page_number: int | None
    section: str
    quality: str
    text: str


class DocumentPassagesResult(BaseModel):
    found: bool
    document_id: str
    info: dict = Field(default_factory=dict)  # title / year / doi / licence when the corpus knows them
    passages: list[DocPassage] = Field(default_factory=list)
    truncated: bool = False


class VerifyArgs(BaseModel):
    passage_id: str = Field(min_length=1)


class VerifyResult(BaseModel):
    ok: bool
    message: str


def build_corpus_tools(index: BM25Index, catalog: Callable[[str], dict | None] | None = None,
                       verifier: Callable[[str], tuple[bool, str]] | None = None) -> list[Tool]:
    """`catalog(document_id)` supplies document metadata (title, year, ...) when a document library is attached;
    `verifier(passage_id)` re-checks a passage against its stored original file. Both are optional."""
    from research.bm25.analysis import analyze

    def search(args: SearchArgs) -> SearchResult:
        docs = set(args.document_ids) if args.document_ids else None
        hits = index.search(args.query, k=args.k, document_ids=docs)
        return SearchResult(
            query_terms=list(dict.fromkeys(analyze(args.query))),
            hits=[
                HitModel(
                    passage_id=h.passage.passage_id, document_id=h.passage.document_id,
                    page_number=h.passage.page_number, section=h.passage.section, score=round(h.score, 6),
                    snippet=sanitize_snippet(h.passage.text), matched_terms=list(h.matched_terms),
                    quality=h.passage.quality,
                )
                for h in hits
            ],
            corpus_documents=index.num_documents,
            corpus_passages=index.num_passages,
        )

    def get_passage(args: GetPassageArgs) -> PassageResult:
        p = index.get(args.passage_id)
        if p is None:
            return PassageResult(found=False, passage_id=args.passage_id)
        return PassageResult(found=True, passage_id=p.passage_id, document_id=p.document_id,
                             page_number=p.page_number, section=p.section, text=p.text, quality=p.quality)

    def document_passages(args: DocumentPassagesArgs) -> DocumentPassagesResult:
        if not index.has_document(args.document_id):
            return DocumentPassagesResult(found=False, document_id=args.document_id)
        ps = index.document_passages(args.document_id)
        info = (catalog(args.document_id) if catalog else None) or {}
        return DocumentPassagesResult(
            found=True, document_id=args.document_id, info=info, truncated=len(ps) > args.limit,
            passages=[DocPassage(passage_id=p.passage_id, page_number=p.page_number, section=p.section, quality=p.quality,
                                 text=" ".join(_CONTROL_RE.sub(" ", p.text).split())) for p in ps[: args.limit]])

    tools = [
        Tool("corpus.search", "BM25 search over indexed passages", SearchArgs, search),
        Tool("corpus.get_passage", "Fetch the full text of one indexed passage by id", GetPassageArgs, get_passage),
        Tool("corpus.document_passages", "All indexed passages of one document, in reading order, with its metadata",
             DocumentPassagesArgs, document_passages),
    ]
    if verifier is not None:
        async def verify(args: VerifyArgs) -> VerifyResult:
            ok, message = await asyncio.to_thread(verifier, args.passage_id)  # re-reads a file: keep the event loop free
            return VerifyResult(ok=ok, message=message)

        tools.append(Tool("corpus.verify_passage", "Check a passage against the stored original file", VerifyArgs, verify))
    return tools


class IndexEvidenceResolver:
    """An agent may only cite what exists. Returns None if the reference is valid, else the reason it is not."""

    def __init__(self, index: BM25Index, require_passage: bool = True):
        self.index, self.require_passage = index, require_passage

    def check(self, ref: EvidenceReference) -> str | None:
        if ref.passage_id is None:
            if self.require_passage:
                return f"evidence for document '{ref.document_id}' has no passage_id"
            return None if self.index.has_document(ref.document_id) else f"unknown document '{ref.document_id}'"
        p = self.index.get(ref.passage_id)
        if p is None:
            return f"unknown passage '{ref.passage_id}'"
        if p.document_id != ref.document_id:
            return f"passage '{ref.passage_id}' belongs to '{p.document_id}', not '{ref.document_id}'"
        if ref.page_number != p.page_number:
            return f"passage '{ref.passage_id}' is on page {p.page_number}, not {ref.page_number}"
        return None
