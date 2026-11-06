"""Plain data types shared by the ingestion pipeline (extraction -> quality -> segmentation -> store)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ExtractionError(Exception):
    """A file that cannot be ingested. `code` is stable and machine-readable; `message` is for people."""

    CODES = ("unsupported_type", "type_mismatch", "too_large", "empty", "corrupt", "encrypted", "unsafe",
             "too_many_pages", "no_text", "timeout", "worker_failed", "invalid_metadata")

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class Block:
    """One paragraph-like unit of text as it appears in the source."""

    text: str
    size: float = 0.0  # dominant font size in points (0 = unknown / not applicable)
    bold: bool = False
    heading_level: int = 0  # 0 = body text; >0 = heading (set by the extractor if the format knows, else by segmentation)
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class Page:
    number: int | None  # 1-based for paged formats (PDF); None for formats without pages
    blocks: list[Block]
    raw_text: str  # the extractor's plain text for the page, untouched: the reference for verifying passages later
    height: float = 0.0
    has_images: bool = False


@dataclass
class ExtractedDocument:
    file_type: str  # pdf | docx | txt | md | html
    method: str  # e.g. "pymupdf-1.28.2"; stored with every passage
    pages: list[Page]
    metadata: dict[str, Any] = field(default_factory=dict)  # whatever the file itself declares (never guessed)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExtractedDocument:
        pages = [
            Page(
                number=p["number"], raw_text=p["raw_text"], height=p.get("height", 0.0), has_images=p.get("has_images", False),
                blocks=[Block(text=b["text"], size=b.get("size", 0.0), bold=b.get("bold", False),
                              heading_level=b.get("heading_level", 0),
                              bbox=tuple(b["bbox"]) if b.get("bbox") else None) for b in p["blocks"]],
            )
            for p in d["pages"]
        ]
        return cls(d["file_type"], d["method"], pages, dict(d.get("metadata", {})), list(d.get("warnings", [])))


@dataclass
class DocumentRecord:
    """A document as stored (blueprint 11.3: hash, method, quality, source and licence travel with the text)."""

    id: str
    file_hash: str
    filename: str
    file_type: str
    size_bytes: int
    title: str
    title_source: str
    authors: list[str]
    year: int | None
    year_source: str | None
    doi: str | None
    page_count: int
    extraction_method: str
    extraction_status: str  # ok | partial  (documents with no text at all are rejected, not stored)
    quality: dict[str, Any]
    warnings: list[str]
    source: str | None
    license: str | None
    allowed_use: list[str]
    ingested_at: str
    stored_path: str
    # Declared by the person ingesting (never guessed from the file): full publication date, source URL, kind of document.
    publication_date: str | None = None
    url: str | None = None
    document_type: str | None = None

    @property
    def searchable(self) -> bool:
        return "search" in self.allowed_use

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestReport:
    path: str
    status: str  # ingested | duplicate | rejected | failed
    document_id: str | None = None
    code: str | None = None  # ExtractionError code when not ingested
    message: str = ""
    pages: int = 0
    passages: int = 0
    quality_status: str | None = None
    warnings: list[str] = field(default_factory=list)
    pages_without_text: list[int] = field(default_factory=list)  # image-only pages (OCR is not available)
    pages_low_quality: list[int] = field(default_factory=list)  # text was extracted but looks unreliable

    @property
    def ok(self) -> bool:
        return self.status in ("ingested", "duplicate")

    def to_dict(self) -> dict:
        return asdict(self)
