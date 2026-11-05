"""PDF text extraction with PyMuPDF (the blueprint's choice). Text layer only: no OCR, no ML layout models.

Per page we keep (a) blocks with font size / boldness / position, used later for heading and header-footer detection, and
(b) the page's plain text, which is the reference passages are verified against.
"""
from __future__ import annotations

import gc
from pathlib import Path

from research.corpus.models import Block, ExtractedDocument, ExtractionError, Page

DEFAULT_MAX_PAGES = 1500
BOLD_FLAG = 16  # PyMuPDF span flag bit for bold


def extract_pdf(path: Path, max_pages: int = DEFAULT_MAX_PAGES) -> ExtractedDocument:
    import pymupdf  # imported lazily: the rest of the package works without it

    problem = None
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:  # PyMuPDF raises several unrelated types for damaged files
        problem = str(e)[:200]
    if problem is not None:
        # Raised outside the `except` block on purpose: a chained exception keeps PyMuPDF's half-built Document alive, and on
        # Windows that leaves the file locked so the caller could not delete it.
        gc.collect()
        raise ExtractionError("corrupt", f"cannot open PDF: {problem}")

    with doc:
        if doc.needs_pass:
            raise ExtractionError("encrypted", "PDF is password-protected; RESEARCHOS does not try to bypass passwords")
        n = doc.page_count
        if n == 0:
            raise ExtractionError("empty", "PDF has no pages")
        if n > max_pages:
            raise ExtractionError("too_many_pages", f"PDF has {n} pages; the limit is {max_pages}")

        pages: list[Page] = []
        warnings: list[str] = []
        for i in range(n):
            try:
                page = doc[i]
                data = page.get_text("dict")
                raw = page.get_text("text")
                # images or lots of vector drawing: a page that may hold text we cannot read (scan, outlined fonts)
                images = bool(page.get_images(full=False)) or len(page.get_drawings()) > 20
                height = float(page.rect.height)
            except Exception as e:
                warnings.append(f"page {i + 1}: extraction failed ({str(e)[:120]})")
                pages.append(Page(i + 1, [], "", 0.0, False))
                continue
            pages.append(Page(i + 1, _blocks(data), raw, height, images))

        meta = {k: v for k, v in (doc.metadata or {}).items() if v and k in ("title", "author", "subject", "keywords", "creationDate")}
        return ExtractedDocument("pdf", f"pymupdf-{pymupdf.VersionBind}", pages, meta, warnings)


def _blocks(data: dict) -> list[Block]:
    out: list[Block] = []
    for b in data.get("blocks", []):
        if b.get("type") != 0:  # 0 = text block; 1 = image
            continue
        lines: list[str] = []
        weight = 0
        size_sum = 0.0
        bold_chars = 0
        for line in b.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", []))
            if text.strip():
                lines.append(text.rstrip())
            for s in line.get("spans", []):
                n = len(s.get("text", "").strip())
                if not n:
                    continue
                weight += n
                size_sum += float(s.get("size", 0.0)) * n
                if int(s.get("flags", 0)) & BOLD_FLAG or "bold" in str(s.get("font", "")).lower():
                    bold_chars += n
        text = "\n".join(lines).strip()
        if not text or weight == 0:
            continue
        bbox = tuple(round(float(v), 1) for v in b.get("bbox", (0, 0, 0, 0)))
        out.append(Block(text=text, size=round(size_sum / weight, 1), bold=bold_chars / weight > 0.5, bbox=bbox))  # type: ignore[arg-type]
    return out
