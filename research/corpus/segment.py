"""Page-aware segmentation: clean the extracted blocks, find headings, and cut passages that never cross a page boundary
(so a page citation is exact). Passage ids are `<document>:pNNN`, numbered in reading order."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from research.bm25.index import _SENTENCE_RE, Passage
from research.corpus.extract_text import looks_like_heading
from research.corpus.models import ExtractedDocument
from research.corpus.quality import PageQuality

MARGIN = 0.10  # top/bottom fraction of a page treated as header/footer territory
MERGE_BELOW = 25  # words: smaller fragments (list items, captions, short paragraphs) are merged into a neighbour
MIN_WORDS = 4  # passages shorter than this (stray labels, page furniture) are dropped
_PAGE_NO = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)


def normalize(text: str) -> str:
    """NFKC (folds ligatures such as 'fi') and collapse all whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def repair_hyphenation(text: str) -> str:
    """Re-join words broken across lines ('elec-\\ntrolyte' -> 'electrolyte'). Known limit: a real compound broken at its
    hyphen ('state-\\nof-the-art') loses that hyphen; verification ignores hyphens so it still checks out."""
    text = text.replace("­", "")
    return re.sub(r"(?<=[A-Za-z])[-‐]\n(?=[a-z])", "", text)


def squash(text: str) -> str:
    """Comparison form used to verify passages against page text: lower-case letters and digits only."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).lower())


def strip_margin_furniture(doc: ExtractedDocument) -> int:
    """Remove repeated running headers/footers and bare page numbers from page margins. Returns the number removed."""
    paged = [p for p in doc.pages if p.height > 0]
    if not paged:
        return 0

    def zone(p, b):
        if not b.bbox:
            return None
        if b.bbox[3] <= MARGIN * p.height:
            return "top"
        if b.bbox[1] >= (1 - MARGIN) * p.height:
            return "bottom"
        return None

    def key(b):
        return re.sub(r"\d+", "#", normalize(b.text).lower())

    seen: Counter = Counter()
    for p in paged:
        seen.update({(zone(p, b), key(b)) for b in p.blocks if zone(p, b)})
    min_pages = max(2, math.ceil(0.5 * len(paged))) if len(paged) >= 2 else None  # a single page cannot show repetition
    removed = 0
    for p in paged:
        kept = []
        for b in p.blocks:
            z = zone(p, b)
            if z and (_PAGE_NO.match(b.text) or (min_pages and seen[(z, key(b))] >= min_pages)):
                removed += 1
                continue
            kept.append(b)
        p.blocks = kept
    return removed


def detect_headings(doc: ExtractedDocument) -> None:
    """Mark heading blocks in sized formats (PDF) from font size, boldness and numbering. Formats whose extractor already
    knows the structure (Markdown, HTML, DOCX) keep what they set."""
    weights: Counter = Counter()
    for p in doc.pages:
        for b in p.blocks:
            if b.size > 0 and not b.heading_level:
                weights[b.size] += len(b.text)
    if not weights:
        return
    half, acc, body = sum(weights.values()) / 2, 0, 0.0
    for size in sorted(weights):
        acc += weights[size]
        if acc >= half:
            body = size
            break
    for p in doc.pages:
        for b in p.blocks:
            if b.heading_level or b.size <= 0:
                continue
            t = normalize(b.text)
            words, lines = len(t.split()), b.text.count("\n") + 1
            if not words or len(t) > 160 or lines > 3:
                continue
            if b.size >= body * 1.5 and words <= 30:
                b.heading_level = 1
            elif b.size >= body * 1.15 and words <= 25:
                b.heading_level = 2
            elif b.bold and lines == 1 and words <= 14 and not t.endswith((".", ",", ";", ":")):
                b.heading_level = 3
            elif lines == 1 and b.size >= body * 0.95 and looks_like_heading(t):
                b.heading_level = 3


def _pieces(text: str, max_words: int) -> list[str]:
    out, cur, count = [], [], 0
    for s in _SENTENCE_RE.split(text):
        w = len(s.split())
        if cur and count + w > max_words:
            out.append(" ".join(cur))
            cur, count = [], 0
        cur.append(s)
        count += w
    if cur:
        out.append(" ".join(cur))
    return out


def build_passages(doc_id: str, doc: ExtractedDocument, page_quality: dict, max_words: int = 110) -> list[Passage]:
    """Cut the cleaned document into passages. `page_quality` maps page number -> PageQuality; pages with no text layer
    produce nothing. Each passage carries its page, the nearest heading above it, and its page's extraction quality."""
    repair = doc.file_type == "pdf"
    raw: list[tuple[int | None, str, str, str]] = []  # (page, section, text, quality)
    section = ""
    for page in doc.pages:
        q: PageQuality | None = page_quality.get(page.number)
        if q is None or q.status in ("no_text", "blank"):
            continue
        quality = "low" if q.status == "low" else "ok"
        buf, buf_section = "", section

        def flush() -> None:
            nonlocal buf
            if buf:
                raw.append((page.number, buf_section, buf, quality))
            buf = ""

        for block in page.blocks:
            text = normalize(repair_hyphenation(block.text) if repair else block.text)
            if not text:
                continue
            if block.heading_level > 0:
                flush()
                section = text[:200]
                buf_section = section
                continue
            for piece in _pieces(text, max_words):
                if buf and len(f"{buf} {piece}".split()) <= max_words and (
                        len(buf.split()) < MERGE_BELOW or len(piece.split()) < MERGE_BELOW):
                    buf = f"{buf} {piece}"
                else:
                    flush()
                    buf, buf_section = piece, section
        flush()
    kept = [r for r in raw if len(r[2].split()) >= MIN_WORDS]
    return [Passage(f"{doc_id}:p{i:03d}", doc_id, text, page, sec, quality)
            for i, (page, sec, text, quality) in enumerate(kept)]
