"""Metadata extraction. Every field records where it came from: a title read off a page is a guess, a title declared in the
file's metadata is a claim by whoever made the file, and a value you typed is yours. Nothing is invented - when a field
cannot be found it stays empty."""
from __future__ import annotations

import datetime
import re
from pathlib import Path

from research.corpus.models import ExtractedDocument
from research.corpus.segment import normalize

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}|\\^`]+)", re.I)
_JUNK_TITLE = re.compile(r"(?i)(\.(pdf|docx?|tex|indd|pages|odt)$|^untitled|^microsoft word|^document\s*\d*$|^slide\s*\d*$)")
_YEAR_CTX = re.compile(r"(?i)(?:©|\(c\)|copyright|published|received|accepted|year|date)[^0-9]{0,15}((?:19|20)\d{2})\b")
_TITLE_LINE = re.compile(r"(?im)^\s*title\s*:\s*(\S.{3,250})$")


def _valid_year(y: int) -> bool:
    return 1900 <= y <= datetime.date.today().year + 1


def _first_year(text: str) -> int | None:
    m = re.search(r"(?:19|20)\d{2}", text or "")
    return int(m.group(0)) if m and _valid_year(int(m.group(0))) else None


def _title(doc: ExtractedDocument, filename: str) -> tuple[str, str]:
    declared = normalize(str(doc.metadata.get("title") or ""))
    if len(declared) >= 4 and not _JUNK_TITLE.search(declared):
        return declared[:300], "file_metadata"
    first = doc.pages[0] if doc.pages else None
    if first is not None:
        if doc.file_type == "pdf":
            heads = [b for b in first.blocks if b.heading_level in (1, 2) and b.size > 0]
            if heads:
                top = max(b.size for b in heads)
                text = normalize(" ".join([b.text for b in heads if b.size == top][:3]))
                if len(text) >= 4:
                    return text[:300], "first_page_largest_text"
        elif doc.file_type == "txt":
            m = _TITLE_LINE.search(first.raw_text[:2000])
            if m:
                return normalize(m.group(1))[:300], "first_page_text"
        else:  # md / html / docx know their headings
            levels = [b for b in first.blocks if b.heading_level > 0]
            if levels:
                best = min(b.heading_level for b in levels)
                return normalize(next(b.text for b in levels if b.heading_level == best))[:300], "first_heading"
    return re.sub(r"[_\-]+", " ", Path(filename).stem).strip() or filename, "filename"


def _authors(doc: ExtractedDocument) -> list[str]:
    raw = str(doc.metadata.get("author") or "")
    parts = re.split(r";|\n|&|\band\b", raw)
    return [normalize(p) for p in parts if normalize(p)][:30]


def _year(doc: ExtractedDocument) -> tuple[int | None, str | None]:
    first = doc.pages[0].raw_text[:3000] if doc.pages else ""
    for m in _YEAR_CTX.finditer(first):
        if _valid_year(int(m.group(1))):
            return int(m.group(1)), "first_page_text"
    for key, source in (("date", "file_date"), ("creationDate", "file_created_date")):
        raw = str(doc.metadata.get(key) or "")
        y = _first_year(raw[2:] if raw.startswith("D:") else raw)
        if y:
            return y, source
    return None, None


def _doi(doc: ExtractedDocument) -> str | None:
    head = "\n".join(p.raw_text for p in doc.pages[:2])[:8000]
    for field in (str(doc.metadata.get("subject") or ""), head):
        m = DOI_RE.search(field)
        if m:
            return m.group(1).rstrip(".,;:)]}'\"").lower()
    return None


def extract_metadata(doc: ExtractedDocument, filename: str, overrides: dict | None = None) -> dict:
    """-> title, title_source, authors, year, year_source, doi. `overrides` (from the person ingesting) win and are
    labelled source 'user'. DOIs are only format-checked here - nothing is looked up online."""
    title, title_source = _title(doc, filename)
    year, year_source = _year(doc)
    out = {"title": title, "title_source": title_source, "authors": _authors(doc), "year": year,
           "year_source": year_source, "doi": _doi(doc)}
    for key, value in (overrides or {}).items():
        if value in (None, "", []):
            continue
        if key == "title":
            out["title"], out["title_source"] = normalize(str(value))[:300], "user"
        elif key == "authors":
            out["authors"] = [normalize(a) for a in ([value] if isinstance(value, str) else value) if normalize(a)]
        elif key == "year":
            if not _valid_year(int(value)):
                raise ValueError(f"year {value} is out of range")
            out["year"], out["year_source"] = int(value), "user"
        elif key == "doi":
            if not DOI_RE.fullmatch(str(value).strip()):
                raise ValueError(f"'{value}' does not look like a DOI (10.xxxx/...)")
            out["doi"] = str(value).strip().lower()
        elif key == "publication_date":
            out["publication_date"] = _valid_date(str(value))
            if "year" not in (overrides or {}):
                out["year"], out["year_source"] = int(out["publication_date"][:4]), "user"
        elif key == "url":
            out["url"] = _valid_url(str(value))
        elif key == "document_type":
            if str(value) not in DOCUMENT_TYPES:
                raise ValueError(f"document_type must be one of {sorted(DOCUMENT_TYPES)}")
            out["document_type"] = str(value)
        else:
            raise ValueError(f"unknown metadata field '{key}'")
    return out


DOCUMENT_TYPES = frozenset({"article", "preprint", "report", "thesis", "book", "chapter", "notes", "other"})
_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def _valid_date(value: str) -> str:
    """YYYY, YYYY-MM or YYYY-MM-DD, and a real calendar date. Returned unchanged (precision is kept, never padded)."""
    m = _DATE_RE.match(value.strip())
    if not m:
        raise ValueError(f"publication_date '{value}' must be YYYY, YYYY-MM or YYYY-MM-DD")
    year, month, day = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
    if not _valid_year(year):
        raise ValueError(f"publication_date year {year} is out of range")
    try:
        datetime.date(year, month, day)
    except ValueError as e:
        raise ValueError(f"publication_date '{value}' is not a real date") from e
    return value.strip()


def _valid_url(value: str) -> str:
    from urllib.parse import urlparse
    value = value.strip()
    u = urlparse(value)
    if u.scheme not in ("http", "https") or not u.netloc or len(value) > 500 or any(c.isspace() for c in value):
        raise ValueError("url must be an http(s) address of at most 500 characters")
    return value
