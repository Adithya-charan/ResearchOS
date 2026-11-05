"""Builds real files for the corpus tests. PDFs are written by PyMuPDF and DOCX files by hand (zip + XML), so the extractors
are exercised on genuine file bytes - nothing about the parsers is mocked.

A PDF page is a list of items, each `(kind, payload)`:

    ("title" | "h1" | "h2" | "p" | "author", "text")     wrapped to the column width
    ("lines", ["exact", "lines"])                          body text with fixed line breaks (to force hyphenation etc.)
    ("columns", [[items...], [items...]])                  side-by-side columns

or one of the strings "blank" (an empty page) / a tuple ("scan", items) (the items rasterised into an image: a scanned page
with no text layer).
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pymupdf

PAGE_W, PAGE_H = 595.0, 842.0
LEFT, RIGHT, TOP, BOTTOM = 72.0, 523.0, 92.0, 748.0
STYLES = {  # kind: (font, size, space before, space after)
    "title": ("hebo", 20.0, 0.0, 16.0),
    "author": ("helv", 11.0, 0.0, 10.0),
    "h1": ("hebo", 14.0, 8.0, 7.0),
    "h2": ("hebo", 12.0, 6.0, 5.0),
    "p": ("helv", 10.5, 0.0, 10.0),
}


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and pymupdf.get_text_length(trial, fontname=font, fontsize=size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _flow(page: pymupdf.Page, items: list, x0: float, x1: float, y: float) -> float:
    for kind, payload in items:
        if kind == "columns":
            gap, mid = 20.0, (x0 + x1) / 2
            ends = []
            for i, col in enumerate(payload):
                ends.append(_flow(page, col, x0 if i == 0 else mid + gap / 2, mid - gap / 2 if i == 0 else x1, y))
            y = max(ends)
            continue
        if kind == "lines":
            font, size, before, after = STYLES["p"]
            lines = list(payload)
        else:
            font, size, before, after = STYLES[kind]
            lines = _wrap(payload, font, size, x1 - x0)
        y += before
        for line in lines:
            y += size
            page.insert_text((x0, y), line, fontname=font, fontsize=size)
            y += size * 0.3
        y += after
        if y > BOTTOM:
            raise ValueError(f"test page overflows ({y:.0f} > {BOTTOM:.0f}); split the content over more pages")
    return y


def _height(item: tuple) -> float:
    kind, payload = item
    if kind == "columns":
        return max(sum(_height(i) for i in col) for col in payload)
    font, size, before, after = STYLES["p" if kind == "lines" else kind]
    n = len(payload) if kind == "lines" else len(_wrap(payload, font, size, RIGHT - LEFT))
    return before + n * size * 1.3 + after


def paginate(items: list[tuple]) -> list[list[tuple]]:
    """Split a flat list of items over as many pages as needed (a heading is never left alone at the bottom of a page)."""
    pages: list[list[tuple]] = [[]]
    used = 0.0
    for i, item in enumerate(items):
        if item[0] == "pagebreak":  # explicit page break
            if pages[-1]:
                pages.append([])
                used = 0.0
            continue
        need = _height(item)
        if item[0] in ("h1", "h2") and i + 1 < len(items) and items[i + 1][0] != "pagebreak":
            need += _height(items[i + 1])
        if pages[-1] and used + need > BOTTOM - TOP:
            pages.append([])
            used = 0.0
        pages[-1].append(item)
        used += _height(item)
    return pages


def _draw(page: pymupdf.Page, items: list) -> None:
    _flow(page, items, LEFT, RIGHT, TOP)


def make_pdf(path: str | Path, pages: list[Any], *, title: str | None = None, author: str | None = None,
             subject: str | None = None, created: str | None = None, header: str | None = None,
             page_numbers: bool = False, password: str | None = None) -> Path:
    """Write a PDF. `header` is repeated at the top of every page and `page_numbers` adds a footer number - the running
    furniture the segmenter is expected to strip."""
    path = Path(path)
    doc = pymupdf.open()
    for n, spec in enumerate(pages, start=1):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        if spec == "blank":
            continue
        if isinstance(spec, tuple) and spec[0] == "scan":
            scratch = pymupdf.open()
            sp = scratch.new_page(width=PAGE_W, height=PAGE_H)
            _draw(sp, spec[1])
            pix = sp.get_pixmap(dpi=60)
            scratch.close()
            page.insert_image(page.rect, pixmap=pix)
        else:
            _draw(page, spec)
            if header:
                page.insert_text((LEFT, 40), header, fontname="helv", fontsize=9)
            if page_numbers:
                page.insert_text((PAGE_W / 2 - 4, 806), str(n), fontname="helv", fontsize=9)
    meta = {"title": title or "", "author": author or "", "subject": subject or "", "creator": "", "producer": "",
            "keywords": ""}
    if created:
        meta["creationDate"] = created  # e.g. "D:20190412101500"
    doc.set_metadata(meta)
    if password:
        doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password + "-owner")
    else:
        doc.save(str(path))
    doc.close()
    return path


def make_docx(path: str | Path, paragraphs: list[tuple[str, str]], *, title: str | None = None, author: str | None = None,
              created: str | None = None, extra_member: tuple[str, bytes] | None = None) -> Path:
    """Write a minimal but valid .docx. Paragraphs are `(style, text)` with style "Title", "Heading1".., or "" for body."""
    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    body = []
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body.append(f"<w:p>{ppr}<w:r><w:t xml:space='preserve'>{esc(text)}</w:t></w:r></w:p>")
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{w}"><w:body>'
                f'{"".join(body)}</w:body></w:document>')
    core = ""
    if title or author or created:
        parts = []
        if title:
            parts.append(f"<dc:title>{esc(title)}</dc:title>")
        if author:
            parts.append(f"<dc:creator>{esc(author)}</dc:creator>")
        if created:
            parts.append(f"<dcterms:created>{esc(created)}</dcterms:created>")
        core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties '
                'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">'
                + "".join(parts) + "</cp:coreProperties>")
    types = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="xml" ContentType="application/xml"/></Types>')
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", types)
        zf.writestr("word/document.xml", document)
        if core:
            zf.writestr("docProps/core.xml", core)
        if extra_member:
            zf.writestr(*extra_member)
    return path
