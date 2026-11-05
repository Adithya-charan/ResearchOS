"""DOCX extraction with the standard library (a .docx is a zip of XML). No macros, embedded objects or external
references are ever followed; DTDs/entities are refused outright."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from research.corpus.models import Block, ExtractedDocument, ExtractionError, Page

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MAX_MEMBER_BYTES = 60 * 1024 * 1024
_HEADING_STYLE = re.compile(r"^(?:heading|überschrift|titre)\s*(\d)$", re.I)


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_MEMBER_BYTES:
        raise ExtractionError("unsafe", f"{name} is {info.file_size / 1e6:.0f} MB uncompressed (limit {MAX_MEMBER_BYTES / 1e6:.0f} MB)")
    with zf.open(info) as f:
        data = f.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES:
        raise ExtractionError("unsafe", f"{name} expands beyond the size limit")
    return data


def _parse(data: bytes) -> ET.Element:
    head = data[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise ExtractionError("unsafe", "XML declares a DTD/entities (possible entity-expansion attack); refusing")
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise ExtractionError("corrupt", f"damaged XML in document: {e}") from e


def _para_text(p: ET.Element) -> str:
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        for child in el:
            tag = child.tag
            if tag in (W + "del", W + "moveFrom"):  # tracked deletions are not part of the text
                continue
            if tag == W + "t":
                parts.append(child.text or "")
            elif tag == W + "tab":
                parts.append(" ")
            elif tag in (W + "br", W + "cr"):
                parts.append("\n")
            else:
                walk(child)

    walk(p)
    return "".join(parts).strip()


def _heading_level(p: ET.Element) -> int:
    style = p.find(f"{W}pPr/{W}pStyle")
    val = style.get(W + "val", "") if style is not None else ""
    if val.lower() == "title":
        return 1
    m = _HEADING_STYLE.match(val.replace("_", " ").strip())
    return int(m.group(1)) if m else 0


def extract_docx(path: Path) -> ExtractedDocument:
    try:
        with zipfile.ZipFile(path) as zf:
            doc_xml = _read_member(zf, "word/document.xml")
            core_xml = _read_member(zf, "docProps/core.xml")
    except zipfile.BadZipFile as e:
        raise ExtractionError("corrupt", f"damaged zip container: {e}") from e
    if doc_xml is None:
        raise ExtractionError("type_mismatch", "no word/document.xml in the container")

    body = _parse(doc_xml).find(W + "body")
    if body is None:
        raise ExtractionError("corrupt", "document has no body")
    blocks: list[Block] = []
    for el in body:
        if el.tag == W + "p":
            text = _para_text(el)
            if text:
                blocks.append(Block(text, heading_level=_heading_level(el)))
        elif el.tag == W + "tbl":  # a table row becomes one block: cells joined by " | "
            for tr in el.iter(W + "tr"):
                cells = [" ".join(_para_text(p) for p in tc.iter(W + "p")).strip() for tc in tr.findall(W + "tc")]
                row = " | ".join(c for c in cells if c)
                if row:
                    blocks.append(Block(row))

    meta: dict = {}
    if core_xml:
        core = _parse(core_xml)
        for tag, key in (("title", "title"), ("creator", "author"), ("created", "date")):
            for node in core.iter():
                if node.tag.endswith("}" + tag) and (node.text or "").strip():
                    meta[key] = node.text.strip()
                    break
    if not blocks:
        raise ExtractionError("no_text", "document contains no text")
    raw = "\n".join(b.text for b in blocks)
    return ExtractedDocument("docx", "docx-xml", [Page(None, blocks, raw)], meta, [])
