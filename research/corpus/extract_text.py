"""Plain text, Markdown and HTML extraction (standard library only). These formats have no pages: one Page(number=None)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from research.corpus.models import Block, ExtractedDocument, ExtractionError, Page

MAX_TEXT_CHARS = 30_000_000

_NUMBERED = re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}\.?\s+[A-Z]")
KNOWN_HEADINGS = frozenset(
    "abstract introduction background related-work methods method methodology materials-and-methods results discussion "
    "conclusion conclusions references acknowledgements acknowledgments appendix limitations open-questions summary".split()
)


def looks_like_heading(text: str) -> bool:
    """Short single line that is a known section name or starts with a section number, and is not a sentence."""
    t = text.strip()
    if not t or "\n" in t or len(t) > 100 or len(t.split()) > 14:
        return False
    key = re.sub(r"^[\d.\sA-Z]{0,8}?(?=[A-Za-z]{3})", "", t).rstrip(".:").lower().replace(" ", "-")
    if key in KNOWN_HEADINGS or t.rstrip(".:").lower().replace(" ", "-") in KNOWN_HEADINGS:
        return True
    return bool(_NUMBERED.match(t)) and not t.endswith((",", ";")) and not (t.endswith(".") and len(t.split()) > 6)


def _decode(data: bytes, warnings: list[str]) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
        warnings.append("file is not valid UTF-8; decoded as Windows-1252 (some characters may be wrong)")
    if len(text) > MAX_TEXT_CHARS:
        raise ExtractionError("too_large", f"text has {len(text):,} characters; the limit is {MAX_TEXT_CHARS:,}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _single_page(file_type: str, method: str, blocks: list[Block], raw: str, meta: dict, warnings: list[str]) -> ExtractedDocument:
    if not blocks:
        raise ExtractionError("no_text", "document contains no text")
    return ExtractedDocument(file_type, method, [Page(None, blocks, raw)], meta, warnings)


def extract_txt(path: Path) -> ExtractedDocument:
    warnings: list[str] = []
    text = _decode(Path(path).read_bytes(), warnings)
    blocks = []
    for chunk in re.split(r"\n\s*\n", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if looks_like_heading(chunk):
            blocks.append(Block(chunk, heading_level=1))
        else:
            blocks.append(Block(chunk))
    return _single_page("txt", "plain-text", blocks, text, {}, warnings)


_HEADING_MD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _md_inline(s: str) -> str:
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"(\*\*|__|`)", "", s)
    return s


def extract_md(path: Path) -> ExtractedDocument:
    warnings: list[str] = []
    text = _decode(Path(path).read_bytes(), warnings)
    meta: dict = {}
    body = text
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    if m:  # simple "key: value" front matter only; anything fancier is ignored rather than guessed at
        for line in m.group(1).split("\n"):
            k, sep, v = line.partition(":")
            if sep and k.strip().lower() in ("title", "author", "authors", "date", "year") and v.strip():
                meta[k.strip().lower()] = v.strip().strip("\"'")
        body = text[m.end():]
    blocks: list[Block] = []
    para: list[str] = []
    in_code = False

    def flush() -> None:
        if para:
            blocks.append(Block(_md_inline("\n".join(para)).strip()))
            para.clear()

    for line in body.split("\n"):
        if _FENCE.match(line):
            flush()
            in_code = not in_code
            continue
        if in_code:
            para.append(line)
            continue
        h = _HEADING_MD.match(line)
        if h:
            flush()
            blocks.append(Block(_md_inline(h.group(2)).strip(), heading_level=len(h.group(1))))
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    blocks = [b for b in blocks if b.text]
    # raw_text is the *rendered* text (markup removed): it is what passages are verified against after re-extraction
    return _single_page("md", "markdown", blocks, "\n".join(b.text for b in blocks), meta, warnings)


class _HtmlText(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg", "iframe", "object"}
    BREAK = {"p", "div", "li", "tr", "br", "section", "article", "blockquote", "pre", "table", "ul", "ol", "dd", "dt",
             "figcaption", "header", "footer", "td", "th", "main", "aside", "nav"}
    HEADINGS = {f"h{i}": i for i in range(1, 7)}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.meta: dict = {}
        self._cur: list[str] = []
        self._skip = 0
        self._heading = 0
        self._in_head = False
        self._in_title = False
        self._title: list[str] = []
        self._authors: list[str] = []

    def _flush(self) -> None:
        text = " ".join("".join(self._cur).split())
        self._cur = []
        if text:
            self.blocks.append(Block(text, heading_level=self._heading))

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "head":
            self._in_head = True
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (a.get("name") or a.get("property") or "").lower()
            content = (a.get("content") or "").strip()
            if content:
                if name in ("author", "citation_author", "dc.creator"):
                    self._authors.append(content)
                elif name in ("citation_title", "dc.title", "og:title") and "title" not in self.meta:
                    self.meta["title"] = content
                elif name in ("citation_publication_date", "citation_date", "dc.date", "date", "article:published_time"):
                    self.meta.setdefault("date", content)
        elif tag in self.SKIP:
            self._skip += 1
        elif not self._in_head:
            if tag in self.HEADINGS:
                self._flush()
                self._heading = self.HEADINGS[tag]
            elif tag in self.BREAK:
                self._flush()

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        elif tag == "title":
            self._in_title = False
        elif tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif not self._in_head:
            if tag in self.HEADINGS:
                self._flush()
                self._heading = 0
            elif tag in self.BREAK:
                self._flush()

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        elif not self._in_head and not self._skip:
            self._cur.append(data)

    def close(self):
        super().close()
        self._flush()
        if self._title and "title" not in self.meta:
            t = " ".join("".join(self._title).split())
            if t:
                self.meta["title"] = t
        if self._authors:
            self.meta["author"] = "; ".join(dict.fromkeys(self._authors))


def extract_html(path: Path) -> ExtractedDocument:
    warnings: list[str] = []
    text = _decode(Path(path).read_bytes(), warnings)
    parser = _HtmlText()
    try:
        parser.feed(text)
        parser.close()
    except Exception as e:  # HTMLParser is lenient, but never let a hostile file raise something odd
        raise ExtractionError("corrupt", f"cannot parse HTML: {str(e)[:150]}") from e
    raw = "\n".join(b.text for b in parser.blocks)
    return _single_page("html", "html-parser", parser.blocks, raw, parser.meta, warnings)
