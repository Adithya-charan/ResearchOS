"""Extraction-quality assessment. Blueprint risk "OCR errors -> show extraction quality": every page gets a status that
travels with its passages, so downstream agents and the user can see when evidence comes from shaky text."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

MIN_CHARS = 25  # fewer non-space characters than this = the page has no usable text layer


@dataclass(frozen=True)
class PageQuality:
    status: str  # ok | low | no_text (image/drawing-only page: needs OCR) | blank (nothing on it at all)
    chars: int
    alpha_ratio: float
    issues: tuple[str, ...] = ()


def assess_text(text: str, has_images: bool = False) -> PageQuality:
    chars = [c for c in text if not c.isspace()]
    n = len(chars)
    if n < MIN_CHARS:
        if not has_images:  # nothing to read and nothing that could hold text: a genuinely blank page
            return PageQuality("blank", n, 0.0)
        return PageQuality("no_text", n, 0.0, ("no extractable text (the page holds images or drawings - probably scanned; "
                                               "OCR was not performed)",))

    issues: list[str] = []
    alpha = sum(c.isalpha() for c in chars) / n
    if text.count("�") / n > 0.005:
        issues.append("garbled characters (U+FFFD replacement characters)")
    odd = sum(unicodedata.category(c) in ("Cc", "Cs", "Co") for c in chars)
    if odd / n > 0.005:
        issues.append("control or private-use characters (likely a broken font encoding)")
    if alpha < 0.5:
        issues.append("mostly non-letter characters (tables, formulas or a bad encoding)")
    words = text.split()
    if len(words) >= 20:
        if sum(len(w) == 1 and w.isalpha() for w in words) / len(words) > 0.4:
            issues.append("letters separated by spaces (broken word spacing)")
        if sum(len(w) for w in words) / len(words) > 18:
            issues.append("words run together (missing spaces)")
    return PageQuality("low" if issues else "ok", n, round(alpha, 3), tuple(issues))


def summarize(qualities: list[tuple[int | None, PageQuality]]) -> dict:
    """Document-level quality over the non-blank pages: ok = all ok; partial = some low or without a text layer;
    no_text = nothing usable."""
    considered = [(n, q) for n, q in qualities if q.status != "blank"]
    ok = sum(q.status == "ok" for _, q in considered)
    low = [n for n, q in considered if q.status == "low"]
    empty = [n for n, q in considered if q.status == "no_text"]
    blank = [n for n, q in qualities if q.status == "blank"]
    if not considered or len(empty) == len(considered):
        status = "no_text"
    elif ok == len(considered):
        status = "ok"
    else:
        status = "partial"
    issues: list[str] = []
    for n, q in considered:
        for issue in q.issues:
            line = f"page {n}: {issue}" if n is not None else issue
            if line not in issues and len(issues) < 12:
                issues.append(line)
    return {"status": status, "pages": len(qualities), "pages_ok": ok, "pages_low": low, "pages_no_text": empty,
            "pages_blank": blank, "issues": issues}
