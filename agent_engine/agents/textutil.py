"""Deterministic text helpers for the analysis agents. Everything here is extractive: sentences are quoted from passages,
never generated or paraphrased."""
from __future__ import annotations

import re
import unicodedata

from research.bm25.index import _SENTENCE_RE

MIN_WORDS, MAX_WORDS = 6, 60


def squash(text: str) -> str:
    """Comparison form for quote checks (same definition the corpus uses to verify passages): letters and digits only."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).lower())


def sentences(text: str) -> list[str]:
    out = [s.strip() for s in _SENTENCE_RE.split(" ".join(text.split()))]
    return [s for s in out if MIN_WORDS <= len(s.split()) <= MAX_WORDS]


_HAS_NUM = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|forty|fifty|hundred|thousand|"
                      r"ninety|eighty|half|percent)\b", re.I)
_LIMIT = re.compile(r"\b(limitations?|limited|however|cannot|can not|do not|does not|did not|not (?:examine|measure|protect|"
                    r"remove|prevent)|assum(?:e|es|ed|ing)|future work|caution|unable|violate[sd]?|although)\b", re.I)
_METHOD = re.compile(r"\b(we (?:propose|introduce|use|used|train|trained|prepare|prepared|synthesi[sz]ed|analy[sz]ed|fitted|"
                     r"combine|measured|recorded|computed|fabricated|deposited|study)|was (?:prepared|deposited|measured|"
                     r"computed|pressed|computed)|were (?:fabricated|pressed|measured|recorded|sampled|deposited|prepared)|"
                     r"each (?:client|query)|our (?:pattern|method|approach)|is computed|are computed)\b", re.I)
_RESULT = re.compile(r"\b(reach(?:es|ed)?|retain(?:ed|s)?|show(?:s|ed|n)?|found|rose|increase[sd]?|decrease[sd]?|reduc\w+|"
                     r"improv\w+|higher|lower|faster|less|more|cycled|achiev\w+|drops?|dropped|bleached|suffered|lost|"
                     r"declin\w+|stayed|followed|used|ran)\b", re.I)


def section_kind(section: str) -> str | None:
    s = section.lower()
    if re.search(r"limitation|future|threat|caveat", s):
        return "limitation"
    if re.search(r"method|material|experiment|approach|setup|procedure", s):
        return "method"
    if re.search(r"result|finding|evaluation|performance", s):
        return "result"
    return None


def classify(sentence: str, section: str) -> str | None:
    """'method' | 'result' | 'limitation' | None, from the section title and cue words. Heuristic, and labelled as such."""
    sec = section_kind(section)
    if sec == "limitation" or _LIMIT.search(sentence) and sec != "method":
        return "limitation"
    if _METHOD.search(sentence) or sec == "method":
        return "method"
    if _RESULT.search(sentence) and _HAS_NUM.search(sentence):
        return "result"
    return None


def has_number(sentence: str) -> bool:
    return bool(_HAS_NUM.search(sentence))


def digit_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


_UP = {"increase", "increased", "increases", "rise", "rose", "higher", "improve", "improved", "improves", "faster", "more",
       "gain", "raises", "raised", "longer", "above"}
_DOWN = {"decrease", "decreased", "decreases", "reduce", "reduced", "reduces", "lower", "less", "slower", "decline",
         "declined", "drop", "dropped", "fell", "fall", "below", "degrade", "degraded", "loss", "lost"}


def polarity(sentence: str) -> int:
    """+1 if the sentence only uses 'up' words, -1 if only 'down' words, else 0 (used to flag possible tensions only)."""
    words = set(re.findall(r"[a-z]+", sentence.lower()))
    up, down = bool(words & _UP), bool(words & _DOWN)
    return 1 if up and not down else -1 if down and not up else 0
