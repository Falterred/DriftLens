"""simple-sent-v1 approximate sentence splitter (Step 9)."""

from __future__ import annotations

import re

SENTENCE_SPLITTER_ID = "simple-sent-v1"

ABBREVIATIONS = frozenset(
    {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Sr",
        "Jr",
        "St",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "Inc",
        "Ltd",
        "Co",
        "No",
        "Fig",
        "al",
    }
)

# ASCII terminators need following whitespace/EOS; CJK terminators always split.
_BOUNDARY_RE = re.compile(r"[.!?](?=\s|$)|[\u3002\uFF01\uFF1F]")
_ASCII_MARKS = ".!?"


def split_sentences(text: str) -> list[str]:
    """Approximate sentence split; used for counts only. Pure, never raises."""

    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        mark = match.start()
        if text[mark] in _ASCII_MARKS:
            cursor = mark
            while cursor > 0 and not text[cursor - 1].isspace():
                cursor -= 1
            candidate = text[cursor:mark]
            if candidate in ABBREVIATIONS:
                continue
            if len(candidate) == 1 and candidate.isupper():
                continue
        end = mark + 1
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def sentence_count(text: str) -> int:
    """Number of sentences in the text."""

    return len(split_sentences(text))
