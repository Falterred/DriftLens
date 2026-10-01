"""simple-v1 word tokenizer (Step 9)."""

from __future__ import annotations

import unicodedata

TOKENIZER_ID_BASE = "simple-v1"

# Inclusive code-point ranges for scripts written without word spaces.
NO_WORD_SPACE_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x1780, 0x17FF),  # Khmer
    (0x1000, 0x109F),  # Myanmar
)

_JOINERS = ("'", "\u2019", "-")


def tokenizer_id() -> str:
    """Identifier of the tokenizer plus the interpreter's Unicode data version."""

    return f"{TOKENIZER_ID_BASE}+u{unicodedata.unidata_version}"


def is_no_word_space_char(ch: str) -> bool:
    """True when the character's code point falls in a no-word-space range."""

    if not ch:
        return False
    point = ord(ch)
    for low, high in NO_WORD_SPACE_RANGES:
        if low <= point <= high:
            return True
    return False


def _flush(run: list[str]) -> str:
    """Return the run as a token, stripping trailing joiners that were never internal."""

    while run and run[-1] in _JOINERS:
        run.pop()
    return "".join(run)


def tokenize(text: str) -> list[str]:
    """simple-v1 word tokenizer for already-normalized text. Pure, never raises."""

    tokens: list[str] = []
    run: list[str] = []
    length = len(text)
    for index, ch in enumerate(text):
        if is_no_word_space_char(ch):
            token = _flush(run)
            if token:
                tokens.append(token)
            run = []
            tokens.append(ch)
        elif ch.isalpha() or ch.isdigit():
            run.append(ch)
        elif (
            ch in _JOINERS
            and run
            and index + 1 < length
            and (text[index + 1].isalpha() or text[index + 1].isdigit())
        ):
            run.append(ch)
        else:
            token = _flush(run)
            if token:
                tokens.append(token)
            run = []
    token = _flush(run)
    if token:
        tokens.append(token)
    return tokens


def token_count(text: str) -> int:
    """Number of tokens in the text using a single streaming pass (no list build)."""

    count = 0
    run: list[str] = []
    length = len(text)
    for index, ch in enumerate(text):
        if is_no_word_space_char(ch):
            if _flush(run):
                count += 1
            run = []
            count += 1
        elif ch.isalpha() or ch.isdigit():
            run.append(ch)
        elif (
            ch in _JOINERS
            and run
            and index + 1 < length
            and (text[index + 1].isalpha() or text[index + 1].isdigit())
        ):
            run.append(ch)
        else:
            if _flush(run):
                count += 1
            run = []
    if _flush(run):
        count += 1
    return count
