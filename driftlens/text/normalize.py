"""Text normalization (nfc-ws-1) and UTF-8 admission checks (Step 8)."""

from __future__ import annotations

import unicodedata
from typing import Union

import re as _regex

from driftlens.errors import InvalidInputError

NORMALIZATION_ID = "nfc-ws-1"


def normalization_version() -> str:
    """Identifier of the normalization algorithm plus the interpreter's Unicode data version."""

    return f"{NORMALIZATION_ID}+u{unicodedata.unidata_version}"


def _collapse_line(line: str) -> str:
    pieces: list[str] = []
    in_run = False
    for char in line:
        if char.isspace():
            in_run = True
            continue
        if in_run:
            pieces.append(" ")
            in_run = False
        pieces.append(char)
    return "".join(pieces).strip(" ")


def normalize_text(raw: str) -> str:
    """Apply the nfc-ws-1 normalization pipeline. Pure function, no I/O."""

    # Pass 1 - BOM removal (every U+FEFF, not only a leading one).
    text = raw.replace("\ufeff", "")
    # Pass 2 - Unicode normalization (NFC, never NFKC).
    text = unicodedata.normalize("NFC", text)
    # Pass 3 - line ending normalization.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Pass 4 - per-line whitespace collapse and trim.
    text = "\n".join(_collapse_line(line) for line in text.split("\n"))
    # Pass 5 - blank-line collapse.
    text = _regex.sub(r"\n{3,}", "\n\n", text)
    # Pass 6 - final trim.
    return text.strip()


def _surrogate_offset(text: str) -> int:
    offset = 0
    for char in text:
        if 0xD800 <= ord(char) <= 0xDFFF:
            return offset
        offset += len(char.encode("utf-8"))
    return offset


def check_utf8_clean(raw_bytes: Union[bytes, str], *, location: str) -> str:
    """Admit raw text from a source, rejecting invalid UTF-8 and lone surrogates."""

    if isinstance(raw_bytes, bytes):
        try:
            text = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise InvalidInputError(message="invalid UTF-8", location=location) from None
    elif isinstance(raw_bytes, str):
        text = raw_bytes
    else:
        raise InvalidInputError(message="invalid UTF-8", location=location)

    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise InvalidInputError(
            message=f"lone surrogate at byte offset {_surrogate_offset(text)}",
            location=location,
        ) from None
    return text
