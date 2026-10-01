"""Text normalization nfc-ws-1 (Step 8)."""

from __future__ import annotations

import re
import unicodedata

import pytest

from driftlens.errors import InvalidInputError
from driftlens.text.normalize import (
    NORMALIZATION_ID,
    check_utf8_clean,
    normalize_text,
    normalization_version,
)


def test_bom_removed_anywhere() -> None:
    assert normalize_text("a\ufeffb") == "ab"


def test_nfc_not_nfkc() -> None:
    assert normalize_text("\uFF21") == "\uFF21"
    assert normalize_text("e\u0301") == "\u00e9"


def test_case_preserved() -> None:
    assert normalize_text("Apple") != normalize_text("apple")


def test_crlf_and_cr_to_lf() -> None:
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_whitespace_collapse_and_trim() -> None:
    assert normalize_text("  a   b\t\tc  ") == "a b c"


def test_blank_line_collapse() -> None:
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_full_trim() -> None:
    assert normalize_text("\n\n  x  \n\n") == "x"


def test_normalization_version_format() -> None:
    version = normalization_version()
    assert re.match(r"^nfc-ws-1\+u\d+\.\d+\.\d+$", version)
    assert version == f"{NORMALIZATION_ID}+u{unicodedata.unidata_version}"


def test_invalid_utf8_bytes_rejected() -> None:
    location = "v1.csv row 3 field text"
    with pytest.raises(InvalidInputError) as excinfo:
        check_utf8_clean(b"\xff\xfe", location=location)
    assert excinfo.value.message == "invalid UTF-8"
    assert excinfo.value.location == location


def test_lone_surrogate_rejected() -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        check_utf8_clean("a" + chr(0xD800) + "b", location="v1.jsonl row 1 field text")
    assert "lone surrogate" in excinfo.value.message


def test_valid_utf8_passthrough() -> None:
    text = "caf\u00e9 \u2014 \u65e5\u672c\u8a9e"
    assert check_utf8_clean(text.encode("utf-8"), location="x") == text
