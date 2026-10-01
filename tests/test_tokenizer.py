"""simple-v1 tokenizer (Step 9)."""

from __future__ import annotations

import re

from driftlens.text.tokenizer import tokenizer_id, tokenize


def test_basic_tokenize() -> None:
    assert tokenize("It's a well-known fact.") == ["It's", "a", "well-known", "fact"]


def test_trailing_joiner_stripped() -> None:
    assert tokenize("rock-") == ["rock"]


def test_leading_joiner_not_joined() -> None:
    assert tokenize("-rock") == ["rock"]


def test_cjk_char_per_token() -> None:
    assert tokenize("\u65e5\u672c\u8a9e\u3067\u3059") == [
        "\u65e5",
        "\u672c",
        "\u8a9e",
        "\u3067",
        "\u3059",
    ]


def test_mixed_script() -> None:
    assert tokenize("hello\u65e5\u672c\u8a9eworld") == [
        "hello",
        "\u65e5",
        "\u672c",
        "\u8a9e",
        "world",
    ]


def test_digits_included() -> None:
    assert tokenize("v2.0 release") == ["v2", "0", "release"]


def test_tokenizer_id_format() -> None:
    assert re.match(r"^simple-v1\+u\d+\.\d+\.\d+$", tokenizer_id())
