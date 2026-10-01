"""simple-sent-v1 sentence splitter (Step 9)."""

from __future__ import annotations

from driftlens.text.sentences import split_sentences


def test_sentence_basic() -> None:
    assert split_sentences("A cat sat. It slept.") == ["A cat sat.", "It slept."]


def test_sentence_abbreviation_no_split() -> None:
    assert split_sentences("See Dr. Smith today.") == ["See Dr. Smith today."]


def test_sentence_single_uppercase_no_split() -> None:
    assert split_sentences("Meet J. Smith now.") == ["Meet J. Smith now."]


def test_sentence_cjk_terminator() -> None:
    assert split_sentences("\u4eca\u65e5\u306f\u6674\u308c\u3002\u660e\u65e5\u306f\u96e8\u3002") == [
        "\u4eca\u65e5\u306f\u6674\u308c\u3002",
        "\u660e\u65e5\u306f\u96e8\u3002",
    ]
