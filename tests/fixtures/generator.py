"""Deterministic corpus-pair generator for the v0.1 benchmark suite (Step 34)."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

WORD_LIST = [f"word{index}" for index in range(200)]
CATEGORIES = ["alpha", "beta", "gamma", "delta", "epsilon"]

ChangeProfile = Literal[
    "none",
    "pct1",
    "pct10",
    "pct50",
    "whitespace_only",
    "replaced",
    "added_removed",
    "chunk_moves",
]


@dataclass
class ExpectedOutcome:
    added: int
    removed: int
    unchanged: int
    format_only: int
    replaced: int
    modified: int


@dataclass
class CorpusPair:
    v1_records: list[dict]
    v2_records: list[dict]
    expected: ExpectedOutcome


def _iso(index: int) -> str:
    minute = (index // 60) % 60
    second = index % 60
    return f"2024-01-01T00:{minute:02d}:{second:02d}+00:00"


def _make_records(
    rng: random.Random, size: int, dim: int, *, start_index: int = 0
) -> list[dict]:
    records: list[dict] = []
    for offset in range(size):
        doc_id = f"doc-{start_index + offset}"
        chunk_total = rng.randint(1, 5)
        for chunk in range(chunk_total):
            tokens = [rng.choice(WORD_LIST) for _ in range(rng.randint(8, 16))]
            vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            records.append(
                {
                    "id": f"{doc_id}-c{chunk}",
                    "document_id": doc_id,
                    "chunk_index": chunk,
                    "text": " ".join(tokens),
                    "embedding": [round(value / norm, 6) for value in vector],
                    "category": rng.choice(CATEGORIES),
                    "score": round(rng.uniform(0.0, 1.0), 6),
                    "created_at": _iso(start_index + offset),
                }
            )
    return records


def _group(records: list[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(record["document_id"], []).append(index)
    return grouped


def _replace_one_word(rng: random.Random, text: str) -> str:
    words = text.split(" ")
    if not words:
        return text + " extra"
    position = rng.randrange(len(words))
    replacement = rng.choice(WORD_LIST)
    while replacement == words[position]:
        replacement = rng.choice(WORD_LIST)
    words[position] = replacement
    return " ".join(words)


def _mutate_documents(
    rng: random.Random, records: list[dict], grouped, chosen: set[int]
) -> list[dict]:
    result = copy.deepcopy(records)
    for offset in sorted(chosen):
        doc_id = f"doc-{offset}"
        indices = grouped.get(doc_id, [])
        if not indices:
            continue
        index = indices[0]
        result[index]["text"] = _replace_one_word(rng, result[index]["text"])
    return result


def _whitespace_documents(records: list[dict]) -> list[dict]:
    result = copy.deepcopy(records)
    for record in result:
        record["text"] = record["text"].replace(" ", "  ", 1)
    return result


def _replace_documents(
    rng: random.Random, records: list[dict], grouped, chosen: set[int]
) -> list[dict]:
    result = copy.deepcopy(records)
    for offset in sorted(chosen):
        doc_id = f"doc-{offset}"
        for index in grouped.get(doc_id, []):
            result[index]["text"] = " ".join(
                rng.choice(WORD_LIST) for _ in range(rng.randint(8, 16))
            )
    return result


def _move_documents(
    rng: random.Random, records: list[dict], grouped, chosen: set[int]
) -> list[dict]:
    result = copy.deepcopy(records)
    for offset in sorted(chosen):
        indices = grouped.get(f"doc-{offset}", [])
        if len(indices) >= 2:
            first, second = result[indices[0]], result[indices[1]]
            first["chunk_index"], second["chunk_index"] = (
                second["chunk_index"],
                first["chunk_index"],
            )
    return result


def generate_corpus_pair(
    seed: int,
    size: int,
    change_profile: ChangeProfile = "none",
    dim: int = 16,
) -> CorpusPair:
    """Build a deterministic snapshot pair with planted expected document counts."""
    rng = random.Random(seed)
    v1_records = _make_records(rng, size, dim)
    grouped = _group(v1_records)

    if change_profile == "none":
        v2_records = copy.deepcopy(v1_records)
        expected = ExpectedOutcome(0, 0, size, 0, 0, 0)
    elif change_profile in ("pct1", "pct10", "pct50"):
        pct = {"pct1": 0.01, "pct10": 0.10, "pct50": 0.50}[change_profile]
        changed = round(size * pct)
        chosen = set(rng.sample(range(size), changed))
        v2_records = _mutate_documents(rng, v1_records, grouped, chosen)
        expected = ExpectedOutcome(0, 0, size - changed, 0, 0, changed)
    elif change_profile == "whitespace_only":
        v2_records = _whitespace_documents(v1_records)
        expected = ExpectedOutcome(0, 0, 0, size, 0, 0)
    elif change_profile == "replaced":
        changed = round(size * 0.10)
        chosen = set(rng.sample(range(size), changed))
        v2_records = _replace_documents(rng, v1_records, grouped, chosen)
        expected = ExpectedOutcome(0, 0, size - changed, 0, changed, 0)
    elif change_profile == "added_removed":
        changed = round(size * 0.10)
        removed = set(rng.sample(range(size), changed))
        v2_records = [
            copy.deepcopy(record)
            for record in v1_records
            if record["document_id"] not in {f"doc-{offset}" for offset in removed}
        ]
        v2_records.extend(_make_records(rng, changed, dim, start_index=size))
        expected = ExpectedOutcome(changed, changed, size - changed, 0, 0, 0)
    elif change_profile == "chunk_moves":
        changed = round(size * 0.10)
        chosen = set(rng.sample(range(size), changed))
        v2_records = _move_documents(rng, v1_records, grouped, chosen)
        expected = ExpectedOutcome(0, 0, size, 0, 0, 0)
    else:
        raise ValueError(f"unknown change_profile {change_profile!r}")

    return CorpusPair(v1_records=v1_records, v2_records=v2_records, expected=expected)


_CSV_COLUMNS = (
    "id",
    "document_id",
    "chunk_index",
    "text",
    "embedding",
    "category",
    "score",
    "created_at",
)


def write_corpus_as(records: list[dict], fmt: str, path: Path) -> Path:
    """Write records to the given format, matching the matching adapter's shape."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        return target
    if fmt == "json":
        target.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return target
    if fmt == "csv":
        import csv as _csv

        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow(_CSV_COLUMNS)
            for record in records:
                row = []
                for column in _CSV_COLUMNS:
                    value = record[column]
                    if column == "embedding":
                        row.append(json.dumps(value))
                    elif column == "score":
                        row.append(repr(value))
                    else:
                        row.append(str(value))
                writer.writerow(row)
        return target
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        dim = len(records[0]["embedding"]) if records else 0
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("document_id", pa.string()),
                ("chunk_index", pa.int64()),
                ("text", pa.string()),
                ("embedding", pa.list_(pa.float32(), dim)),
                ("category", pa.string()),
                ("score", pa.float64()),
                ("created_at", pa.timestamp("us", tz="UTC")),
            ]
        )
        table = pa.table(
            {
                "id": [record["id"] for record in records],
                "document_id": [record["document_id"] for record in records],
                "chunk_index": [record["chunk_index"] for record in records],
                "text": [record["text"] for record in records],
                "embedding": [record["embedding"] for record in records],
                "category": [record["category"] for record in records],
                "score": [record["score"] for record in records],
                "created_at": [
                    datetime.fromisoformat(record["created_at"]).astimezone(
                        timezone.utc
                    )
                    for record in records
                ],
            },
            schema=schema,
        )
        pq.write_table(table, target)
        return target
    raise ValueError(f"unknown format {fmt!r}")
