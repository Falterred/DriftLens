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

# --- Step 51: duplicate / retrieval benchmark fixtures ----------------------

_DUPLICATE_GROUP_SIZE = 3
_NEAR_BAND_THRESHOLD = 0.90
_NEAR_BAND_CEILING = 0.99
_SEMANTIC_BAND_THRESHOLD = 0.98
_NEAR_TEXT_WORDS = 120


@dataclass
class DuplicatePlan:
    exact_groups: int
    near_groups: int
    semantic_groups: int
    group_size: int


@dataclass
class RetrievalFixturePlan:
    removed_documents: int
    unmappable_chunks: int
    edited_documents: int
    rechunked_documents: int
    partial_cases: int
    valid_documents: int


def _unit_vector(rng: random.Random, dim: int) -> list[float]:
    vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _random_text(rng: random.Random, words: int = _NEAR_TEXT_WORDS) -> str:
    return ' '.join(rng.choice(WORD_LIST) for _ in range(words))


def _mutate_words(rng: random.Random, text: str, count: int) -> str:
    parts = text.split(' ')
    if len(parts) < 2:
        return text
    count = max(1, min(count, len(parts) - 1))
    positions = rng.sample(range(len(parts)), count)
    for position in positions:
        replacement = rng.choice(WORD_LIST)
        while replacement == parts[position]:
            replacement = rng.choice(WORD_LIST)
        parts[position] = replacement
    return ' '.join(parts)


def _fixture_record(
    record_id: str,
    *,
    text: str,
    vector: list[float],
    document_id: str | None = None,
    chunk_index: int = 0,
    category: str = 'alpha',
    score: float = 0.0,
    created_at: str = '2024-01-01T00:00:00+00:00',
) -> dict:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return {
        'id': record_id,
        'document_id': record_id if document_id is None else document_id,
        'chunk_index': chunk_index,
        'text': text,
        'embedding': [round(value / norm, 6) for value in vector],
        'category': category,
        'score': score,
        'created_at': created_at,
    }


def _cosine(first: list[float], second: list[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second))
    norm_a = math.sqrt(sum(a * a for a in first)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in second)) or 1.0
    return dot / (norm_a * norm_b)


def _shingle_set(text: str):
    from driftlens.text.shingles import shingles
    from driftlens.text.tokenizer import tokenize

    return shingles(tokenize(text), 5, casefold=True)


def _near_variant(
    rng: random.Random,
    base: str,
    *,
    threshold: float = _NEAR_BAND_THRESHOLD,
    ceiling: float = _NEAR_BAND_CEILING,
    attempts: int = 64,
) -> str:
    from driftlens.text.shingles import jaccard

    base_shingles = _shingle_set(base)
    for _ in range(attempts):
        candidate = _mutate_words(rng, base, 1)
        score = jaccard(base_shingles, _shingle_set(candidate))
        if threshold <= score < ceiling:
            return candidate
    raise ValueError('could not draw a near-duplicate variant inside the band')


def _semantic_variant(
    rng: random.Random,
    base: list[float],
    *,
    threshold: float = _SEMANTIC_BAND_THRESHOLD,
    attempts: int = 64,
) -> list[float]:
    for _ in range(attempts):
        candidate = [value + rng.gauss(0.0, 0.01) for value in base]
        if _cosine(base, candidate) >= threshold:
            return candidate
    return list(base)


def _groups_for_rate(size: int, rate: float, group_size: int) -> int:
    if rate <= 0.0:
        return 0
    return max(1, round(size * rate / group_size))


def duplicate_corpus_plan(
    size: int,
    *,
    exact_rate: float,
    near_rate: float,
    semantic_rate: float,
    group_size: int = _DUPLICATE_GROUP_SIZE,
) -> DuplicatePlan:
    return DuplicatePlan(
        exact_groups=_groups_for_rate(size, exact_rate, group_size),
        near_groups=_groups_for_rate(size, near_rate, group_size),
        semantic_groups=_groups_for_rate(size, semantic_rate, group_size),
        group_size=group_size,
    )


def generate_duplicate_corpus(
    seed: int,
    size: int,
    *,
    exact_rate: float,
    near_rate: float,
    semantic_rate: float,
    chain_case: bool = False,
    dim: int = 16,
) -> CorpusPair:
    rng = random.Random(seed)
    plan = duplicate_corpus_plan(
        size,
        exact_rate=exact_rate,
        near_rate=near_rate,
        semantic_rate=semantic_rate,
    )
    records: list[dict] = []
    counter = 0

    def _append(text: str, vector: list[float]) -> None:
        nonlocal counter
        record_id = f'dup-{counter:06d}'
        counter += 1
        records.append(_fixture_record(record_id, text=text, vector=vector))

    for _ in range(plan.exact_groups):
        text = _random_text(rng)
        for _ in range(plan.group_size):
            _append(text, _unit_vector(rng, dim))

    for _ in range(plan.near_groups):
        base = _random_text(rng)
        variants = [base]
        while len(variants) < plan.group_size:
            variants.append(_near_variant(rng, base))
        for text in variants:
            _append(text, _unit_vector(rng, dim))

    for _ in range(plan.semantic_groups):
        base_vector = _unit_vector(rng, dim)
        for _ in range(plan.group_size):
            _append(_random_text(rng), _semantic_variant(rng, base_vector))

    if chain_case:
        anchor = _random_text(rng)
        _append(anchor, _unit_vector(rng, dim))
        _append(anchor, _unit_vector(rng, dim))
        _append(_near_variant(rng, anchor), _unit_vector(rng, dim))

    while len(records) < size:
        _append(
            _random_text(rng, words=rng.randint(8, 16)),
            _unit_vector(rng, dim),
        )

    v2_records = copy.deepcopy(records)
    return CorpusPair(
        v1_records=records,
        v2_records=v2_records,
        expected=ExpectedOutcome(0, 0, len(records), 0, 0, 0),
    )


def retrieval_fixture_plan(
    corpus_size: int,
    *,
    ground_truth_removed_rate: float,
    edited_rate: float,
    rechunked_rate: float,
    partial_gt_rate: float,
) -> RetrievalFixturePlan:
    removed = max(1, round(corpus_size * ground_truth_removed_rate))
    edited = max(1, round(corpus_size * edited_rate))
    rechunked = max(1, round(corpus_size * rechunked_rate))
    valid = max(1, corpus_size - removed - edited - rechunked)
    partial = max(1, round(corpus_size * partial_gt_rate))
    partial = min(partial, removed, valid)
    return RetrievalFixturePlan(
        removed_documents=removed,
        unmappable_chunks=removed,
        edited_documents=edited,
        rechunked_documents=rechunked,
        partial_cases=partial,
        valid_documents=valid,
    )


def generate_retrieval_eval_fixture(
    seed: int,
    corpus_size: int,
    *,
    ground_truth_removed_rate: float,
    edited_rate: float,
    rechunked_rate: float,
    partial_gt_rate: float,
    dim: int = 16,
) -> tuple[CorpusPair, 'object']:
    import numpy as np

    from driftlens.schema.eval_set import (
        EvalCase,
        EvalSet,
        QueryEmbeddings,
        derive_query_id,
    )

    rng = random.Random(seed)
    plan = retrieval_fixture_plan(
        corpus_size,
        ground_truth_removed_rate=ground_truth_removed_rate,
        edited_rate=edited_rate,
        rechunked_rate=rechunked_rate,
        partial_gt_rate=partial_gt_rate,
    )
    v1_records: list[dict] = []
    v2_records: list[dict] = []
    cases: list[EvalCase] = []

    def _base_text() -> str:
        return _random_text(rng, words=rng.randint(8, 16))

    def _add_query(
        query: str,
        document_ids: list[str],
        chunk_ids: list[str],
        v1_vector,
        v2_vector,
    ) -> None:
        cases.append(
            EvalCase(
                query_id=derive_query_id(query),
                query_id_was_explicit=False,
                query=query,
                relevant_document_ids=list(document_ids),
                relevant_chunk_ids=list(chunk_ids),
                query_embeddings=QueryEmbeddings(
                    v1=None if v1_vector is None else np.asarray(v1_vector, dtype=np.float64),
                    v2=None if v2_vector is None else np.asarray(v2_vector, dtype=np.float64),
                ),
            )
        )

    for index in range(plan.valid_documents):
        doc_id = f'valid-{index:04d}'
        chunk_id = f'{doc_id}-c0'
        text = _base_text()
        vector = _unit_vector(rng, dim)
        v1_records.append(
            _fixture_record(chunk_id, text=text, vector=vector, document_id=doc_id)
        )
        v2_records.append(
            _fixture_record(chunk_id, text=text, vector=vector, document_id=doc_id)
        )
        _add_query(f'query {text}', [doc_id], [], vector, vector)

    for index in range(plan.edited_documents):
        doc_id = f'edited-{index:04d}'
        chunk_id = f'{doc_id}-c0'
        text = _base_text()
        v1_records.append(
            _fixture_record(
                chunk_id, text=text, vector=_unit_vector(rng, dim), document_id=doc_id
            )
        )
        v2_records.append(
            _fixture_record(
                chunk_id,
                text=_mutate_words(rng, text, 1),
                vector=_unit_vector(rng, dim),
                document_id=doc_id,
            )
        )

    for index in range(plan.removed_documents):
        doc_id = f'removed-{index:04d}'
        chunk_id = f'{doc_id}-c0'
        text = _base_text()
        vector = _unit_vector(rng, dim)
        v1_records.append(
            _fixture_record(chunk_id, text=text, vector=vector, document_id=doc_id)
        )
        _add_query(f'query {text}', [doc_id], [], vector, None)
        _add_query(f'removed chunk query {text}', [], [chunk_id], vector, None)

    for index in range(plan.rechunked_documents):
        doc_id = f'rechunked-{index:04d}'
        v1_chunk = f'{doc_id}-legacy'
        tokens = [f'r{index}x{position}' for position in range(8)]
        v1_records.append(
            _fixture_record(
                v1_chunk,
                text=' '.join(tokens),
                vector=_unit_vector(rng, dim),
                document_id=doc_id,
            )
        )
        v2_records.append(
            _fixture_record(
                f'{doc_id}-c0',
                text=' '.join(tokens[:5]),
                vector=_unit_vector(rng, dim),
                document_id=doc_id,
                chunk_index=0,
            )
        )
        v2_records.append(
            _fixture_record(
                f'{doc_id}-c1',
                text=' '.join(tokens[3:]),
                vector=_unit_vector(rng, dim),
                document_id=doc_id,
                chunk_index=1,
            )
        )
        _add_query(
            f'rechunked query {index}', [], [v1_chunk], None, None
        )

    removed_ids = [f'removed-{index:04d}' for index in range(plan.removed_documents)]
    valid_ids = [f'valid-{index:04d}' for index in range(plan.valid_documents)]
    for index in range(plan.partial_cases):
        present = valid_ids[index % len(valid_ids)]
        missing = removed_ids[index % len(removed_ids)]
        _add_query(f'partial query {index}', [present, missing], [], None, None)

    pair = CorpusPair(
        v1_records=v1_records,
        v2_records=v2_records,
        expected=ExpectedOutcome(
            0, plan.removed_documents, plan.valid_documents, 0, 0, 0
        ),
    )
    return pair, EvalSet(cases=cases)
