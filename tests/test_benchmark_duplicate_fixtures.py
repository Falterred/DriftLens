"""Step 51 full-scale duplicate-corpus fixture generators (Release 1.0)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from driftlens.analysis.duplicates import compute_all_duplicate_stages
from driftlens.config.schema import DuplicatesCfg, TextCfg
from driftlens.identity.chunks import compute_text_hashes
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.text.shingles import jaccard, shingles
from driftlens.text.tokenizer import tokenize
from fixtures.generator import (
    duplicate_corpus_plan,
    generate_corpus_pair,
    generate_duplicate_corpus,
)

NEAR_THRESHOLD = 0.90
SEMANTIC_THRESHOLD = 0.98
CFG = DuplicatesCfg()
TEXT_CFG = TextCfg()


def _canonical(record: dict) -> Record:
    item = Record(
        record_id=record["id"],
        document_id=record["document_id"],
        text=record["text"],
        chunk_index=record["chunk_index"],
    )
    compute_text_hashes(item)
    vector = np.asarray(record["embedding"], dtype=np.float64)
    item.embedding = EmbeddingInfo(
        vector=vector,
        dtype="float64",
        dimension=int(vector.shape[0]),
        norm=float(np.linalg.norm(vector)),
        validity="ok",
        model=None,
        provider=None,
        similarity_metric="cosine",
        normalized=False,
    )
    item._embedding_digest_hex = record["id"]
    return item


def _stages(pair):
    records = [_canonical(record) for record in pair.v1_records]
    return compute_all_duplicate_stages(
        records,
        cfg=CFG,
        near_threshold=NEAR_THRESHOLD,
        semantic_threshold=SEMANTIC_THRESHOLD,
        similarity_metric_known=True,
        seed=42,
        text_cfg=TEXT_CFG,
    )


def _shingle_set(text: str):
    return shingles(tokenize(text), 5, casefold=True)


def test_duplicate_generator_self_check_near_group():
    pair = generate_duplicate_corpus(
        11, 3, exact_rate=0.0, near_rate=1.0, semantic_rate=0.0
    )
    base = pair.v1_records[0]["text"]
    for record in pair.v1_records[1:3]:
        score = jaccard(_shingle_set(base), _shingle_set(record["text"]))
        assert NEAR_THRESHOLD <= score < 1.0

    measured = _stages(pair)
    assert len(measured.near_groups) == 1
    assert len(measured.near_groups[0].member_ids) == 3


def test_duplicate_generator_self_check_semantic_group():
    pair = generate_duplicate_corpus(
        13, 3, exact_rate=0.0, near_rate=0.0, semantic_rate=1.0
    )
    vectors = [
        np.asarray(record["embedding"], dtype=np.float64)
        for record in pair.v1_records[:3]
    ]
    base = vectors[0]
    for vector in vectors[1:]:
        cosine = float(np.dot(base, vector)) / (
            float(np.linalg.norm(base)) * float(np.linalg.norm(vector))
        )
        assert cosine >= SEMANTIC_THRESHOLD

    measured = _stages(pair)
    assert len(measured.semantic_groups) == 1
    assert len(measured.semantic_groups[0].member_ids) == 3


def test_exact_near_semantic_rates_match_generator_intent():
    size = 24
    rates = {"exact_rate": 0.25, "near_rate": 0.25, "semantic_rate": 0.25}
    pair = generate_duplicate_corpus(17, size, **rates)
    plan = duplicate_corpus_plan(size, **rates)
    assert plan.exact_groups >= 1
    assert plan.near_groups >= 1
    assert plan.semantic_groups >= 1

    measured = _stages(pair)
    assert len(measured.groups) == plan.exact_groups
    assert len(measured.near_groups) == plan.near_groups
    assert len(measured.semantic_groups) == plan.semantic_groups


def test_chain_case_classified_correctly():
    pair = generate_duplicate_corpus(
        19,
        12,
        exact_rate=0.0,
        near_rate=0.0,
        semantic_rate=0.0,
        chain_case=True,
    )
    measured = _stages(pair)
    assert len(measured.groups) == 1
    assert sorted(measured.groups[0].member_ids) == ["dup-000000", "dup-000001"]
    assert len(measured.near_groups) == 1
    assert sorted(measured.near_groups[0].member_ids) == ["dup-000000", "dup-000002"]
    assert measured.semantic_groups == []
    assert measured.redundant_count == 1
    assert measured.redundant_near == 1


@pytest.mark.benchmark_scale
def test_1m_scale_fixture_completes_and_records_time():
    seed = 20240501
    documents = 333_333
    started = time.perf_counter()
    pair = generate_corpus_pair(seed, documents)
    elapsed = time.perf_counter() - started

    results_path = Path(__file__).resolve().parent / "fixtures" / "benchmark_results.json"
    results_path.write_text(
        json.dumps(
            {
                "tier": "1m",
                "seed": seed,
                "documents": documents,
                "v1_chunks": len(pair.v1_records),
                "seconds": round(elapsed, 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert len(pair.v1_records) > 0
    assert results_path.exists()
