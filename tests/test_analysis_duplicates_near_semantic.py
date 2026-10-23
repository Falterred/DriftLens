"""Near and semantic duplicate stages (Step 37)."""

from __future__ import annotations

import numpy as np
import pytest

from driftlens.analysis.duplicates import (
    DuplicateUnit,
    compare_duplicates,
    compute_all_duplicate_stages,
    compute_near_groups,
    compute_semantic_groups,
    lsh_band_rows,
    lsh_candidate_pairs,
    minhash_signature,
)
from driftlens.config.schema import DuplicatesCfg, TextCfg, default_config
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import (
    DocumentTable,
    Snapshot,
    StatisticsAccumulator,
)
from driftlens.text.shingles import jaccard, shingles
from driftlens.text.tokenizer import tokenize

_BASE_WORDS = [f"w{index}" for index in range(40)]


def _base_text() -> str:
    return " ".join(_BASE_WORDS)


def _one_word_changed() -> str:
    words = list(_BASE_WORDS)
    words[19] = "changed"
    return " ".join(words)


def _record(record_id, *, text=None, document_id=None, chunk_index=None):
    record = Record(record_id=record_id, text=text, normalized_text=text)
    if text is not None:
        record.normalized_text_hash = text
    record.document_id = document_id
    record.chunk_index = chunk_index
    return record


def _embedding_record(record_id, vector, *, metric="cosine", validity="ok"):
    record = Record(record_id=record_id)
    array = np.asarray(vector, dtype=float)
    record.embedding = EmbeddingInfo(
        vector=array,
        dtype="float64",
        dimension=array.shape[0],
        norm=float(np.linalg.norm(array)),
        validity=validity,
        model=None,
        provider=None,
        similarity_metric=metric,
        normalized=False,
    )
    record._embedding_digest_hex = record_id
    return record


def _unit(record_id, *, text=None, vector=None, metric="cosine",
          document_id=None, chunk_index=None):
    record = _record(
        record_id, text=text, document_id=document_id, chunk_index=chunk_index
    )
    if vector is not None:
        record.embedding = _embedding_record(record_id, vector, metric=metric).embedding
    return DuplicateUnit(unit_id=record_id, representative=record, members=[record])


def _manifest(record_ids, *, similarity_metric=None):
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
        driftlens_version="0.1.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity="CHUNK",
        granularity_source="detected",
        document_count=len(record_ids),
        chunk_count=len(record_ids),
        embedding_dimension=None,
        embedding_dtype=None,
        embedding_model=None,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric=similarity_metric,
        embedding_normalized=None,
        chunk_identity_levels={record_id: [4] for record_id in record_ids},
        identity_summary={"HIGH": len(record_ids), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _snapshot(records, *, similarity_metric=None):
    return Snapshot(
        manifest=_manifest(
            [record.record_id for record in records],
            similarity_metric=similarity_metric,
        ),
        records=list(records),
        documents=DocumentTable(rows={}),
        statistics=StatisticsAccumulator(),
    )


def _three_stage_records():
    exact_text = "e0 e1 e2 e3 e4 e5 e6 e7 e8 e9"
    return [
        _record("e1", text=exact_text),
        _record("e2", text=exact_text),
        _record("e3", text=exact_text),
        _record("n1", text=_base_text()),
        _record("n2", text=_one_word_changed()),
        _embedding_record("s1", [1.0, 0.0]),
        _embedding_record("s2", [3.0, 0.0]),
    ]


def test_minhash_signature_deterministic() -> None:
    shingle_set = shingles(tokenize(_base_text()), 5)
    first = minhash_signature(shingle_set, seed=42)
    second = minhash_signature(shingle_set, seed=42)
    assert first == second
    assert len(first) == 128


def test_minhash_signature_empty_set_sentinel() -> None:
    signature = minhash_signature(frozenset(), seed=42)
    assert signature == tuple(2 ** 256 for _ in range(128))


def test_lsh_band_rows() -> None:
    assert lsh_band_rows(0.9) == 4
    assert lsh_band_rows(0.5) == 4
    assert lsh_band_rows(0.3) == 2


def test_lsh_candidates_find_similar_pairs() -> None:
    left = minhash_signature(shingles(tokenize(_base_text()), 5), seed=42)
    right = minhash_signature(shingles(tokenize(_one_word_changed()), 5), seed=42)
    candidates = lsh_candidate_pairs({"a": left, "b": right}, r=lsh_band_rows(0.5))
    assert frozenset({"a", "b"}) in candidates


def test_near_group_verified_by_exact_jaccard(monkeypatch) -> None:
    from driftlens.analysis import duplicates as module

    constant = tuple(1 for _ in range(128))
    monkeypatch.setattr(module, "minhash_signature", lambda *args, **kwargs: constant)

    left_text = "a0 a1 a2 a3 a4 a5 a6 a7 a8 a9"
    right_text = "a0 a1 a2 a3 a4 a5 a6 a7 a8 zz"
    left_shingles = shingles(tokenize(left_text), 5)
    right_shingles = shingles(tokenize(right_text), 5)
    assert jaccard(left_shingles, right_shingles) < 0.9

    units = [_unit("a", text=left_text), _unit("b", text=right_text)]
    groups, _next_units, _sample = compute_near_groups(
        units, cfg=DuplicatesCfg(), threshold=0.9, seed=42
    )
    assert groups == []


def test_near_adjacent_chunks_excluded_by_default() -> None:
    units = [
        _unit("a", text=_base_text(), document_id="doc", chunk_index=0),
        _unit("b", text=_one_word_changed(), document_id="doc", chunk_index=1),
    ]
    groups, _next_units, _sample = compute_near_groups(
        units, cfg=DuplicatesCfg(), threshold=0.5, seed=42
    )
    assert groups == []


def test_near_adjacent_chunks_included_when_disabled() -> None:
    units = [
        _unit("a", text=_base_text(), document_id="doc", chunk_index=0),
        _unit("b", text=_one_word_changed(), document_id="doc", chunk_index=1),
    ]
    groups, _next_units, _sample = compute_near_groups(
        units,
        cfg=DuplicatesCfg(ignore_adjacent_chunks=False),
        threshold=0.5,
        seed=42,
    )
    assert len(groups) == 1
    assert groups[0].member_ids == ["a", "b"]


def test_semantic_requires_known_metric(monkeypatch) -> None:
    from driftlens.analysis import duplicates as module

    calls: list[int] = []
    monkeypatch.setattr(
        module, "_cosine", lambda *args, **kwargs: calls.append(1) or 1.0
    )
    units = [
        _unit("a", text="alpha", vector=[1.0, 0.0]),
        _unit("b", text="beta", vector=[1.0, 0.0]),
    ]
    groups, next_units = compute_semantic_groups(
        units,
        cfg=DuplicatesCfg(),
        threshold=0.9,
        similarity_metric_known=False,
        seed=42,
    )
    assert groups == []
    assert next_units == units
    assert calls == []


def test_semantic_uses_normalized_cosine_regardless_of_declared_metric() -> None:
    records = [
        _embedding_record("a", [1.0, 0.0], metric="l2"),
        _embedding_record("b", [3.0, 0.0], metric="l2"),
    ]
    assert records[0].embedding.similarity_metric == "l2"
    snapshot = _snapshot(records, similarity_metric="l2")
    _, _, metrics = compare_duplicates(
        snapshot,
        snapshot,
        cfg=DuplicatesCfg(),
        drift_cfg=default_config().drift,
        similarity_metric_known=True,
    )
    by_name = {metric.name: metric for metric in metrics}
    assert by_name["duplicate_rate_semantic"].v2 == 0.5


def test_three_classes_disjoint_sum() -> None:
    records = _three_stage_records()
    result = compute_all_duplicate_stages(
        records,
        cfg=DuplicatesCfg(),
        near_threshold=0.5,
        semantic_threshold=0.98,
        similarity_metric_known=True,
        seed=42,
    )
    assert result.redundant_count == 2
    assert result.redundant_near == 1
    assert result.redundant_semantic == 1
    assert len(result.semantic_units) == 3
    assert (
        result.redundant_count + result.redundant_near + result.redundant_semantic
        == len(records) - len(result.semantic_units)
    )


def test_compute_all_duplicate_stages_chains_correctly() -> None:
    records = _three_stage_records()
    result = compute_all_duplicate_stages(
        records,
        cfg=DuplicatesCfg(),
        near_threshold=0.5,
        semantic_threshold=0.98,
        similarity_metric_known=True,
        seed=42,
    )
    assert result.duplicate_rate_exact > 0.0
    assert result.duplicate_rate_near > 0.0
    assert result.duplicate_rate_semantic > 0.0
    assert result.duplicate_rate_total == pytest.approx(
        result.duplicate_rate_exact
        + result.duplicate_rate_near
        + result.duplicate_rate_semantic
    )


def test_compare_duplicates_aggregate_now_populated() -> None:
    records = _three_stage_records()
    snapshot = _snapshot(records, similarity_metric="cosine")
    _, _, metrics = compare_duplicates(
        snapshot,
        snapshot,
        cfg=DuplicatesCfg(),
        drift_cfg=default_config().drift,
        near_threshold=0.5,
        semantic_threshold=0.98,
        similarity_metric_known=True,
    )
    by_name = {metric.name: metric for metric in metrics}
    for name in ("duplicate_rate_near", "duplicate_rate_semantic"):
        assert by_name[name].v2 is not None
        assert by_name[name].v2 > 0.0
        assert by_name[name].status != "N/A"
        assert by_name[name].status == "UNBASELINED"
