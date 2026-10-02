"""Temporal / freshness-shift analysis tests (Step 44)."""

from __future__ import annotations

from driftlens.analysis import temporal
from driftlens.analysis.content import (
    DocumentDiffRow,
    _reconstruct_all,
    compare_content,
    diff_documents,
)
from driftlens.analysis.temporal import (
    compare_temporal,
    compute_as_of,
    compute_freshness_shift,
    compute_timestamp_inconsistent_flags,
    detect_future_timestamps,
)
from driftlens.config.schema import default_config
from driftlens.errors import DiagnosticSink
from driftlens.identity.chunks import compute_text_hashes
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import Record
from driftlens.schema.snapshot import (
    DocumentRow,
    DocumentTable,
    Snapshot,
    StatisticsAccumulator,
)


def _record(record_id, *, document_id="d1", created_at=None, updated_at=None):
    return Record(
        record_id=record_id,
        document_id=document_id,
        created_at=created_at,
        updated_at=updated_at,
    )


def _row(document_id, status, *, metadata_changed=False, timestamp_inconsistent=False):
    return DocumentDiffRow(
        document_id=document_id,
        status=status,
        metadata_changed=metadata_changed,
        timestamp_inconsistent=timestamp_inconsistent,
        reconstruction_confidence_v1="HIGH",
        reconstruction_confidence_v2="HIGH",
        similarity=None,
    )


def _manifest(levels, *, granularity="DOCUMENT"):
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={
            "min_overlap_chars": 16,
            "overlap_window": 2000,
            "casefold": True,
        },
        driftlens_version="0.1.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity=granularity,
        granularity_source="detected" if granularity == "CHUNK" else "assumed",
        document_count=len(levels),
        chunk_count=len(levels),
        embedding_dimension=None,
        embedding_dtype=None,
        embedding_model=None,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric=None,
        embedding_normalized=None,
        chunk_identity_levels=levels,
        identity_summary={"HIGH": len(levels), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _snapshot(entries):
    # entries: dict document_id -> (text, created_at, updated_at)
    records = []
    rows = {}
    for index, (document_id, (text, created_at, updated_at)) in enumerate(entries.items()):
        record = Record(
            record_id=f"{document_id}::0",
            document_id=document_id,
            text=text,
            created_at=created_at,
            updated_at=updated_at,
            row_number=index,
        )
        compute_text_hashes(record)
        records.append(record)
        rows[document_id] = DocumentRow(
            document_id=document_id, chunk_count=1, identity_confidence="HIGH"
        )
    manifest = _manifest(
        {record.record_id: [4] for record in records}, granularity="DOCUMENT"
    )
    return Snapshot(
        manifest=manifest,
        records=records,
        documents=DocumentTable(rows=rows),
        statistics=StatisticsAccumulator(),
    )


def test_compute_as_of_pools_both_fields():
    records = [
        _record("r1", created_at="2023-05-01T00:00:00Z"),
        _record("r2", updated_at="2024-06-01T00:00:00Z"),
        _record("r3", created_at="2024-01-01T00:00:00Z"),
    ]
    assert compute_as_of(records) == "2024-06-01T00:00:00Z"


def test_compute_as_of_empty():
    assert compute_as_of([_record("r1"), _record("r2")]) is None


def test_freshness_shift_independent_as_of_by_default(monkeypatch):
    v1 = [
        _record("a", created_at="2024-01-01T00:00:00Z"),
        _record("b", created_at="2024-01-04T00:00:00Z"),
    ]
    v2 = [
        _record("c", created_at="2024-02-01T00:00:00Z"),
        _record("d", created_at="2024-02-02T00:00:00Z"),
    ]
    assert compute_as_of(v1) == "2024-01-04T00:00:00Z"
    assert compute_as_of(v2) == "2024-02-02T00:00:00Z"

    captured = {}

    def _spy(ages_v1, ages_v2):
        captured["v1"] = list(ages_v1)
        captured["v2"] = list(ages_v2)
        return (0.25, None)

    monkeypatch.setattr(temporal, "ks_statistic", _spy)
    result = compute_freshness_shift(
        v1, v2, field="created_at", as_of_override=None
    )
    assert captured["v1"] == [259200.0, 0.0]
    assert captured["v2"] == [86400.0, 0.0]
    assert result.ks_d == 0.25
    assert result.reason is None
    assert result.field == "created_at"


def test_freshness_shift_shared_as_of_when_overridden(monkeypatch):
    v1 = [
        _record("a", created_at="2024-01-01T00:00:00Z"),
        _record("b", created_at="2024-01-04T00:00:00Z"),
    ]
    v2 = [
        _record("c", created_at="2024-02-01T00:00:00Z"),
        _record("d", created_at="2024-02-02T00:00:00Z"),
    ]
    captured = {}

    def _spy(ages_v1, ages_v2):
        captured["v1"] = list(ages_v1)
        captured["v2"] = list(ages_v2)
        return (0.5, None)

    monkeypatch.setattr(temporal, "ks_statistic", _spy)
    result = compute_freshness_shift(
        v1,
        v2,
        field="created_at",
        as_of_override="2024-03-01T00:00:00Z",
    )
    assert captured["v1"] == [5184000.0, 4924800.0]
    assert captured["v2"] == [2505600.0, 2419200.0]
    assert result.ks_d == 0.5


def test_freshness_shift_empty_sample():
    v1 = [_record("a")]
    v2 = [_record("b", created_at="2024-01-01T00:00:00Z")]
    result = compute_freshness_shift(
        v1, v2, field="created_at", as_of_override=None
    )
    assert result.ks_d is None
    assert result.reason == "EMPTY_SAMPLE"
    assert result.field == "created_at"


def test_future_timestamp_none_without_override():
    records = [
        _record("old", created_at="2020-01-01T00:00:00Z"),
        _record("new", updated_at="2999-01-01T00:00:00Z"),
    ]
    assert detect_future_timestamps(records, as_of_override=None) == []


def test_future_timestamp_detected_with_override():
    records = [
        _record("old", created_at="2024-01-01T00:00:00Z"),
        _record("new", updated_at="2024-06-01T00:00:00Z"),
    ]
    assert detect_future_timestamps(
        records, as_of_override="2024-03-01T00:00:00Z"
    ) == ["new"]


def test_timestamp_inconsistent_content_changed_timestamp_not():
    rows = [_row("d1", "MODIFIED")]
    v1 = {"d1": [_record("c1", document_id="d1", updated_at="2024-01-01T00:00:00Z")]}
    v2 = {"d1": [_record("c2", document_id="d1", updated_at="2024-01-01T00:00:00Z")]}
    assert compute_timestamp_inconsistent_flags(rows, None, v1, v2) == {"d1"}


def test_timestamp_inconsistent_timestamp_changed_content_not():
    rows = [_row("d1", "UNCHANGED")]
    v1 = {"d1": [_record("c1", document_id="d1", updated_at="2024-01-01T00:00:00Z")]}
    v2 = {"d1": [_record("c2", document_id="d1", updated_at="2024-02-01T00:00:00Z")]}
    assert compute_timestamp_inconsistent_flags(rows, None, v1, v2) == {"d1"}


def test_timestamp_inconsistent_both_changed_together_not_flagged():
    rows = [_row("d1", "MODIFIED")]
    v1 = {"d1": [_record("c1", document_id="d1", updated_at="2024-01-01T00:00:00Z")]}
    v2 = {"d1": [_record("c2", document_id="d1", updated_at="2024-02-01T00:00:00Z")]}
    assert compute_timestamp_inconsistent_flags(rows, None, v1, v2) == set()


def test_timestamp_inconsistent_neither_changed_not_flagged():
    rows = [_row("d1", "UNCHANGED")]
    v1 = {"d1": [_record("c1", document_id="d1", updated_at="2024-01-01T00:00:00Z")]}
    v2 = {"d1": [_record("c2", document_id="d1", updated_at="2024-01-01T00:00:00Z")]}
    assert compute_timestamp_inconsistent_flags(rows, None, v1, v2) == set()


def test_timestamp_inconsistent_skips_added_removed():
    rows = [_row("added", "ADDED"), _row("removed", "REMOVED")]
    v1 = {
        "added": [],
        "removed": [_record("c1", document_id="removed", updated_at="2024-01-01T00:00:00Z")],
    }
    v2 = {
        "added": [_record("c2", document_id="added", updated_at="2024-02-01T00:00:00Z")],
        "removed": [],
    }
    assert compute_timestamp_inconsistent_flags(rows, None, v1, v2) == set()


def test_compare_temporal_report_only_no_threshold():
    v1 = _snapshot({
        "d1": ("hello world", "2024-01-01T00:00:00Z", "2024-01-15T00:00:00Z"),
    })
    v2 = _snapshot({
        "d1": ("hello world", "2024-03-01T00:00:00Z", "2024-03-15T00:00:00Z"),
    })
    metrics = compare_temporal(
        v1, v2, as_of_override=None, sink=DiagnosticSink()
    )
    assert [metric.name for metric in metrics] == [
        "freshness_ks_created_at",
        "freshness_ks_updated_at",
    ]
    for metric in metrics:
        assert metric.is_threshold_key is False
        assert metric.status == "REPORT_ONLY"


def test_content_py_two_phase_flag_patch():
    cfg = default_config()
    changed_text_v1 = " ".join(f"w{index}" for index in range(50))
    changed_text_v2 = " ".join([f"w{index}" for index in range(49)] + ["zz"])
    v1 = _snapshot({
        "d_unchanged": ("one two three four five", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        "d_modified": (changed_text_v1, "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z"),
    })
    v2 = _snapshot({
        "d_unchanged": ("one two three four five", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        "d_modified": (changed_text_v2, "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z"),
    })

    rows, _chunks, _metrics = compare_content(v1, v2, cfg=cfg, sink=DiagnosticSink())
    base = {row.document_id: row for row in rows}
    assert base["d_unchanged"].status == "UNCHANGED"
    assert base["d_modified"].status in ("MODIFIED", "REPLACED")
    assert all(not row.metadata_changed for row in rows)
    assert all(not row.timestamp_inconsistent for row in rows)

    v1_by_doc = {record.document_id: [record] for record in v1.records}
    v2_by_doc = {record.document_id: [record] for record in v2.records}
    ts_flags = compute_timestamp_inconsistent_flags(rows, None, v1_by_doc, v2_by_doc)
    assert ts_flags == {"d_modified"}
    md_flags = {"d_unchanged"}

    patched = diff_documents(
        v1.documents,
        v2.documents,
        _reconstruct_all(v1, cfg),
        _reconstruct_all(v2, cfg),
        cfg=cfg.documents,
        metadata_changed_ids=md_flags,
        timestamp_inconsistent_ids=ts_flags,
    )
    final = {row.document_id: row for row in patched}
    assert final["d_modified"].timestamp_inconsistent is True
    assert final["d_modified"].metadata_changed is False
    assert final["d_unchanged"].metadata_changed is True
    assert final["d_unchanged"].timestamp_inconsistent is False
