"""Default validation for Release 1.0 (Step 52).

Section 21's release rule is mechanical: a control pair must show no content
change and SAME_SPACE, and a planted change at the stated magnitude must be
detected in the expected class in at least 95% of seeded repetitions.  This
module compiles the explicit set of classification-gating defaults and runs that
rule across many seeds, unlike the rest of the plan which is single-seeded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pytest

from driftlens.analysis.chunking import classify_chunk_anomaly
from driftlens.analysis.chunk_matching import match_chunks_full
from driftlens.analysis.clusters import (
    ClusterAssignment,
    ClusteringResult,
    classify_cluster_transitions,
)
from driftlens.analysis.compatibility import run_space_consistency_test
from driftlens.analysis.content import (
    ChunkDiffResult,
    ComparisonLevelResult,
    DocumentDiffRow,
    ReconstructedDocument,
    diff_documents,
)
from driftlens.analysis.duplicates import compute_all_duplicate_stages
from driftlens.analysis.metadata import infer_field_type
from driftlens.config.schema import Config, default_config
from driftlens.identity.chunks import compute_text_hashes
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import DocumentRow, DocumentTable
from fixtures.generator import generate_corpus_pair, generate_duplicate_corpus

FixtureKind = Literal[
    "content_change",
    "chunking_anomaly",
    "metadata_drift",
    "duplicate_rate",
    "embedding_space_test",
    "chunk_match",
    "cluster_stability",
]

_LEVEL = ComparisonLevelResult(level=1, hash_kind="text")


@dataclass
class DefaultValidationCase:
    """One classification-gating default and the class it must produce."""

    config_key: str
    fixture_kind: FixtureKind
    magnitude_param: str
    expected_class: str
    repetitions: int = 30


# Implementation Logic 1's table, compiled once here.  Keys are the real schema
# keys (the table's "duplicates.*" prefix is a spec typo; both thresholds live on
# AnalysisCfg).  A key may appear in more than one case when the table names an
# above/below pair (e.g. REPLACED for "replaced" and MODIFIED for a small edit).
DEFAULT_VALIDATION_SUITE: list[DefaultValidationCase] = [
    # documents.replace_threshold (0.20)
    DefaultValidationCase(
        "documents.replace_threshold", "content_change", "replaced", "REPLACED"
    ),
    DefaultValidationCase(
        "documents.replace_threshold", "content_change", "small_change", "MODIFIED"
    ),
    # chunk_match.min_similarity (0.60)
    DefaultValidationCase(
        "chunk_match.min_similarity", "chunk_match", "jaccard_above", "MODIFIED"
    ),
    DefaultValidationCase(
        "chunk_match.min_similarity", "chunk_match", "jaccard_below", "UNMATCHED"
    ),
    # chunk_match.coverage (0.80) / split_contribution (0.10)
    DefaultValidationCase(
        "chunk_match.coverage", "chunk_match", "split_above", "SPLIT_PART"
    ),
    DefaultValidationCase(
        "chunk_match.coverage", "chunk_match", "split_below", "UNGROUPED"
    ),
    DefaultValidationCase(
        "chunk_match.split_contribution", "chunk_match", "split_above", "SPLIT_PART"
    ),
    DefaultValidationCase(
        "chunk_match.split_contribution", "chunk_match", "split_below", "UNGROUPED"
    ),
    # chunk_match.rechunk_exact_fraction (0.50)
    DefaultValidationCase(
        "chunk_match.rechunk_exact_fraction", "chunk_match", "rechunk_below", "RECHUNKED"
    ),
    DefaultValidationCase(
        "chunk_match.rechunk_exact_fraction",
        "chunk_match",
        "rechunk_above",
        "NOT_RECHUNKED",
    ),
    # chunking.oversized (2048) / suspicious_short (20) / short (3)
    DefaultValidationCase("chunking.short", "chunking_anomaly", "very_short", "very_short"),
    DefaultValidationCase(
        "chunking.short", "chunking_anomaly", "short_boundary", "suspicious_short"
    ),
    DefaultValidationCase(
        "chunking.suspicious_short",
        "chunking_anomaly",
        "suspicious_short",
        "suspicious_short",
    ),
    DefaultValidationCase(
        "chunking.suspicious_short", "chunking_anomaly", "suspicious_boundary", "NONE"
    ),
    DefaultValidationCase("chunking.oversized", "chunking_anomaly", "oversized", "oversized"),
    DefaultValidationCase(
        "chunking.oversized", "chunking_anomaly", "oversized_boundary", "NONE"
    ),
    # metadata.mixed_type_share (0.05)
    DefaultValidationCase("metadata.mixed_type_share", "metadata_drift", "mixed", "mixed"),
    DefaultValidationCase("metadata.mixed_type_share", "metadata_drift", "dominant", "int"),
    # analysis.near_duplicate_threshold (0.90) / analysis.duplicate_threshold (0.98)
    DefaultValidationCase(
        "analysis.near_duplicate_threshold", "duplicate_rate", "near_above", "GROUPED"
    ),
    DefaultValidationCase(
        "analysis.near_duplicate_threshold", "duplicate_rate", "near_below", "UNGROUPED"
    ),
    DefaultValidationCase(
        "analysis.duplicate_threshold", "duplicate_rate", "semantic_above", "GROUPED"
    ),
    DefaultValidationCase(
        "analysis.duplicate_threshold", "duplicate_rate", "semantic_below", "UNGROUPED"
    ),
    # embeddings.consistency_same (0.95) / consistency_different (0.05)
    DefaultValidationCase(
        "embeddings.consistency_same", "embedding_space_test", "same", "SAME_SPACE"
    ),
    DefaultValidationCase(
        "embeddings.consistency_same", "embedding_space_test", "partial", "PARTIAL_MIGRATION"
    ),
    DefaultValidationCase(
        "embeddings.consistency_same", "embedding_space_test", "same_boundary", "SAME_SPACE"
    ),
    DefaultValidationCase(
        "embeddings.consistency_same",
        "embedding_space_test",
        "sameness_below",
        "PARTIAL_MIGRATION",
    ),
    DefaultValidationCase(
        "embeddings.consistency_different",
        "embedding_space_test",
        "different",
        "DIFFERENT_SPACE",
    ),
    DefaultValidationCase(
        "embeddings.consistency_different",
        "embedding_space_test",
        "different_boundary",
        "DIFFERENT_SPACE",
    ),
    DefaultValidationCase(
        "embeddings.consistency_different",
        "embedding_space_test",
        "difference_below",
        "PARTIAL_MIGRATION",
    ),
    # cluster.stable_jaccard (0.70) / stable_centroid_cosine (0.95)
    DefaultValidationCase(
        "cluster.stable_jaccard", "cluster_stability", "jaccard_above", "STABLE"
    ),
    DefaultValidationCase(
        "cluster.stable_jaccard", "cluster_stability", "jaccard_below", "SHIFTED"
    ),
    DefaultValidationCase(
        "cluster.stable_centroid_cosine", "cluster_stability", "centroid_above", "STABLE"
    ),
    DefaultValidationCase(
        "cluster.stable_centroid_cosine", "cluster_stability", "centroid_below", "SHIFTED"
    ),
    # Borderline: the anchor-count gates decide SPACE_UNTESTABLE vs a real test,
    # which arguably is a classification boundary (see the annotation below).
    DefaultValidationCase(
        "embeddings.consistency_min_anchors",
        "embedding_space_test",
        "anchors_min_testable",
        "SAME_SPACE",
    ),
    DefaultValidationCase(
        "embeddings.consistency_min_anchors",
        "embedding_space_test",
        "anchors_min_untestable",
        "SPACE_UNTESTABLE",
    ),
    DefaultValidationCase(
        "embeddings.consistency_max_anchors",
        "embedding_space_test",
        "anchors_max_at",
        "SAME_SPACE",
        repetitions=3,
    ),
    DefaultValidationCase(
        "embeddings.consistency_max_anchors",
        "embedding_space_test",
        "anchors_max_above",
        "SAME_SPACE",
        repetitions=3,
    ),
]

# Exactly the config keys Implementation Logic 1's table lists (real schema keys).
DOCUMENTED_DEFAULT_KEYS = frozenset(
    {
        "documents.replace_threshold",
        "chunk_match.min_similarity",
        "chunk_match.coverage",
        "chunk_match.split_contribution",
        "chunk_match.rechunk_exact_fraction",
        "chunking.oversized",
        "chunking.suspicious_short",
        "chunking.short",
        "metadata.mixed_type_share",
        "analysis.near_duplicate_threshold",
        "analysis.duplicate_threshold",
        "embeddings.consistency_same",
        "embeddings.consistency_different",
        "embeddings.consistency_min_anchors",
        "embeddings.consistency_max_anchors",
        "cluster.stable_jaccard",
        "cluster.stable_centroid_cosine",
    }
)

# The borderline-inclusion decision from the Inconsistency Patch, made visible in
# the suite's own output instead of only in prose.
BOUNDARY_OF_SCOPE_ANNOTATIONS = {
    "embeddings.consistency_min_anchors": "boundary-of-scope, included for completeness",
    "embeddings.consistency_max_anchors": "boundary-of-scope, included for completeness",
}


# --- fixture helpers -------------------------------------------------------


def _record_from_generator(row: dict) -> Record:
    record = Record(
        record_id=row["id"],
        document_id=row["document_id"],
        text=row["text"],
        chunk_index=row["chunk_index"],
    )
    compute_text_hashes(record)
    vector = np.asarray(row["embedding"], dtype=np.float64)
    record.embedding = EmbeddingInfo(
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
    record._embedding_digest_hex = row["id"]
    return record


def _recon(document_id: str, text: str, raw: str) -> ReconstructedDocument:
    return ReconstructedDocument(
        document_id=document_id,
        text=text,
        document_raw_hash=raw,
        document_content_hash=raw,
        reconstruction_confidence="HIGH",
        order_unknown=False,
    )


def _table(document_ids) -> DocumentTable:
    return DocumentTable(
        rows={
            document_id: DocumentRow(
                document_id=document_id, chunk_count=1, identity_confidence="HIGH"
            )
            for document_id in document_ids
        }
    )


def _chunk(
    record_id: str, text: str, *, document_id: str = "d", chunk_index: int = 0
) -> Record:
    record = Record(
        record_id=record_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
    )
    compute_text_hashes(record)
    return record


def _shared_doc_row() -> DocumentDiffRow:
    return DocumentDiffRow(
        document_id="d",
        status="UNCHANGED",
        metadata_changed=False,
        timestamp_inconsistent=False,
        reconstruction_confidence_v1="HIGH",
        reconstruction_confidence_v2="HIGH",
        similarity=None,
    )


def _match(v1, v2, cfg: Config, *, pass1=None):
    initial = ChunkDiffResult(
        pairs=[],
        unmatched_v1=list(v1),
        unmatched_v2=list(v2),
        pass1_match_counts=pass1 or {},
    )
    return match_chunks_full(
        v1,
        v2,
        initial_diff=initial,
        document_diff=[_shared_doc_row()],
        cfg=cfg.chunk_match,
        text_cfg=cfg.text,
    )


# --- per-kind classifiers --------------------------------------------------


def _content_change_class(cfg: Config, magnitude: str) -> str:
    if magnitude == "replaced":
        first = " ".join("a%d" % index for index in range(20))
        second = " ".join("b%d" % index for index in range(20))
    else:
        first = " ".join("c%d" % index for index in range(20))
        second = " ".join(["c%d" % index for index in range(19)] + ["z0"])
    rows = diff_documents(
        _table({"d"}),
        _table({"d"}),
        {"d": _recon("d", first, first)},
        {"d": _recon("d", second, second)},
        cfg=cfg.documents,
    )
    return rows[0].status


def _chunk_match_class(cfg: Config, magnitude: str) -> str:
    if magnitude in ("jaccard_above", "jaccard_below"):
        first_text = " ".join("a%d" % index for index in range(1, 19))
        v1 = [_chunk("a", first_text, chunk_index=0)]
        if magnitude == "jaccard_above":
            second_text = " ".join(
                ["a%d" % index for index in range(1, 17)]
                + ["b%d" % index for index in range(1, 6)]
            )
        else:
            second_text = " ".join(
                ["a%d" % index for index in range(1, 15)]
                + ["b%d" % index for index in range(1, 6)]
            )
        v2 = [_chunk("b", second_text, chunk_index=0)]
        result = _match(v1, v2, cfg)
        statuses = [status for _first, _second, status in result.pairs]
        if statuses and statuses[0] == "MODIFIED":
            return "MODIFIED"
        return "UNMATCHED"
    if magnitude in ("split_above", "split_below"):
        source = " ".join("w%d" % index for index in range(1, 9))
        v1 = [_chunk("s", source, chunk_index=0)]
        if magnitude == "split_above":
            first = " ".join("w%d" % index for index in range(1, 6))
            second = " ".join("w%d" % index for index in range(4, 9))
        else:
            first = " ".join("w%d" % index for index in range(1, 5))
            second = " ".join("w%d" % index for index in range(5, 9))
        v2 = [
            _chunk("c1", first, chunk_index=0),
            _chunk("c2", second, chunk_index=1),
        ]
        result = _match(v1, v2, cfg)
        if result.splits:
            statuses = [status for _first, _second, status in result.pairs]
            if statuses and all(status == "SPLIT_PART" for status in statuses):
                return "SPLIT_PART"
            return "OTHER"
        return "UNGROUPED"
    v1 = [_chunk("d%d" % index, "token%d alpha beta" % index, chunk_index=index) for index in range(5)]
    v2 = [_chunk("e%d" % index, "token%d alpha beta" % index, chunk_index=index) for index in range(5)]
    pass1 = {"d": 2} if magnitude == "rechunk_below" else {"d": 3}
    result = _match(v1, v2, cfg, pass1=pass1)
    return "RECHUNKED" if "d" in result.rechunked_document_ids else "NOT_RECHUNKED"


def _chunking_class(cfg: Config, magnitude: str) -> str:
    counts = {
        "very_short": 2,
        "short_boundary": 3,
        "suspicious_short": 19,
        "suspicious_boundary": 20,
        "oversized": 2049,
        "oversized_boundary": 2048,
    }
    record = Record(record_id="r", text="x")
    record.token_count = counts[magnitude]
    flags = classify_chunk_anomaly(record, cfg=cfg.chunking)
    return flags.size_category or "NONE"


def _metadata_class(cfg: Config, magnitude: str) -> str:
    if magnitude == "mixed":
        values = [1] * 94 + ["alpha"] * 6
    else:
        values = [1] * 96 + ["alpha"] * 4
    return infer_field_type(values, cfg=cfg.metadata)


def _duplicate_class(cfg: Config, magnitude: str, seed: int) -> str:
    near_rate = 1.0 if magnitude == "near_above" else 0.0
    semantic_rate = 1.0 if magnitude == "semantic_above" else 0.0
    pair = generate_duplicate_corpus(
        seed, 6, exact_rate=0.0, near_rate=near_rate, semantic_rate=semantic_rate
    )
    records = [_record_from_generator(row) for row in pair.v1_records]
    result = compute_all_duplicate_stages(
        records,
        cfg=cfg.duplicates,
        near_threshold=cfg.analysis.near_duplicate_threshold,
        semantic_threshold=cfg.analysis.duplicate_threshold,
        similarity_metric_known=True,
        seed=seed,
        text_cfg=cfg.text,
    )
    if magnitude.startswith("near"):
        return "GROUPED" if result.near_groups else "UNGROUPED"
    return "GROUPED" if result.semantic_groups else "UNGROUPED"


def _space_record(record_id: str, vector) -> Record:
    value = np.asarray(vector, dtype=np.float32)
    info = EmbeddingInfo(
        vector=value,
        dtype="float32",
        dimension=int(value.shape[0]),
        norm=float(np.linalg.norm(value)),
        validity="ok",
        model=None,
        provider=None,
        similarity_metric=None,
        normalized=None,
    )
    record = Record(record_id=record_id, embedding=info)
    record.raw_text_hash = "h"
    return record


def _space_pair(count: int, moved: int):
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0], dtype=np.float32)
    v1 = []
    v2 = []
    for index in range(count):
        v1.append(_space_record("r%d" % index, first))
        v2.append(_space_record("r%d" % index, second if index < moved else first))
    return v1, v2


def _embedding_space_class(cfg: Config, magnitude: str, seed: int) -> str:
    specs = {
        "same": (100, 0),
        "partial": (100, 28),
        "different": (100, 100),
        "same_boundary": (100, 5),
        "sameness_below": (100, 6),
        "different_boundary": (100, 95),
        "difference_below": (100, 94),
        "anchors_min_testable": (30, 0),
        "anchors_min_untestable": (29, 0),
        "anchors_max_at": (5000, 0),
        "anchors_max_above": (5001, 0),
    }
    count, moved = specs[magnitude]
    v1, v2 = _space_pair(count, moved)
    result = run_space_consistency_test(
        v1, v2, comparison_level=_LEVEL, cfg=cfg.embeddings, seed=seed
    )
    return result.state


def _cluster_class(cfg: Config, magnitude: str) -> str:
    if magnitude in ("jaccard_above", "jaccard_below"):
        first_ids = ["r%d" % index for index in range(50)]
        if magnitude == "jaccard_above":
            second_ids = ["r%d" % index for index in range(8, 58)]
        else:
            second_ids = ["r%d" % index for index in range(9, 59)]
        v1 = ClusteringResult([ClusterAssignment(i, 0) for i in first_ids], 0.0, 2)
        v2 = ClusteringResult([ClusterAssignment(i, 0) for i in second_ids], 0.0, 2)
        vectors = {i: np.array([1.0, 0.0]) for i in set(first_ids) | set(second_ids)}
        transitions = classify_cluster_transitions(
            v1, v2, vectors, vectors, cfg=cfg.cluster, cosine_space=False
        )
        return transitions[0].kind
    cosine = 0.96 if magnitude == "centroid_above" else 0.92
    sine = math.sqrt(1.0 - cosine * cosine)
    first_ids = ["m0", "m1"]
    second_ids = ["n0", "n1"]
    v1 = ClusteringResult([ClusterAssignment(i, 0) for i in first_ids], 0.0, 2)
    v2 = ClusteringResult([ClusterAssignment(i, 0) for i in second_ids], 0.0, 2)
    first_vectors = {i: np.array([1.0, 0.0]) for i in first_ids}
    second_vectors = {i: np.array([cosine, sine]) for i in second_ids}
    transitions = classify_cluster_transitions(
        v1, v2, first_vectors, second_vectors, cfg=cfg.cluster, cosine_space=False
    )
    return transitions[0].kind


_DISPATCH: dict[str, Callable[[Config, str, int], str]] = {
    "content_change": lambda cfg, magnitude, _seed: _content_change_class(cfg, magnitude),
    "chunking_anomaly": lambda cfg, magnitude, _seed: _chunking_class(cfg, magnitude),
    "metadata_drift": lambda cfg, magnitude, _seed: _metadata_class(cfg, magnitude),
    "duplicate_rate": _duplicate_class,
    "embedding_space_test": _embedding_space_class,
    "chunk_match": lambda cfg, magnitude, _seed: _chunk_match_class(cfg, magnitude),
    "cluster_stability": lambda cfg, magnitude, _seed: _cluster_class(cfg, magnitude),
}


def run_default_validation_case(case: DefaultValidationCase, *, cfg: Config) -> float:
    """Fraction of seeded trials that produce the expected class."""

    runner = _DISPATCH[case.fixture_kind]
    matches = 0
    for seed in range(case.repetitions):
        if runner(cfg, case.magnitude_param, seed) == case.expected_class:
            matches += 1
    return matches / case.repetitions if case.repetitions else 0.0


# --- control validation ----------------------------------------------------


def _control_rows(v1, v2, cfg: Config):
    def recon_map(records):
        grouped: dict[str, list[str]] = {}
        for record in records:
            grouped.setdefault(record.document_id, []).append(record.text or "")
        return {
            document_id: _recon(document_id, " ".join(texts), " ".join(texts))
            for document_id, texts in grouped.items()
        }

    first = _table({record.document_id for record in v1})
    second = _table({record.document_id for record in v2})
    return diff_documents(first, second, recon_map(v1), recon_map(v2), cfg=cfg.documents)


def _content_change_rate(rows) -> float:
    if not rows:
        return 0.0
    changed = sum(1 for row in rows if row.status != "UNCHANGED")
    return changed / len(rows)


def _control_components(cfg: Config, seed: int):
    pair = generate_corpus_pair(seed, 40, change_profile="none")
    v1 = [_record_from_generator(row) for row in pair.v1_records]
    v2 = [_record_from_generator(row) for row in pair.v2_records]
    rows = _control_rows(v1, v2, cfg)
    space = run_space_consistency_test(
        v1, v2, comparison_level=_LEVEL, cfg=cfg.embeddings, seed=seed
    )
    return rows, space


def run_control_validation(*, cfg: Config, repetitions: int = 30) -> float:
    """Control pairs: zero content change AND SAME_SPACE, every trial."""

    matches = 0
    for seed in range(repetitions):
        rows, space = _control_components(cfg, seed)
        if _content_change_rate(rows) == 0.0 and space.state == "SAME_SPACE":
            matches += 1
    return matches / repetitions if repetitions else 0.0


def run_control_validation_space(*, cfg: Config, repetitions: int = 30) -> float:
    """The space-test half of the control check, reported separately."""

    matches = 0
    for seed in range(repetitions):
        _rows, space = _control_components(cfg, seed)
        if space.state == "SAME_SPACE":
            matches += 1
    return matches / repetitions if repetitions else 0.0


# --- tests -----------------------------------------------------------------


def test_default_validation_suite_covers_documented_keys() -> None:
    assert {case.config_key for case in DEFAULT_VALIDATION_SUITE} == DOCUMENTED_DEFAULT_KEYS


@pytest.mark.parametrize(
    "case",
    DEFAULT_VALIDATION_SUITE,
    ids=[case.config_key + "::" + case.magnitude_param for case in DEFAULT_VALIDATION_SUITE],
)
def test_default_validation(case: DefaultValidationCase) -> None:
    rate = run_default_validation_case(case, cfg=default_config())
    assert rate >= 0.95, (
        "default %s (%s, magnitude=%s) produced expected class %s in only %.3f of "
        "%d seeded trials (required >= 0.95)"
        % (
            case.config_key,
            case.fixture_kind,
            case.magnitude_param,
            case.expected_class,
            rate,
            case.repetitions,
        )
    )


def test_control_validation_zero_content_change_always() -> None:
    cfg = default_config()
    for seed in range(30):
        rows, _space = _control_components(cfg, seed)
        assert _content_change_rate(rows) == 0.0
    assert run_control_validation(cfg=cfg) == 1.0


def test_control_validation_same_space_always() -> None:
    assert run_control_validation_space(cfg=default_config()) == 1.0


def test_borderline_anchor_counts_documented() -> None:
    suite_keys = {case.config_key for case in DEFAULT_VALIDATION_SUITE}
    assert set(BOUNDARY_OF_SCOPE_ANNOTATIONS) <= suite_keys
    for annotation in BOUNDARY_OF_SCOPE_ANNOTATIONS.values():
        assert "boundary-of-scope" in annotation
