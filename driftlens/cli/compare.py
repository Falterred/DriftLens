"""The driftlens compare pipeline (Step 33)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driftlens import __version__
from driftlens.adapters.chroma import load_chroma
from driftlens.adapters.csv import load_csv
from driftlens.adapters.detect import InputFormat, detect_format
from driftlens.adapters.faiss import load_faiss
from driftlens.adapters.json import load_json
from driftlens.adapters.jsonl import LoadedInput, load_jsonl
from driftlens.adapters.parquet import load_parquet
from driftlens.analysis.chunking import compare_chunking
from driftlens.analysis.compatibility import (
    apply_strict_compat,
    determine_compatibility_state,
)
from driftlens.analysis.content import (
    check_doc_identity_mismatch,
    compare_content,
    determine_comparison_level,
)
from driftlens.analysis.duplicates import compare_duplicates
from driftlens.analysis.embeddings import analyze_embeddings
from driftlens.analysis.health import Metric, build_health_report
from driftlens.analysis.metadata import compare_metadata
from driftlens.analysis.mmd import compare_mmd
from driftlens.config.loader import effective_side_options, load_config
from driftlens.config.schema import Config, SideOptions
from driftlens.errors import (
    DependencyMissingError,
    DiagnosticSink,
    DriftLensError,
    FatalErrors,
    InvalidInputError,
    UnsupportedFormatError,
    guarded,
)
from driftlens.evaluation.ground_truth import ResolvedCase, resolve_eval_set
from driftlens.evaluation.metrics import first_relevant_rank
from driftlens.evaluation.retrieval import run_retrieval_comparison
from driftlens.evaluation.retrievers import BM25Index, DenseProvidedIndex
from driftlens.identity.snapshot_id import (
    EmbeddingDescriptor,
    build_descriptor,
    compute_snapshot_id,
)
from driftlens.reports.diff_export import (
    BreakdownRow,
    compute_breakdown,
    top_n_changed_documents,
    write_breakdown,
    write_chunks_diff,
    write_documents_diff,
    write_duplicates_diff,
    write_retrieval_cases_diff,
)
from driftlens.reports.json import (
    BorderlineCounter,
    build_report_json,
    write_report_json,
)
from driftlens.reports.terminal import (
    render_ci_report,
    render_quiet_report,
    render_terminal_report,
)
from driftlens.schema.eval_set import load_eval_set
from driftlens.schema.manifest import (
    SNAPSHOT_SCHEMA_VERSION,
    FieldMappingSummary,
)
from driftlens.schema.snapshot import Snapshot, build_snapshot
from driftlens.security.limits import RuntimeLimits
from driftlens.security.redaction import apply_redaction_to_metric
from driftlens.text.normalize import normalization_version
from driftlens.text.tokenizer import tokenizer_id

_UNIMPLEMENTED_FLAGS = ("baseline",)

_SIDE_OPTION_FLAGS = (
    "id_field",
    "chunk_id_field",
    "text_field",
    "embedding_field",
    "model_field",
    "sidecar",
    "timestamp_unit",
    "chroma_collection",
    "format",
    "embedding_format",
    "embedding_prefix",
    "model",
    "similarity_metric",
)


@dataclass
class CompareContext:
    cfg: Config
    side_a: SideOptions
    side_b: SideOptions
    sink: DiagnosticSink
    limits: RuntimeLimits
    borderline: BorderlineCounter


def build_arg_parser() -> argparse.ArgumentParser:
    """The driftlens argument parser with compare/snapshot/baseline subcommands."""
    parser = argparse.ArgumentParser(prog="driftlens")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    compare = subparsers.add_parser("compare", help="compare two snapshots")
    compare.add_argument("input_a")
    compare.add_argument("input_b")
    compare.add_argument("--config")
    compare.add_argument("--quiet", action="store_true")
    compare.add_argument("--json-out", dest="json_out")
    compare.add_argument("--report-out", dest="report_out")
    compare.add_argument("--diff-out", dest="diff_out")
    compare.add_argument("--breakdown-by", dest="breakdown_by", action="append")
    compare.add_argument("--eval-set", dest="eval_set")
    compare.add_argument("--baseline", dest="baseline")
    compare.add_argument("--redact-ids", dest="redact_ids", action="store_true")
    compare.add_argument("--diagnostic", dest="diagnostic", action="store_true")
    compare.add_argument("--top-n", dest="privacy.top_n", type=int)
    compare.add_argument(
        "--force-cross-model", dest="force_cross_model", action="store_true"
    )
    compare.add_argument("--verify-full", dest="verify_full", action="store_true")
    compare.add_argument("--expect-a", dest="expect_a")
    compare.add_argument("--expect-b", dest="expect_b")
    compare.add_argument(
        "--allow-unsafe-pickle", dest="allow_unsafe_pickle", action="store_true"
    )
    compare.add_argument("--ci", action="store_true")
    compare.add_argument("--strict-compat", dest="strict_compat", action="store_true")
    compare.add_argument("--clusters", dest="cluster.mode", choices=["auto", "on", "off"])
    compare.add_argument(
        "--granularity",
        dest="schema.granularity",
        choices=["AUTO", "CHUNK", "DOCUMENT", "auto", "chunk", "document"],
    )
    for flag in _SIDE_OPTION_FLAGS:
        compare.add_argument(f"--{flag.replace('_', '-')}", dest=f"schema.{flag}")
    compare.add_argument(
        "--allow-lossy-vectors", dest="schema.allow_lossy_vectors", action="store_true"
    )
    compare.add_argument("--embedding-model", dest="embeddings.model")
    compare.add_argument(
        "--embedding-similarity-metric", dest="embeddings.similarity_metric"
    )
    compare.add_argument("--workers", dest="analysis.workers", type=int)
    compare.add_argument("--sample-size", dest="analysis.sample_size", type=int)
    compare.add_argument("--seed", dest="analysis.seed", type=int)
    compare.add_argument("--bandwidth", dest="analysis.bandwidth", type=float)
    compare.add_argument(
        "--embedding-max-tokens", dest="chunking.embedding_max_tokens", type=int
    )
    compare.add_argument("--max-records", dest="limits.max_records", type=int)
    compare.add_argument("--max-dimension", dest="limits.max_dimension", type=int)
    compare.add_argument("--max-text-length", dest="limits.max_text_length", type=int)
    compare.add_argument("--max-sample-size", dest="limits.max_sample_size", type=int)
    compare.add_argument("--max-workers", dest="limits.max_workers")
    compare.add_argument("--max-input-bytes", dest="limits.max_input_bytes", type=int)

    from driftlens.cli.snapshot import build_arg_parser_snapshot_subcommand

    build_arg_parser_snapshot_subcommand(subparsers)

    baseline = subparsers.add_parser("baseline", help="baseline management")
    baseline.add_argument("input", nargs="?")
    baseline.add_argument("--out")

    return parser


def _probe(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_dependencies(
    format_a: InputFormat, format_b: InputFormat, *, clusters_mode: str
) -> None:
    """Raise DependencyMissingError when an optional extra is required but absent."""
    formats = {format_a, format_b}
    if "chroma" in formats and not _probe("chromadb"):
        raise DependencyMissingError(
            message="Chroma support requires the [chroma] extra",
            hint="pip install 'driftlens[chroma]'",
        )
    if "faiss" in formats and not _probe("faiss"):
        raise DependencyMissingError(
            message="FAISS support requires the [faiss] extra",
            hint="pip install 'driftlens[faiss]'",
        )
    if clusters_mode == "on" and not (_probe("sklearn") and _probe("hdbscan")):
        raise DependencyMissingError(
            message="cluster analysis requires the [cluster] extra",
            hint="pip install 'driftlens[cluster]'",
        )


def _cli_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if "." in key and value is not None
    }


def _mapping_summary(mapping) -> FieldMappingSummary:
    roles: dict[str, str] = {}
    for role in (
        "text",
        "document_id",
        "chunk_id",
        "embedding",
        "chunk_index",
        "model",
        "provider",
        "dtype",
        "metric",
    ):
        value = getattr(mapping, role, None)
        if value is not None:
            roles[role] = value
    return FieldMappingSummary(roles=roles, ignored=dict(mapping.ignored))


def _load_input(
    format_: InputFormat,
    path: Path,
    side: SideOptions,
    *,
    limits: RuntimeLimits,
    allow_unsafe_pickle: bool,
    verify_full: bool = False,
) -> tuple[LoadedInput, bool]:
    if format_ == "jsonl":
        return load_jsonl(path, side=side, limits=limits), False
    if format_ == "json":
        return load_json(path, side=side, limits=limits), False
    if format_ == "csv":
        return load_csv(path, side=side, limits=limits), False
    if format_ == "parquet":
        return load_parquet(path, side=side, limits=limits), False
    if format_ == "faiss":
        result = load_faiss(
            path,
            side=side,
            limits=limits,
            allow_lossy_vectors=side.allow_lossy_vectors,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )
        return result, bool(getattr(result, "unsafe_pickle", False))
    if format_ == "chroma":
        return load_chroma(path, side=side, limits=limits), False
    if format_ == "snapshot":
        from driftlens.adapters.snapshot import load_snapshot_as_input

        return load_snapshot_as_input(path, limits=limits, verify_full=verify_full), False
    raise UnsupportedFormatError(message=f"unsupported format {format_!r}")


def _granularity_override(side: SideOptions) -> str | None:
    value = (side.granularity or "AUTO").upper()
    return None if value == "AUTO" else value


def _build_side(
    loaded: LoadedInput,
    side: SideOptions,
    *,
    cfg: Config,
    sink: DiagnosticSink,
    limits: RuntimeLimits,
    unsafe_pickle: bool,
) -> Snapshot:
    return build_snapshot(
        loaded.records,
        side=side,
        cfg=cfg,
        source_type=loaded.source_type,
        sink=sink,
        limits=limits,
        granularity_override=_granularity_override(side),
        field_mapping_summary=_mapping_summary(loaded.field_mapping),
        unsafe_pickle=unsafe_pickle,
    )


def _verify_full(snapshot: Snapshot) -> None:
    manifest = snapshot.manifest
    descriptor_bytes = build_descriptor(
        SNAPSHOT_SCHEMA_VERSION,
        normalization_version(),
        tokenizer_id(),
        manifest.granularity,
        EmbeddingDescriptor(
            model=manifest.embedding_model,
            provider=manifest.embedding_provider,
            dtype=manifest.embedding_dtype,
            dimension=manifest.embedding_dimension,
            similarity_metric=manifest.similarity_metric,
            normalized=manifest.embedding_normalized,
        ),
    )
    recomputed, _content = compute_snapshot_id(snapshot.records, descriptor_bytes)
    if recomputed != manifest.snapshot_id:
        raise InvalidInputError(message="--verify-full snapshot id mismatch")


def _reject_unimplemented(args: argparse.Namespace) -> None:
    for flag_name in _UNIMPLEMENTED_FLAGS:
        if getattr(args, flag_name, None):
            dashed = "--" + flag_name.replace("_", "-")
            raise InvalidInputError(
                message=f"{dashed} is not available in this release"
            )


def _text_by_document(snapshot: Snapshot) -> dict[str, str]:
    """First non-empty text per document, for --diagnostic excerpts."""
    texts: dict[str, str] = {}
    for record in snapshot.records:
        text = record.text if record.text is not None else record.normalized_text
        if text is None:
            continue
        texts.setdefault(record.document_id, text)
    return texts


def _lexical_runnable(snapshot: Snapshot) -> bool:
    return any(
        record.text is not None or record.normalized_text is not None
        for record in snapshot.records
    )


def _safe_field(field: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in field)


def _breakdown_to_json(row: BreakdownRow) -> dict[str, Any]:
    return {
        "segment_value": row.segment_value,
        "is_other": row.is_other,
        "document_count": row.document_count,
        "status_counts": dict(row.status_counts),
        "chunk_profile": row.chunk_profile,
        "mmd": row.mmd,
    }


def _retrieval_rank_maps(
    mode: str,
    v1: Snapshot,
    v2: Snapshot,
    resolved_cases: list[ResolvedCase],
    *,
    cfg: Config,
    v1_summary: Any,
    v2_summary: Any,
) -> tuple[dict[str, int | None], dict[str, int | None]]:
    """First-relevant chunk rank per case for the exported retrieval diff.

    ResolvedCase carries no ranks and run_retrieval_comparison does not return
    them, so the CLI recomputes the primary mode's ranking here for the
    --diff-out artifact only.
    """
    if mode == "lexical":
        index_v1 = BM25Index(
            v1.records, cfg=cfg.retrieval.bm25, casefold=cfg.text.casefold
        )
        index_v2 = BM25Index(
            v2.records, cfg=cfg.retrieval.bm25, casefold=cfg.text.casefold
        )
    elif mode == "dense":
        index_v1 = DenseProvidedIndex(v1.records, metric=v1_summary.similarity_metric)
        index_v2 = DenseProvidedIndex(v2.records, metric=v2_summary.similarity_metric)
    else:
        return {}, {}
    kmax = max(cfg.retrieval.k)
    rank_v1: dict[str, int | None] = {}
    rank_v2: dict[str, int | None] = {}
    for case in resolved_cases:
        if case.state not in ("VALID", "PARTIAL"):
            continue
        if mode == "lexical":
            query_v1 = query_v2 = case.case.query
        else:
            embeddings = case.case.query_embeddings
            if embeddings is None or embeddings.v1 is None or embeddings.v2 is None:
                continue
            query_v1, query_v2 = embeddings.v1, embeddings.v2
        rank_v1[case.case.query_id] = first_relevant_rank(
            index_v1.search(query_v1, top_k=kmax),
            case.present_document_ids,
            kmax,
        )
        rank_v2[case.case.query_id] = first_relevant_rank(
            index_v2.search(query_v2, top_k=kmax),
            case.present_document_ids,
            kmax,
        )
    return rank_v1, rank_v2


def run_compare(args: argparse.Namespace) -> int:
    """Run the full compare pipeline and return the process exit code."""
    # Stage 0 - early flag validation.
    if getattr(args, "allow_unsafe_pickle", False) and getattr(args, "ci", False):
        raise InvalidInputError(
            message="--allow-unsafe-pickle is rejected under --ci"
        )

    # Stage 1 - config.
    cli = _cli_dict(args)
    config_path = Path(args.config) if getattr(args, "config", None) else None
    cfg = load_config(config_path, cli, Path.cwd())
    _reject_unimplemented(args)
    sink = DiagnosticSink()
    limits = RuntimeLimits(cfg.limits)
    borderline = BorderlineCounter()
    side_a = effective_side_options(cfg, "a", cli)
    side_b = effective_side_options(cfg, "b", cli)

    # Stage 2 - format detection (both sides collected before raising).
    errors: list[DriftLensError] = []
    format_a: InputFormat | None = None
    format_b: InputFormat | None = None
    try:
        format_a = detect_format(Path(args.input_a), forced=side_a.format, limits=limits)
    except DriftLensError as exc:
        errors.append(exc)
    try:
        format_b = detect_format(Path(args.input_b), forced=side_b.format, limits=limits)
    except DriftLensError as exc:
        errors.append(exc)
    if errors:
        raise FatalErrors(errors)

    # Stage 3 - dependency check.
    clusters_mode = (
        cfg.cluster.mode
        if getattr(args, "cluster.mode", None) is None
        else getattr(args, "cluster.mode")
    )
    check_dependencies(format_a, format_b, clusters_mode=clusters_mode)

    # Stage 4 - load and validation (sequential; one side fully completes first).
    # A written snapshot directory already carries a Snapshot; it is passed
    # through unchanged instead of rebuilding identity.
    loaded_a, unsafe_a = _load_input(
        format_a,
        Path(args.input_a),
        side_a,
        limits=limits,
        allow_unsafe_pickle=getattr(args, "allow_unsafe_pickle", False),
        verify_full=getattr(args, "verify_full", False),
    )
    v1_snapshot = getattr(loaded_a, "_prebuilt_snapshot", None)
    if v1_snapshot is None:
        v1_snapshot = _build_side(
            loaded_a, side_a, cfg=cfg, sink=sink, limits=limits, unsafe_pickle=unsafe_a
        )

    loaded_b, unsafe_b = _load_input(
        format_b,
        Path(args.input_b),
        side_b,
        limits=limits,
        allow_unsafe_pickle=getattr(args, "allow_unsafe_pickle", False),
        verify_full=getattr(args, "verify_full", False),
    )
    v2_snapshot = getattr(loaded_b, "_prebuilt_snapshot", None)
    if v2_snapshot is None:
        v2_snapshot = _build_side(
            loaded_b, side_b, cfg=cfg, sink=sink, limits=limits, unsafe_pickle=unsafe_b
        )

    expect_a = getattr(args, "expect_a", None)
    if expect_a and expect_a != v1_snapshot.manifest.snapshot_id:
        raise InvalidInputError(
            message="--expect-a mismatch",
            location=str(args.input_a),
        )
    expect_b = getattr(args, "expect_b", None)
    if expect_b and expect_b != v2_snapshot.manifest.snapshot_id:
        raise InvalidInputError(
            message="--expect-b mismatch",
            location=str(args.input_b),
        )
    if getattr(args, "verify_full", False):
        _verify_full(v1_snapshot)
        _verify_full(v2_snapshot)

    # Stage 5 - compatibility.
    comparison_level = determine_comparison_level(
        v1_snapshot.manifest, v2_snapshot.manifest, v1_snapshot.records, v2_snapshot.records
    )
    check_doc_identity_mismatch(v1_snapshot.records, v2_snapshot.records, sink)
    v1_summary = analyze_embeddings(
        v1_snapshot.records,
        file_descriptors=loaded_a.file_descriptors,
        side=side_a,
        cfg=cfg.embeddings,
        sink=sink,
    )
    v2_summary = analyze_embeddings(
        v2_snapshot.records,
        file_descriptors=loaded_b.file_descriptors,
        side=side_b,
        cfg=cfg.embeddings,
        sink=sink,
    )
    compat = determine_compatibility_state(
        v1_summary,
        v2_summary,
        v1_snapshot.records,
        v2_snapshot.records,
        v1_dimension=v1_snapshot.manifest.embedding_dimension,
        v2_dimension=v2_snapshot.manifest.embedding_dimension,
        comparison_level=comparison_level,
        cfg=cfg,
        sink=sink,
    )
    apply_strict_compat(compat, strict_compat=getattr(args, "strict_compat", False))

    # Stage 6 - analysis.
    with guarded("analysis:content"):
        _document_rows, chunk_result, content_metrics = compare_content(
            v1_snapshot, v2_snapshot, cfg=cfg, sink=sink
        )
    with guarded("analysis:chunking"):
        chunking_metrics = compare_chunking(v1_snapshot, v2_snapshot, cfg=cfg)

    common_record_ids = {
        first.record_id
        for first, second, _status in chunk_result.pairs
        if first is not None and second is not None
    }
    if not common_record_ids:
        common_record_ids = {record.record_id for record in v1_snapshot.records} & {
            record.record_id for record in v2_snapshot.records
        }

    with guarded("analysis:metadata"):
        _field_drifts, metadata_metrics = compare_metadata(
            v1_snapshot,
            v2_snapshot,
            cfg=cfg.metadata,
            drift_cfg=cfg.drift,
            common_record_ids=common_record_ids,
        )
    with guarded("analysis:duplication"):
        _dups_v1, _dups_v2, duplication_metrics = compare_duplicates(
            v1_snapshot,
            v2_snapshot,
            cfg=cfg.duplicates,
            drift_cfg=cfg.drift,
            seed=cfg.analysis.seed,
            text_cfg=cfg.text,
            near_threshold=cfg.analysis.near_duplicate_threshold,
            semantic_threshold=cfg.analysis.duplicate_threshold,
            similarity_metric_known=(
                v1_snapshot.manifest.similarity_metric is not None
                and v2_snapshot.manifest.similarity_metric is not None
            ),
        )
    with guarded("analysis:mmd"):
        mmd_metric = compare_mmd(
            v1_snapshot,
            v2_snapshot,
            compat=compat,
            v1_summary=v1_summary,
            v2_summary=v2_summary,
            cfg=cfg,
            sink=sink,
        )

    eval_set = None
    resolved_cases: list[ResolvedCase] = []
    retrieval_modes: list[str] = []
    retrieval_metrics: list[Metric] = []
    if getattr(args, "eval_set", None):
        eval_set = load_eval_set(Path(args.eval_set), limits=limits)
        with guarded("analysis:retrieval"):
            retrieval_result = run_retrieval_comparison(
                v1_snapshot,
                v2_snapshot,
                eval_set,
                cfg=cfg,
                compat=compat,
                v1_summary=v1_summary,
                v2_summary=v2_summary,
                comparison_level=comparison_level,
                sink=sink,
            )
        for mode_result in retrieval_result.modes:
            label = mode_result.retriever or "retrieval"
            if mode_result.retriever is not None:
                retrieval_modes.append(mode_result.retriever)
            for name, metric in mode_result.metrics.items():
                metric.name = f"{label}:{name}"
                retrieval_metrics.append(metric)
        resolved_cases = resolve_eval_set(
            eval_set,
            v1_documents=v1_snapshot.documents,
            v2_documents=v2_snapshot.documents,
            v1_records_by_id={
                record.record_id: record for record in v1_snapshot.records
            },
            v2_records_by_id={
                record.record_id: record for record in v2_snapshot.records
            },
            comparison_level=comparison_level,
        )

    dimensions = {
        "content": content_metrics,
        "chunking": chunking_metrics,
        "metadata": metadata_metrics,
        "embedding": [mmd_metric],
        "semantic": [],
        "duplication": duplication_metrics,
        "retrieval": retrieval_metrics,
    }
    health = build_health_report(dimensions, cfg=cfg)

    if getattr(args, "redact_ids", False):
        redaction_key = v2_snapshot.manifest.snapshot_id
        for dimension in health.dimensions.values():
            dimension.metrics = [
                apply_redaction_to_metric(
                    metric, snapshot_id=redaction_key, redact=True
                )
                for metric in dimension.metrics
            ]

    breakdown_rows: dict[str, list[BreakdownRow]] = {
        field: compute_breakdown(
            field,
            v1=v1_snapshot,
            v2=v2_snapshot,
            document_diff=_document_rows,
            cfg=cfg,
        )
        for field in (getattr(args, "breakdown_by", None) or [])
    }

    top_changed = top_n_changed_documents(_document_rows, n=cfg.privacy.top_n)

    diagnostic_active = bool(getattr(args, "diagnostic", False))
    diff_artifact_paths: dict[str, str] = {}
    if getattr(args, "diff_out", None):
        diff_dir = Path(args.diff_out)
        documents_path = diff_dir / "documents.parquet"
        chunks_path = diff_dir / "chunks.parquet"
        write_documents_diff(
            _document_rows,
            path=documents_path,
            diagnostic=diagnostic_active,
            v1_text_by_doc=_text_by_document(v1_snapshot),
            v2_text_by_doc=_text_by_document(v2_snapshot),
        )
        write_chunks_diff(chunk_result, path=chunks_path, diagnostic=diagnostic_active)
        diff_artifact_paths["documents"] = str(documents_path)
        diff_artifact_paths["chunks"] = str(chunks_path)
        write_duplicates_diff(_dups_v1, path=diff_dir, side="v1")
        write_duplicates_diff(_dups_v2, path=diff_dir, side="v2")
        diff_artifact_paths["duplicates_v1"] = str(diff_dir / "duplicates_v1.parquet")
        diff_artifact_paths["duplicates_v2"] = str(diff_dir / "duplicates_v2.parquet")
        if eval_set is not None:
            rank_v1: dict[str, int | None] = {}
            rank_v2: dict[str, int | None] = {}
            if retrieval_modes:
                rank_v1, rank_v2 = _retrieval_rank_maps(
                    retrieval_modes[0],
                    v1_snapshot,
                    v2_snapshot,
                    resolved_cases,
                    cfg=cfg,
                    v1_summary=v1_summary,
                    v2_summary=v2_summary,
                )
            retrieval_path = diff_dir / "retrieval_cases.parquet"
            write_retrieval_cases_diff(
                resolved_cases,
                path=retrieval_path,
                rank_v1=rank_v1,
                rank_v2=rank_v2,
            )
            diff_artifact_paths["retrieval_cases"] = str(retrieval_path)
        for field, rows in breakdown_rows.items():
            breakdown_path = diff_dir / ("breakdown_" + _safe_field(field) + ".parquet")
            write_breakdown(rows, path=breakdown_path)
            diff_artifact_paths["breakdown_" + field] = str(breakdown_path)

    # Stage 7 - report rendering.
    exit_code_ci = 0
    if diagnostic_active and not getattr(args, "ci", False):
        print("DIAGNOSTIC MODE: document text excerpts are included in --diff-out")
    if getattr(args, "ci", False):
        text, exit_code_ci = render_ci_report(health)
    elif getattr(args, "quiet", False):
        text = render_quiet_report(health)
    else:
        text = render_terminal_report(
            health=health,
            v1_manifest=v1_snapshot.manifest,
            v2_manifest=v2_snapshot.manifest,
            sink=sink,
            determinism_level="TOLERANCE",
            isatty=sys.stdout.isatty(),
            seed=cfg.analysis.seed,
            top_changed=top_changed,
        )
    print(text)

    if getattr(args, "json_out", None):
        breakdowns_json = {
            field: [_breakdown_to_json(row) for row in rows]
            for field, rows in breakdown_rows.items()
        }
        report = build_report_json(
            health=health,
            v1_manifest=v1_snapshot.manifest,
            v2_manifest=v2_snapshot.manifest,
            sink=sink,
            cfg=cfg,
            borderline=borderline,
            diff_artifact_paths=diff_artifact_paths,
            driftlens_version=__version__,
            breakdowns=breakdowns_json or None,
        )
        write_report_json(report, Path(args.json_out))

    if getattr(args, "report_out", None):
        raise InvalidInputError(
            message="--report-out (HTML report) is not available until release 0.2"
        )

    return 0 if not getattr(args, "ci", False) else exit_code_ci
