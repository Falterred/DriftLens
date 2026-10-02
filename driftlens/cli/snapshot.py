"""The driftlens snapshot command (Step 36)."""

from __future__ import annotations

import argparse
from pathlib import Path

from driftlens.adapters.detect import InputFormat, detect_format
from driftlens.config.loader import effective_side_options, load_config
from driftlens.errors import DiagnosticSink, InvalidInputError
from driftlens.schema.snapshot import build_snapshot, write_snapshot
from driftlens.security.limits import RuntimeLimits

_GRANULARITY_CHOICES = ["AUTO", "CHUNK", "DOCUMENT", "auto", "chunk", "document"]


def build_arg_parser_snapshot_subcommand(subparsers) -> None:
    """Register the snapshot subcommand and all of its flags."""
    from driftlens.cli.compare import _SIDE_OPTION_FLAGS

    snapshot = subparsers.add_parser("snapshot", help="build a snapshot from a source")
    snapshot.add_argument("source")
    snapshot.add_argument("--out", required=True)
    snapshot.add_argument("--config")
    snapshot.add_argument("--no-text", dest="no_text", action="store_true")
    snapshot.add_argument(
        "--granularity", dest="schema.granularity", choices=_GRANULARITY_CHOICES
    )
    for flag in _SIDE_OPTION_FLAGS:
        snapshot.add_argument(f"--{flag.replace('_', '-')}", dest=f"schema.{flag}")
    snapshot.add_argument(
        "--allow-lossy-vectors", dest="schema.allow_lossy_vectors", action="store_true"
    )
    snapshot.add_argument("--embedding-model", dest="embeddings.model")
    snapshot.add_argument(
        "--embedding-similarity-metric", dest="embeddings.similarity_metric"
    )
    snapshot.add_argument(
        "--allow-unsafe-pickle", dest="allow_unsafe_pickle", action="store_true"
    )
    snapshot.add_argument("--workers", dest="analysis.workers", type=int)
    snapshot.add_argument("--sample-size", dest="analysis.sample_size", type=int)
    snapshot.add_argument("--seed", dest="analysis.seed", type=int)
    snapshot.add_argument("--max-records", dest="limits.max_records", type=int)
    snapshot.add_argument("--max-dimension", dest="limits.max_dimension", type=int)
    snapshot.add_argument("--max-text-length", dest="limits.max_text_length", type=int)
    snapshot.add_argument("--max-sample-size", dest="limits.max_sample_size", type=int)
    snapshot.add_argument("--max-workers", dest="limits.max_workers")
    snapshot.add_argument("--max-input-bytes", dest="limits.max_input_bytes", type=int)


def run_snapshot(args: argparse.Namespace) -> int:
    """Build a snapshot directory from a source file/directory."""
    from driftlens.cli.compare import (
        _cli_dict,
        _granularity_override,
        _load_input,
        _mapping_summary,
        check_dependencies,
    )

    cli = _cli_dict(args)
    config_path = Path(args.config) if getattr(args, "config", None) else None
    cfg = load_config(config_path, cli, Path.cwd())
    sink = DiagnosticSink()
    limits = RuntimeLimits(cfg.limits)
    side = effective_side_options(cfg, "a", cli)

    source = Path(args.source)
    format_: InputFormat = detect_format(source, forced=side.format, limits=limits)
    if format_ == "snapshot":
        raise InvalidInputError(
            location=str(source),
            message="input is already a snapshot directory",
        )
    check_dependencies(format_, format_, clusters_mode="off")

    loaded, unsafe = _load_input(
        format_,
        source,
        side,
        limits=limits,
        allow_unsafe_pickle=getattr(args, "allow_unsafe_pickle", False),
    )
    snapshot = build_snapshot(
        loaded.records,
        side=side,
        cfg=cfg,
        source_type=format_,
        sink=sink,
        limits=limits,
        granularity_override=_granularity_override(side),
        field_mapping_summary=_mapping_summary(loaded.field_mapping),
        unsafe_pickle=unsafe,
    )
    write_snapshot(snapshot, Path(args.out), no_text=bool(getattr(args, "no_text", False)))
    print(
        f"Snapshot written to {args.out} "
        f"({snapshot.manifest.document_count} documents, "
        f"{snapshot.manifest.chunk_count} chunks)"
    )
    return 0
