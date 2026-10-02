"""The driftlens baseline build command (Step 48)."""

from __future__ import annotations

import argparse
from pathlib import Path

from driftlens.analysis.baselines import build_baseline, write_baseline
from driftlens.config.loader import effective_side_options, load_config
from driftlens.config.schema import THRESHOLD_KEYS
from driftlens.errors import DiagnosticSink, InvalidInputError
from driftlens.schema.eval_set import load_eval_set
from driftlens.schema.snapshot import read_snapshot
from driftlens.security.limits import RuntimeLimits

_GRANULARITY_CHOICES = ["AUTO", "CHUNK", "DOCUMENT", "auto", "chunk", "document"]


def build_arg_parser_baseline_subcommand(subparsers) -> None:
    """Register the baseline subcommand and all of its flags."""
    from driftlens.cli.compare import _SIDE_OPTION_FLAGS

    baseline = subparsers.add_parser("baseline", help="baseline management")
    baseline.add_argument("input", nargs="*")
    baseline.add_argument("--list")
    baseline.add_argument("--out", required=True)
    baseline.add_argument("--config")
    baseline.add_argument("--eval-set", dest="eval_set")
    baseline.add_argument("--verify-full", dest="verify_full", action="store_true")
    baseline.add_argument(
        "--granularity", dest="schema.granularity", choices=_GRANULARITY_CHOICES
    )
    for flag in _SIDE_OPTION_FLAGS:
        baseline.add_argument(f"--{flag.replace('_', '-')}", dest=f"schema.{flag}")
    baseline.add_argument(
        "--allow-lossy-vectors", dest="schema.allow_lossy_vectors", action="store_true"
    )
    baseline.add_argument("--embedding-model", dest="embeddings.model")
    baseline.add_argument(
        "--embedding-similarity-metric", dest="embeddings.similarity_metric"
    )
    baseline.add_argument(
        "--allow-unsafe-pickle", dest="allow_unsafe_pickle", action="store_true"
    )
    baseline.add_argument("--workers", dest="analysis.workers", type=int)
    baseline.add_argument("--sample-size", dest="analysis.sample_size", type=int)
    baseline.add_argument("--seed", dest="analysis.seed", type=int)
    baseline.add_argument("--bandwidth", dest="analysis.bandwidth", type=float)
    baseline.add_argument("--max-records", dest="limits.max_records", type=int)
    baseline.add_argument("--max-dimension", dest="limits.max_dimension", type=int)
    baseline.add_argument("--max-text-length", dest="limits.max_text_length", type=int)
    baseline.add_argument("--max-sample-size", dest="limits.max_sample_size", type=int)
    baseline.add_argument("--max-workers", dest="limits.max_workers")
    baseline.add_argument("--max-input-bytes", dest="limits.max_input_bytes", type=int)


def _resolve_paths(args: argparse.Namespace) -> list[Path]:
    list_path = getattr(args, "list", None)
    if list_path:
        lines = Path(list_path).read_text(encoding="utf-8").splitlines()
        entries = [line.strip() for line in lines if line.strip()]
    else:
        entries = [str(item) for item in (getattr(args, "input", None) or [])]
    if not entries:
        raise InvalidInputError(
            location="baseline",
            message="no snapshot paths were provided",
            hint="pass snapshot paths positionally or via --list",
        )
    return [Path(entry) for entry in entries]


def _load_or_build(
    path: Path,
    *,
    side,
    cfg,
    sink: DiagnosticSink,
    limits: RuntimeLimits,
    allow_unsafe_pickle: bool,
    verify_full: bool,
    clusters_mode: str,
):
    from driftlens.adapters.detect import detect_format
    from driftlens.cli.compare import _build_side, _load_input, check_dependencies

    format_ = detect_format(path, forced=side.format, limits=limits)
    if format_ == "snapshot":
        return read_snapshot(path, limits=limits, verify_full=verify_full)
    check_dependencies(format_, format_, clusters_mode=clusters_mode)
    loaded, unsafe = _load_input(
        format_,
        path,
        side,
        limits=limits,
        allow_unsafe_pickle=allow_unsafe_pickle,
        verify_full=verify_full,
    )
    return _build_side(
        loaded,
        side,
        cfg=cfg,
        sink=sink,
        limits=limits,
        unsafe_pickle=unsafe,
    )


def run_baseline_build(args: argparse.Namespace) -> int:
    """Build a baseline file from an ordered list of snapshots."""
    from driftlens.cli.compare import _cli_dict

    cli = _cli_dict(args)
    config_path = Path(args.config) if getattr(args, "config", None) else None
    cfg = load_config(config_path, cli, Path.cwd())
    sink = DiagnosticSink()
    limits = RuntimeLimits(cfg.limits)
    side = effective_side_options(cfg, "a", cli)
    clusters_mode = getattr(cfg.cluster, "mode", "auto")

    snapshots = [
        _load_or_build(
            path,
            side=side,
            cfg=cfg,
            sink=sink,
            limits=limits,
            allow_unsafe_pickle=getattr(args, "allow_unsafe_pickle", False),
            verify_full=getattr(args, "verify_full", False),
            clusters_mode=clusters_mode,
        )
        for path in _resolve_paths(args)
    ]
    eval_set = None
    if getattr(args, "eval_set", None):
        eval_set = load_eval_set(Path(args.eval_set), limits=limits)

    result = build_baseline(snapshots, eval_set=eval_set, cfg=cfg)
    write_baseline(result, Path(args.out))
    print(
        f"Baseline written to {args.out}: "
        f"{len(result.thresholds)}/{len(THRESHOLD_KEYS)} threshold keys "
        f"established from {result.pair_count} pair(s)"
    )
    return 0
