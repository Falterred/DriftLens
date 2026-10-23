"""Step 51 schema-freeze, CLI-reference, and configuration contract checks.

The private docs/ tree is gitignored and absent from a clean checkout, so these
tests validate the frozen contracts against the tracked authoritative sources:
the published version constants, the argparse definitions that drive the CLI
reference generator, and the configuration schema itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from driftlens.cli.compare import build_arg_parser
from driftlens.config.schema import Config, default_config, validate_config
from driftlens.reports.json import REPORT_SCHEMA_VERSION
from driftlens.schema.manifest import SNAPSHOT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _iter_option_strings(parser: argparse.ArgumentParser) -> Iterator[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _iter_option_strings(subparser)
        else:
            yield from action.option_strings


def _iter_subcommands(parser: argparse.ArgumentParser) -> Iterator[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            yield from action.choices


def _iter_config_paths(value: Any, prefix: str = "") -> Iterator[str]:
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            path = prefix + "." + field.name if prefix else field.name
            yield path
            yield from _iter_config_paths(getattr(value, field.name), path)
    elif isinstance(value, dict):
        for key, item in value.items():
            path = prefix + "." + str(key) if prefix else str(key)
            yield path
            yield from _iter_config_paths(item, path)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_config_paths(item, prefix)


def _load_cli_reference_generator() -> Any:
    """Load the tracked CLI-reference generator without touching private docs."""
    path = ROOT / "tools" / "generate_cli_docs.py"
    spec = importlib.util.spec_from_file_location("_driftlens_cli_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_freeze_literals() -> None:
    """1.0 stays frozen: both published schema versions are literal MAJOR.MINOR."""
    assert SNAPSHOT_SCHEMA_VERSION == "1.0"
    assert REPORT_SCHEMA_VERSION == "1.1"
    for version in (SNAPSHOT_SCHEMA_VERSION, REPORT_SCHEMA_VERSION):
        assert re.fullmatch(r"[0-9]+\.[0-9]+", version)


def test_cli_reference_matches_current_parser() -> None:
    """The tracked CLI-reference generator covers every parser definition."""
    parser = build_arg_parser()
    options = sorted(set(_iter_option_strings(parser)))
    assert options
    assert set(_iter_subcommands(parser)) == {"compare", "snapshot", "baseline"}
    generator = _load_cli_reference_generator()
    reference = generator.render()
    missing = [option for option in options if option not in reference]
    assert not missing, missing
    assert reference == generator.render()


def test_configuration_doc_covers_every_field() -> None:
    """The default configuration materialises every declared schema field."""
    cfg = default_config()
    paths = set(_iter_config_paths(cfg))
    assert "analysis.workers" in paths
    assert "retrieval.ann_threshold" in paths
    for block in fields(Config):
        assert block.name in paths, block.name
        value = getattr(cfg, block.name)
        if is_dataclass(value):
            for nested in fields(value):
                assert f"{block.name}.{nested.name}" in paths, (block.name, nested.name)
    validate_config(cfg)
