"""Step 51 schema-freeze, CLI-reference, and configuration-doc checks."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from driftlens.cli.compare import build_arg_parser
from driftlens.config.schema import default_config
from driftlens.reports.json import REPORT_SCHEMA_VERSION
from driftlens.schema.manifest import SNAPSHOT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _iter_option_strings(parser: argparse.ArgumentParser) -> Iterator[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _iter_option_strings(subparser)
        else:
            yield from action.option_strings


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


def test_schema_freeze_literals():
    assert SNAPSHOT_SCHEMA_VERSION == "1.0"
    assert REPORT_SCHEMA_VERSION == "1.0"
    doc = _read("docs/schema-versioning.md")
    assert "SNAPSHOT_SCHEMA_VERSION" in doc
    assert "REPORT_SCHEMA_VERSION" in doc
    assert "1.0" in doc


def test_cli_reference_matches_current_parser():
    doc = _read("docs/cli-reference.md")
    options = sorted(set(_iter_option_strings(build_arg_parser())))
    assert options
    missing = [option for option in options if option not in doc]
    assert not missing, missing


def test_configuration_doc_covers_every_field():
    doc = _read("docs/configuration.md")
    paths = sorted(set(_iter_config_paths(default_config())))
    assert "analysis.workers" in paths
    assert "retrieval.ann_threshold" in paths
    missing = [path for path in paths if path not in doc]
    assert not missing, missing
