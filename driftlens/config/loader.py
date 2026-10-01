"""Configuration loading: YAML file, CLI overrides and per-side resolution."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from driftlens.config.schema import (
    Config,
    DIMENSION_OF_KEY,
    DIMENSIONS,
    SideOptions,
    THRESHOLD_KEYS,
    Threshold,
    default_config,
    validate_config,
)
from driftlens.errors import InvalidInputError

DEFAULT_CONFIG_NAME = "driftlens.yaml"

_SIDE_OPTION_FIELDS = {item.name for item in fields(SideOptions)}
_UNSAFE_PICKLE_MESSAGE = (
    "not a valid configuration key; use --allow-unsafe-pickle on the CLI"
)
_THRESHOLD_BOUNDS = ("warning", "critical")


def _invalid(message: str, path: str) -> InvalidInputError:
    return InvalidInputError(message=message, location=f"config: {path}")


def _read_yaml(config_path: Path | None, cwd: Path) -> dict:
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise InvalidInputError(
                message=f"config file not found: {path}", location="config"
            )
    else:
        candidate = Path(cwd) / DEFAULT_CONFIG_NAME
        if not candidate.is_file():
            return {}
        path = candidate

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise InvalidInputError(
            message=f"cannot read config file: {path.name}", location="config"
        ) from None
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        raise InvalidInputError(message="malformed YAML", location="config") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise _invalid("config top level must be a mapping", "")
    return loaded


def _threshold_value(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid("threshold must be a number or null", path)
    return float(value)


def _apply_drift(cfg: Config, value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise _invalid("drift must be a mapping", path)
    for key, entry in value.items():
        if key not in THRESHOLD_KEYS:
            raise _invalid("unknown key", f"{path}.{key}")
        if entry is None:
            cfg.drift[key] = None
            continue
        if not isinstance(entry, dict):
            raise _invalid("threshold must be a mapping or null", f"{path}.{key}")
        for bound in entry:
            if bound not in _THRESHOLD_BOUNDS:
                raise _invalid("unknown key", f"{path}.{key}.{bound}")
        cfg.drift[key] = Threshold(
            warning=_threshold_value(entry.get("warning"), f"{path}.{key}.warning"),
            critical=_threshold_value(entry.get("critical"), f"{path}.{key}.critical"),
        )


def _apply_mapping(
    target: Any, mapping: dict, prefix: str, user_fail_on: dict[str, Any]
) -> None:
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if key == "allow_unsafe_pickle":
            raise _invalid(_UNSAFE_PICKLE_MESSAGE, path)
        if not isinstance(key, str) or not hasattr(target, key):
            raise _invalid("unknown key", path)
        if key == "fail_on":
            if not isinstance(value, dict):
                raise _invalid("fail_on must be a mapping", path)
            for sub_key, sub_value in value.items():
                user_fail_on[sub_key] = sub_value
            continue
        if key in ("schema_a", "schema_b"):
            if not isinstance(value, dict):
                raise _invalid(f"{key} must be a mapping", path)
            for sub_key in value:
                if sub_key not in _SIDE_OPTION_FIELDS:
                    raise _invalid("unknown key", f"{path}.{sub_key}")
            setattr(target, key, dict(value))
            continue
        if key == "drift":
            _apply_drift(target, value, path)
            continue
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise _invalid(f"{key} must be a mapping", path)
            _apply_mapping(current, value, f"{path}.", user_fail_on)
            continue
        setattr(target, key, value)


def _apply_cli(cfg: Config, cli: dict[str, Any], user_fail_on: dict[str, Any]) -> None:
    for path, value in cli.items():
        if value is None:
            continue
        parts = path.split(".")
        if parts[0] in ("schema_a", "schema_b"):
            if len(parts) != 2 or parts[1] not in _SIDE_OPTION_FIELDS:
                raise _invalid("unknown key", path)
            getattr(cfg, parts[0])[parts[1]] = value
            continue
        if parts[0] == "drift":
            if (
                len(parts) != 3
                or parts[1] not in THRESHOLD_KEYS
                or parts[2] not in _THRESHOLD_BOUNDS
            ):
                raise _invalid("unknown key", path)
            existing = cfg.drift.get(parts[1])
            if existing is None:
                existing = Threshold()
                cfg.drift[parts[1]] = existing
            setattr(existing, parts[2], value)
            continue
        if len(parts) == 3 and parts[0] == "ci" and parts[1] == "fail_on":
            user_fail_on[parts[2]] = value
            continue
        obj: Any = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise _invalid("unknown key", path)
            obj = getattr(obj, part)
        if not hasattr(obj, parts[-1]):
            raise _invalid("unknown key", path)
        setattr(obj, parts[-1], value)


def _process_fail_on(cfg: Config, user_fail_on: dict[str, Any]) -> None:
    fail_on = dict(cfg.ci.fail_on)
    explicit_warning = {
        key
        for key, value in user_fail_on.items()
        if isinstance(value, str) and value == "warning"
    }
    for key, value in user_fail_on.items():
        if isinstance(value, bool):
            raise _invalid("fail_on value must be a dimension policy or a number", f"ci.fail_on.{key}")
        if isinstance(value, str):
            if value not in ("warning", "critical"):
                raise _invalid("fail_on policy must be warning or critical", f"ci.fail_on.{key}")
            if key not in DIMENSIONS:
                raise _invalid("fail_on dimension is unknown", f"ci.fail_on.{key}")
            fail_on[key] = value
            continue
        if isinstance(value, (int, float)):
            if key not in THRESHOLD_KEYS:
                raise _invalid("fail_on threshold key is unknown", f"ci.fail_on.{key}")
            existing = cfg.drift.get(key)
            if existing is not None and existing.critical is not None:
                raise _invalid(
                    f"ci.fail_on.{key} and drift.{key}.critical both set",
                    f"ci.fail_on.{key}",
                )
            warning = existing.warning if existing is not None else None
            cfg.drift[key] = Threshold(warning=warning, critical=float(value))
            fail_on.pop(key, None)
            cfg.ci.critical_gates.add(key)
            dimension = DIMENSION_OF_KEY[key]
            if dimension not in explicit_warning:
                fail_on[dimension] = "critical"
            continue
        raise _invalid("fail_on value must be a dimension policy or a number", f"ci.fail_on.{key}")
    cfg.ci.fail_on = fail_on


def load_config(config_path: Path | None, cli: dict[str, Any], cwd: Path) -> Config:
    """Build, override and validate the configuration for one run."""

    data = _read_yaml(config_path, cwd)
    cfg = default_config()
    user_fail_on: dict[str, Any] = {}
    _apply_mapping(cfg, data, "", user_fail_on)
    _apply_cli(cfg, cli, user_fail_on)
    _process_fail_on(cfg, user_fail_on)
    validate_config(cfg)
    return cfg


def _coerce_side_value(name: str, value: Any) -> Any:
    if name == "metadata_fields" and isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def effective_side_options(cfg: Config, side: str, cli: dict[str, Any]) -> SideOptions:
    """Resolve the per-side options for side a or b, highest layer wins."""

    if side not in ("a", "b"):
        raise _invalid(f"unknown side {side!r}", "")

    options = replace(cfg.schema)
    for key, value in getattr(cfg, f"schema_{side}").items():
        setattr(options, key, value)

    if options.model is None:
        options.model = cfg.embeddings.model
    if options.similarity_metric is None:
        options.similarity_metric = cfg.embeddings.similarity_metric

    for path, value in cli.items():
        if value is None:
            continue
        parts = path.split(".")
        if len(parts) != 2:
            continue
        if parts[0] == "schema" and parts[1] in _SIDE_OPTION_FIELDS:
            setattr(options, parts[1], _coerce_side_value(parts[1], value))
        elif parts[0] == "embeddings" and parts[1] in ("model", "similarity_metric"):
            setattr(options, parts[1], value)

    suffix = f"schema_{side}"
    for path, value in cli.items():
        if value is None:
            continue
        parts = path.split(".")
        if (
            len(parts) == 2
            and parts[0] == suffix
            and parts[1] in _SIDE_OPTION_FIELDS
        ):
            setattr(options, parts[1], _coerce_side_value(parts[1], value))

    return options
