#!/usr/bin/env python3
"""Generate docs/cli-reference.md from the DriftLens argparse definitions.

This script is deliberately stdlib-only.  It imports the real argument parser
(driftlens.cli.compare.build_arg_parser) so the generated reference can never
drift from the shipped CLI.

Usage:

    python tools/generate_cli_docs.py          # (re)write docs/cli-reference.md
    python tools/generate_cli_docs.py --check  # exit non-zero if it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "docs" / "cli-reference.md"

sys.path.insert(0, str(REPO_ROOT))

from driftlens.cli.compare import build_arg_parser  # noqa: E402

TICK = chr(96)


def _takes_value(action: argparse.Action) -> bool:
    """False only for store_true/store_false/count style flags."""

    if action.option_strings and action.nargs == 0:
        return False
    return True


def _describe_value(action: argparse.Action) -> str:
    bits: list[str] = []
    metavar = action.metavar
    if metavar:
        if isinstance(metavar, tuple):
            bits.append(" ".join(str(part) for part in metavar))
        else:
            bits.append(str(metavar))
    if action.choices:
        bits.append("choices: " + ", ".join(str(choice) for choice in action.choices))
    type_name = getattr(action.type, "__name__", None)
    if type_name:
        bits.append("type: " + str(type_name))
    if action.nargs in ("*", "+"):
        bits.append("nargs: " + str(action.nargs))
    return ", ".join(bits)


def _render_action(action: argparse.Action) -> str:
    if isinstance(action, argparse._SubParsersAction):
        return ""
    if action.option_strings:
        label = ", ".join(TICK + opt + TICK for opt in action.option_strings)
    else:
        label = TICK + str(action.metavar or action.dest) + TICK
    if _takes_value(action):
        described = _describe_value(action)
        note = "takes a value" + (" (" + described + ")" if described else "")
    else:
        note = "no value"
    help_text = (action.help or "").strip()
    line = "- " + label + " \u2014 " + note
    if help_text:
        line += ". " + help_text
    return line


def _subparsers(parser: argparse.ArgumentParser) -> list[argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return list(action.choices.values())
    return []


def render() -> str:
    parser = build_arg_parser()
    lines: list[str] = []
    lines.append("# DriftLens CLI Reference")
    lines.append("")
    lines.append(
        "> Generated from the live " + TICK + "argparse" + TICK + " definitions by "
        + TICK + "tools/generate_cli_docs.py" + TICK + "."
    )
    lines.append(
        "> Do not edit by hand \u2014 run " + TICK + "python tools/generate_cli_docs.py"
        + TICK + " to regenerate."
    )
    lines.append("")

    ordered = [parser, *_subparsers(parser)]
    for command in ordered:
        lines.append("## " + TICK + command.prog + TICK)
        lines.append("")
        lines.append("```")
        lines.append(command.format_usage().strip())
        lines.append("```")
        lines.append("")
        for action in command._actions:
            rendered = _render_action(action)
            if rendered:
                lines.append(rendered)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Every option string listed above is generated from the parser built by "
        + TICK + "driftlens.cli.compare.build_arg_parser()" + TICK + "."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    content = render()
    if "--check" in args:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != content:
            print(
                "docs/cli-reference.md is out of date; "
                "run " + TICK + "python tools/generate_cli_docs.py" + TICK,
                file=sys.stderr,
            )
            return 1
        print("docs/cli-reference.md is up to date")
        return 0
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print("wrote " + str(OUTPUT_PATH.relative_to(REPO_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
