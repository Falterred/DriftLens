"""Console entry point for DriftLens (Step 33)."""

from __future__ import annotations

import sys

from driftlens import __version__
from driftlens.cli.compare import build_arg_parser, run_compare
from driftlens.errors import (
    DriftLensError,
    FatalErrors,
    InvalidInputError,
    format_fatal,
)


def main(argv: list[str] | None = None) -> int:
    """Run the DriftLens command line and return the process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        print(f"driftlens {__version__}")
        return 0

    command = getattr(args, "command", None)
    if command is None:
        parser.print_usage(sys.stderr)
        return 2

    try:
        if command == "compare":
            return run_compare(args)
        raise InvalidInputError(
            message=f"the '{command}' command is not available until a later release"
        )
    except FatalErrors as exc:
        print(format_fatal(exc.errors), file=sys.stderr)
        return exc.exit_code
    except DriftLensError as exc:
        print(format_fatal([exc]), file=sys.stderr)
        return exc.exit_code
    except Exception:
        print(
            "DRIFTLENS ERROR INTERNAL_ERROR (exit 6)\nan unexpected error occurred",
            file=sys.stderr,
        )
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
