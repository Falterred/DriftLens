"""Console entry point for DriftLens.

Step 1 implements only `--version`; Step 33 replaces this module with the full
`compare` pipeline.
"""

from __future__ import annotations

import argparse
import sys

from driftlens import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the DriftLens command line and return the process exit code."""
    parser = argparse.ArgumentParser(prog="driftlens")
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the DriftLens version and exit",
    )
    args = parser.parse_args(argv)
    if args.version:
        print(f"driftlens {__version__}")
        return 0
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
