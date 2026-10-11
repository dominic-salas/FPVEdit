"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from fpv_editor import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpv-editor",
        description="Turn a raw FPV freestyle recording into one highlight MP4.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print("fpv-editor: editing is not implemented yet (milestone 2).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
