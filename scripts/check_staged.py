"""Refuse to commit footage, flight logs, analysis output, weights, secrets or big files.

Usage:
    python scripts/check_staged.py            # check files staged for commit
    python scripts/check_staged.py --all      # check every tracked file (CI)
    python scripts/check_staged.py PATH ...   # check the given paths

Exits 1 and lists every offending path when something is prohibited.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 5 * 1024 * 1024

PROHIBITED_SUFFIXES: dict[str, str] = {
    **dict.fromkeys(
        (".ts", ".m2ts", ".mts", ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"),
        "video file",
    ),
    **dict.fromkeys((".wav", ".mp3", ".aac", ".flac"), "audio file"),
    **dict.fromkeys(
        (".safetensors", ".onnx", ".pt", ".pth", ".ckpt", ".bin", ".gguf", ".h5"),
        "model weights",
    ),
    **dict.fromkeys((".pem", ".key", ".p12", ".pfx"), "credential"),
    # DVR sidecars and flight-controller logs can hold GPS positions and pilot data.
    **dict.fromkeys((".srt", ".osd"), "DVR sidecar"),
    **dict.fromkeys((".bbl", ".bfl", ".bbs", ".event"), "blackbox flight log"),
    **dict.fromkeys((".gpx", ".kml"), "GPS track"),
    **dict.fromkeys((".drp", ".drt"), "editor project"),
    **dict.fromkeys((".npy", ".npz"), "analysis output"),
}

# Folders refused at the top of the repository / at any depth (mirrors .gitignore).
TOP_LEVEL_DIRECTORIES: dict[str, str] = {"res": "local test footage folder"}
ANY_LEVEL_DIRECTORIES: dict[str, str] = {
    **dict.fromkeys(("work", "output", "outputs", "footage", "frames"), "local working folder"),
    ".cache": "cache folder",
}
# blackbox_decode writes LOG00001.01.csv, LOG00001.01.gps.csv, ... next to the log.
BLACKBOX_CSV = re.compile(r"^(btfl_.*|log\d+.*)\.csv$")

PROHIBITED_NAMES: dict[str, str] = {
    ".env": "credential",
    "id_rsa": "credential",
    "id_ed25519": "credential",
}


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str


def classify(path: str, size: int | None) -> str | None:
    """Return why ``path`` must not be committed, or None if it is allowed."""
    pure = PurePosixPath(path.replace("\\", "/"))
    name = pure.name.lower()
    if name in PROHIBITED_NAMES:
        return PROHIBITED_NAMES[name]
    if name.startswith(".env."):
        return "credential"
    if name.endswith(".analysis.json"):
        return "analysis output"
    if BLACKBOX_CSV.match(name):
        return "blackbox flight log"
    folders = [part.lower() for part in pure.parts[:-1]]
    if folders and folders[0] in TOP_LEVEL_DIRECTORIES:
        return TOP_LEVEL_DIRECTORIES[folders[0]]
    for folder in folders:
        if folder in ANY_LEVEL_DIRECTORIES:
            return ANY_LEVEL_DIRECTORIES[folder]
    suffix = pure.suffix.lower()
    if suffix in PROHIBITED_SUFFIXES:
        return PROHIBITED_SUFFIXES[suffix]
    if size is not None and size > MAX_FILE_BYTES:
        return f"file larger than {MAX_FILE_BYTES // (1024 * 1024)} MB"
    return None


def find_violations(paths: Iterable[str], root: Path) -> list[Violation]:
    violations = []
    for path in paths:
        file = root / path
        size = file.stat().st_size if file.is_file() else None
        reason = classify(path, size)
        if reason is not None:
            violations.append(Violation(path, reason))
    return violations


def _git_paths(args: Sequence[str], root: Path) -> list[str]:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="paths to check (default: staged files)")
    parser.add_argument("--all", action="store_true", help="check every tracked file")
    args = parser.parse_args(argv)

    root = Path.cwd()
    try:
        if args.paths:
            paths = list(args.paths)
        elif args.all:
            paths = _git_paths(["ls-files"], root)
        else:
            paths = _git_paths(["diff", "--cached", "--name-only", "--diff-filter=ACMR"], root)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"could not list files with git: {detail.strip()}", file=sys.stderr)
        return 2

    violations = find_violations(paths, root)
    for violation in violations:
        print(f"refused: {violation.path} ({violation.reason})", file=sys.stderr)
    if violations:
        print(
            "Footage, renders, model weights and credentials must not be committed. "
            "Unstage them with: git restore --staged <path>",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
