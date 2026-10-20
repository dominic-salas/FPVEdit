"""The measurement cache survives code edits, so measuring-code edits must be decided."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fpvedit
from fpvedit.analysis import MEASUREMENT_MODULES, MEASUREMENT_VERSION

# The measuring code that MEASUREMENT_VERSION describes (line endings normalised).
PINNED = {1: "4a967dc07c35988bb22116b28d9e3e64abb8c48cc429c3e330fba9cf77f8c539"}


def test_measuring_code_matches_its_measurement_version() -> None:
    package = Path(fpvedit.__file__).parent
    digest = hashlib.sha256()
    for name in MEASUREMENT_MODULES:
        digest.update((package / name).read_bytes().replace(b"\r\n", b"\n"))
    assert PINNED.get(MEASUREMENT_VERSION) == digest.hexdigest(), (
        "The measuring code changed. If cached measurements would now differ, bump "
        "analysis.MEASUREMENT_VERSION; either way, pin this hash for the version: "
        + digest.hexdigest()
    )
