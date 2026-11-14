"""The measurement cache survives code edits, so measuring-code edits must be decided."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fpvedit
from fpvedit.analysis import MEASUREMENT_MODULES, MEASUREMENT_VERSION

# The measuring code that MEASUREMENT_VERSION describes (line endings normalised).
PINNED = {1: "96a6bcb1245a0d5a00cd9f4dc1b68646439e205e8279d0403b83cc98cbed4a1f"}


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
