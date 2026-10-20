"""The measurement cache survives code edits, so measuring-code edits must be decided."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fpvedit
from fpvedit.analysis import MEASUREMENT_MODULES, MEASUREMENT_VERSION

# The measuring code that MEASUREMENT_VERSION describes (line endings normalised).
PINNED = {1: "2af2de30beee18b112baacc225572f3f53910b18db925e3e1f228f959bb8d37d"}


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
