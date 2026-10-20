"""The measurement cache survives code edits, so measuring-code edits must be decided."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fpv_editor
from fpv_editor.analysis import MEASUREMENT_MODULES, MEASUREMENT_VERSION

# The measuring code that MEASUREMENT_VERSION describes (line endings normalised).
PINNED = {1: "2454ee803af76288571fcd2373ee1a1318bf21b2e0271ea83bf769beb321f605"}


def test_measuring_code_matches_its_measurement_version() -> None:
    package = Path(fpv_editor.__file__).parent
    digest = hashlib.sha256()
    for name in MEASUREMENT_MODULES:
        digest.update((package / name).read_bytes().replace(b"\r\n", b"\n"))
    assert PINNED.get(MEASUREMENT_VERSION) == digest.hexdigest(), (
        "The measuring code changed. If cached measurements would now differ, bump "
        "analysis.MEASUREMENT_VERSION; either way, pin this hash for the version: "
        + digest.hexdigest()
    )
