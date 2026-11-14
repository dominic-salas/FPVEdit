"""The measurement cache survives code edits, so measuring-code edits must be decided."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fpvedit
from fpvedit.analysis import MEASUREMENT_MODULES, MEASUREMENT_VERSION

# The measuring code that MEASUREMENT_VERSION describes (line endings normalised). Version 1
# kept its number when recordings above 1080p got a faster decode (no loop filter, GPU
# scaling): their measurements move by under 1 %, and recordings up to 1080p are unchanged.
PINNED = {1: "b901c133cdda109e30277c94c5707e295136b636d974110a08732be534812073"}


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
