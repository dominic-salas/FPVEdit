from __future__ import annotations

import sys
from pathlib import Path

# Make the fixture generator importable as a plain module from the tests directory.
sys.path.insert(0, str(Path(__file__).parent))


import pytest


@pytest.fixture(autouse=True)
def _no_image_model(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never load the image or gravity model unless marked ``model`` (they download
    weights and use the GPU); those steps then behave as if the ml extra were missing."""
    if request.node.get_closest_marker("model") is None:
        from fpvedit import attitude, scenery

        monkeypatch.setattr(scenery, "available", lambda: False)
        monkeypatch.setattr(attitude, "available", lambda: False)


def pytest_sessionstart(session: pytest.Session) -> None:
    """On CI (GitHub sets ``CI``), a missing FFmpeg or Qt is a broken runner, not a reason
    to skip: the integration tests would skip and the run would still look green."""
    import importlib.util
    import os

    from media_fixtures import ffmpeg_available

    if os.environ.get("CI"):
        missing = [
            name
            for name, present in (
                ("FFmpeg", ffmpeg_available()),
                ("PySide6", importlib.util.find_spec("PySide6") is not None),
            )
            if not present
        ]
        if missing:
            pytest.exit(f"CI is missing {', '.join(missing)}; tests would skip", returncode=1)
