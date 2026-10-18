"""Dom's own star ratings of shots, kept so the rating can be tuned to his taste.

Pressing 1-5 while a shot plays (or clicking the stars in the shot panel) appends one line
of JSON to ``my-ratings.jsonl`` in the app's data folder: the recording, the shot's times,
his stars, and the app's own rating with its parts and reasons, so the two can be compared
later. Nothing leaves the computer. A later rating of the same shot replaces the earlier one
when the file is read.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fpv_editor.montage import Shot


def shot_key(path: Path, shot: Shot) -> str:
    return f"{path.resolve()}|{shot.start:.2f}|{shot.end:.2f}"


class MyRatings:
    def __init__(self, file: Path | None) -> None:
        self.file = file
        self._stars: dict[str, int] = {}
        if file is not None and file.is_file():
            for line in file.read_text(encoding="utf-8").splitlines():
                try:
                    entry: dict[str, Any] = json.loads(line)
                    self._stars[str(entry["key"])] = int(entry["stars"])
                except (ValueError, KeyError, TypeError):
                    continue  # a damaged line; the others still count

    @property
    def count(self) -> int:
        return len(self._stars)

    def stars(self, path: Path, shot: Shot) -> int | None:
        return self._stars.get(shot_key(path, shot))

    def rate(self, path: Path, shot: Shot, stars: int) -> None:
        """Remember ``stars`` (1-5) for ``shot`` of the recording at ``path``."""
        if not 1 <= stars <= 5:
            raise ValueError("stars are 1 to 5")
        key = shot_key(path, shot)
        self._stars[key] = stars
        if self.file is None:
            return
        quality = shot.quality
        entry = {
            "key": key,
            "recording": str(path),
            "start": round(shot.start, 2),
            "end": round(shot.end, 2),
            "stars": stars,
            "app_stars": quality.stars if quality is not None else None,
            "app_parts": asdict(quality) if quality is not None else None,
            "tricks": [m.name for m in shot.moments],
            "rated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
