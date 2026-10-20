"""Dom's own star ratings of shots, kept so the rating can be tuned to his taste.

Pressing 1-5 while a shot plays (or clicking the stars in the shot panel) appends one line
of JSON to ``my-ratings.jsonl`` in the app's data folder: the recording, the shot's times,
his stars, and the app's own rating with its parts and reasons, so the two can be compared
later. Nothing leaves the computer. A later rating of the same shot replaces the earlier one
when the file is read, and an undo is a line with 0 stars (the file is only appended to).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fpv_editor.montage import Shot
from fpv_editor.rating import Quality
from fpv_editor.taste import RatedShot


def shot_key(path: Path, shot: Shot) -> str:
    return f"{path.resolve()}|{shot.start:.2f}|{shot.end:.2f}"


class MyRatings:
    def __init__(self, file: Path | None) -> None:
        self.file = file
        self._stars: dict[str, int] = {}
        self._entries: dict[str, dict[str, Any]] = {}  # the latest rating of each shot
        if file is not None and file.is_file():
            for line in file.read_text(encoding="utf-8").splitlines():
                try:
                    entry: dict[str, Any] = json.loads(line)
                    stars = int(entry["stars"])
                    if stars == 0:  # an undone rating
                        self._stars.pop(str(entry["key"]), None)
                        self._entries.pop(str(entry["key"]), None)
                    else:
                        self._stars[str(entry["key"])] = stars
                        self._entries[str(entry["key"])] = entry
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
        self._entries[key] = entry
        if self.file is None:
            return
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def unrate(self, path: Path, shot: Shot, why: str = "undone") -> None:
        """Forget the rating of ``shot`` (appends an undo line; nothing is deleted)."""
        key = shot_key(path, shot)
        self.unrate_key(key, why)

    def unrate_key(self, key: str, why: str = "undone") -> None:
        self._stars.pop(key, None)
        self._entries.pop(key, None)
        if self.file is None:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {"key": key, "stars": 0, "why": why, "rated_at": stamp}
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def rated_shots(self) -> list[RatedShot]:
        """Every rating that recorded the app's own rating of the shot, to learn from."""
        rated = []
        for entry in self._entries.values():
            parts = entry.get("app_parts")
            if not isinstance(parts, dict):
                continue
            try:
                quality = Quality(
                    float(parts["trick"]),
                    float(parts["clean"]),
                    float(parts["impressive"]),
                    float(parts["watchable"]),
                    tuple(str(r) for r in parts.get("reasons") or ()),
                    None if parts.get("indoorness") is None else float(parts["indoorness"]),
                    picture=(
                        tuple(float(v) for v in parts["picture"])
                        if isinstance(parts.get("picture"), list)
                        else None
                    ),
                )
                start = float(entry["start"])
                duration = float(entry["end"]) - start
                tricks = tuple(str(t) for t in entry.get("tricks") or ())
                recording = Path(str(entry["recording"])) if entry.get("recording") else None
                rated.append(
                    RatedShot(quality, tricks, duration, int(entry["stars"]), recording, start)
                )
            except (KeyError, TypeError, ValueError):
                continue
        return rated

    def clear(self) -> None:
        """Forget every rating: the file is deleted (nothing else is kept)."""
        self._stars.clear()
        self._entries.clear()
        if self.file is not None:
            self.file.unlink(missing_ok=True)
