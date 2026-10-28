"""The pilot's own star ratings of shots, kept so the rating can learn their taste.

Pressing 1-5 while a shot plays (or clicking the stars in the shot panel) appends one line
of JSON to ``my-ratings.jsonl`` in the app's data folder: the recording, the shot's times,
their stars, and the app's own rating with its parts and reasons, so the two can be compared
later. Nothing leaves the computer. A later rating of the same shot replaces the earlier one
when the file is read, and an undo is a line with 0 stars (the file is only appended to).

Exporting a montage appends what the edit said about its shots to ``my-edits.jsonl`` beside
it (`model.ShotList.outcome`: removed, exported as it was, trimmed, added, marked), in the
same form. The taste learns from those more weakly (`taste.EDITS`); a shot's own stars
always win over what an edit said about it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fpvedit.montage import Shot
from fpvedit.rating import Quality
from fpvedit.taste import EDITS, RatedShot

EDITS_FILE = "my-edits.jsonl"


def shot_key(path: Path, shot: Shot) -> str:
    return f"{path.resolve()}|{shot.start:.2f}|{shot.end:.2f}"


class MyRatings:
    def __init__(self, file: Path | None) -> None:
        self.file = file
        self.edits_file = file.with_name(EDITS_FILE) if file is not None else None
        self._stars: dict[str, int] = {}
        self._entries: dict[str, dict[str, Any]] = {}  # the latest rating of each shot
        self._edits: dict[str, dict[str, Any]] = {}  # the latest edit of each shot
        for entry in _read(file):
            try:
                stars = int(entry["stars"])
                if stars == 0:  # an undone rating
                    self._stars.pop(str(entry["key"]), None)
                    self._entries.pop(str(entry["key"]), None)
                else:
                    self._stars[str(entry["key"])] = stars
                    self._entries[str(entry["key"])] = entry
            except (ValueError, KeyError, TypeError):
                continue  # a damaged line; the others still count
        for entry in _read(self.edits_file):
            if isinstance(entry.get("key"), str):
                self._edits[entry["key"]] = entry

    @property
    def count(self) -> int:
        return len(self._stars)

    @property
    def edit_count(self) -> int:
        """Shots an edit has said something about (trims are only kept)."""
        return sum(1 for e in self._edits.values() if e.get("edit") in EDITS)

    def stars(self, path: Path, shot: Shot) -> int | None:
        return self._stars.get(shot_key(path, shot))

    def rate(self, path: Path, shot: Shot, stars: int) -> None:
        """Remember ``stars`` (1-5) for ``shot`` of the recording at ``path``."""
        if not 1 <= stars <= 5:
            raise ValueError("stars are 1 to 5")
        entry = {**_describe(path, shot), "stars": stars}
        entry["rated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._stars[entry["key"]] = stars
        self._entries[entry["key"]] = entry
        _append(self.file, [entry])

    def unrate(self, path: Path, shot: Shot, why: str = "undone") -> None:
        """Forget the rating of ``shot`` (appends an undo line; nothing is deleted)."""
        key = shot_key(path, shot)
        self.unrate_key(key, why)

    def unrate_key(self, key: str, why: str = "undone") -> None:
        self._stars.pop(key, None)
        self._entries.pop(key, None)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _append(self.file, [{"key": key, "stars": 0, "why": why, "rated_at": stamp}])

    def note_edits(self, outcome: list[tuple[Path, Shot, str]]) -> None:
        """Remember what an exported montage's edit said: (recording, shot, what happened).

        Keeping a shot again says less than adding or removing it did before, so an
        "exported" never replaces an earlier edit of the same shot.
        """
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entries = []
        for path, shot, what in outcome:
            entry = {**_describe(path, shot), "edit": what, "edited_at": stamp}
            earlier = self._edits.get(entry["key"], {}).get("edit")
            if what == "exported" and earlier in EDITS and earlier != "exported":
                continue
            self._edits[entry["key"]] = entry
            entries.append(entry)
        _append(self.edits_file, entries)

    def rated_shots(self) -> list[RatedShot]:
        """Every rating that recorded the app's own rating of the shot, then what the edits
        said about the shots without a rating, to learn from."""
        edits = [
            {**e, "stars": EDITS[e["edit"]][0], "weight": EDITS[e["edit"]][1]}
            for key, e in self._edits.items()
            if key not in self._entries and e.get("edit") in EDITS
        ]
        rated = []
        for entry in [*self._entries.values(), *edits]:
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
                stars, weight = int(entry["stars"]), float(entry.get("weight", 1.0))
                rated.append(RatedShot(quality, tricks, duration, stars, recording, start, weight))
            except (KeyError, TypeError, ValueError):
                continue
        return rated

    def clear(self) -> None:
        """Forget every rating and edit: the files are deleted (nothing else is kept)."""
        self._stars.clear()
        self._entries.clear()
        self._edits.clear()
        for file in (self.file, self.edits_file):
            if file is not None:
                file.unlink(missing_ok=True)


def _describe(path: Path, shot: Shot) -> dict[str, Any]:
    quality = shot.quality
    return {
        "key": shot_key(path, shot),
        "recording": str(path),
        "start": round(shot.start, 2),
        "end": round(shot.end, 2),
        "app_stars": quality.stars if quality is not None else None,
        "app_parts": asdict(quality) if quality is not None else None,
        "tricks": [m.name for m in shot.moments],
    }


def _read(file: Path | None) -> list[dict[str, Any]]:
    if file is None or not file.is_file():
        return []
    entries = []
    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a damaged line; the others still count
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _append(file: Path | None, entries: list[dict[str, Any]]) -> None:
    if file is None or not entries:
        return
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
