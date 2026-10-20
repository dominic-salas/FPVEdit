"""What the clip list says about each recording, without Qt: when it was recorded and a
one-line description.

DVR recordings carry no creation time in their metadata (checked on Walksnail and
HDZero files), but Walksnail names them after the start of the recording
(``20261010-203859.ts``). Other names fall back to the file's modified time, labelled as
such because a copy changes it.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fpvedit.media import MediaInfo

_STAMP = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[-_ ]?(\d{2})(\d{2})(\d{2})(?!\d)")


@dataclass(frozen=True)
class Recorded:
    when: datetime
    from_name: bool  # False: the file's modified time

    def label(self) -> str:
        """'REC 10 OCT 2026 20:38' or 'FILE 9 OCT 2026 22:57'."""
        kind = "REC" if self.from_name else "FILE"
        return f"{kind} {self.when.day} {self.when:%b %Y %H:%M}".upper()


def recorded_at(path: Path) -> Recorded | None:
    """When ``path`` was recorded: from a date-time in its name, else its modified time."""
    match = _STAMP.search(path.stem)
    if match is not None:
        try:
            year, month, day, hour, minute, second = (int(part) for part in match.groups())
            return Recorded(datetime(year, month, day, hour, minute, second), True)
        except ValueError:  # digits that only look like a date
            pass
    try:
        return Recorded(datetime.fromtimestamp(path.stat().st_mtime), False)
    except OSError:
        return None


def clock(seconds: float) -> str:
    """'3:10' (minutes and whole seconds); hours as '1:02:03'."""
    whole = round(max(0.0, seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def describe_clip(info: MediaInfo) -> str:
    """'3:10   720P 60   REC 10 OCT 2026 20:38' for the clip list."""
    parts = [clock(info.duration) if info.duration else "LENGTH UNKNOWN"]
    rate = info.video.nominal_rate
    parts.append(f"{info.video.height}P {round(float(rate))}" if rate else f"{info.video.height}P")
    recorded = recorded_at(info.path)
    if recorded is not None:
        parts.append(recorded.label())
    return "   ".join(parts)


def is_removable(path: Path) -> bool:
    """Whether ``path`` is on a removable drive (an SD card or USB stick), on Windows.

    Montages are never saved there unless the user picks it: it is usually the card the
    goggles record to, holding the original footage.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    drive = path.resolve().anchor
    if not drive:
        return False
    removable = 2  # DRIVE_REMOVABLE from GetDriveTypeW
    return bool(ctypes.windll.kernel32.GetDriveTypeW(drive) == removable)


def default_output_folder(remembered: Path | None, sources: list[Path], videos: Path) -> Path:
    """Where a montage is saved unless the user picks a place: the folder they used last,
    unless it holds the clips or is on a removable card; otherwise ``videos``."""
    if remembered is not None and remembered.is_dir() and not is_removable(remembered):
        folders = {source.resolve().parent for source in sources}
        if remembered.resolve() not in folders:
            return remembered
    return videos
