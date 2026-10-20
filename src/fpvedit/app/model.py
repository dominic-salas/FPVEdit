"""The montage being edited in the app: its shots in playing order, and the edits on them.

No Qt here, so the editing rules are tested like the rest of the engine. Shots from one
clip never overlap (each moment is shown once); any other order or length is the user's
choice, since they are the editor. Every edit can be undone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from fpvedit.montage import Shot, strength

MIN_SHOT_S = 1.0  # a trim never makes a shot shorter than this


def overlaps(a: Shot, b: Shot) -> bool:
    return a.clip == b.clip and a.start < b.end and b.start < a.end


class ShotList:
    def __init__(
        self, shots: Sequence[Shot], candidates: Sequence[Shot], durations: dict[int, float]
    ) -> None:
        """``durations`` gives each clip's analysed length, which trims stay inside."""
        self._shots = tuple(shots)
        self._candidates = tuple(candidates)
        self._durations = dict(durations)
        self._history: list[tuple[Shot, ...]] = []
        moments = {(s.clip, m) for s in (*shots, *candidates) for m in s.moments}
        self._moments = sorted(moments, key=lambda cm: (cm[0], cm[1].time))

    @property
    def shots(self) -> tuple[Shot, ...]:
        return self._shots

    @property
    def duration(self) -> float:
        return sum(shot.duration for shot in self._shots)

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def undo(self) -> None:
        if self._history:
            self._shots = self._history.pop()

    def _set(self, shots: Iterable[Shot]) -> None:
        new = tuple(shots)
        if new != self._shots:
            self._history.append(self._shots)
            self._shots = new

    def remove(self, index: int) -> None:
        self._set(s for i, s in enumerate(self._shots) if i != index)

    def move(self, index: int, to: int) -> None:
        """Move the shot at ``index`` so it ends up at position ``to``."""
        shots = list(self._shots)
        shot = shots.pop(index)
        shots.insert(max(0, min(to, len(shots))), shot)
        self._set(shots)

    def trim(self, index: int, *, start: float = 0.0, end: float = 0.0) -> None:
        """Move the shot's start and end by these many seconds (negative is earlier).

        The ends stop at the clip's ends and at the neighbouring shots of the same clip; a
        trim that would leave less than `MIN_SHOT_S` does nothing.
        """
        shot = self._shots[index]
        others = [s for i, s in enumerate(self._shots) if i != index and s.clip == shot.clip]
        low = max([0.0, *(s.end for s in others if s.end <= shot.start)])
        clip_end = self._durations.get(shot.clip, shot.end)
        high = min([clip_end, *(s.start for s in others if s.start >= shot.end)])
        new_start = max(low, shot.start + start)
        new_end = min(high, shot.end + end)
        if new_end - new_start >= MIN_SHOT_S:
            moments = tuple(
                m for c, m in self._moments if c == shot.clip and new_start <= m.time < new_end
            )
            retimed = dataclasses.replace(shot, start=new_start, end=new_end, moments=moments)
            self._replace_at(index, retimed)

    def replace(self, index: int, candidate: Shot) -> None:
        """Swap the shot at ``index`` for ``candidate`` (it must fit beside the others)."""
        if any(overlaps(candidate, s) for i, s in enumerate(self._shots) if i != index):
            raise ValueError("that shot overlaps another shot of the same clip")
        self._replace_at(index, candidate)

    def add(self, candidate: Shot) -> int:
        """Add ``candidate`` after the shots that come before it in clip and time order;
        returns its position."""
        if any(overlaps(candidate, s) for s in self._shots):
            raise ValueError("that shot overlaps another shot of the same clip")
        key = (candidate.clip, candidate.start)
        position = sum(1 for s in self._shots if (s.clip, s.start) < key)
        shots = list(self._shots)
        shots.insert(position, candidate)
        self._set(shots)
        return position

    def _replace_at(self, index: int, shot: Shot) -> None:
        self._set(shot if i == index else s for i, s in enumerate(self._shots))

    def alternatives(self, index: int | None = None, limit: int = 12) -> list[Shot]:
        """Candidates that fit beside the current shots, best first, without near-copies.

        With ``index``: those from the same clip as that shot, which could replace it.
        Without: those from every clip, which could be added.
        """
        current = self._shots[index] if index is not None else None
        kept = [s for i, s in enumerate(self._shots) if i != index]
        pool = [
            c
            for c in self._candidates
            if (current is None or c.clip == current.clip)
            and c != current
            and not any(overlaps(c, s) for s in kept)
            and (current is None or not _mostly_same(c, current))
        ]
        chosen: list[Shot] = []
        for candidate in sorted(pool, key=lambda c: (-strength(c), c.clip, c.start)):
            if any(overlaps(candidate, c) for c in chosen):
                continue  # candidates overlap each other a lot; show each moment once
            chosen.append(candidate)
            if len(chosen) == limit:
                break
        return chosen


def _mostly_same(a: Shot, b: Shot) -> bool:
    shared = min(a.end, b.end) - max(a.start, b.start)
    return a.clip == b.clip and shared > 0.5 * min(a.duration, b.duration)


def shot_starts(shots: Sequence[Shot]) -> list[float]:
    """Where each shot starts on the montage's own timeline (seconds from its start)."""
    starts: list[float] = []
    total = 0.0
    for shot in shots:
        starts.append(total)
        total += shot.duration
    return starts


def locate(shots: Sequence[Shot], seconds: float) -> tuple[int, float]:
    """The shot playing ``seconds`` into the montage and the matching time in its clip.

    Times before the start give the first shot's start; times at or after the end give
    the last shot's end. Raises ValueError for an empty montage.
    """
    if not shots:
        raise ValueError("the montage has no shots")
    for index, start in enumerate(shot_starts(shots)):
        shot = shots[index]
        if seconds < start + shot.duration:
            return index, shot.start + max(0.0, seconds - start)
    return len(shots) - 1, shots[-1].end


def montage_time(shots: Sequence[Shot], index: int, clip_seconds: float) -> float:
    """``clip_seconds`` in shot ``index`` as seconds into the montage."""
    shot = shots[index]
    offset = min(max(clip_seconds, shot.start), shot.end) - shot.start
    return shot_starts(shots)[index] + offset


def drop_position(dragged: int, gap: int) -> int:
    """`ShotList.move` target for a shot at ``dragged`` dropped into ``gap`` (0 = before
    the first shot, n = after the last)."""
    return gap if gap <= dragged else gap - 1


def rating(shot: Shot, montage: Sequence[Shot]) -> int:
    """1-5 stars for ``shot``: its own rating (`rating.py`) when it has one; otherwise
    where its strength falls among the shots of ``montage`` (one shot alone gets 3)."""
    if shot.quality is not None:
        return shot.quality.stars
    others = [strength(s) for s in montage if s != shot]
    if not others:
        return 3
    value = strength(shot)
    below = sum(1 for o in others if o < value) + 0.5 * sum(1 for o in others if o == value)
    return min(5, 1 + int(5 * below / len(others)))
