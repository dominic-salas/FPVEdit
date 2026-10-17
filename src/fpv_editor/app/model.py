"""The montage being edited in the app: its shots in playing order, and the edits on them.

No Qt here, so the editing rules are tested like the rest of the engine. Shots from one
clip never overlap (each moment is shown once); any other order or length is the user's
choice, since they are the editor. Every edit can be undone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from fpv_editor.montage import Shot

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

    def reorder(self, order: Sequence[int]) -> None:
        """Put the shots in this order (a permutation of their positions)."""
        if sorted(order) != list(range(len(self._shots))):
            raise ValueError("the new order must list every shot once")
        self._set(self._shots[i] for i in order)

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
        for candidate in sorted(pool, key=lambda c: (-c.score / c.duration**0.5, c.clip, c.start)):
            if any(overlaps(candidate, c) for c in chosen):
                continue  # candidates overlap each other a lot; show each moment once
            chosen.append(candidate)
            if len(chosen) == limit:
                break
        return chosen


def _mostly_same(a: Shot, b: Shot) -> bool:
    shared = min(a.end, b.end) - max(a.start, b.start)
    return a.clip == b.clip and shared > 0.5 * min(a.duration, b.duration)
