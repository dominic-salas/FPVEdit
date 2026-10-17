"""Editing rules of the app's shot list (no Qt needed)."""

from __future__ import annotations

import pytest

from fpv_editor.app.model import MIN_SHOT_S, ShotList
from fpv_editor.montage import Moment, MomentKind, Shot


def shot(clip: int, start: float, end: float, score: float = 5.0) -> Shot:
    return Shot(clip, start, end, score, (Moment(start + 2, MomentKind.FLIP, 3.0),))


A, B, C = shot(0, 10, 20), shot(0, 40, 50), shot(1, 5, 15)
DURATIONS = {0: 100.0, 1: 60.0}


def make(*candidates: Shot) -> ShotList:
    return ShotList([A, B, C], [A, B, C, *candidates], DURATIONS)


def test_remove_and_undo() -> None:
    shots = make()
    shots.remove(1)
    assert list(shots.shots) == [A, C]
    assert shots.duration == pytest.approx(20)
    shots.undo()
    assert list(shots.shots) == [A, B, C]
    assert not shots.can_undo


def test_move_and_reorder() -> None:
    shots = make()
    shots.move(2, 0)
    assert list(shots.shots) == [C, A, B]
    shots.reorder([2, 1, 0])
    assert list(shots.shots) == [B, A, C]
    with pytest.raises(ValueError, match="every shot once"):
        shots.reorder([0, 0, 1])


def test_trim_moves_ends_and_drops_moments_outside() -> None:
    shots = make()
    shots.trim(0, start=3.0, end=1.0)
    trimmed = shots.shots[0]
    assert (trimmed.start, trimmed.end) == (13.0, 21.0)
    assert trimmed.moments == ()  # the flip at 12 s is no longer shown


def test_trim_stops_at_the_clip_and_at_neighbouring_shots() -> None:
    shots = make()
    shots.trim(0, start=-50.0)  # clip start
    assert shots.shots[0].start == 0.0
    shots.trim(0, end=30.0)  # stops where the next shot of clip 0 starts
    assert shots.shots[0].end == B.start
    shots.trim(2, end=100.0)  # clip 1 is 60 s long
    assert shots.shots[2].end == 60.0


def test_trim_never_leaves_a_too_short_shot() -> None:
    shots = make()
    shots.trim(0, start=9.5)
    assert shots.shots[0] == A
    shots.trim(0, start=10 - MIN_SHOT_S)
    assert shots.shots[0].duration == pytest.approx(MIN_SHOT_S)


def test_replace_and_add_refuse_overlaps_within_a_clip() -> None:
    other = shot(0, 60, 70)
    shots = make(other)
    with pytest.raises(ValueError, match="overlaps"):
        shots.replace(0, shot(0, 45, 55))
    shots.replace(0, other)
    assert list(shots.shots) == [other, B, C]
    with pytest.raises(ValueError, match="overlaps"):
        shots.add(shot(1, 10, 20))
    position = shots.add(shot(1, 30, 40))
    assert position == 3
    assert list(shots.shots)[3] == shot(1, 30, 40)


def test_add_goes_in_clip_and_time_order() -> None:
    shots = make()
    assert shots.add(shot(0, 25, 35)) == 1
    assert shots.add(shot(1, 0, 4)) == 3


def test_alternatives_fit_and_are_not_near_copies() -> None:
    near_copy = shot(0, 11, 21, score=9.0)  # mostly the same footage as A
    fits = shot(0, 60, 70, score=4.0)
    overlaps_b = shot(0, 45, 55, score=8.0)
    overlapping_each_other = shot(0, 62, 72, score=3.0)
    shots = make(near_copy, fits, overlaps_b, overlapping_each_other)
    assert shots.alternatives(0) == [fits]
    added = shots.alternatives()
    assert near_copy not in added  # overlaps A, already in the montage
    assert added == [fits]


def test_extending_a_trimmed_shot_shows_its_moment_again() -> None:
    shots = make()
    shots.trim(0, start=3.0)
    assert shots.shots[0].moments == ()
    shots.trim(0, start=-3.0)
    assert shots.shots[0].moments == A.moments
