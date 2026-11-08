"""The flight score rewards what it claims to and refuses what it cannot judge."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fpvedit.rating import ClipSignals
from fpvedit.scoring import (
    ScoringConfig,
    Status,
    crash_count,
    find_turns,
    score_flight,
    sparc,
)

FPS = 30.0


def signals(seconds: float = 60.0, turns: tuple[tuple[float, int, float], ...] = ()) -> ClipSignals:
    """Steady forward flight with half-second turns: (time, axis, signed deg/s)."""
    count = round(seconds * FPS)
    rotation = np.zeros((count, 3))
    for time, axis, rate in turns:
        start = round(time * FPS)
        rotation[start : start + round(0.5 * FPS), axis] = rate
    return ClipSignals(
        fps=FPS,
        rate=np.linalg.norm(rotation, axis=1) + 40.0,
        rotation=rotation,
        speed=np.full(count, 0.5),
        approach=np.zeros(count),
        confidence=np.full(count, 0.9),
        brightness=np.full(count, 0.6),
        crashes=(),
    )


def usable(clip: ClipSignals) -> np.ndarray:
    return np.ones(len(clip.rate), dtype=bool)


VARIED = tuple((5.0 + 5.0 * i, i % 3, 500.0 if i % 2 else -500.0) for i in range(10))
SAME = tuple((5.0 + 5.0 * i, 2, 500.0) for i in range(10))


def test_same_measurements_give_the_same_score() -> None:
    clip = signals(turns=VARIED)
    assert score_flight(clip, usable(clip)) == score_flight(clip, usable(clip))


def test_short_recording_gets_no_score() -> None:
    clip = signals(seconds=20.0)
    assert score_flight(clip, usable(clip)).status is Status.TOO_SHORT


def test_poor_tracking_gets_no_score() -> None:
    clip = replace(signals(), confidence=np.full(round(60 * FPS), 0.1))
    assert score_flight(clip, usable(clip)).status is Status.LOW_TRACKING


def test_unusable_frames_do_not_count_as_turns() -> None:
    clip = signals(turns=VARIED)
    assert (
        find_turns(clip.rotation, np.zeros(len(clip.rate), dtype=bool), FPS, ScoringConfig()) == ()
    )


def test_more_turns_score_higher() -> None:
    few, many = signals(turns=VARIED[:3]), signals(turns=VARIED)
    assert score_flight(many, usable(many)).freestyle > score_flight(few, usable(few)).freestyle


def test_repeating_one_turn_is_worth_less_than_varied_turns() -> None:
    same, varied = signals(turns=SAME), signals(turns=VARIED)
    a, b = score_flight(same, usable(same)), score_flight(varied, usable(varied))
    assert sum(t.value for t in a.turns) < sum(t.value for t in b.turns)
    assert a.variety == 0.0 < b.variety
    assert a.freestyle < b.freestyle


def test_linked_turns_are_counted() -> None:
    spaced = signals(turns=VARIED)
    linked = signals(turns=tuple((5.0 + 1.5 * i, i % 3, 500.0) for i in range(10)))
    assert score_flight(spaced, usable(spaced)).linking == 0.0
    assert score_flight(linked, usable(linked)).linking == 100.0


def test_crashes_lower_flow() -> None:
    clean = signals(turns=VARIED)
    crashed = replace(clean, crashes=(10.0, 20.0))
    before, after = score_flight(clean, usable(clean)), score_flight(crashed, usable(crashed))
    assert after.crashes == 2
    assert after.flow == pytest.approx(before.flow - 2 * ScoringConfig().crash_penalty)


def test_hovering_lowers_continuity() -> None:
    moving = signals(turns=VARIED)
    hovering = replace(moving, rate=np.zeros(len(moving.rate)), speed=np.zeros(len(moving.rate)))
    assert score_flight(hovering, usable(hovering)).continuity == 0.0
    assert score_flight(moving, usable(moving)).continuity > 90.0


def test_long_cruising_lowers_continuity() -> None:
    busy, cruising = signals(turns=VARIED), signals(turns=VARIED[:2])
    a, b = score_flight(busy, usable(busy)), score_flight(cruising, usable(cruising))
    assert b.continuity < a.continuity


def test_crashes_close_together_count_once() -> None:
    assert crash_count((10.0, 12.0, 15.0, 40.0), ScoringConfig()) == 2
    assert crash_count((), ScoringConfig()) == 0


def test_sparc_prefers_one_smooth_movement_to_a_shaky_one() -> None:
    time = np.linspace(0.0, 2.0, 60, endpoint=False)
    smooth = np.sin(np.pi * time / 2.0) ** 2
    shaky = smooth * (1.0 + 0.5 * np.sin(2 * np.pi * 7.0 * time))
    assert sparc(smooth, FPS) > sparc(shaky, FPS)


def test_sparc_of_no_movement_is_zero() -> None:
    assert sparc(np.zeros(60), FPS) == pytest.approx(0.0)
