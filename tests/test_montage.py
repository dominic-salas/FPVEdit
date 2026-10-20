from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from flight_fixtures import FPS, Timeline
from fpvedit.errors import NoFlightContentError
from fpvedit.flight import detect_flight
from fpvedit.highlights import Selection, select_highlights
from fpvedit.measure import Measurements
from fpvedit.montage import (
    MomentKind,
    MontageConfig,
    Shot,
    candidate_shots,
    find_moments,
    pick_shots,
    usable_footage,
)
from fpvedit.motion import CONFIDENCE, DIVERGENCE, PITCH

CONFIG = MontageConfig()


def analyse(timeline: Timeline) -> tuple[Measurements, Selection]:
    measurements = timeline.measurements()
    return measurements, select_highlights(measurements, detect_flight(measurements))


def flight(seconds: float = 120, *, seed: int = 0) -> Timeline:
    """Armed at 2 s, disarmed (landing) at the end, calm cruising in between."""
    return Timeline(seconds, seed=seed).armed_message(2).stats(seconds - 5, seconds)


def shot(clip: int, start: float, end: float, score: float) -> Shot:
    return Shot(clip, start, end, score, ())


def test_a_flip_is_a_moment_and_calm_cruising_is_not() -> None:
    timeline = flight()
    timeline.motion[:, DIVERGENCE] = 0.2  # no close passes
    measurements, selection = analyse(timeline.flip(40))
    moments = find_moments(measurements, usable_footage(measurements, selection))
    flips = [m for m in moments if m.kind is MomentKind.FLIP]
    assert len(flips) == 1 and abs(flips[0].time - 40.25) < 0.5
    assert flips[0].weight == CONFIG.flip_weight + CONFIG.big_flip_weight
    assert all(abs(m.time - 40) < 3 for m in moments)  # nothing in the calm cruising


def test_a_sudden_approach_is_a_pass_but_steady_closeness_is_not() -> None:
    timeline = flight()
    timeline.motion[:, DIVERGENCE] = 0.2
    timeline.motion[round(30 * FPS) : round(50 * FPS), DIVERGENCE] = 2.5  # a long hallway
    timeline.motion[round(80 * FPS) : round(80.5 * FPS), DIVERGENCE] = 2.5  # a gap
    measurements, selection = analyse(timeline)
    moments = find_moments(measurements, usable_footage(measurements, selection))
    passes = [m.time for m in moments if m.kind is MomentKind.PROXIMITY]
    assert passes and all(abs(t - 80.25) < 0.5 or 29 < t < 31 or 49 < t < 51 for t in passes)
    assert not any(32 < t < 48 for t in passes)
    assert any(abs(t - 80.25) < 0.5 for t in passes)


def test_nothing_is_usable_outside_clean_flying() -> None:
    measurements, selection = analyse(flight().still(10, 20).tumble(60, 61).stats(61, 62))
    usable = usable_footage(measurements, selection)
    for t in (1, 15, 60.5, 118):  # before arming, on the ground, crash, stats screen
        assert not usable[round(t * FPS)], t
    assert usable[round(40 * FPS)]


def test_short_input_has_no_shots() -> None:
    # Armed and flipping, but only 5 s long: too short for any shot.
    measurements, selection = analyse(Timeline(5).armed_message(0.5).flip(2.5))
    assert not usable_footage(measurements, selection).any()
    assert candidate_shots(0, measurements, selection) == []


def test_shots_are_long_enough_and_show_the_flip_whole() -> None:
    measurements, selection = analyse(flight().flip(40).flip(80))
    shots = candidate_shots(0, measurements, selection)
    assert shots
    usable = usable_footage(measurements, selection)
    for s in shots:
        assert CONFIG.shot_min_s - 1e-9 <= s.duration <= CONFIG.shot_max_s + 1e-9
        assert usable[round(s.start * FPS) : round(s.end * FPS)].all()
        for moment in s.moments:
            assert s.start + CONFIG.edge_s <= moment.time < s.end - CONFIG.tail_s
    flips = [s for s in shots if any(m.kind is MomentKind.FLIP for m in s.moments)]
    assert {round(m.time) for s in flips for m in s.moments if m.kind is MomentKind.FLIP} == {
        40,
        80,
    }
    # A shot opens on the set-up: the flip is never in its first second.
    assert all(m.time - s.start >= 1.0 for s in flips for m in s.moments)


def test_a_flip_that_crashes_gives_no_shot() -> None:
    timeline = flight().flip(20).flip(58)
    timeline.tumble(59, 60).stats(60, 62).armed_message(70)
    measurements, selection = analyse(timeline)
    shots = candidate_shots(0, measurements, selection)
    assert any(s.start <= 19 and s.end >= 21 for s in shots)  # the clean flip is a shot
    assert not any(s.start < 60 and s.end > 56 for s in shots)  # the crashing one is not


def test_a_short_flight_between_crashes_is_one_whole_shot() -> None:
    timeline = flight()
    timeline.tumble(28.5, 30).stats(30, 31).armed_message(31.5)  # crash, quick re-arm
    timeline.flip(34).tumble(36.5, 38).stats(38, 39).armed_message(39.5)  # 6 s, crash
    measurements, selection = analyse(timeline)
    shots = [s for s in candidate_shots(0, measurements, selection) if 31 < s.start < 38]
    assert len(shots) == 1
    assert shots[0].start <= 33 and shots[0].end >= 34.5  # the whole flip is in it ...
    assert shots[0].end < 37  # ... and the impact is not


def test_a_quad_on_the_desk_gives_no_shots() -> None:
    timeline = Timeline(60, osd=False).still(0, 60)
    timeline.motion[:, DIVERGENCE] = np.linspace(0, 3, timeline.count)  # noise, not passes
    measurements = timeline.measurements()
    with pytest.raises(NoFlightContentError):
        select_highlights(measurements, detect_flight(measurements))


def test_pick_stops_at_the_target_length_best_first() -> None:
    candidates = [shot(0, 10 * i, 10 * i + 10, score=i) for i in range(20)]
    config = dataclasses.replace(CONFIG, length_s=30, length_slack_s=0, min_gap_s=0)
    picked = pick_shots(candidates, 1, config)
    assert [s.start for s in picked] == [170, 180, 190]


def test_pick_never_overlaps_and_keeps_a_gap_within_a_clip() -> None:
    candidates = [shot(0, 0, 10, 9), shot(0, 5, 15, 8), shot(0, 12, 22, 7), shot(0, 30, 40, 1)]
    picked = pick_shots(candidates, 1, CONFIG)
    assert [(s.start, s.end) for s in picked] == [(0, 10), (30, 40)]


def test_pick_shares_the_montage_between_clips_in_clip_order() -> None:
    great = [shot(1, 20 * i, 20 * i + 10, score=10) for i in range(10)]
    fine = [shot(0, 20 * i, 20 * i + 10, score=1) for i in range(10)]
    config = dataclasses.replace(CONFIG, length_s=60, length_slack_s=0)
    picked = pick_shots(great + fine, 2, config)
    per_clip = {c: sum(s.duration for s in picked if s.clip == c) for c in (0, 1)}
    assert per_clip == {0: 30, 1: 30}
    assert [s.clip for s in picked] == [0, 0, 0, 1, 1, 1]
    assert [s.start for s in picked[:3]] == sorted(s.start for s in picked[:3])


def test_a_lone_clip_may_fill_the_whole_montage() -> None:
    candidates = [shot(1, 20 * i, 20 * i + 10, score=1) for i in range(10)]
    config = dataclasses.replace(CONFIG, length_s=60, length_slack_s=0)
    assert sum(s.duration for s in pick_shots(candidates, 3, config)) == 60


def test_without_a_length_every_distinct_shot_is_kept() -> None:
    candidates = [shot(0, 0, 10, 9), shot(0, 5, 15, 8), shot(1, 0, 10, 0.1), shot(0, 40, 50, 1)]
    picked = pick_shots(candidates, 2, dataclasses.replace(CONFIG, length_s=None))
    assert [(s.clip, s.start) for s in picked] == [(0, 0), (0, 40), (1, 0)]


def test_pick_handles_many_clips() -> None:
    candidates = [
        shot(c, 20 * i, 20 * i + 10, score=(c * 7 + i) % 11) for c in range(60) for i in range(5)
    ]
    config = dataclasses.replace(CONFIG, length_s=90)
    picked = pick_shots(candidates, 60, config)
    assert 90 - config.length_slack_s <= sum(s.duration for s in picked)
    assert sum(s.duration for s in picked) <= 90 + config.length_slack_s
    assert [(s.clip, s.start) for s in picked] == sorted((s.clip, s.start) for s in picked)


def test_no_shot_starts_or_ends_in_a_flip() -> None:
    # Flips close together, so many candidate ends would land in one.
    timeline = flight()
    for at in (30, 38, 39.5, 46, 47.2, 55, 63.5, 64.6, 72):
        timeline.flip(at)
    measurements, selection = analyse(timeline)
    shots = candidate_shots(0, measurements, selection)
    assert shots
    flipping = np.abs(measurements.motion[:, 0]) > 360  # PITCH: the flips
    for candidate in shots:
        assert 8 <= candidate.duration <= 15
        for edge in (candidate.start, candidate.end):
            near = flipping[max(0, round((edge - 0.4) * FPS)) : round((edge + 0.4) * FPS)]
            assert not near.any(), f"a cut at {edge:.2f} s is in or next to a flip"


def test_no_shot_starts_or_ends_in_a_blurred_flip() -> None:
    # Real flips blur: the motion fit stays on but with low confidence and noisy rates
    # (measured on Dom's clips: under 0.1 % of frames are lost outright, 2-12 % are low
    # confidence). Flips must still be found and cuts must still avoid them.
    timeline = flight()
    flips = (30, 38, 46, 55, 63.5, 72)
    rng = np.random.default_rng(1)
    for at in flips:
        timeline.flip(at)
        span = slice(round(at * FPS), round((at + 0.5) * FPS))
        timeline.motion[span, PITCH] += rng.normal(0, 150, span.stop - span.start)
        timeline.motion[span, CONFIDENCE] = 0.15
    measurements, selection = analyse(timeline)
    shots = candidate_shots(0, measurements, selection)
    found = {round(m.time) for s in shots for m in s.moments if m.kind is MomentKind.FLIP}
    assert found == {round(at) for at in flips}
    for candidate in shots:
        for edge in (candidate.start, candidate.end):
            for at in flips:
                assert not (at - 0.4 < edge < at + 0.9), (
                    f"a cut at {edge:.2f} s cuts the flip at {at}"
                )
