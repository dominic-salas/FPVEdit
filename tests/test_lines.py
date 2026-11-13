"""Longest lines on synthetic signals: a `CutMap` built by hand, or a `Timeline` run through
the real flight rules."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np

from flight_fixtures import FPS, Timeline
from fpvedit.flight import detect_flight
from fpvedit.highlights import CutReason, SelectionMode, select_highlights
from fpvedit.lines import (
    Line,
    LineEnd,
    LinesConfig,
    find_lines,
    no_lines_message,
    pick_lines,
)
from fpvedit.montage import CutMap, Moment, MomentKind, MontageConfig, cut_map, find_moments
from fpvedit.rating import ClipSignals

CONFIG = LinesConfig(fast_flying=False)
NO_SNAP = MontageConfig(snap_s=0.0)  # line ends land exactly on the lead and tail


def flip(at: float) -> Moment:
    return Moment(at, MomentKind.FLIP, 3.0, "flip")


def cuts_for(
    seconds: float,
    *,
    unusable: tuple[tuple[float, float], ...] = (),
    crashes: tuple[float, ...] = (),
    speed: float = 0.0,
    rate: float = 0.0,
    fps: float = FPS,
) -> CutMap:
    count = round(seconds * fps)
    usable = np.ones(count, dtype=bool)
    for start, end in unusable:
        usable[round(start * fps) : round(end * fps)] = False
    signals = ClipSignals(
        fps=fps,
        rate=np.full(count, rate),
        rotation=np.zeros((count, 3)),
        speed=np.full(count, speed),
        approach=np.zeros(count),
        confidence=np.full(count, 0.9),
        brightness=np.zeros(count),
        crashes=crashes,
    )
    return CutMap(usable, np.ones(count), np.zeros(count, dtype=bool), signals)


def lines_of(cuts: CutMap, moments: list[Moment], config: LinesConfig = CONFIG) -> tuple[Line, ...]:
    return find_lines(0, cuts, moments, config, NO_SNAP)


def every(first: float, last: float, step: float = 1.0) -> list[Moment]:
    return [flip(first + i * step) for i in range(round((last - first) / step) + 1)]


def line(clip: int, start: float, length: float, events: int = 0, active: float = 0.0) -> Line:
    return Line(clip, start, start + length, events, active, LineEnd.LULL)


# ---- the line's ends ------------------------------------------------------------------


def test_a_crash_keeps_the_flying_up_to_the_crash_cut() -> None:
    cuts = cuts_for(60, unusable=((40, 60),), crashes=(40.0,))
    (found,) = lines_of(cuts, [*every(10, 36), flip(38.5)])
    assert found.ended_by is LineEnd.CRASH
    assert (found.start, found.end) == (8.5, 40.0)  # the last trick is kept, the impact is not
    assert found.events == 28


def test_a_failed_move_before_a_crash_does_not_shorten_the_line() -> None:
    timeline = Timeline(100).armed_message(2)
    for at in range(10, 57, 3):  # the last flips fall in the 3.5-8 s before the disarm
        timeline.flip(at)
    timeline.tumble(58.5, 60).stats(60, 64).armed_message(70).stats(95, 98)
    measurements = timeline.measurements()
    selection = select_highlights(measurements, detect_flight(measurements))
    assert any(c.reason is CutReason.FAILED_MOVE for c in selection.cuts)
    cuts = cut_map(measurements, selection)
    moments = find_moments(measurements, cuts.usable)
    first = find_lines(0, cuts, moments, CONFIG)[0]
    assert first.ended_by is LineEnd.CRASH
    crash = next(c.start for c in selection.cuts if c.reason is CutReason.CRASH)
    assert first.end == crash  # up to the impact ...
    failed = next(c for c in selection.cuts if c.reason is CutReason.FAILED_MOVE)
    assert first.end > failed.start  # ... which is past where the failed move begins
    assert any(failed.start <= m.time < failed.end for m in first.moments)


def test_a_landing_ends_a_line() -> None:
    cuts = cuts_for(60, unusable=((40, 60),))
    (found,) = lines_of(cuts, every(10, 39))
    assert found.ended_by is LineEnd.LANDED and found.end == 40.0


def test_a_lull_ends_a_line_and_the_end_comes_a_second_after_the_last_trick() -> None:
    (found,) = lines_of(cuts_for(80), every(10, 20))
    assert found.ended_by is LineEnd.LULL
    assert (found.start, found.end) == (8.5, 21.5)  # a second around the active frames


def test_the_end_of_the_recording_ends_a_line() -> None:
    (found,) = lines_of(cuts_for(30), every(10, 29.5, 1.5))
    assert found.ended_by is LineEnd.RECORDING_END and found.end == 30.0


def test_lead_and_tail_stop_at_the_recording_s_ends() -> None:
    (found,) = lines_of(cuts_for(14), every(0.3, 13.5, 1.0))
    assert (found.start, found.end) == (0.0, 14.0)


def test_a_line_does_not_start_inside_footage_the_flight_rules_cut() -> None:
    cuts = cuts_for(60, unusable=((0, 20),))
    (found,) = lines_of(cuts, every(20.2, 40))
    assert found.start == 20.0  # the lead stops at the end of the cut


def test_every_end_reason_is_reachable() -> None:
    reasons = set()
    for cuts in (
        cuts_for(60, unusable=((30, 60),), crashes=(30.0,)),
        cuts_for(60, unusable=((30, 60),)),
        cuts_for(80),
        cuts_for(30),
    ):
        reasons |= {found.ended_by for found in lines_of(cuts, every(12, 29))}
    assert reasons == set(LineEnd)


# ---- chaining -------------------------------------------------------------------------


def test_an_8_5_s_lull_splits_a_line_and_a_7_5_s_lull_does_not() -> None:
    def moments(gap: float) -> list[Moment]:
        second = 20.5 + gap + 0.5  # the pause is between the two spans of activity
        return [*every(10, 20, 1.0), *every(second, second + 10, 1.0)]

    assert len(lines_of(cuts_for(80), moments(8.5))) == 2
    (joined,) = lines_of(cuts_for(80), moments(7.5))
    assert joined.start == 8.5 and joined.end == 40.0


def test_a_cut_is_never_bridged_by_a_short_gap() -> None:
    cuts = cuts_for(80, unusable=((25, 28),))  # 3 s the flight rules cut
    first, second = lines_of(cuts, [*every(10, 24, 1.0), *every(29, 43, 1.0)])
    assert first.ended_by is LineEnd.LANDED and first.end == 25.0
    assert second.start == 28.0


def test_a_quick_landing_and_re_arm_does_not_split_a_line() -> None:
    timeline = Timeline(90).armed_message(2).still(28, 30).stats(30, 31).armed_message(31.5)
    timeline.stats(85, 88)
    for at in (10, 13, 16, 19, 22, 25, 33, 36, 39, 42):
        timeline.flip(at)
    measurements = timeline.measurements()
    selection = select_highlights(measurements, detect_flight(measurements))
    assert selection.mode is SelectionMode.FLIGHT_STATE
    cuts = cut_map(measurements, selection)
    assert cuts.usable[round(29 * FPS)]  # the selection keeps the quick landing as flight
    found = find_lines(0, cuts, find_moments(measurements, cuts.usable), CONFIG)
    assert len(found) == 1 and found[0].start < 10 and found[0].end > 42


def test_a_re_arm_churn_stretch_ends_a_line() -> None:
    timeline = (
        Timeline(100)
        .armed_message(2)
        .stats(20, 22)
        .armed_message(24)
        .stats(27, 29)
        .armed_message(31)
        .stats(33, 35)
        .armed_message(37)
        .stats(95, 98)
    )
    for at in (6, 9, 12, 15, 18, 55, 58, 61, 64, 67, 70):
        timeline.flip(at)
    measurements = timeline.measurements()
    selection = select_highlights(measurements, detect_flight(measurements))
    assert any(c.reason is CutReason.REARM_CHURN for c in selection.cuts)
    cuts = cut_map(measurements, selection)
    found = find_lines(0, cuts, find_moments(measurements, cuts.usable), CONFIG)
    assert found[0].ended_by is LineEnd.LANDED
    assert found[0].end <= 24 and found[-1].start > 45  # nothing is bridged over the churn


# ---- what makes a line -----------------------------------------------------------------


def test_a_line_needs_10_s() -> None:
    assert lines_of(cuts_for(60), every(10, 16)) == ()  # 9 s long
    assert len(lines_of(cuts_for(60), every(10, 17))) == 1  # 10 s: enough


def test_pauses_up_to_the_gap_keep_one_line_however_calm_it_is_between() -> None:
    # Flips 7 s apart over a long flight: mostly calm flying, but never 8 s of nothing,
    # so it is one line (a real street flight read 86 s active in a 193 s line).
    sparse = [flip(t) for t in range(5, 200, 7)]
    (found,) = lines_of(cuts_for(210), sparse)
    assert found.start < 5 and found.end > 194 and found.events == len(sparse)


def test_a_longer_gap_never_loses_a_line() -> None:
    moments = [flip(10), flip(16), flip(22), flip(40), flip(47), flip(52)]
    short = find_lines(0, cuts_for(60), moments, LinesConfig(), NO_SNAP)
    long = find_lines(0, cuts_for(60), moments, LinesConfig(gap_s=20.0), NO_SNAP)
    covered = sum(line.duration for line in short)
    assert long and sum(line.duration for line in long) >= covered


def test_fast_flying_alone_makes_a_line_unless_it_is_switched_off() -> None:
    cuts = cuts_for(60, speed=1.0, rate=100.0)
    (found,) = find_lines(0, cuts, [], LinesConfig(), NO_SNAP)
    assert found.events == 0 and found.start == 0.0 and found.end == 60.0
    assert find_lines(0, cuts, [], LinesConfig(fast_flying=False), NO_SNAP) == ()


def test_cruising_is_not_fast_flying() -> None:
    assert find_lines(0, cuts_for(60, speed=0.3, rate=20.0), [], LinesConfig(), NO_SNAP) == ()
    blurred = dataclasses.replace(
        cuts_for(60, speed=1.0, rate=100.0).signals, confidence=np.full(1200, 0.1)
    )
    cuts = dataclasses.replace(cuts_for(60), signals=blurred)
    assert find_lines(0, cuts, [], LinesConfig(), NO_SNAP) == ()


def test_no_usable_frames_give_no_lines() -> None:
    cuts = cuts_for(60, unusable=((0, 60),))
    assert find_lines(0, cuts, [], LinesConfig(), NO_SNAP) == ()
    assert find_lines(0, cuts, every(10, 30), LinesConfig(), NO_SNAP) == ()


def test_short_input_gives_no_lines() -> None:
    timeline = Timeline(5).armed_message(0.5).flip(2.5)
    measurements = timeline.measurements()
    selection = select_highlights(measurements, detect_flight(measurements))
    assert selection.mode is SelectionMode.SHORT_INPUT
    cuts = cut_map(measurements, selection)
    assert find_lines(0, cuts, find_moments(measurements, cuts.usable), LinesConfig()) == ()


def test_a_recording_without_an_osd_follows_the_same_rules() -> None:
    timeline = Timeline(70, osd=False)
    for at in range(10, 28, 2):
        timeline.flip(at)
    timeline.tumble(28.5, 30).still(30, 30.8)  # a crash without an OSD: a dead stop
    measurements = timeline.measurements()
    selection = select_highlights(measurements, detect_flight(measurements))
    assert selection.mode is SelectionMode.NO_OSD
    cuts = cut_map(measurements, selection)
    found = find_lines(0, cuts, find_moments(measurements, cuts.usable), CONFIG)
    assert found and found[0].ended_by is LineEnd.CRASH
    assert found[0].end <= 30.0


def test_lead_and_tail_are_set_by_the_config() -> None:
    config = LinesConfig(fast_flying=False, lead_s=2.0, tail_s=0.5)
    (found,) = find_lines(0, cuts_for(80), every(10, 20, 1.0), config, NO_SNAP)
    assert (found.start, found.end) == (7.5, 21.0)


def test_line_ends_snap_to_calm_frames_and_never_into_a_turn() -> None:
    cuts = cuts_for(80)
    calm = np.ones(len(cuts.usable))
    calm[round(8.9 * FPS)] = 0.0  # a calmer frame inside the start's reach
    turning = np.zeros(len(cuts.usable), dtype=bool)
    turning[round(21.25 * FPS) : round(22.0 * FPS)] = True  # where the end would be, 21.5
    cuts = dataclasses.replace(cuts, calm=calm, turning=turning)
    (found,) = find_lines(0, cuts, every(10, 20, 1.0), CONFIG, MontageConfig())
    assert found.start == 8.9
    assert not turning[round(found.end * FPS)]
    assert 20.75 <= found.end <= 22.25  # within the snap distance of 21.5 s


def test_events_and_active_time_count_what_the_line_shows() -> None:
    (found,) = lines_of(cuts_for(60), [*every(10, 20, 1.0), Moment(40.0, MomentKind.SPIN, 1.5)])
    assert found.events == 11 and len(found.moments) == 11
    assert 10.9 < found.active_s < 11.2  # one second per trick, overlapping


# ---- choosing lines ---------------------------------------------------------------------


def test_the_order_is_whole_seconds_then_events_then_active_time() -> None:
    a = line(0, 0, 28.8, events=8, active=20)
    b = line(0, 100, 29.4, events=5, active=20)
    c = line(0, 200, 30.0, events=2, active=20)
    # 30.0 s, 29.4 s and 28.8 s are 30, 29 and 28 whole seconds: that order, not by events.
    picked = pick_lines([a, b, c])
    assert set(picked) == {a, b, c}
    d = line(0, 300, 30.9, events=1, active=10)  # also 30 whole seconds
    e = line(0, 400, 30.2, events=3, active=10)  # 30 s with more events beats d
    f = line(0, 500, 30.5, events=3, active=12)  # ... and more active time beats e
    assert pick_lines([d, e, f], LinesConfig(per_recording=1)) == (f,)
    assert pick_lines([d, e], LinesConfig(per_recording=1)) == (e,)
    assert pick_lines([c, b, a], LinesConfig(per_recording=1)) == (c,)
    # the top line is the first of the order, so a shorter one cannot take its place by events
    assert pick_lines([a, b], LinesConfig(per_recording=1)) == (b,)


def test_a_runner_up_needs_80_percent_of_the_best_lines_length() -> None:
    best = line(0, 0, 50)
    close = line(0, 100, 40)  # exactly 80%
    short = line(0, 200, 39.9)
    assert pick_lines([short, close, best]) == (best, close)


def test_at_most_three_lines_per_recording() -> None:
    lines = [line(0, 100 * i, 50 - i) for i in range(5)]
    assert pick_lines(lines) == tuple(lines[:3])


def test_each_recording_keeps_its_own_best_and_the_montage_order_is_the_input_order() -> None:
    a1, a2 = line(1, 0, 40), line(1, 100, 15)
    b1 = line(0, 50, 12)
    assert pick_lines([a1, a2, b1]) == (b1, a1)  # recording 1's 15 s is under 80% of its 40 s


def test_a_cleaned_copy_and_its_original_give_one_line_from_the_newer_file() -> None:
    moments = tuple(flip(t) for t in (12.0, 20.0, 30.0))
    original = Line(0, 10.0, 40.0, 3, 20.0, LineEnd.CRASH, moments)
    shifted = tuple(flip(t + 0.3) for t in (12.0, 20.0, 30.0))
    copy = Line(1, 10.3, 40.2, 3, 20.0, LineEnd.CRASH, shifted)
    assert pick_lines([original, copy], modified={0: 100.0, 1: 200.0}) == (copy,)
    assert pick_lines([original, copy], modified={0: 200.0, 1: 100.0}) == (original,)
    assert pick_lines([original, copy], modified={0: 100.0, 1: 100.0}) == (original,)


def test_different_recordings_of_the_same_length_keep_their_own_lines() -> None:
    first = Line(0, 10.0, 40.0, 3, 20.0, LineEnd.LULL, tuple(flip(t) for t in (12, 20, 30)))
    other = Line(1, 55.0, 85.0, 3, 20.0, LineEnd.LULL, tuple(flip(t) for t in (57, 65, 75)))
    assert pick_lines([first, other]) == (first, other)
    # the same times but other tricks are not a copy either
    unlike = Line(1, 10.0, 40.0, 3, 20.0, LineEnd.LULL, tuple(flip(t) for t in (12, 25, 31)))
    assert pick_lines([first, unlike]) == (first, unlike)


def test_a_line_never_crosses_a_recording() -> None:
    first = find_lines(3, cuts_for(60), every(10, 30), CONFIG, NO_SNAP)
    second = find_lines(4, cuts_for(60), every(30, 50), CONFIG, NO_SNAP)
    assert {f.clip for f in first} == {3} and {f.clip for f in second} == {4}
    # Back to back in time, but two recordings: two lines, each on its own.
    assert [f.clip for f in pick_lines([*first, *second])] == [3, 4]


def test_lines_survive_pickling_and_the_empty_message_follows_the_config() -> None:
    (found,) = lines_of(cuts_for(60), every(10, 30))
    assert pickle.loads(pickle.dumps(found)) == found  # noqa: S301 (our own bytes)
    assert "fast flying" in no_lines_message(LinesConfig())
    assert "fast flying" not in no_lines_message(LinesConfig(fast_flying=False))
    assert no_lines_message().startswith("No lines found. A line needs 10 s or more of")
