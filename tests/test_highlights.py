from __future__ import annotations

import numpy as np
import pytest

from flight_fixtures import Timeline
from fpvedit.errors import NoFlightContentError
from fpvedit.flight import detect_flight
from fpvedit.highlights import (
    CutReason,
    Selection,
    SelectionMode,
    mask_to_intervals,
    select_highlights,
    window_sum,
)
from fpvedit.motion import PITCH, ROLL, SPEED


def select(timeline: Timeline) -> Selection:
    measurements = timeline.measurements()
    return select_highlights(measurements, detect_flight(measurements))


def kept(selection: Selection, t: float) -> bool:
    return any(i.start <= t < i.end for i in selection.intervals)


def cut_reason(selection: Selection, t: float) -> CutReason | None:
    return next((c.reason for c in selection.cuts if c.start <= t < c.end), None)


def test_window_sum_and_intervals() -> None:
    assert window_sum(np.ones(5), 3).tolist() == [2, 3, 3, 3, 2]
    intervals = mask_to_intervals(np.array([0, 1, 1, 0, 1], dtype=bool), 2.0)
    assert [(i.start, i.end) for i in intervals] == [(0.5, 1.5), (2.0, 2.5)]


def test_short_input_is_kept_whole() -> None:
    selection = select(Timeline(4))
    assert selection.mode is SelectionMode.SHORT_INPUT
    assert selection.retention == 1.0


def test_keeps_armed_flight_only() -> None:
    selection = select(Timeline(60).clear(0, 2).armed_message(5).stats(50, 55))
    assert selection.mode is SelectionMode.FLIGHT_STATE
    assert not kept(selection, 1) and not kept(selection, 3) and not kept(selection, 52)
    assert kept(selection, 6) and kept(selection, 49)
    assert cut_reason(selection, 3) is CutReason.NOT_FLYING
    assert cut_reason(selection, 57) is CutReason.NOT_FLYING


def test_short_clear_screen_while_armed_is_not_downtime() -> None:
    selection = select(Timeline(60).armed_message(5).clear(25, 25.5).stats(50, 55))
    assert kept(selection, 25.2)


def test_long_static_is_cut_and_a_blip_is_kept() -> None:
    selection = select(Timeline(60).armed_message(2).static(20, 22).static(30, 30.4))
    assert cut_reason(selection, 21) is CutReason.STATIC
    assert kept(selection, 30.2)


def test_sitting_on_the_ground_is_cut_up_to_lift_off() -> None:
    selection = select(Timeline(60).armed_message(2).still(2.5, 10).stats(50, 55))
    assert cut_reason(selection, 6) is CutReason.ON_GROUND
    assert kept(selection, 9.8)


def test_a_calm_hover_is_not_the_ground() -> None:
    timeline = Timeline(60).armed_message(2).stats(50, 55)
    hover = slice(25 * 20, 27 * 20)
    timeline.motion[hover, PITCH : ROLL + 1] = 3.0  # hardly turning or moving ...
    timeline.motion[hover, SPEED] = 0.05  # ... but the picture still shakes
    selection = select(timeline)
    assert kept(selection, 26)


def test_crash_is_cut_with_the_failed_move_before_it() -> None:
    timeline = (
        Timeline(60).armed_message(2).tumble(18.5, 20).stats(20, 24).armed_message(30).stats(55, 58)
    )
    selection = select(timeline)
    assert cut_reason(selection, 19) is CutReason.CRASH
    assert cut_reason(selection, 17) is CutReason.FAILED_MOVE
    assert not any(kept(selection, t) for t in (17, 19, 22, 27, 30.2))
    assert kept(selection, 12) and kept(selection, 35)


def test_a_crash_ends_at_the_re_arm_and_the_next_flight_stays() -> None:
    # Crash, re-armed at once, then a whole flight that ends in a landing.
    timeline = (
        Timeline(60)
        .armed_message(2)
        .tumble(18.5, 20)
        .stats(20, 21)
        .armed_message(21.5)
        .still(32, 33)
        .stats(33, 35)
        .armed_message(36)
        .stats(55, 58)
    )
    selection = select(timeline)
    assert cut_reason(selection, 20.5) is not None
    assert kept(selection, 27)


def test_a_re_arm_that_soon_disarms_again_is_part_of_the_crash() -> None:
    # Crash, re-arm, turtle mode and a second disarm, then flying on.
    timeline = (
        Timeline(60)
        .armed_message(2)
        .tumble(18.5, 20)
        .stats(20, 21)
        .armed_message(21.5)
        .stats(24, 25)
        .armed_message(25.5)
        .stats(55, 58)
    )
    selection = select(timeline)
    assert not kept(selection, 23)
    assert kept(selection, 35)


def test_gentle_landing_is_not_a_crash() -> None:
    selection = select(Timeline(60).armed_message(2).still(37, 40).stats(40, 45))
    assert not any(c.reason is CutReason.CRASH for c in selection.cuts)
    assert kept(selection, 34)


def test_quick_landing_and_rearm_stays_in_the_edit() -> None:
    timeline = (
        Timeline(60).armed_message(2).still(18, 20).stats(20, 21).armed_message(21.5).stats(55, 58)
    )
    selection = select(timeline)
    assert all(kept(selection, t) for t in (17, 19, 20.5, 21.7, 23))


def test_repeated_disarms_are_cut_as_a_whole() -> None:
    timeline = (
        Timeline(70)
        .armed_message(2)
        .stats(20, 22)
        .armed_message(24)
        .stats(27, 29)
        .armed_message(31)
        .stats(33, 35)
        .armed_message(37)
        .stats(65, 68)
    )
    selection = select(timeline)
    assert cut_reason(selection, 32.5) is CutReason.REARM_CHURN
    assert cut_reason(selection, 25.5) is CutReason.REARM_CHURN
    assert kept(selection, 10) and kept(selection, 45)


def test_dull_stretch_is_cut_but_a_flip_protects_it() -> None:
    dull = select(Timeline(60).armed_message(2).cruise_straight(10, 32))
    assert all(cut_reason(dull, t) is CutReason.DULL for t in (15, 20, 25))
    assert kept(dull, 6) and kept(dull, 40)

    flipped = select(Timeline(60).armed_message(2).cruise_straight(10, 32).flip(21))
    assert not any(c.reason is CutReason.DULL for c in flipped.cuts)
    assert kept(flipped, 21.2)


def test_without_osd_only_flying_rules_apply() -> None:
    selection = select(Timeline(40, osd=False))
    assert selection.mode is SelectionMode.NO_OSD
    assert selection.warnings
    assert not any(c.reason is CutReason.NOT_FLYING for c in selection.cuts)
    assert selection.retention > 0.9


def test_without_arm_events_a_quad_on_a_desk_gives_nothing() -> None:
    timeline = Timeline(30, osd=False)
    timeline.motion[:, :3] = 4.0  # barely turning: carried, or a hand in front of it
    with pytest.raises(NoFlightContentError):
        select(timeline)


def test_nothing_worth_keeping_raises() -> None:
    with pytest.raises(NoFlightContentError):
        select(Timeline(30).still(0, 30))


def test_without_osd_a_tumble_into_a_dead_stop_is_a_crash() -> None:
    # A crash without an OSD: tumbling, then lying dead still for under a second.
    timeline = Timeline(60, osd=False).tumble(28.5, 30).still(30, 30.8)
    selection = select(timeline)
    assert cut_reason(selection, 29) is CutReason.CRASH
    assert cut_reason(selection, 27) is CutReason.FAILED_MOVE
    assert kept(selection, 15) and kept(selection, 45)


def test_without_osd_a_flip_into_a_hover_is_not_a_crash() -> None:
    timeline = Timeline(60, osd=False).flip(29)
    hover = slice(30 * 20, 31 * 20)
    timeline.motion[hover, PITCH : ROLL + 1] = 3.0  # calm, but the picture still moves
    timeline.motion[hover, SPEED] = 0.12
    selection = select(timeline)
    assert not any(c.reason is CutReason.CRASH for c in selection.cuts)
    assert kept(selection, 29)
