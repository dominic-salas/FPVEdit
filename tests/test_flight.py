from __future__ import annotations

import pytest

from flight_fixtures import LOGO_BANDS, Timeline
from fpv_editor.flight import Screen, armed_intervals, detect_flight, runs


# The flicker filter widens the layout by a frame, so screen edges move by one frame.
def flat(spans: tuple[tuple[float, float], ...]) -> list[float]:
    return [t for span in spans for t in span]


def kinds(timeline: Timeline) -> list[Screen]:
    return [e.kind for e in detect_flight(timeline.measurements()).events]


def test_runs_finds_half_open_ranges() -> None:
    import numpy as np

    assert runs(np.array([0, 1, 1, 0, 0, 1, 1], dtype=bool)) == [(1, 3), (5, 7)]
    assert runs(np.zeros(3, dtype=bool)) == []


def test_armed_from_armed_message_to_stats_screen() -> None:
    timeline = Timeline(60).clear(0, 2).armed_message(5).stats(40, 45)
    flight = detect_flight(timeline.measurements())
    assert flight.has_osd
    assert [e.kind for e in flight.events] == [Screen.BOOT, Screen.ARMED, Screen.STATS]
    assert flat(flight.armed) == pytest.approx([5.5, 40.0], abs=0.1)
    assert [e.start for e in flight.disarms] == pytest.approx([40.0], abs=0.1)


def test_missed_arm_is_recovered_from_the_next_stats_screen() -> None:
    timeline = Timeline(60).armed_message(5).stats(20, 24).stats(50, 54)
    flight = detect_flight(timeline.measurements())
    assert flat(flight.armed) == pytest.approx([5.5, 20.0, 24.0, 50.0], abs=0.1)


def test_arming_logo_is_not_a_disarm() -> None:
    timeline = (
        Timeline(60)
        .clear(0, 2, LOGO_BANDS)  # boot logo
        .clear(2.1, 5, LOGO_BANDS)  # straight after the boot screen: still booting
        .clear(10, 11.5, LOGO_BANDS)  # arming logo ...
        .clear(11.5, 12)  # ... then the ARMED message
        .stats(40, 45)
    )
    flight = detect_flight(timeline.measurements())
    assert [e.kind for e in flight.events] == [
        Screen.BOOT,
        Screen.BOOT,
        Screen.LOGO,
        Screen.ARMED,
        Screen.STATS,
    ]
    assert flat(flight.armed) == pytest.approx([12.0, 40.0], abs=0.1)


def test_arming_logo_without_armed_message() -> None:
    timeline = Timeline(60).clear(0, 2, LOGO_BANDS).clear(10, 12, LOGO_BANDS).stats(40, 45)
    flight = detect_flight(timeline.measurements())
    assert flat(flight.armed) == pytest.approx([12.0, 40.0], abs=0.1)


def test_partial_layout_dip_does_not_end_the_flight() -> None:
    timeline = Timeline(60).armed_message(5).dip(20, 25).stats(40, 45)
    flight = detect_flight(timeline.measurements())
    assert kinds(timeline) == [Screen.ARMED, Screen.STATS]
    assert flat(flight.armed) == pytest.approx([5.5, 40.0], abs=0.1)


def test_static_and_grey_screen_are_marked_without_changing_armed_state() -> None:
    timeline = Timeline(60).armed_message(5).static(20, 22).grey(30, 40).stats(50, 52)
    flight = detect_flight(timeline.measurements())
    static = [e for e in flight.events if e.kind is Screen.STATIC]
    assert len(static) == 2
    assert static[0].start == pytest.approx(20, abs=0.2)
    assert static[0].end == pytest.approx(22, abs=0.2)
    assert static[1].start < 31 and static[1].end > 39
    assert flat(flight.armed) == pytest.approx([5.5, 50.0], abs=0.1)
    assert not flight.static[(10 * 20)]


def test_without_osd_the_whole_recording_counts_as_armed() -> None:
    flight = detect_flight(Timeline(30, osd=False).measurements())
    assert not flight.has_osd
    assert flight.armed == ((0.0, 30.0),)
    assert flight.events == ()


def test_recording_that_starts_mid_flight() -> None:
    assert armed_intervals([], 30.0) == [(0.0, 30.0)]
    flight = detect_flight(Timeline(60).stats(40, 45).measurements())
    assert flat(flight.armed) == pytest.approx([0.0, 40.0], abs=0.1)


def test_never_armed_after_boot_screen() -> None:
    flight = detect_flight(Timeline(30).clear(0, 2).measurements())
    assert flight.armed == ()
