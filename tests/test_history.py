"""Saved flights keep their order, never count twice, and cheating does not count."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fpvedit.history import FlightHistory, summarise
from fpvedit.scoring import SCORER_VERSION, FlightScore, Status, Turn, disqualify

TURNS = tuple(Turn(float(i), 240.0, "roll+", 1.0) for i in range(10))
GOOD = FlightScore(Status.SCORED, SCORER_VERSION, 60.0, "daylight", freestyle=40.0, flow=50.0)


def clip(folder: Path, name: str, body: bytes) -> Path:
    path = folder / name
    path.write_bytes(body)
    return path


def test_same_content_under_a_new_name_counts_once(tmp_path: Path) -> None:
    history = FlightHistory(":memory:")
    assert history.record([(clip(tmp_path, "a.mp4", b"one"), GOOD)]) == 1
    assert history.record([(clip(tmp_path, "b.mp4", b"one"), GOOD)]) == 0
    assert len(history.flights()) == 1


def test_order_is_first_scored_then_file_name(tmp_path: Path) -> None:
    history = FlightHistory(":memory:")
    history.record([(clip(tmp_path, "b.mp4", b"2"), GOOD), (clip(tmp_path, "a.mp4", b"1"), GOOD)])
    history.record([(clip(tmp_path, "0.mp4", b"3"), GOOD)])
    assert [f.name for f in history.flights()] == ["a.mp4", "b.mp4", "0.mp4"]


def test_first_import_is_the_baseline(tmp_path: Path) -> None:
    history = FlightHistory(":memory:")
    history.record([(clip(tmp_path, "a.mp4", b"1"), GOOD)])
    history.record([(clip(tmp_path, "b.mp4", b"2"), replace(GOOD, freestyle=60.0))])
    summary = summarise(history.flights())
    assert summary is not None
    assert summary.baseline == 1
    assert summary.start_freestyle == 40.0


def test_disqualified_flight_leaves_the_trend_until_counted(tmp_path: Path) -> None:
    history = FlightHistory(":memory:")
    spam = replace(GOOD, turns=TURNS, top_share=1.0)
    history.record([(clip(tmp_path, "a.mp4", b"1"), spam)])
    flight = history.flights()[0]
    assert flight.disqualified and summarise(history.flights()) is None
    history.count_anyway(flight.id, True)
    assert summarise(history.flights()) is not None


def test_disqualify_rules() -> None:
    assert disqualify(GOOD, []) is None
    assert disqualify(replace(GOOD, pace=30.0), []) is not None
    past = [30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 31.0, 33.0, 35.0, 37.0]
    assert disqualify(replace(GOOD, freestyle=95.0), past) is not None
    assert disqualify(GOOD, past) is None
