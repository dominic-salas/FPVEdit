"""Saved flights keep their order, never count twice, and cheating does not count."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fpvedit.history import FlightHistory, leaderboard, summarise
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


def test_leaderboard_ranks_only_flights_that_count(tmp_path: Path) -> None:
    history = FlightHistory(":memory:")
    spam = replace(GOOD, freestyle=99.0, turns=TURNS, top_share=1.0)
    history.record(
        [
            (clip(tmp_path, "a.mp4", b"1"), GOOD),
            (clip(tmp_path, "b.mp4", b"2"), replace(GOOD, freestyle=70.0, flow=20.0)),
            (clip(tmp_path, "c.mp4", b"3"), spam),
        ]
    )
    flights = history.flights()
    assert [f.name for f in leaderboard(flights, "freestyle")] == ["b.mp4", "a.mp4"]
    assert [f.name for f in leaderboard(flights, "flow")] == ["a.mp4", "b.mp4"]


def test_an_old_database_gains_the_longest_line_column_and_keeps_its_rows(tmp_path: Path) -> None:
    import sqlite3

    from fpvedit.history import SCHEMA

    database = tmp_path / "old.db"
    old = sqlite3.connect(str(database))
    old.execute(SCHEMA.replace(",\n    longest_line_s REAL", ""))
    old.execute(
        "INSERT INTO flights (key, name, batch, version, status, conditions, flying_s, freestyle, "
        "flow, difficulty, variety, linking, continuity, smoothness, crashes, turns, disqualified) "
        "VALUES ('k', 'old.mp4', 1, 1, 'scored', 'daylight', 60, 40, 50, 1, 1, 1, 1, 1, 0, 3, '')"
    )
    old.commit()
    old.close()
    history = FlightHistory(database)
    (flight,) = history.flights()
    assert flight.name == "old.mp4" and flight.freestyle == 40.0
    assert flight.longest_line_s is None


def test_a_longest_line_changes_nothing_else_about_the_flight(tmp_path: Path) -> None:
    history = FlightHistory(":memory:")
    spam = replace(GOOD, freestyle=99.0, turns=TURNS, top_share=1.0)
    paths = [clip(tmp_path, "a.mp4", b"1"), clip(tmp_path, "b.mp4", b"2")]
    history.record([(paths[0], GOOD), (paths[1], spam)])
    before = history.flights()
    history.set_longest_line(paths[0], 42.5)
    history.set_longest_line(paths[1], 80.0)
    history.set_longest_line(clip(tmp_path, "never.mp4", b"3"), 5.0)
    after = history.flights()
    assert [f.longest_line_s for f in after] == [42.5, 80.0]
    assert [replace(f, longest_line_s=None) for f in after] == before
    assert [f.counts for f in after] == [True, False]


def test_best_line_skips_disqualified_flights_and_is_none_without_values(tmp_path: Path) -> None:
    from fpvedit.history import best_line

    history = FlightHistory(":memory:")
    spam = replace(GOOD, turns=TURNS, top_share=1.0)
    paths = [clip(tmp_path, "a.mp4", b"1"), clip(tmp_path, "b.mp4", b"2")]
    history.record([(paths[0], GOOD), (paths[1], spam)])
    assert best_line(history.flights()) is None
    history.set_longest_line(paths[0], 30.0)
    history.set_longest_line(paths[1], 90.0)
    assert best_line(history.flights()) == 30.0
