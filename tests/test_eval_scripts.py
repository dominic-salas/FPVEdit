"""Tests for the evaluation scripts (scripts/evaluate.py and scripts/match_edit.py)."""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from fpv_editor.media import MediaInfo, VideoStream

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load("evaluate")
match_edit = _load("match_edit")


@pytest.fixture
def moments(tmp_path: Path) -> Path:
    path = tmp_path / "flight.json"
    path.write_text(
        json.dumps(
            {
                "flight": "flight",
                "duration_s": 100,
                "events": [
                    {"start": "0:10", "end": "0:20", "label": "flip", "keep": "include"},
                    {"start": 50, "end": 60, "label": "crash", "keep": "exclude"},
                ],
            }
        )
    )
    return path


def test_random_edits_keep_segment_lengths_inside_the_flight() -> None:
    intervals = [(5.0, 15.0), (40.0, 43.0), (70.0, 90.0)]
    rng = random.Random(3)  # noqa: S311
    for _ in range(50):
        edit = evaluate.random_edit(intervals, 100.0, rng)
        assert sorted(round(b - a, 6) for a, b in edit) == [3.0, 10.0, 20.0]
        assert all(0.0 <= a < b <= 100.0 for a, b in edit)
        assert all(b1 <= a2 for (_, b1), (a2, _) in itertools.pairwise(edit))


def test_a_perfect_edit_beats_random_edits(moments: Path) -> None:
    beaten = evaluate.compare_with_random(moments, [(9.0, 21.0)], trials=200, seed=1)
    assert beaten["event_recall"] > 0.8
    assert beaten["exclude_time_kept"] >= 0.5


def test_scores_for_moments(moments: Path) -> None:
    summary = evaluate.evaluate_intervals(moments, [(9.0, 21.0), (55.0, 58.0)])
    assert summary["event_recall"] == 1.0
    assert summary["exclude_time_kept"] == pytest.approx(0.3)


def test_matching_finds_hard_cuts_and_offsets() -> None:
    rate = 10.0
    rng = np.random.default_rng(0)
    raw_a = rng.standard_normal((300, 16)).astype(np.float32)
    raw_b = rng.standard_normal((200, 16)).astype(np.float32)
    raws = [raw_a / np.linalg.norm(raw_a, axis=1, keepdims=True)]
    raws.append(raw_b / np.linalg.norm(raw_b, axis=1, keepdims=True))
    # Edit: raw A frames 50-99, then raw B frames 120-159, then raw A frames 200-229.
    edit = np.concatenate([raws[0][50:100], raws[1][120:160], raws[0][200:230]])
    which, frame, score = match_edit.best_matches(edit, raws)
    segments = match_edit.segments_from_matches(which, frame, score, rate)
    found = [(s.raw, round(s.raw_start, 2), round(s.raw_end, 2)) for s in segments]
    assert found == [(0, 5.0, 10.0), (1, 12.0, 16.0), (0, 20.0, 23.0)]
    assert all(s.mean_correlation > 0.99 for s in segments)


def _segment(raw_start: float, raw_end: float, correlation: float = 0.99) -> object:
    return match_edit.Segment(0, 0.0, raw_end - raw_start, raw_start, raw_end, correlation)


def test_annotation_marks_weak_matches_as_unknown(tmp_path: Path) -> None:
    raw = MediaInfo(
        path=tmp_path / "flight.ts",
        format_name="mpegts",
        duration=100.0,
        start_time=0.0,
        video=VideoStream(0, "hevc", 1280, 720, Fraction(1), 60.0),
        audio=None,
    )
    edit = MediaInfo(raw.path.with_name("edit.mp4"), "mp4", 20.0, 0.0, raw.video, None)
    document = match_edit.annotation(
        raw, [_segment(10, 20)], edit, 60.0, unsure=[_segment(50, 52, correlation=0.5)]
    )
    events = [(e["start"], e["end"], e["keep"]) for e in document["events"]]
    assert events == [
        (0.0, 10.0, "editor_cut"),
        (10.0, 20.0, "include"),
        (20.0, 50.0, "editor_cut"),
        (50.0, 52.0, "ignore"),
        (52.0, 100.0, "editor_cut"),
    ]
    assert document["complete"] is False


def test_only_reference_edits_count_as_replaceable(tmp_path: Path) -> None:
    hand = tmp_path / "hand.json"
    hand.write_text(json.dumps({"flight": "x", "events": []}))
    reference = tmp_path / "ref.json"
    reference.write_text(json.dumps({"kind": "reference_edit", "events": []}))
    assert not match_edit._is_reference_edit(hand)
    assert match_edit._is_reference_edit(reference)


def test_ignored_time_does_not_count_against_a_reference_edit(tmp_path: Path) -> None:
    path = tmp_path / "ref.json"
    path.write_text(
        json.dumps(
            {
                "flight": "ref",
                "duration_s": 100,
                "kind": "reference_edit",
                "events": [
                    {"start": 0, "end": 40, "label": "editor_cut", "keep": "editor_cut"},
                    {"start": 40, "end": 60, "label": "editor_kept", "keep": "include"},
                    {"start": 60, "end": 70, "label": "unmatched", "keep": "ignore"},
                    {"start": 70, "end": 100, "label": "editor_cut", "keep": "editor_cut"},
                ],
            }
        )
    )
    summary = evaluate.evaluate_intervals(path, [(40.0, 70.0)])
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["iou"] == pytest.approx(1.0)
