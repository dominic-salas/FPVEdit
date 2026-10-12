from __future__ import annotations

import numpy as np
import pytest

from fpv_editor.errors import NoFlightContentError
from fpv_editor.highlights import (
    SelectionConfig,
    SelectionMode,
    clean_idle,
    dilate,
    extend_short_runs,
    fill_short_gaps,
    finalize,
    hysteresis,
    mask_to_intervals,
    median_filter,
    remove_short_runs,
    runs,
    select_highlights,
)

FPS = 4.0


def mask(text: str) -> np.ndarray:
    return np.array([c == "1" for c in text], dtype=np.bool_)


def text(values: np.ndarray) -> str:
    return "".join("1" if v else "0" for v in values)


def test_runs_finds_half_open_ranges() -> None:
    assert runs(mask("0110011")) == [(1, 3), (5, 7)]
    assert runs(mask("000")) == []
    assert runs(mask("1")) == [(0, 1)]


def test_remove_short_runs() -> None:
    assert text(remove_short_runs(mask("1101110"), 3)) == "0001110"


def test_fill_short_gaps_only_between_runs() -> None:
    assert text(fill_short_gaps(mask("0110011000"), 3)) == "0111111000"
    assert text(fill_short_gaps(mask("1100011"), 3)) == "1100011"


def test_dilate_is_clipped_to_bounds() -> None:
    assert text(dilate(mask("0010000"), 1, 2)) == "0111100"
    assert text(dilate(mask("1000001"), 2, 2)) == "1110111"


def test_median_filter_removes_single_sample_flicker() -> None:
    assert text(median_filter(mask("1111011110000100"), 3)) == "1111111110000000"


def test_extend_short_runs_grows_symmetrically_within_bounds() -> None:
    assert text(extend_short_runs(mask("0000100000"), 5)) == "0011111000"
    assert text(extend_short_runs(mask("1000000000"), 4)) == "1111000000"


def test_hysteresis_needs_a_peak_and_holds_until_low() -> None:
    score = np.array([0.1, 0.45, 0.6, 0.45, 0.42, 0.3, 0.45, 0.48, 0.2])
    assert text(hysteresis(score, high=0.55, low=0.4)) == "011110000"


def test_hysteresis_does_not_flicker_around_a_single_threshold() -> None:
    score = np.array([0.0, 0.52, 0.49, 0.51, 0.48, 0.53, 0.0])
    assert text(hysteresis(score, high=0.5, low=0.45)) == "0111110"


def test_clean_idle_ignores_short_pauses() -> None:
    config = SelectionConfig(min_idle_s=2.0, idle_median_s=0.0)
    short_pause = mask("0000" + "1111" + "0000")  # 1 s at 4 fps
    long_pause = mask("0000" + "1" * 12 + "0000")  # 3 s
    assert not clean_idle(short_pause, FPS, config).any()
    assert clean_idle(long_pause, FPS, config).sum() == 12


def test_finalize_adds_context_but_only_a_little_into_idle() -> None:
    config = SelectionConfig(
        lead_in_s=1.0, aftermath_s=1.0, idle_pad_s=0.5, merge_gap_s=0.0, min_segment_s=0.0
    )
    idle = mask("1" * 8 + "0" * 12 + "1" * 8)
    keep = mask("0" * 12 + "1111" + "0" * 12)
    result = finalize(keep, idle, FPS, config)
    # 1 s of non-idle context each side, then 0.5 s (2 samples) into idle.
    assert text(result) == "0" * 6 + "1" * 16 + "0" * 6


def test_select_keeps_all_flight_and_cuts_idle() -> None:
    raw_idle = mask("1" * 40 + "0" * 40 + "1" * 40)  # 10 s idle, 10 s flying, 10 s idle
    score = np.where(raw_idle, 0.0, 0.5)
    selection = select_highlights(score, raw_idle, FPS, SelectionConfig())
    assert selection.mode is SelectionMode.ALL_CONTENT
    assert len(selection.intervals) == 1
    interval = selection.intervals[0]
    assert interval.start == pytest.approx(10.0 - SelectionConfig().idle_pad_s)
    assert interval.end == pytest.approx(20.0 + SelectionConfig().idle_pad_s)


def test_select_trims_least_interesting_flight_to_soft_target() -> None:
    n = 240  # 60 s of flying, no idle
    raw_idle = np.zeros(n, dtype=np.bool_)
    score = np.full(n, 0.2)
    score[40:80] = 0.8  # two clear highlights
    score[160:200] = 0.7
    selection = select_highlights(score, raw_idle, FPS, SelectionConfig())
    assert selection.mode is SelectionMode.TRIMMED
    assert selection.retention <= SelectionConfig().max_retention
    kept = np.zeros(n, dtype=np.bool_)
    for interval in selection.intervals:
        kept[round(interval.start * FPS) : round(interval.end * FPS)] = True
    assert kept[40:80].all() and kept[160:200].all()


def test_select_keeps_everything_when_score_is_flat() -> None:
    n = 240
    selection = select_highlights(
        np.full(n, 0.4), np.zeros(n, dtype=np.bool_), FPS, SelectionConfig()
    )
    assert selection.mode is SelectionMode.FLAT_SCORE
    assert selection.retention == pytest.approx(1.0)
    assert selection.warnings


def test_select_keeps_short_inputs_whole() -> None:
    selection = select_highlights(np.zeros(8), np.ones(8, dtype=np.bool_), FPS, SelectionConfig())
    assert selection.mode is SelectionMode.SHORT_INPUT
    assert selection.intervals[0].start == 0.0
    assert selection.intervals[0].end == pytest.approx(2.0)


def test_select_refuses_all_idle_recordings() -> None:
    with pytest.raises(NoFlightContentError):
        select_highlights(np.zeros(100), np.ones(100, dtype=np.bool_), FPS, SelectionConfig())


def test_mask_to_intervals_converts_to_seconds() -> None:
    intervals = mask_to_intervals(mask("0011100011"), FPS)
    assert [(i.start, i.end) for i in intervals] == [(0.5, 1.25), (2.0, 2.5)]


def test_select_rejects_non_finite_scores() -> None:
    score = np.full(100, 0.5)
    score[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        select_highlights(score, np.zeros(100, dtype=np.bool_), FPS, SelectionConfig())
