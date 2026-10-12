"""Choose which parts of the flight to keep, as a list of time intervals.

Selection works on boolean masks over the analysis samples, which keeps every step
(idle cleanup, hysteresis, context padding, gap merging, minimum length) small and
independently testable. The policy is conservative:

1. Idle or blank stretches of at least ``min_idle_s`` are cut.
2. Everything else is kept, with short context into idle stretches at the edges.
3. Only if that keeps more than ``max_retention`` of the flight, and the interest score
   actually varies, are the least interesting stretches trimmed, using hysteresis so
   the cut points do not flicker. The retention bound is a soft target: when the
   score is flat (low confidence), nothing more is cut and a warning is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from fpv_editor.errors import NoFlightContentError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SelectionMode(Enum):
    SHORT_INPUT = "short-input"
    ALL_CONTENT = "all-content"
    FLAT_SCORE = "flat-score"
    TRIMMED = "trimmed"


@dataclass(frozen=True)
class SelectionConfig:
    min_idle_s: float = 2.0
    idle_median_s: float = 1.0
    idle_pad_s: float = 0.5
    lead_in_s: float = 1.5
    aftermath_s: float = 1.5
    merge_gap_s: float = 3.0
    min_segment_s: float = 3.0
    max_retention: float = 0.65
    flat_score_spread: float = 0.08
    hysteresis_gap: float = 0.05
    short_input_s: float = 6.0


@dataclass(frozen=True)
class Selection:
    intervals: tuple[Interval, ...]
    idle: tuple[Interval, ...]
    mode: SelectionMode
    retention: float
    threshold: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def kept_duration(self) -> float:
        return sum(interval.duration for interval in self.intervals)


def runs(mask: BoolArray) -> list[tuple[int, int]]:
    """Half-open index ranges [start, end) of consecutive True values."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def remove_short_runs(mask: BoolArray, min_length: int) -> BoolArray:
    result = mask.copy()
    for start, end in runs(mask):
        if end - start < min_length:
            result[start:end] = False
    return result


def fill_short_gaps(mask: BoolArray, max_gap: int) -> BoolArray:
    """Set False gaps shorter than ``max_gap`` that lie between True runs to True."""
    result = mask.copy()
    kept = runs(mask)
    for (_, previous_end), (next_start, _) in pairwise(kept):
        if next_start - previous_end < max_gap:
            result[previous_end:next_start] = True
    return result


def dilate(mask: BoolArray, before: int, after: int) -> BoolArray:
    result = mask.copy()
    for start, end in runs(mask):
        result[max(0, start - before) : min(len(mask), end + after)] = True
    return result


def median_filter(mask: BoolArray, window: int) -> BoolArray:
    """Majority vote over an odd window, repeating edge values."""
    if window <= 1 or len(mask) == 0:
        return mask.copy()
    window += 1 - window % 2
    half = window // 2
    padded = np.pad(mask.astype(np.int32), half, mode="edge")
    sums = np.convolve(padded, np.ones(window, dtype=np.int32), mode="valid")
    return np.asarray(sums > half, dtype=np.bool_)


def hysteresis(score: FloatArray, high: float, low: float) -> BoolArray:
    """Keep stretches where the score stays >= low and somewhere reaches >= high."""
    candidate = score >= low
    result = np.zeros_like(candidate)
    for start, end in runs(candidate):
        if np.any(score[start:end] >= high):
            result[start:end] = True
    return result


def extend_short_runs(mask: BoolArray, min_length: int) -> BoolArray:
    """Grow runs shorter than ``min_length`` symmetrically (clipped to the bounds)."""
    result = mask.copy()
    n = len(mask)
    for start, end in runs(mask):
        if end - start >= min_length:
            continue
        new_start = max(0, start - (min_length - (end - start)) // 2)
        new_end = min(n, new_start + min_length)
        new_start = max(0, new_end - min_length)
        result[new_start:new_end] = True
    return result


def clean_idle(raw_idle: BoolArray, fps: float, config: SelectionConfig) -> BoolArray:
    smoothed = median_filter(raw_idle, round(config.idle_median_s * fps))
    return remove_short_runs(smoothed, round(config.min_idle_s * fps))


def finalize(keep: BoolArray, idle: BoolArray, fps: float, config: SelectionConfig) -> BoolArray:
    """Add context around kept stretches, merge small gaps, enforce a minimum length."""
    context = dilate(keep, round(config.lead_in_s * fps), round(config.aftermath_s * fps))
    padded = keep | (context & ~idle)
    pad = round(config.idle_pad_s * fps)
    padded = dilate(padded, pad, pad)
    merged = fill_short_gaps(padded, round(config.merge_gap_s * fps))
    return extend_short_runs(merged, round(config.min_segment_s * fps))


def mask_to_intervals(mask: BoolArray, fps: float) -> tuple[Interval, ...]:
    return tuple(Interval(start / fps, end / fps) for start, end in runs(mask))


def select_highlights(
    score: FloatArray, raw_idle: BoolArray, fps: float, config: SelectionConfig
) -> Selection:
    n = len(score)
    if not np.isfinite(score).all():
        raise ValueError("interest score contains NaN or infinite values")
    if n == 0:
        raise NoFlightContentError("no analysis samples")
    everything = np.ones(n, dtype=np.bool_)

    if n / fps < config.short_input_s:
        return Selection(
            intervals=mask_to_intervals(everything, fps),
            idle=(),
            mode=SelectionMode.SHORT_INPUT,
            retention=1.0,
            warnings=(f"input is shorter than {config.short_input_s:g} s; kept all of it",),
        )

    idle = clean_idle(raw_idle, fps, config)
    if idle.all():
        raise NoFlightContentError("the recording contains only idle or blank footage")
    active = ~idle
    base = finalize(active, idle, fps, config)
    idle_intervals = mask_to_intervals(idle, fps)
    base_retention = float(base.mean())

    if base_retention <= config.max_retention:
        return Selection(
            intervals=mask_to_intervals(base, fps),
            idle=idle_intervals,
            mode=SelectionMode.ALL_CONTENT,
            retention=base_retention,
        )

    active_scores = score[active]
    spread = float(np.subtract(*np.percentile(active_scores, [75, 25])))
    if spread < config.flat_score_spread:
        return Selection(
            intervals=mask_to_intervals(base, fps),
            idle=idle_intervals,
            mode=SelectionMode.FLAT_SCORE,
            retention=base_retention,
            warnings=(
                f"interest score is nearly flat (spread {spread:.3f}); kept all flight "
                f"footage ({base_retention:.0%}) rather than cutting on weak evidence",
            ),
        )

    threshold, trimmed = _trim_to_retention(score, active, idle, fps, config)
    retention = float(trimmed.mean())
    warnings: tuple[str, ...] = ()
    if retention > config.max_retention + 0.05:
        warnings = (f"kept {retention:.0%}, above the {config.max_retention:.0%} soft target",)
    return Selection(
        intervals=mask_to_intervals(trimmed, fps),
        idle=idle_intervals,
        mode=SelectionMode.TRIMMED,
        retention=retention,
        threshold=threshold,
        warnings=warnings,
    )


def _trim_to_retention(
    score: FloatArray, active: BoolArray, idle: BoolArray, fps: float, config: SelectionConfig
) -> tuple[float, BoolArray]:
    """Lowest hysteresis threshold whose final selection keeps <= max_retention."""
    candidates = np.unique(np.percentile(score[active], np.arange(0, 100, 2.5)))
    best: tuple[float, BoolArray] | None = None
    low, high = 0, len(candidates) - 1
    while low <= high:
        middle = (low + high) // 2
        threshold = float(candidates[middle])
        keep = hysteresis(score, threshold, threshold - config.hysteresis_gap) & active
        selected = finalize(keep, idle, fps, config)
        if selected.mean() <= config.max_retention:
            best = (threshold, selected)
            high = middle - 1
        else:
            low = middle + 1
    if best is None:
        threshold = float(candidates[-1])
        keep = hysteresis(score, threshold, threshold - config.hysteresis_gap) & active
        best = (threshold, finalize(keep, idle, fps, config))
    return best
