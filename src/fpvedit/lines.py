"""Longest lines: a pilot's unbroken runs of flying where something keeps happening.

A line is one stretch of usable flying (`montage.CutMap.usable`: the montage's mask, which
keeps the failed move just before a crash) in which a flip, a spin, a close pass or fast
flying keeps happening. Runs of that activity chain into one line while the pauses between
them are at most ``gap_s``. A line ends at the first of

- a crash: everything up to the impact stays, only the impact is cut (`CRASH`),
- a landing, or any other footage the montage cannot use (`LANDED`),
- more than ``gap_s`` with nothing happening (`LULL`),
- the end of the recording (`RECORDING_END`).

A line never spans two recordings. `pick_lines` keeps each recording's longest line and any
close runner-up, and drops a line that another file (an AnaHD-cleaned copy of the same
flight) already gives.

Pure functions over the per-frame map a clip keeps (`CutMap`) and its moments; no video,
no Qt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from fpvedit.flight import runs
from fpvedit.montage import CutMap, Moment, MontageConfig, snap_cut

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class LineEnd(Enum):
    CRASH = "crash"
    LANDED = "landed"
    LULL = "lull"
    RECORDING_END = "recording end"

    @property
    def description(self) -> str:
        return {
            LineEnd.CRASH: "Ended in a crash",
            LineEnd.LANDED: "Landed",
            LineEnd.LULL: "Eased off",
            LineEnd.RECORDING_END: "End of recording",
        }[self]


@dataclass(frozen=True)
class Line:
    clip: int  # index of the clip in the list given to the montage
    start: float  # seconds into the recording
    end: float
    events: int  # flips, spins and close passes inside the line
    active_s: float  # seconds of the line spent active
    ended_by: LineEnd
    moments: tuple[Moment, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class LinesConfig:
    gap_s: float = 8.0  # more than this with nothing happening ends a line
    min_s: float = 10.0  # the shortest line ...
    active_share: float = 0.5  # ... and the share of it that must be active
    close_ratio: float = 0.8  # a runner-up is kept when this long compared to the best line
    per_recording: int = 3  # most lines kept from one recording
    lead_s: float = 1.0  # a line starts this long before its first active frame
    tail_s: float = 1.0  # ... and ends this long after its last
    moment_reach_s: float = 0.5  # a frame this close to a flip, spin or pass is active
    # Fast flying (starting values, tuned on real footage): over one second, the image
    # moves at least free_speed picture widths/s, the camera turns at least free_rate deg/s
    # and tracking holds at least `confidence`.
    fast_flying: bool = True
    free_speed: float = 0.5
    free_rate: float = 60.0
    confidence: float = 0.2
    # Two lines of different files are one flight when every time matches within this.
    same_s: float = 0.5


def no_lines_message(config: LinesConfig | None = None) -> str:
    config = config or LinesConfig()
    what = "tricks, close passes or fast flying" if config.fast_flying else "tricks or close passes"
    return f"No lines found. A line needs {config.min_s:g} s or more of {what}."


def _rolling_mean(values: FloatArray, window: int) -> FloatArray:
    """Mean over a centred window of ``window`` samples (the edges repeat their value)."""
    half = max(1, window) // 2
    padded = np.pad(values, half, mode="edge")
    sums = np.cumsum(np.insert(padded, 0, 0.0))
    size = 2 * half + 1
    return np.asarray((sums[size:] - sums[:-size]) / size, dtype=np.float64)[: len(values)]


def active_frames(
    cuts: CutMap, moments: Sequence[Moment], config: LinesConfig | None = None
) -> BoolArray:
    """Frames where something is happening, inside usable footage only."""
    config = config or LinesConfig()
    fps, count = cuts.fps, len(cuts.usable)
    active = np.zeros(count, dtype=bool)
    reach = config.moment_reach_s
    for moment in moments:
        active[max(0, round((moment.time - reach) * fps)) : round((moment.time + reach) * fps)] = (
            True
        )
    if config.fast_flying:
        window = max(1, round(fps))
        signals = cuts.signals
        active |= (
            (_rolling_mean(signals.speed, window) >= config.free_speed)
            & (_rolling_mean(signals.rate, window) >= config.free_rate)
            & (_rolling_mean(signals.confidence, window) >= config.confidence)
        )
    return active & cuts.usable


def _run_end(cuts: CutMap, high: int) -> LineEnd:
    """Why usable flying ends at frame ``high``."""
    fps = cuts.fps
    if any(abs(round(start * fps) - high) <= 1 for start in cuts.signals.crashes):
        return LineEnd.CRASH
    return LineEnd.RECORDING_END if high >= len(cuts.usable) else LineEnd.LANDED


def find_lines(
    clip: int,
    cuts: CutMap,
    moments: Sequence[Moment],
    config: LinesConfig | None = None,
    montage: MontageConfig | None = None,
) -> tuple[Line, ...]:
    """The lines of one recording, in time order. ``moments`` come from
    `montage.find_moments` (on the same ``cuts.usable``); ``montage`` supplies `snap_s`."""
    config = config or LinesConfig()
    montage = montage or MontageConfig()
    fps = cuts.fps
    if not cuts.usable.any():
        return ()
    active = active_frames(cuts, moments, config)
    gap, lead, tail = (round(s * fps) for s in (config.gap_s, config.lead_s, config.tail_s))
    found: list[Line] = []
    for low, high in runs(cuts.usable):
        chains: list[list[int]] = []
        for first, last in runs(active[low:high]):
            first, last = first + low, last + low
            if chains and first - chains[-1][1] <= gap:
                chains[-1][1] = last
            else:
                chains.append([first, last])
        floor = low
        for number, (first, last) in enumerate(chains):
            start = snap_cut(
                cuts, max(floor, first - lead), floor, first, montage, out_of_turns=True
            )
            reason = _run_end(cuts, high)
            if high - last <= gap and reason is LineEnd.CRASH:
                end, ended_by = high, LineEnd.CRASH  # keep the run up to the impact
            elif last + tail >= high:
                end, ended_by = high, reason
            else:
                end = snap_cut(cuts, last + tail, last, high, montage, out_of_turns=True)
                last_chain = number == len(chains) - 1
                ended_by = reason if last_chain and high - last <= gap else LineEnd.LULL
            floor = end
            length = (end - start) / fps
            seconds = float(active[start:end].sum()) / fps
            if length < config.min_s or seconds < config.active_share * length:
                continue
            shown = tuple(m for m in moments if start <= m.time * fps < end)
            found.append(Line(clip, start / fps, end / fps, len(shown), seconds, ended_by, shown))
    return tuple(found)


def _same_flight(a: Line, b: Line, tolerance: float) -> bool:
    if abs(a.start - b.start) > tolerance or abs(a.end - b.end) > tolerance:
        return False
    if len(a.moments) != len(b.moments):
        return False
    first = sorted(m.time for m in a.moments)
    second = sorted(m.time for m in b.moments)
    return all(abs(x - y) <= tolerance for x, y in zip(first, second, strict=True))


def _without_copies(
    lines: Sequence[Line], modified: Mapping[int, float], tolerance: float
) -> list[Line]:
    """Drop a line that a line of another file already gives; of two copies the newer file
    (modification time; the earlier input on a tie) is kept."""
    preferred = sorted(
        lines, key=lambda line: (-modified.get(line.clip, 0.0), line.clip, line.start)
    )
    kept: list[Line] = []
    for line in preferred:
        if not any(
            other.clip != line.clip and _same_flight(line, other, tolerance) for other in kept
        ):
            kept.append(line)
    return kept


def _order(line: Line) -> tuple[int, int, float, int, float]:
    # Whole seconds, not a pairwise "within a second" tie, so the order is a total one.
    return (-math.floor(line.duration + 1e-6), -line.events, -line.active_s, line.clip, line.start)


def pick_lines(
    lines: Sequence[Line],
    config: LinesConfig | None = None,
    modified: Mapping[int, float] | None = None,
) -> tuple[Line, ...]:
    """The lines of the montage, in montage order (recording order, then time).

    Copies of one flight in different files count once (`modified` maps a clip index to its
    file's modification time). Each recording then keeps its top line (longest in whole
    seconds, then most events, then most active time) plus any other line at least
    ``close_ratio`` as long as it, up to ``per_recording``."""
    config = config or LinesConfig()
    distinct = _without_copies(lines, modified or {}, config.same_s)
    kept: list[Line] = []
    for clip in sorted({line.clip for line in distinct}):
        ranked = sorted((line for line in distinct if line.clip == clip), key=_order)
        best = ranked[0].duration
        close = [line for line in ranked if line.duration >= config.close_ratio * best - 1e-9]
        kept.extend(close[: config.per_recording])
    return tuple(sorted(kept, key=lambda line: (line.clip, line.start)))
