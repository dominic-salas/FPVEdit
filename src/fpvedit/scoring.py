"""A Freestyle and a Flow score for one recording, from the motion the analysis measured.

The score reads rotation, tracking and crashes only. It does not name tricks, because the
names are not reliable enough to score on. A turn is described by its size and its axis,
and repeating the same turn soon after itself earns less. Poorly tracked or very short
recordings get no score instead of a misleading one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from math import ceil, log2

import numpy as np
from numpy.typing import NDArray

from fpvedit.rating import ClipSignals

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SCORER_VERSION = 1
AXES = ("pitch", "yaw", "roll")
SIGNATURES = 7  # each axis in both directions, and mixed


@dataclass(frozen=True)
class ScoringConfig:
    min_flying_s: float = 30.0
    min_confidence: float = 0.35
    turn_deg: float = 120.0  # net turn within a second that counts as a turn
    full_turn_deg: float = 240.0  # what a full flip measures: blur loses the fastest frames
    largest_turn: float = 2.25  # value of the biggest turn, in full turns
    axis_ratio: float = 1.5  # the top axis must lead by this much, or the turn is mixed
    repeat_window_s: float = 30.0
    repeat_free: int = 2  # the same turn counts in full this many times per window
    repeat_decay: float = 0.5
    link_gap_s: float = 2.0  # turns this close are linked
    difficulty_scale: float = 8.0  # turn value per minute that reaches about 63 of 100
    still_rate: float = 25.0  # deg/s; below this and below `still_speed` the quad is idling
    still_speed: float = 0.12  # picture widths/s
    smooth_window_s: float = 6.0
    smooth_low: float = -4.5  # SPARC mapped to 0
    smooth_high: float = -2.5  # SPARC mapped to 100
    lull_gap_s: float = 8.0  # longer than this without a turn is cruising, not flow
    crash_merge_s: float = 8.0
    crash_penalty: float = 8.0  # Flow points lost for every crash
    spam_turns: int = 8
    spam_share: float = 0.75  # real test flights reached 0.69 with 8 or more turns
    pace_cap: float = 9.0  # real test flights reached 5.9; the same footage at 2x read 10.5
    outlier_history: int = 10
    outlier_z: float = 3.5
    bright: float = 0.35  # median brightness from which a flight counts as daylight


class Status(Enum):
    SCORED = "scored"
    TOO_SHORT = "too short"
    LOW_TRACKING = "tracking too poor"


@dataclass(frozen=True)
class Turn:
    time: float
    degrees: float
    signature: str
    value: float


@dataclass(frozen=True)
class FlightScore:
    status: Status
    version: int
    flying_s: float
    conditions: str  # "daylight" or "low light"
    freestyle: float = 0.0
    flow: float = 0.0
    difficulty: float = 0.0
    variety: float = 0.0
    linking: float = 0.0
    continuity: float = 0.0
    smoothness: float = 0.0
    crashes: int = 0
    pace: float = 0.0  # turn value per minute of flying
    top_share: float = 0.0  # share of the turns that are the most common one
    turns: tuple[Turn, ...] = ()


def sparc(speed: FloatArray, fps: float, cutoff_hz: float = 10.0, amplitude: float = 0.05) -> float:
    """Spectral arc length of a speed profile: closer to zero is smoother."""
    if len(speed) < 4 or float(speed.max(initial=0.0)) <= 0.0:
        return 0.0
    size = 2 ** (ceil(log2(len(speed))) + 4)
    spectrum = np.abs(np.fft.rfft(speed, size))
    frequencies = np.fft.rfftfreq(size, 1.0 / fps)
    spectrum = spectrum / float(spectrum.max())
    inside = frequencies <= min(cutoff_hz, fps / 2.0)
    frequencies, spectrum = frequencies[inside], spectrum[inside]
    last = int(np.flatnonzero(spectrum >= amplitude)[-1])
    if last == 0:
        return 0.0
    frequencies, spectrum = frequencies[: last + 1], spectrum[: last + 1]
    steps = np.diff(frequencies) / float(frequencies[-1])
    return -float(np.sum(np.hypot(steps, np.diff(spectrum))))


def find_turns(
    rotation: FloatArray, usable: BoolArray, fps: float, config: ScoringConfig
) -> tuple[Turn, ...]:
    """The separate turns of a recording, with what each is worth after repeats."""
    width = max(1, round(fps))
    kernel = np.ones(width) / fps
    trusted = np.where(usable[:, None], rotation, 0.0)
    net = np.stack([np.convolve(trusted[:, k], kernel, "same") for k in range(3)], axis=1)
    turned = np.linalg.norm(net, axis=1)
    taken = np.zeros(len(turned), dtype=bool)
    peaks: list[int] = []
    for index in np.argsort(-turned, kind="stable"):
        if turned[index] < config.turn_deg:
            break
        if usable[index] and not taken[max(0, index - width) : index + width + 1].any():
            taken[index] = True
            peaks.append(int(index))
    seen: dict[str, list[float]] = {}
    turns: list[Turn] = []
    for index in sorted(peaks):
        time = index / fps
        signature = _signature(net[index], config)
        earlier = [t for t in seen.get(signature, []) if time - t <= config.repeat_window_s]
        extra = max(0, len(earlier) + 1 - config.repeat_free)
        size = min(float(turned[index]) / config.full_turn_deg, config.largest_turn)
        turns.append(Turn(time, float(turned[index]), signature, size * config.repeat_decay**extra))
        seen.setdefault(signature, []).append(time)
    return tuple(turns)


def _signature(net: FloatArray, config: ScoringConfig) -> str:
    order = np.argsort(-np.abs(net), kind="stable")
    top, second = abs(float(net[order[0]])), abs(float(net[order[1]]))
    if top < config.axis_ratio * second:
        return "mixed"
    return f"{AXES[int(order[0])]}{'+' if net[order[0]] > 0 else '-'}"


def _percent(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0) * 100.0)


def _variety(turns: tuple[Turn, ...]) -> float:
    if not turns:
        return 0.0
    _, counts = np.unique([t.signature for t in turns], return_counts=True)
    shares = counts / counts.sum()
    return float(-np.sum(shares * np.log(shares)) / np.log(SIGNATURES))


def _top_share(turns: tuple[Turn, ...]) -> float:
    named = [t.signature for t in turns if t.signature != "mixed"]
    if not named:
        return 0.0
    return float(np.unique(named, return_counts=True)[1].max()) / len(named)


def _linking(turns: tuple[Turn, ...], config: ScoringConfig) -> float:
    if len(turns) < 2:
        return 0.0
    gaps = np.diff([t.time for t in turns])
    return float(np.mean(gaps <= config.link_gap_s))


def _continuity(
    turns: tuple[Turn, ...],
    rate: FloatArray,
    speed: FloatArray,
    usable: BoolArray,
    fps: float,
    config: ScoringConfig,
) -> float:
    """Share of the flying that is not idling or a long stretch between turns."""
    lull = (rate < config.still_rate) & (speed < config.still_speed)
    edges = [0.0, *(t.time for t in turns), len(rate) / fps]
    margin = config.lull_gap_s / 2.0
    for before, after in pairwise(edges):
        if after - before > config.lull_gap_s:
            lull[round((before + margin) * fps) : round((after - margin) * fps)] = True
    return 1.0 - float(lull[usable].mean())


def crash_count(crashes: tuple[float, ...], config: ScoringConfig) -> int:
    """Crashes close together are one crash: a tumble, a re-arm and another tumble."""
    if not crashes:
        return 0
    return 1 + int(np.sum(np.diff(sorted(crashes)) > config.crash_merge_s))


def _smoothness(rate: FloatArray, usable: BoolArray, fps: float, config: ScoringConfig) -> float:
    width = max(4, round(config.smooth_window_s * fps))
    values = [
        sparc(rate[start : start + width], fps)
        for start in range(0, len(rate) - width + 1, width)
        if usable[start : start + width].all()
    ]
    if not values:
        return 0.0
    return (float(np.median(values)) - config.smooth_low) / (config.smooth_high - config.smooth_low)


def disqualify(
    score: FlightScore, earlier: Sequence[float], config: ScoringConfig | None = None
) -> str | None:
    """Why a scored flight should not count, or None. ``earlier``: past Freestyle scores."""
    config = config or ScoringConfig()
    if score.status is not Status.SCORED:
        return None
    if len(score.turns) >= config.spam_turns and score.top_share >= config.spam_share:
        return "most of the flight is the same turn repeated"
    if score.pace > config.pace_cap:
        return "far more turns than this much flying should hold, so the video may be sped up"
    if len(earlier) >= config.outlier_history:
        middle = float(np.median(earlier))
        spread = float(np.median(np.abs(np.asarray(earlier) - middle)))
        if spread > 0 and 0.6745 * (score.freestyle - middle) / spread > config.outlier_z:
            return "the score is far above anything in your history"
    return None


def score_flight(
    signals: ClipSignals, usable: BoolArray, config: ScoringConfig | None = None
) -> FlightScore:
    """Score one recording. The same measurements always give the same score."""
    config = config or ScoringConfig()
    fps = signals.fps
    rotation = np.nan_to_num(signals.rotation)
    rate = np.nan_to_num(signals.rate)
    speed = np.nan_to_num(signals.speed)
    confidence = np.nan_to_num(signals.confidence)
    flying_s = float(usable.sum()) / fps
    level = float(np.median(signals.brightness[usable])) if usable.any() else 0.0
    conditions = "daylight" if level >= config.bright else "low light"
    if flying_s < config.min_flying_s:
        return FlightScore(Status.TOO_SHORT, SCORER_VERSION, flying_s, conditions)
    if float(np.mean(confidence[usable])) < config.min_confidence:
        return FlightScore(Status.LOW_TRACKING, SCORER_VERSION, flying_s, conditions)

    turns = find_turns(rotation, usable, fps, config)
    per_minute = sum(t.value for t in turns) * 60.0 / flying_s
    difficulty = float(1.0 - np.exp(-per_minute / config.difficulty_scale))
    variety = _variety(turns)
    linking = _linking(turns, config)
    freestyle = 0.6 * difficulty + 0.2 * variety + 0.2 * linking

    continuity = _continuity(turns, rate, speed, usable, fps, config)
    smoothness = float(np.clip(_smoothness(rate, usable, fps, config), 0.0, 1.0))
    crashes = crash_count(signals.crashes, config)
    flow = max(
        0.0, _percent(0.55 * continuity + 0.45 * smoothness) - config.crash_penalty * crashes
    )

    return FlightScore(
        status=Status.SCORED,
        version=SCORER_VERSION,
        flying_s=flying_s,
        conditions=conditions,
        freestyle=_percent(freestyle),
        flow=flow,
        difficulty=_percent(difficulty),
        variety=_percent(variety),
        linking=_percent(linking),
        continuity=_percent(continuity),
        smoothness=_percent(smoothness),
        crashes=crashes,
        pace=per_minute,
        top_share=_top_share(turns),
        turns=turns,
    )
