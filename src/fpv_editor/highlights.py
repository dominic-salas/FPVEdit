"""Choose which parts of a recording to keep, as a list of time intervals.

The edit is built by removing what is not worth watching, in this order:

1. Everything outside armed flight: boot and menus, time before arming, stats screens
   after a disarm, and static or no signal (from `flight.py`). Without a detectable OSD
   the whole recording counts as flight.
2. Sitting armed on the ground, except the moment before lift-off.
3. Crashes: a disarm right after fast, tumbling motion is a crash. The failed move that
   led to it goes too, back to the calmest moment a few seconds earlier, so the edit
   does not run into a crash it then hides. Several disarms in a short time (re-arm
   churn after a failed landing or a stuck quad) are cut as a whole.
   A quick, gentle disarm and re-arm (landing on a slide or a car roof) is part of the
   flying and stays.
4. Long dull stretches: flying without any trick, spin or close pass for ``dull_s``
   seconds. A margin is kept at each end so sessions do not start or stop abruptly.
5. Leftover pieces shorter than ``min_segment_s``.
6. Short pieces in which the camera hardly turns (median under ``flying_rate``): a quad
   on a desk or carried by hand, not flying. Long pieces between an arm and a disarm are
   exempt, as a long calm cruise with one flip in it is still a flight. Without any arm
   or disarm on screen (no OSD, or none shown) there is no other proof of flight, so
   every piece must turn that much and last at least ``unconfirmed_min_s``.

Tricks are found from the camera rotation (`motion.py`): a flip turns the camera through
180 degrees or more within a second, a fast snap shows as a very high rotation rate, and
spins and orbits as sustained turning well above the flight's usual level. Close passes
are the top approach-rate (expansion) moments of the flight. These only protect footage
from the dull-stretch rule; they never bring back footage that steps 1-3 removed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from fpv_editor.errors import NoFlightContentError
from fpv_editor.flight import FlightState, Screen, ScreenEvent, rolling_median, runs
from fpv_editor.measure import Measurements
from fpv_editor.motion import DIVERGENCE, PITCH, ROLL, SPEED

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
    SHORT_INPUT = "short-input"  # too short to edit: kept whole
    FLIGHT_STATE = "flight-state"  # arm/disarm read from the OSD
    NO_OSD = "no-osd"  # no OSD found: the whole recording counted as flight


class CutReason(Enum):
    NOT_FLYING = "not flying"  # boot, menus, disarmed, stats screen
    STATIC = "static"
    ON_GROUND = "on the ground"
    CRASH = "crash"
    FAILED_MOVE = "move before a crash"  # cut from the edit; a montage shot may use it
    REARM_CHURN = "repeated disarms"
    DULL = "dull"
    SHORT = "short leftover"
    NOT_CLEARLY_FLYING = "not clearly flying"


@dataclass(frozen=True)
class Cut:
    start: float
    end: float
    reason: CutReason


@dataclass(frozen=True)
class SelectionConfig:
    short_input_s: float = 6.0
    # on the ground
    still_speed: float = 0.1  # median image speed (widths/s) below which the camera is still
    still_rate: float = 12.0  # deg/s
    still_min_s: float = 3.0  # a calm hover can look still for a second; not for this long
    # A shorter still run counts when the picture barely changes either: a hover still
    # shakes (frame change ~0.004 and up), a quad on the ground does not (~0.001).
    still_change: float = 0.002
    still_quiet_min_s: float = 1.0
    liftoff_keep_s: float = 0.5
    landing_still_s: float = 3.0  # stillness this close to a landing re-arm is part of it
    static_min_s: float = 1.0  # shorter static (analog breakup) stays in the edit
    # crashes
    crash_speed: float = 1.2  # 90th percentile image speed just before a disarm ...
    crash_rate: float = 90.0  # ... or mean rotation rate (deg/s): a crash, not a landing
    crash_window_s: tuple[float, float] = (1.5, 0.2)  # measured this long before the disarm
    landing_rearm_s: float = 3.0  # a landing disarm re-armed this fast is part of the flight
    crash_lead_s: float = 3.5  # the failed move starts at least this long before the disarm
    crash_lookback_s: float = 8.0  # ... and at most this long
    # The impact itself: the last this long before a crash's disarm is cut as the crash;
    # what comes before it is the failed move (a pass that went well just before stays
    # usable for a montage shot).
    crash_impact_s: float = 1.5
    calm_slack: float = 20.0  # deg/s above the calmest moment that still counts as calm
    # without an OSD, a dead stop this still and long stands in for a disarm
    impact_speed: float = 0.08  # widths/s
    impact_rate: float = 10.0  # deg/s
    impact_still_s: float = 0.5
    impact_tumble_rate: float = 300.0  # 90th percentile rotation (deg/s) just before it
    crash_after_s: float = 12.0  # a crash is over by the re-arm or take-off within this
    crash_still_gap_s: float = 2.0  # lying still this soon after the re-arm is still the crash
    crash_refly_s: float = 4.0  # ... as is a re-armed stretch this short ending in a disarm
    churn_disarms: int = 3  # this many disarms ...
    churn_window_s: float = 15.0  # ... within this time are cut as a whole
    calm_rate: float = 60.0  # deg/s: calm flying that ends a churn cut ...
    calm_hold_s: float = 2.0  # ... once held this long
    # what protects flying from the dull-stretch rule
    flip_deg_1s: float = 180.0
    turn_deg_4s: float = 270.0
    snap_rate: float = 400.0  # deg/s
    spin_factor: float = 1.6  # sustained turning this many times the flight's median
    pass_quantile: float = 0.9  # approach rate above this quantile of the flight
    protect_margin_s: float = 1.5
    # dull stretches and leftovers
    dull_s: float = 15.0
    dull_margin_s: float = 1.0
    min_segment_s: float = 3.0
    flying_rate: float = 18.0  # median rotation rate (deg/s) of a kept piece ...
    flying_check_s: float = 20.0  # ... shorter than this
    unconfirmed_min_s: float = 6.0  # shortest piece without arm/disarm events


@dataclass(frozen=True)
class Selection:
    intervals: tuple[Interval, ...]
    cuts: tuple[Cut, ...]
    mode: SelectionMode
    retention: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def kept_duration(self) -> float:
        return sum(interval.duration for interval in self.intervals)


def window_sum(values: FloatArray, window: int) -> FloatArray:
    """Sum over a centred window of ``window`` samples (truncated at the edges)."""
    count = len(values)
    totals = np.concatenate((np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)))
    index = np.arange(count)
    low = np.clip(index - window // 2, 0, count)
    high = np.clip(index + window - window // 2, 0, count)
    return np.asarray(totals[high] - totals[low], dtype=np.float64)


def mask_to_intervals(mask: BoolArray, fps: float) -> tuple[Interval, ...]:
    return tuple(Interval(start / fps, end / fps) for start, end in runs(mask))


def time_mask(count: int, fps: float, start: float, end: float) -> BoolArray:
    mask = np.zeros(count, dtype=bool)
    mask[max(0, round(start * fps)) : min(count, round(end * fps))] = True
    return mask


class _Motion:
    """Per-frame motion summaries used by the policy."""

    def __init__(self, measurements: Measurements) -> None:
        fps = measurements.fps
        motion = np.nan_to_num(measurements.motion)
        self.fps = fps
        self.rotation = motion[:, PITCH : ROLL + 1]
        self.rate = np.linalg.norm(self.rotation, axis=1)
        self.speed = motion[:, SPEED]
        self.divergence = motion[:, DIVERGENCE]
        half = max(1, round(0.5 * fps))
        self.rate_smooth = rolling_median(self.rate, half)
        self.speed_smooth = rolling_median(self.speed, half)
        self.change_smooth = rolling_median(np.nan_to_num(measurements.change), half)
        second = max(1, round(fps))
        self.rate_1s = window_sum(self.rate, second) / second


def select_highlights(
    measurements: Measurements, flight: FlightState, config: SelectionConfig | None = None
) -> Selection:
    config = config or SelectionConfig()
    fps = measurements.fps
    count = len(measurements)
    if count == 0:
        raise NoFlightContentError("no frames to analyse")
    if count / fps < config.short_input_s:
        return Selection(
            intervals=(Interval(0.0, count / fps),),
            cuts=(),
            mode=SelectionMode.SHORT_INPUT,
            retention=1.0,
            warnings=(f"input is shorter than {config.short_input_s:g} s; kept all of it",),
        )
    motion = _Motion(measurements)
    cuts: list[Cut] = []

    def cut(mask: BoolArray, removed: BoolArray, reason: CutReason) -> BoolArray:
        for start, end in runs(mask & removed):
            cuts.append(Cut(start / fps, end / fps, reason))
        return mask & ~removed

    events = list(flight.events)
    landings = _landing_rearms(events, motion, config)
    keep = np.ones(count, dtype=bool)
    keep = cut(keep, _not_flying(events, flight, landings, count, fps), CutReason.NOT_FLYING)
    keep = cut(keep, _long_runs(flight.static, round(config.static_min_s * fps)), CutReason.STATIC)
    ground = _on_ground(motion, landings, config)
    keep = cut(keep, ground, CutReason.ON_GROUND)
    # Without an OSD there are no disarms to read; coming to rest on the ground stands in
    # for them, so a tumble just before it is still found as a crash.
    stops = events if flight.has_osd else _ground_stops(ground, motion, config)
    crash, failed, churn = _failures(stops, motion, keep.copy(), _still(motion, config), config)
    keep = cut(keep, crash, CutReason.CRASH)
    keep = cut(keep, failed, CutReason.FAILED_MOVE)
    keep = cut(keep, churn, CutReason.REARM_CHURN)
    landed = np.zeros(count, dtype=bool)
    for start, end in landings:
        landed |= time_mask(
            count, fps, start - config.landing_still_s, end + config.landing_still_s
        )
    keep = cut(keep, _dull(motion, keep, landed, config), CutReason.DULL)
    short = np.zeros(count, dtype=bool)
    for start, end in runs(keep):
        if (end - start) / fps < config.min_segment_s:
            short[start:end] = True
    keep = cut(keep, short, CutReason.SHORT)
    confirmed = any(e.kind in (Screen.ARMED, Screen.LOGO, Screen.STATS) for e in events)
    unsure = np.zeros(count, dtype=bool)
    for start, end in runs(keep):
        seconds = (end - start) / fps
        calm = float(np.median(motion.rate[start:end])) < config.flying_rate
        if confirmed:
            not_flying = calm and seconds < config.flying_check_s
        else:
            not_flying = calm or seconds < config.unconfirmed_min_s
        if not_flying:
            unsure[start:end] = True
    keep = cut(keep, unsure, CutReason.NOT_CLEARLY_FLYING)

    if not keep.any():
        raise NoFlightContentError(
            "found no flying worth keeping (no armed flight, or only crashes, static or "
            "time on the ground)"
        )
    warnings: tuple[str, ...] = ()
    if not flight.has_osd:
        warnings = (
            "no Betaflight OSD found, so flight state comes from the picture alone: static, "
            "time on the ground, crashes (a tumble before coming to rest) and dull stretches "
            "were cut",
        )
    elif not confirmed:
        warnings = (
            "the OSD never showed arming or disarming, so the whole recording was treated "
            "as one flight",
        )
    return Selection(
        intervals=mask_to_intervals(keep, fps),
        cuts=tuple(sorted(cuts, key=lambda c: c.start)),
        mode=SelectionMode.FLIGHT_STATE if flight.has_osd else SelectionMode.NO_OSD,
        retention=float(keep.mean()),
        warnings=warnings,
    )


def _not_flying(
    events: Sequence[ScreenEvent],
    flight: FlightState,
    landings: Sequence[tuple[float, float]],
    count: int,
    fps: float,
) -> BoolArray:
    armed = np.zeros(count, dtype=bool)
    for start, end in (*flight.armed, *landings):
        armed |= time_mask(count, fps, start, end)
    down = ~armed
    for event in events:
        if any(start <= event.start < end for start, end in landings):
            continue
        index = min(count - 1, max(0, round(event.start * fps)))
        screen = event.kind in (Screen.BOOT, Screen.LOGO, Screen.STATS)
        # A short clear screen while armed is an OSD glitch, not downtime.
        idle_screen = event.kind in (Screen.ARMED, Screen.BLANK) and not armed[index]
        if screen or idle_screen:
            down |= time_mask(count, fps, event.start, event.end)
    return down


def _long_runs(mask: BoolArray, min_length: int) -> BoolArray:
    result = np.zeros(len(mask), dtype=bool)
    for start, end in runs(mask):
        if end - start >= min_length:
            result[start:end] = True
    return result


def _still(motion: _Motion, config: SelectionConfig) -> BoolArray:
    return np.asarray(
        (motion.speed_smooth < config.still_speed) & (motion.rate_smooth < config.still_rate)
    )


def _on_ground(
    motion: _Motion, landings: Sequence[tuple[float, float]], config: SelectionConfig
) -> BoolArray:
    """Camera still for a while: sitting on the ground. Keep the moment before lift-off.

    The stop of a quick landing and re-arm (on a car roof, say) is part of that stunt.
    """
    fps = motion.fps
    count = len(motion.rate)
    still = _still(motion, config)
    for start, end in landings:
        still &= ~time_mask(
            count, fps, start - config.landing_still_s, end + config.landing_still_s
        )
    quiet = still & (motion.change_smooth < config.still_change)
    result = np.zeros(len(still), dtype=bool)
    keep_end = round(config.liftoff_keep_s * fps)
    for mask, min_s in ((still, config.still_min_s), (quiet, config.still_quiet_min_s)):
        for start, end in runs(mask):
            if (end - start) / fps >= min_s:
                result[start : max(start, end - keep_end)] = True
    return result


def _ground_stops(ground: BoolArray, motion: _Motion, config: SelectionConfig) -> list[ScreenEvent]:
    """Each time the quad comes to rest right after tumbling, as a disarm (stats screen)
    event: the start of time on the ground, or a short dead stop (a crash often lies
    still for under a second before the pilot flips it over). Without the OSD's disarm,
    speed alone is not enough: a fast flight into a deliberate stop is not a crash."""
    fps = motion.fps
    window = max(1, round(0.25 * fps))
    dead = (rolling_median(motion.speed, window) < config.impact_speed) & (
        rolling_median(motion.rate, window) < config.impact_rate
    )
    starts = {start for start, _ in runs(ground)}
    starts |= {s for s, e in runs(dead) if (e - s) / fps >= config.impact_still_s}
    sustained = rolling_median(motion.rate, window)  # one-frame glitches are not a tumble
    before, after = config.crash_window_s
    stops = []
    for start in sorted(starts):
        first, last = round(start - before * fps), round(start - after * fps)
        tumble = sustained[first:last] if first >= 0 else np.zeros(0)
        if len(tumble) and float(np.percentile(tumble, 90)) >= config.impact_tumble_rate:
            stops.append(ScreenEvent(Screen.STATS, start / fps, start / fps))
    return stops


def _is_crash(disarm: float, motion: _Motion, config: SelectionConfig) -> bool:
    fps = motion.fps
    before, after = config.crash_window_s
    start, end = round((disarm - before) * fps), round((disarm - after) * fps)
    if start < 0 or end <= start:
        return False
    fast = float(np.percentile(motion.speed[start:end], 90)) >= config.crash_speed
    tumbling = float(motion.rate[start:end].mean()) >= config.crash_rate
    return fast or tumbling


def _landing_rearms(
    events: Sequence[ScreenEvent], motion: _Motion, config: SelectionConfig
) -> list[tuple[float, float]]:
    """Quick, gentle disarm and re-arm, as (stats start, ARMED end) spans to keep."""
    spans = []
    for event, following in pairwise(events):
        if (
            event.kind is Screen.STATS
            and following.kind is Screen.ARMED
            and following.end - event.start <= config.landing_rearm_s
            and not _is_crash(event.start, motion, config)
        ):
            spans.append((event.start, following.end))
    return spans


def _failure_start(
    disarm: float, motion: _Motion, flying: BoolArray, config: SelectionConfig
) -> float:
    """The calmest recent moment before a crash: where the failed move began."""
    fps = motion.fps
    high = round((disarm - config.crash_lead_s) * fps)
    low = max(0, round((disarm - config.crash_lookback_s) * fps))
    candidates = np.flatnonzero(flying[low : max(low, high)]) + low
    if len(candidates) == 0:
        return low / fps
    calm = motion.rate_1s[candidates]
    latest = candidates[np.flatnonzero(calm <= float(np.min(calm)) + config.calm_slack)[-1]]
    return float(latest) / fps


def _crash_end(
    disarm: float,
    events: Sequence[ScreenEvent],
    still: BoolArray,
    fps: float,
    config: SelectionConfig,
) -> float:
    """When the crash is over: the first re-arm, or the end of lying still (turtle mode,
    getting the quad upright) before it takes off again, whichever is later. A later
    re-arm or rest belongs to the next flight (a quick pilot re-arms within seconds),
    unless that flight ends in another disarm within ``crash_refly_s`` (turtle mode)."""
    horizon = disarm + config.crash_after_s
    end = disarm + 0.5
    current = disarm
    while True:
        rearm = next(
            (e.end for e in events if e.kind is Screen.ARMED and current < e.start <= horizon),
            None,
        )
        if rearm is None:
            break
        end = max(end, rearm)
        again = next(
            (e.start for e in events if e.kind is Screen.STATS and rearm <= e.start <= horizon),
            None,
        )
        if again is None or again - rearm >= config.crash_refly_s:
            break
        current = again
    first, last = round(disarm * fps), min(len(still), round(horizon * fps))
    for start, stop in runs(still[first:last]):
        if (first + start) / fps > end + config.crash_still_gap_s:
            break  # flying again in between: a new flight
        if (stop - start) / fps >= config.still_quiet_min_s:
            end = max(end, (first + stop) / fps - config.liftoff_keep_s)
    return end


def _failures(
    events: Sequence[ScreenEvent],
    motion: _Motion,
    flying: BoolArray,
    still: BoolArray,
    config: SelectionConfig,
) -> tuple[BoolArray, BoolArray, BoolArray]:
    """Masks of crashes, of the failed moves before them and of re-arm churn."""
    fps = motion.fps
    count = len(flying)
    crash = np.zeros(count, dtype=bool)
    failed = np.zeros(count, dtype=bool)
    churn = np.zeros(count, dtype=bool)
    disarms = [e.start for e in events if e.kind is Screen.STATS and e.start > 0]
    crashes = {t for t in disarms if _is_crash(t, motion, config)}
    for disarm in crashes:
        start = _failure_start(disarm, motion, flying, config)
        impact = max(start, disarm - config.crash_impact_s)
        failed |= time_mask(count, fps, start, impact)
        crash |= time_mask(count, fps, impact, _crash_end(disarm, events, still, fps, config))
    calm = (
        (motion.rate_smooth < config.calm_rate)
        & (motion.speed_smooth >= config.still_speed)
        & flying
    )
    hold = max(1, round(config.calm_hold_s * fps))
    calm_held = window_sum(calm.astype(np.float64), hold) >= hold - 1
    for first in disarms:
        group = [t for t in disarms if first <= t <= first + config.churn_window_s]
        if len(group) < config.churn_disarms:
            continue
        start = _failure_start(first, motion, flying, config)
        index = round(group[-1] * fps)
        while index < count and not calm_held[index]:
            index += 1
        churn |= time_mask(count, fps, start, index / fps)
    return crash, failed & ~crash, churn


def _dull(
    motion: _Motion, keep: BoolArray, landed: BoolArray, config: SelectionConfig
) -> BoolArray:
    """Kept stretches of at least ``dull_s`` without a trick, spin, close pass or stunt
    landing."""
    fps = motion.fps
    count = len(keep)
    if not keep.any():
        return np.zeros(count, dtype=bool)
    turned_1s = np.linalg.norm(window_sum(motion.rotation, round(fps)), axis=1) / fps
    turned_4s = np.linalg.norm(window_sum(motion.rotation, round(4 * fps)), axis=1) / fps
    half = max(1, round(0.5 * fps))
    smoothed = window_sum(motion.rotation, half) / half
    sustained = window_sum(np.linalg.norm(smoothed, axis=1), round(4 * fps)) / fps
    snap = rolling_median(motion.rate, 3) >= config.snap_rate
    approach = rolling_median(motion.divergence, max(1, round(0.25 * fps)))
    interesting = (
        (turned_1s >= config.flip_deg_1s)
        | (turned_4s >= config.turn_deg_4s)
        | snap
        | (sustained >= config.spin_factor * float(np.median(sustained[keep])))
        | (approach >= float(np.quantile(approach[keep], config.pass_quantile)))
        | landed
    ) & keep
    reach = round(2 * config.protect_margin_s * fps)
    protected = window_sum(interesting.astype(np.float64), reach) > 0
    margin = round(config.dull_margin_s * fps)
    dull = np.zeros(count, dtype=bool)
    for start, end in runs(keep & ~protected):
        if (end - start) / fps < config.dull_s:
            continue
        inner_start = start + margin if start > 0 and keep[start - 1] else start
        inner_end = end - margin if end < count and keep[end] else end
        dull[inner_start:inner_end] = True
    return dull
