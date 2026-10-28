"""Pick the best shots from several clips for one montage.

Each clip is first cleaned with the flight-state rules of `highlights.py`: boot and menus,
time disarmed, static, sitting on the ground, crashes (the impact; a pass that went well
just before one stays usable) and re-arm churn are never usable. What remains is searched
for *moments*:

- flips and rolls: the camera turns 180 degrees or more within a second (a big flip,
  330 degrees or more, counts extra); `tricks.py` names each one by its axis (back or
  front flip, roll, yaw spin, doubles), and a slow full pitch turn away from any flip
  is a power loop;
- spins and orbits: turning sustained over four seconds, well above the clip's usual
  level and at least ``spin_floor`` on average (a flip in calm flying also counts as
  one);
- close passes: the fastest approach (flow expansion) moments of the clip, above
  ``pass_floor``, that also stand out from how close the flying around them is. Flying
  indoors keeps everything close, so a door or a hallway looks like a pass by the first
  test alone; only a sudden approach, such as a gap, passes the second.

A shot is 8-15 s of usable footage that starts ``lead_s`` before a moment, so the viewer
sees the set-up. Both ends are moved to the calmest frame nearby, so a cut never lands in
the middle of a trick, and a moment too close to the end of a shot (cut short) does not
count. A shot's score is the weight of the moments it shows plus a bonus for fast flying.

The montage takes the best shots across all clips, ranked by score per square root of
length (a long shot has to show more), until the target length is reached. Shots never
overlap, keep a gap from other shots of the same clip, and no clip fills more than
``clip_share`` of the montage while others have something to offer. Shots play in the
order the clips were given, then in recording order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from fpvedit.flight import rolling_median, runs
from fpvedit.highlights import (
    CutReason,
    Interval,
    MotionSummary,
    Selection,
    SelectionConfig,
    SelectionMode,
    window_sum,
)
from fpvedit.measure import Measurements
from fpvedit.motion import CONFIDENCE
from fpvedit.rating import ClipSignals, Quality, rate_shot
from fpvedit.tricks import TrickConfig, name_fast_turn, power_loop_frames

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Cut reasons that make footage unusable in a montage. "Dull" and "short leftover" only
# say the footage would not carry a whole-flight edit; a shot can still use it.
UNUSABLE = frozenset(
    {
        CutReason.NOT_FLYING,
        CutReason.STATIC,
        CutReason.ON_GROUND,
        CutReason.CRASH,
        CutReason.REARM_CHURN,
        CutReason.NOT_CLEARLY_FLYING,
    }
)


class MomentKind(Enum):
    FLIP = "flip"
    SPIN = "spin"
    PROXIMITY = "close pass"


@dataclass(frozen=True)
class Moment:
    time: float
    kind: MomentKind
    weight: float
    trick: str = ""  # what the turn looks like (`tricks.py`), e.g. "back flip", "roll"

    @property
    def name(self) -> str:
        return self.trick or self.kind.value


@dataclass(frozen=True)
class Shot:
    clip: int  # index of the clip in the list given to the montage
    start: float
    end: float
    score: float
    moments: tuple[Moment, ...]
    quality: Quality | None = None  # the 1-5 star rating's parts (`rating.py`)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)


@dataclass(frozen=True)
class MontageConfig:
    # None: every distinct shot with a moment, however long the montage gets. A number
    # makes the montage about that long, best shots first.
    length_s: float | None = None
    length_slack_s: float = 4.0  # the montage may run this much over or stop this short
    shot_min_s: float = 8.0
    shot_max_s: float = 15.0
    # A flight shorter than shot_min_s (re-armed between crashes) is one shot, shown
    # whole, when it lasts this long and has a moment anywhere in it.
    short_flight_s: float = 4.0
    lead_s: float = 2.0  # a shot starts this long before its first moment
    edge_s: float = 0.5  # a moment this close to the start of a shot is not shown whole
    tail_s: float = 2.0  # ... nor one this close to its end (it would be cut short)
    snap_s: float = 0.75  # each end moves to the calmest frame within this distance
    turning_rate: float = 90.0  # deg/s (0.3 s median): a cut never falls inside such a turn
    fast_turn_rate: float = 150.0  # deg/s (0.1 s median): nor this close to one this fast
    fast_turn_gap_s: float = 0.4
    # moments
    flip_deg: float = 180.0  # turned within one second
    big_flip_deg: float = 330.0
    flip_gap_s: float = 0.8
    spin_factor: float = 1.8  # sustained turning this many times the clip's median ...
    spin_floor: float = 90.0  # ... and at least this (deg/s over four seconds)
    spin_gap_s: float = 2.0
    pass_quantile: float = 0.95  # approach rate above this quantile of the clip ...
    pass_floor: float = 1.0  # ... and above this (1/s)
    pass_gap_s: float = 1.5
    pass_prominence: float = 0.75  # ... and this far above the median over ...
    pass_baseline_s: float = 4.0  # ... this long around it
    loop_gap_s: float = 3.0  # between two power loops
    loop_flip_gap_s: float = 1.5  # a slow pitch turn this close to a flip is part of it
    # scores
    flip_weight: float = 3.0
    big_flip_weight: float = 1.5  # added to flip_weight
    spin_weight: float = 1.5
    pass_weight: float = 1.0
    speed_weight: float = 3.0  # per unit of median image speed (picture widths/s) ...
    speed_cap: float = 1.5  # ... up to this speed
    # picking
    clip_share: float = 0.5  # most of the montage one clip may fill (unless it is alone)
    min_gap_s: float = 4.0  # between two shots of the same clip


MIN_MARKED_S = 2.0  # the shortest shot a mark builds


def usable_footage(
    measurements: Measurements, selection: Selection, config: SelectionConfig | None = None
) -> BoolArray:
    """Frames a shot may use: cleaned flight that is clearly flying."""
    config = config or SelectionConfig()
    fps = measurements.fps
    count = len(measurements)
    if selection.mode is SelectionMode.SHORT_INPUT:
        return np.zeros(count, dtype=bool)  # too short for any shot
    usable = np.ones(count, dtype=bool)
    for cut in selection.cuts:
        if cut.reason in UNUSABLE:
            usable[max(0, round(cut.start * fps)) : min(count, round(cut.end * fps))] = False
    # The whole-flight edit checks for a quad on a desk or in a hand only in what it keeps;
    # stretches it cut as dull are checked here.
    rate = MotionSummary(measurements).rate
    for start, end in runs(usable):
        calm = float(np.median(rate[start:end])) < config.flying_rate
        if calm and (end - start) / fps < config.flying_check_s:
            usable[start:end] = False
    return usable


def find_moments(
    measurements: Measurements, usable: BoolArray, config: MontageConfig | None = None
) -> list[Moment]:
    """Flips, spins and close passes inside usable footage, in time order."""
    config = config or MontageConfig()
    if not usable.any():
        return []
    fps = measurements.fps
    motion = MotionSummary(measurements)
    second = max(1, round(fps))
    turned_1s = np.linalg.norm(window_sum(motion.rotation, second), axis=1) / fps
    half = max(1, round(0.5 * fps))
    smoothed = window_sum(motion.rotation, half) / half
    four = max(1, round(4 * fps))
    sustained = window_sum(np.linalg.norm(smoothed, axis=1), four) / four
    approach = rolling_median(motion.divergence, max(1, round(0.25 * fps)))

    confidence = np.nan_to_num(measurements.motion[:, CONFIDENCE])
    moments = []
    flips = _peaks(np.where(usable, turned_1s, 0.0), config.flip_deg, config.flip_gap_s, fps)
    for index in flips:
        big = turned_1s[index] >= config.big_flip_deg
        weight = config.flip_weight + (config.big_flip_weight if big else 0.0)
        trick = name_fast_turn(motion.rotation, confidence, index, fps)
        moments.append(Moment(index / fps, MomentKind.FLIP, weight, trick))
    # Power loops: a full pitch turn too slow to count as a flip, away from any flip.
    loops = power_loop_frames(motion.rotation, usable, fps)
    near_flip = np.zeros(len(loops), dtype=bool)
    reach = round(config.loop_flip_gap_s * fps)
    for index in flips:
        near_flip[max(0, index - reach) : index + reach] = True
    loops[near_flip] = 0.0
    for index in _peaks(loops, TrickConfig().loop_deg, config.loop_gap_s, fps):
        moments.append(Moment(index / fps, MomentKind.FLIP, config.flip_weight, "power loop"))
    spin = max(config.spin_floor, config.spin_factor * float(np.median(sustained[usable])))
    for index in _peaks(np.where(usable, sustained, 0.0), spin, config.spin_gap_s, fps):
        moments.append(Moment(index / fps, MomentKind.SPIN, config.spin_weight, "spin"))
    close = max(config.pass_floor, float(np.quantile(approach[usable], config.pass_quantile)))
    baseline = rolling_median(approach, max(1, round(config.pass_baseline_s * fps)))
    sudden = usable & (approach - baseline >= config.pass_prominence)
    for index in _peaks(np.where(sudden, approach, -np.inf), close, config.pass_gap_s, fps):
        moments.append(Moment(index / fps, MomentKind.PROXIMITY, config.pass_weight, "close pass"))
    return sorted(moments, key=lambda m: m.time)


def _peaks(values: FloatArray, threshold: float, gap_s: float, fps: float) -> list[int]:
    """Highest values at or above ``threshold``, at least ``gap_s`` apart."""
    gap = max(1, round(gap_s * fps))
    candidates = np.flatnonzero(values >= threshold)
    blocked = np.zeros(len(values), dtype=bool)
    chosen = []
    for index in candidates[np.argsort(-values[candidates], kind="stable")]:
        if blocked[index]:
            continue
        chosen.append(int(index))
        blocked[max(0, index - gap) : index + gap] = True
    return sorted(chosen)


@dataclass(frozen=True, eq=False)
class CutMap:
    """What building a shot in one clip needs, per frame, once its measurements are gone:
    where a shot may be, where it can be cut, and what its rating reads. Kept with the
    clip so the pilot can build a shot at any moment (`shot_at`)."""

    usable: BoolArray
    calm: FloatArray  # deg/s, a cut goes to the calmest frame nearby
    turning: BoolArray  # a cut never falls in these frames
    signals: ClipSignals

    @property
    def fps(self) -> float:
        return self.signals.fps


def cut_map(
    measurements: Measurements,
    selection: Selection,
    config: MontageConfig | None = None,
    selection_config: SelectionConfig | None = None,
) -> CutMap:
    config = config or MontageConfig()
    fps = measurements.fps
    usable = usable_footage(measurements, selection, selection_config)
    motion = MotionSummary(measurements)
    # Frames whose motion could not be measured (blur in a fast flip) are not calm.
    rate = np.where(np.isnan(measurements.motion).any(axis=1), np.inf, motion.rate)
    calm = rolling_median(rate, max(1, round(0.3 * fps)))
    # A cut inside a turn, or right next to a fast one, shows a flip cut short.
    fast = rolling_median(rate, max(1, round(0.1 * fps))) >= config.fast_turn_rate
    near_fast = window_sum(fast.astype(np.float64), 2 * round(config.fast_turn_gap_s * fps) + 1) > 0
    turning = (calm >= config.turning_rate) | near_fast
    signals = ClipSignals(
        fps=fps,
        rate=rolling_median(motion.rate, max(1, round(0.1 * fps))),
        rotation=motion.rotation,
        speed=motion.speed_smooth,
        approach=motion.divergence,
        confidence=np.nan_to_num(measurements.motion[:, CONFIDENCE]),
        brightness=np.nan_to_num(measurements.brightness),
        crashes=tuple(c.start for c in selection.cuts if c.reason is CutReason.CRASH),
    )
    return CutMap(usable, calm, turning, signals)


def candidate_shots(
    clip: int,
    measurements: Measurements,
    selection: Selection,
    config: MontageConfig | None = None,
    selection_config: SelectionConfig | None = None,
    cuts: CutMap | None = None,
) -> list[Shot]:
    """Every shot worth considering in one clip (they overlap; `pick_shots` chooses)."""
    config = config or MontageConfig()
    cuts = cuts or cut_map(measurements, selection, config, selection_config)
    fps = measurements.fps
    usable = cuts.usable
    moments = find_moments(measurements, usable, config)
    if not moments:
        return []
    motion = MotionSummary(measurements)
    calm, turning, signals = cuts.calm, cuts.turning, cuts.signals
    shortest = round(config.shot_min_s * fps)
    longest = round(config.shot_max_s * fps)
    reach = round(config.snap_s * fps)
    lengths = {shortest, round((shortest + longest) / 2), longest}
    shots: dict[tuple[int, int], Shot] = {}
    for low, high in runs(usable):
        if high - low < shortest:
            if high - low >= round(config.short_flight_s * fps):
                whole = dataclasses.replace(config, edge_s=0.0, tail_s=0.0)
                shot = _score(clip, low / fps, high / fps, moments, motion, whole)
                if shot is not None:
                    named = [(m.time, m.name) for m in shot.moments]
                    quality = rate_shot(shot.start, shot.end, named, signals)
                    shots[(low, high)] = dataclasses.replace(shot, quality=quality)
            continue
        inside = [m for m in moments if low <= m.time * fps < high]
        for moment in inside:
            for length in sorted(lengths):
                start = min(round(moment.time * fps) - round(config.lead_s * fps), high - length)
                if start < low:
                    continue
                start = _calmest(calm, start, reach, low, high)
                end = _calmest(calm, start + length, reach, low, high)
                end = min(max(end, start + shortest), start + longest, high)
                edges = _clear_of_turns(turning, start, end, shortest, longest, low, high)
                if edges is None:
                    continue
                start, end = edges
                shot = _score(clip, start / fps, end / fps, inside, motion, config)
                if shot is not None:
                    named = [(m.time, m.name) for m in shot.moments]
                    quality = rate_shot(shot.start, shot.end, named, signals)
                    shots[(start, end)] = dataclasses.replace(shot, quality=quality)
    return list(shots.values())


def shot_at(
    clip: int,
    time: float,
    cuts: CutMap,
    moments: Sequence[Moment],
    within: tuple[float, float],
    config: MontageConfig | None = None,
) -> Shot | None:
    """A shot of ``clip`` built around ``time`` (seconds), as the pilot marked it: centred
    on it, ``shot_min_s``-``shot_max_s`` long where there is room, its ends in calm frames
    and out of any turn, and inside ``within`` (the room between the
    clip's other shots). It stays in the stretch of usable flying around ``time`` when
    there is one; a mark outside any (a ground tap the analysis cut) uses ``within`` alone.
    ``moments`` are those found in the clip; the ones it shows are rated with it. None
    when ``within`` has no room for `MIN_MARKED_S` around ``time``.
    """
    config = config or MontageConfig()
    fps, count = cuts.fps, len(cuts.usable)
    low, high = max(0, round(within[0] * fps)), min(count, round(within[1] * fps))
    at = min(max(round(time * fps), low), high - 1)
    if high - low < round(MIN_MARKED_S * fps) or not low <= at < high:
        return None
    if cuts.usable[at]:
        run_low, run_high = next((a, b) for a, b in runs(cuts.usable) if a <= at < b)
        if run_high - run_low >= round(MIN_MARKED_S * fps):
            low, high = max(low, run_low), min(high, run_high)
    shortest = min(round(config.shot_min_s * fps), high - low)
    longest = min(round(config.shot_max_s * fps), high - low)
    length = round((shortest + longest) / 2)
    start = min(max(at - length // 2, low), high - length)
    reach = round(config.snap_s * fps)
    start = _calmest(cuts.calm, start, reach, low, high)
    end = _calmest(cuts.calm, start + length, reach, low, high)
    end = min(max(end, start + shortest), start + longest, high)
    # Each end out of any turn: past it where there is room (the whole turn shows),
    # otherwise back before it. A mark is the pilot's choice, so the length may give.
    turning = cuts.turning
    first, last = start, end
    if turning[last - 1]:
        after = _turn_end(turning, last - 1) + 1
        last = after if after <= high else _turn_start(turning, last - 1)
    if turning[first]:
        before = _turn_start(turning, first) - 1
        first = before if before >= low else _turn_end(turning, first)
    if first <= at < last:
        start, end = first, last
    shown = tuple(m for m in moments if start <= m.time * fps < end)
    a, b = start / fps, end / fps
    speed = float(np.median(cuts.signals.speed[start:end]))
    score = sum(m.weight for m in shown) + config.speed_weight * min(speed, config.speed_cap)
    quality = rate_shot(a, b, [(m.time, m.name) for m in shown], cuts.signals)
    return Shot(clip, a, b, score, shown, quality)


def _clear_of_turns(
    turning: BoolArray, start: int, end: int, shortest: int, longest: int, low: int, high: int
) -> tuple[int, int] | None:
    """Move a shot's ends out of any turn (a flip or roll) they would cut through.

    The start moves back to before the turn (the viewer sees all of it) or, failing
    that, after it; the end moves on to after the turn or, failing that, back to before
    it. None if the shot cannot then stay ``shortest``-``longest`` frames long.
    """
    if turning[start]:
        before = _turn_start(turning, start)
        if before > low and end - (before - 1) <= longest:
            start = before - 1
        else:
            start = _turn_end(turning, start)
    if end - 1 >= start and turning[end - 1]:
        after = _turn_end(turning, end - 1)
        fits = after <= high and after - start <= longest
        end = after if fits else _turn_start(turning, end - 1)
    if not shortest <= end - start <= longest or turning[start] or turning[end - 1]:
        return None
    return start, end


def _turn_start(turning: BoolArray, index: int) -> int:
    """First frame of the turn that ``index`` is in."""
    while index > 0 and turning[index - 1]:
        index -= 1
    return index


def _turn_end(turning: BoolArray, index: int) -> int:
    """First frame after the turn that ``index`` is in."""
    while index < len(turning) and turning[index]:
        index += 1
    return index


def _calmest(calm: FloatArray, index: int, reach: int, low: int, high: int) -> int:
    first, last = max(low, index - reach), min(high, index + reach + 1)
    if last <= first:
        return index
    return first + int(np.argmin(calm[first:last]))


def _score(
    clip: int,
    start: float,
    end: float,
    moments: Sequence[Moment],
    motion: MotionSummary,
    config: MontageConfig,
) -> Shot | None:
    shown = tuple(m for m in moments if start + config.edge_s <= m.time < end - config.tail_s)
    if not shown:
        return None
    fps = motion.fps
    speed = float(np.median(motion.speed_smooth[round(start * fps) : round(end * fps)]))
    score = sum(m.weight for m in shown) + config.speed_weight * min(speed, config.speed_cap)
    return Shot(clip, start, end, score, shown)


def strength(shot: Shot) -> float:
    """What shots are ranked by: their moments and speed per square root of length, so a
    long shot needs more in it than a short one to rank as high."""
    return float(shot.score / shot.duration**0.5)


def pick_shots(
    candidates: Sequence[Shot], clip_count: int, config: MontageConfig | None = None
) -> tuple[Shot, ...]:
    """Non-overlapping shots in montage order (clip order, then time), best first where
    they compete for the same footage.

    Without a target length every distinct shot is kept: each one is built around a
    flip, spin or close pass, and a score threshold would also drop good shots that score
    low (weaves of close passes). With ``length_s`` the montage stops near that length
    and no clip fills more than its share.
    """
    config = config or MontageConfig()
    target = config.length_s
    contributing = len({shot.clip for shot in candidates})
    chosen: list[Shot] = []
    used = 0.0
    per_clip = [0.0] * max(1, clip_count)
    ranked = sorted(candidates, key=lambda s: (-strength(s), s.clip, s.start))
    for shot in ranked:
        if target is not None:
            cap = target * max(config.clip_share, 1.0 / max(1, contributing))
            if used >= target - config.length_slack_s:
                break
            if used + shot.duration > target + config.length_slack_s:
                continue
            if per_clip[shot.clip] + shot.duration > cap + config.length_slack_s:
                continue
        if any(
            other.clip == shot.clip
            and shot.start < other.end + config.min_gap_s
            and other.start < shot.end + config.min_gap_s
            for other in chosen
        ):
            continue
        chosen.append(shot)
        used += shot.duration
        per_clip[shot.clip] += shot.duration
    return tuple(sorted(chosen, key=lambda s: (s.clip, s.start)))
