"""Pick the best shots from several clips for one montage.

Each clip is first cleaned with the flight-state rules of `highlights.py`: boot and menus,
time disarmed, static, sitting on the ground, crashes with the failed move before them
and re-arm churn are never usable. What remains is searched for *moments*:

- flips and rolls: the camera turns 180 degrees or more within a second (a big flip,
  330 degrees or more, counts extra);
- spins and orbits: turning sustained over four seconds, well above the clip's usual
  level and at least ``spin_floor`` on average (a flip in calm flying also counts as
  one: on Dom's labelled flights, scoring it both ways picked his tricks more often);
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

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from fpv_editor.flight import rolling_median, runs
from fpv_editor.highlights import (
    CutReason,
    Interval,
    Selection,
    SelectionConfig,
    SelectionMode,
    _Motion,
    window_sum,
)
from fpv_editor.measure import Measurements

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


@dataclass(frozen=True)
class Shot:
    clip: int  # index of the clip in the list given to the montage
    start: float
    end: float
    score: float
    moments: tuple[Moment, ...]

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
    # scenery (with the image model): an indoor-looking shot stays only if it is as good
    # as the best quarter of the session's outdoor shots
    indoor_margin: float = 0.03  # indoorness above this looks indoor (`scenery.py`)
    indoor_bar_quantile: float = 0.75


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
    rate = _Motion(measurements).rate
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
    motion = _Motion(measurements)
    second = max(1, round(fps))
    turned_1s = np.linalg.norm(window_sum(motion.rotation, second), axis=1) / fps
    half = max(1, round(0.5 * fps))
    smoothed = window_sum(motion.rotation, half) / half
    four = max(1, round(4 * fps))
    sustained = window_sum(np.linalg.norm(smoothed, axis=1), four) / four
    approach = rolling_median(motion.divergence, max(1, round(0.25 * fps)))

    moments = []
    for index in _peaks(np.where(usable, turned_1s, 0.0), config.flip_deg, config.flip_gap_s, fps):
        big = turned_1s[index] >= config.big_flip_deg
        weight = config.flip_weight + (config.big_flip_weight if big else 0.0)
        moments.append(Moment(index / fps, MomentKind.FLIP, weight))
    spin = max(config.spin_floor, config.spin_factor * float(np.median(sustained[usable])))
    for index in _peaks(np.where(usable, sustained, 0.0), spin, config.spin_gap_s, fps):
        moments.append(Moment(index / fps, MomentKind.SPIN, config.spin_weight))
    close = max(config.pass_floor, float(np.quantile(approach[usable], config.pass_quantile)))
    baseline = rolling_median(approach, max(1, round(config.pass_baseline_s * fps)))
    sudden = usable & (approach - baseline >= config.pass_prominence)
    for index in _peaks(np.where(sudden, approach, -np.inf), close, config.pass_gap_s, fps):
        moments.append(Moment(index / fps, MomentKind.PROXIMITY, config.pass_weight))
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


def candidate_shots(
    clip: int,
    measurements: Measurements,
    selection: Selection,
    config: MontageConfig | None = None,
    selection_config: SelectionConfig | None = None,
) -> list[Shot]:
    """Every shot worth considering in one clip (they overlap; `pick_shots` chooses)."""
    config = config or MontageConfig()
    fps = measurements.fps
    usable = usable_footage(measurements, selection, selection_config)
    moments = find_moments(measurements, usable, config)
    if not moments:
        return []
    motion = _Motion(measurements)
    # Frames whose motion could not be measured (blur in a fast flip) are not calm.
    rate = np.where(np.isnan(measurements.motion).any(axis=1), np.inf, motion.rate)
    calm = rolling_median(rate, max(1, round(0.3 * fps)))
    # A cut inside a turn, or right next to a fast one, shows a flip cut short.
    fast = rolling_median(rate, max(1, round(0.1 * fps))) >= config.fast_turn_rate
    near_fast = window_sum(fast.astype(np.float64), 2 * round(config.fast_turn_gap_s * fps) + 1) > 0
    turning = (calm >= config.turning_rate) | near_fast
    shortest = round(config.shot_min_s * fps)
    longest = round(config.shot_max_s * fps)
    reach = round(config.snap_s * fps)
    lengths = {shortest, round((shortest + longest) / 2), longest}
    shots: dict[tuple[int, int], Shot] = {}
    for low, high in runs(usable):
        if high - low < shortest:
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
                    shots[(start, end)] = shot
    return list(shots.values())


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
    motion: _Motion,
    config: MontageConfig,
) -> Shot | None:
    shown = tuple(m for m in moments if start + config.edge_s <= m.time < end - config.tail_s)
    if not shown:
        return None
    fps = motion.fps
    speed = float(np.median(motion.speed_smooth[round(start * fps) : round(end * fps)]))
    score = sum(m.weight for m in shown) + config.speed_weight * min(speed, config.speed_cap)
    return Shot(clip, start, end, score, shown)


def pick_shots(
    candidates: Sequence[Shot], clip_count: int, config: MontageConfig | None = None
) -> tuple[Shot, ...]:
    """Non-overlapping shots in montage order (clip order, then time), best first where
    they compete for the same footage.

    Without a target length every distinct shot is kept: each one is built around a
    flip, spin or close pass, and on Dom's hand-labelled recordings any score threshold
    also dropped shots he had marked as worth keeping (weaves of close passes score
    low). With ``length_s`` the montage stops near that length and no clip fills more
    than its share.
    """
    config = config or MontageConfig()
    target = config.length_s
    contributing = len({shot.clip for shot in candidates})
    chosen: list[Shot] = []
    used = 0.0
    per_clip = [0.0] * max(1, clip_count)
    ranked = sorted(candidates, key=lambda s: (-s.score / s.duration**0.5, s.clip, s.start))
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


def leave_out_weak_indoor(
    shots: Sequence[Shot], indoorness: Sequence[float | None], config: MontageConfig | None = None
) -> tuple[tuple[Shot, ...], tuple[Shot, ...]]:
    """Split ``shots`` into those kept and the indoor-looking ones left out.

    Indoor shots are compared with the outdoor shots of the same session: one stays only
    if its score per sqrt(second), the ranking `pick_shots` uses, reaches the
    ``indoor_bar_quantile`` of the outdoor shots. A session with no outdoor shots keeps
    everything; a shot without an indoorness (no readable frame) counts as outdoor.
    """
    config = config or MontageConfig()
    if len(indoorness) != len(shots):
        raise ValueError("one indoorness per shot is required")
    indoor = [v is not None and v > config.indoor_margin for v in indoorness]
    outdoor = [s.score / s.duration**0.5 for s, i in zip(shots, indoor, strict=True) if not i]
    if not outdoor:
        return tuple(shots), ()
    bar = float(np.quantile(outdoor, config.indoor_bar_quantile))
    kept: list[Shot] = []
    left_out: list[Shot] = []
    for shot, inside in zip(shots, indoor, strict=True):
        weak = inside and shot.score / shot.duration**0.5 < bar
        (left_out if weak else kept).append(shot)
    return tuple(kept), tuple(left_out)
