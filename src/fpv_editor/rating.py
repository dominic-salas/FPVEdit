"""How good a shot is: what was done, how cleanly, how impressively, how watchably.

The criteria Dom approved on 2026-10-18 (``docs/rating-plan.md``). Each part is 0-1 and
comes from signals the analysis already measures; none of them recognises a trick's
quality directly, they are measured stand-ins:

- **trick** (what): the tricks in the shot by type (doubles above flips, rolls and power
  loops, above yaw spins, above close passes), with a bonus for tricks linked together and
  less for the same trick repeated;
- **clean** (how well): no crash right after the shot (Dom's failed tricks ended in one
  within a few seconds; such a shot gets 2 stars at most), little wobble after each trick,
  speed kept through it (his kept moments held 1.2x their speed, the cut ones 0.6x), and a
  full turn;
- **impressive** (context): how close the flying got (approach rate) and how fast it was;
  the image model's scenery is added later as a small nudge, never as a filter;
- **watchable** (picture): how much of the shot the motion fit could follow (blur and
  breakup), and that it is not too dark.

`Quality.score` weighs them 0.35 / 0.35 / 0.2 / 0.1 and `stars` turns the score into 1-5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

WEIGHTS = {"trick": 0.35, "clean": 0.35, "impressive": 0.2, "watchable": 0.1}
TRICK_VALUES = {
    "double back flip": 1.2,
    "double front flip": 1.2,
    "double roll": 1.2,
    "back flip": 0.8,
    "front flip": 0.8,
    "flip": 0.8,
    "roll": 0.8,
    "power loop": 0.8,
    "yaw spin": 0.6,
    "spin": 0.35,
    "close pass": 0.35,
}
# Score at which 2, 3, 4 and 5 stars begin: the 15/35/65/85 % points of the 128 distinct
# shots of the 16 test recordings, so a typical session spreads over all five.
STAR_BOUNDS = (0.56, 0.64, 0.71, 0.76)
FAILED_CLEAN = 0.3  # clean below this caps the rating at 2 stars


@dataclass(frozen=True)
class RatingConfig:
    combo_gap_s: float = 3.0  # tricks this close together form one line
    combo_bonus: float = 0.2  # per link in a line
    repeat_after: int = 2  # more of the same trick in one shot than this count half
    trick_scale: float = 1.5  # trick part = 1 - exp(-total / trick_scale)
    crash_near_s: float = 6.0  # a crash this soon after the shot: a failed attempt
    crash_soon_s: float = 10.0  # ... this soon: probably part of it
    wobble_ok: float = 8.0  # deg/s frame-to-frame rate change after a trick that is fine
    wobble_bad: float = 28.0  # ... and that looks like a messy recovery
    full_turn_deg: float = 200.0  # a full flip or roll measures at least about this
    close_low: float = 3.0  # approach rate (1/s) that counts as nothing special ...
    close_high: float = 9.0  # ... and as very close
    fast: float = 0.6  # picture widths/s that count as fast flying
    dark: float = 0.12  # mean brightness below this is too dark to enjoy


@dataclass(frozen=True)
class Quality:
    trick: float
    clean: float
    impressive: float
    watchable: float
    reasons: tuple[str, ...]  # short notes for the shot panel, best first
    indoorness: float | None = None  # from the image model, added after picking

    @property
    def score(self) -> float:
        nudge = 0.0
        if self.indoorness is not None:
            nudge = -0.1 if self.indoorness > 0.03 else 0.05  # cramped down, open up
        impressive = min(1.0, max(0.0, self.impressive + nudge))
        return float(
            WEIGHTS["trick"] * self.trick
            + WEIGHTS["clean"] * self.clean
            + WEIGHTS["impressive"] * impressive
            + WEIGHTS["watchable"] * self.watchable
        )

    @property
    def stars(self) -> int:
        stars = 1 + sum(1 for bound in STAR_BOUNDS if self.score >= bound)
        # A failed attempt (ending in a crash) is never more than 2 stars, however big.
        return min(stars, 2) if self.clean < FAILED_CLEAN else stars


@dataclass(frozen=True)
class ClipSignals:
    """Per-frame signals of one clip that the rating reads (from `montage._Motion`)."""

    fps: float
    rate: FloatArray  # deg/s, lightly smoothed
    rotation: FloatArray  # (frames, 3) deg/s
    speed: FloatArray  # picture widths/s, smoothed
    approach: FloatArray  # 1/s
    confidence: FloatArray  # 0-1
    brightness: FloatArray  # 0-1
    crashes: tuple[float, ...]  # start times of crash cuts


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def trick_part(moments: Sequence[tuple[float, str]], config: RatingConfig) -> tuple[float, str]:
    """Value of the tricks in a shot; ``moments`` are (time, name) in time order."""
    seen: dict[str, int] = {}
    total = 0.0
    for _, name in moments:
        seen[name] = seen.get(name, 0) + 1
        value = TRICK_VALUES.get(name, 0.35)
        total += value if seen[name] <= config.repeat_after else value / 2
    tricks = [t for t, name in moments if name not in ("close pass", "spin")]
    links = sum(1 for a, b in pairwise(tricks) if b - a <= config.combo_gap_s)
    total += config.combo_bonus * links
    names = sorted(seen, key=lambda n: -TRICK_VALUES.get(n, 0.35))
    note = (f"{links + 1} tricks linked" if links else names[0]) if names else "flying"
    return 1.0 - float(np.exp(-total / config.trick_scale)), note


def rate_shot(
    start: float,
    end: float,
    moments: Sequence[tuple[float, str]],
    signals: ClipSignals,
    config: RatingConfig | None = None,
) -> Quality:
    """Rate the shot ``start``-``end`` (seconds) showing ``moments`` (time, trick name)."""
    config = config or RatingConfig()
    fps = signals.fps
    a, b = max(0, round(start * fps)), min(len(signals.rate), round(end * fps))
    reasons: list[str] = []

    trick, note = trick_part(moments, config)
    reasons.append(note)

    clean = 1.0
    crash = min((c - end for c in signals.crashes if c >= start), default=None)
    if crash is not None and crash <= config.crash_near_s:
        clean *= 0.25
        reasons.append("ends in a crash")
    elif crash is not None and crash <= config.crash_soon_s:
        clean *= 0.7
        reasons.append("a crash soon after")
    turns = [t for t, name in moments if name not in ("close pass", "spin")]
    wobbles, kept, fulls = [], [], []
    for time in turns:
        i = round(time * fps)
        after = signals.rate[i + round(0.6 * fps) : i + round(1.6 * fps)]
        if len(after) > 2:
            wobbles.append(float(np.std(np.diff(after))))
        before = signals.speed[max(0, i - round(2 * fps)) : max(0, i - round(0.5 * fps))]
        later = signals.speed[i + round(0.5 * fps) : i + round(2 * fps)]
        if len(before) and len(later):
            kept.append(float(np.median(later)) / max(1e-3, float(np.median(before))))
        half = round(0.5 * fps)
        window = slice(max(0, i - half), i + half)
        if float(np.mean(signals.confidence[window])) >= 0.45:  # blur hides the turn
            fulls.append(float(np.abs(signals.rotation[window].sum(axis=0) / fps).max()))
    if wobbles:
        wobble = float(np.median(wobbles))
        span = config.wobble_bad - config.wobble_ok
        clean *= _clip(1 - (wobble - config.wobble_ok) / span, 0.4)
        reasons.append("smooth recovery" if wobble <= config.wobble_ok else "wobbly recovery")
    if kept:
        held = min(kept)
        clean *= _clip(0.5 + 0.5 * held, 0.5)
        if held >= 1.0:
            reasons.append("keeps its speed")
        elif held < 0.6:
            reasons.append("loses speed")
    if fulls and min(fulls) < config.full_turn_deg:
        clean *= _clip(min(fulls) / config.full_turn_deg, 0.6)
    close = _clip(
        (float(np.max(signals.approach[a:b], initial=0.0)) - config.close_low)
        / (config.close_high - config.close_low)
    )
    fast = _clip(float(np.median(signals.speed[a:b])) / config.fast) if b > a else 0.0
    impressive = 0.5 * close + 0.5 * fast
    if close >= 0.6:
        reasons.append("very close")
    if fast >= 0.9:
        reasons.append("fast")

    followed = float(np.mean(signals.confidence[a:b] >= 0.3)) if b > a else 0.0
    bright = float(np.mean(signals.brightness[a:b])) if b > a else 0.0
    watchable = _clip(1.0 - 2.0 * (1.0 - followed)) * (0.5 if bright < config.dark else 1.0)
    if bright < config.dark:
        reasons.append("dark")
    return Quality(trick, _clip(clean), impressive, watchable, tuple(reasons))
