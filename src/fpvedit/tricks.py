"""Names for the fast turns the montage finds, from which way the camera turned.

The camera is fixed to the quad, tilted up, so a turn of the quad shows up in the
camera's own rotation rates (`motion.py`: pitch, yaw, roll): a flip is pitch, a roll is
roll, and because of the up-tilt a roll also shows some yaw *in the same direction*,
while a yaw spin shows some roll *in the opposite direction*. Angles come from optical
flow through a fisheye model with
an assumed field of view and lose some of the fastest frames to motion blur, so a full
360-degree flip measures about 200-280 degrees; the thresholds below allow for that.

These are descriptions of the rotation, not a trained recogniser: the names are what
the turn looks like. A turn whose axes mix, or that the motion fit could not follow
through the blur, stays a plain "flip". Power loops are found by their shape (a slow
full pitch turn away from any flip).

When gravity measured a turn (`attitude.py`), `name_measured` names it from the quad's own
axes and its real angle instead: whole turns by count ("double back flip"), part turns by
how far they went ("half roll", "3/4 front flip").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fpvedit.attitude import Attitude, Turn

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrickConfig:
    dominant: float = 2.0  # one axis turned at least this many times more than the others
    lateral_partner: float = 0.25  # yaw/roll below this share of the other counts as zero
    double_deg: float = 420.0  # measured turn of a double flip or roll (a single is ~200-300)
    min_confidence: float = 0.45  # mean motion-fit confidence needed to name the axis
    loop_deg: float = 250.0  # pitch over `loop_s` for a power loop ...
    loop_s: float = 2.5
    loop_fast_deg: float = 180.0  # ... while no single second turns this far (not a flip)
    measured_dominant: float = 1.5  # with the up-tilt taken out, the axes separate better


def name_turn(pitch: float, yaw: float, roll: float, config: TrickConfig | None = None) -> str:
    """Name one fast turn from the net degrees it turned on each camera axis."""
    config = config or TrickConfig()
    lateral = float(np.hypot(yaw, roll))
    if abs(pitch) >= config.dominant * lateral:
        kind = "back flip" if pitch > 0 else "front flip"
        return f"double {kind}" if abs(pitch) >= config.double_deg else kind
    if lateral >= config.dominant * abs(pitch):
        small = min(abs(yaw), abs(roll)) < config.lateral_partner * max(abs(yaw), abs(roll))
        # Without a partner axis the larger one decides; with one, the same direction is
        # the up-tilt seen in a roll, the opposite direction a yaw spin.
        is_roll = abs(roll) >= abs(yaw) if small else (yaw > 0) == (roll > 0)
        if is_roll:
            return "double roll" if lateral >= config.double_deg else "roll"
        return "yaw spin"
    return "flip"  # pitch and a sideways turn together: the axis is unclear


def name_fast_turn(
    rotation: FloatArray,
    confidence: FloatArray,
    index: int,
    fps: float,
    config: TrickConfig | None = None,
) -> str:
    """Name the turn peaking at frame ``index`` of ``rotation`` (deg/s: pitch, yaw, roll).

    When the motion fit was unsure through most of the turn (blur), or the axes mix, the
    name stays a plain "flip".
    """
    config = config or TrickConfig()
    half = max(1, round(0.5 * fps))
    low, high = max(0, index - half), index + half
    if float(np.mean(confidence[low:high])) < config.min_confidence:
        return "flip"
    pitch, yaw, roll = (rotation[low:high].sum(axis=0) / fps).tolist()
    return name_turn(pitch, yaw, roll, config)


def power_loop_frames(
    rotation: FloatArray, usable: NDArray[np.bool_], fps: float, config: TrickConfig | None = None
) -> FloatArray:
    """Per frame: net pitch over the `loop_s` around it where that is a slow full turn
    (no second within it turns `loop_fast_deg`), else 0. Peaks are power loops."""
    config = config or TrickConfig()
    count = len(rotation)
    width = max(1, round(config.loop_s * fps))
    second = max(1, round(fps))
    total = np.concatenate([[0.0], np.cumsum(rotation[:, 0])])
    magnitude = np.concatenate([[0.0], np.cumsum(np.linalg.norm(rotation, axis=1))])
    indices = np.arange(count)
    lows = np.maximum(0, indices - width // 2)
    highs = np.minimum(count, indices + width // 2)
    net = np.abs(total[highs] - total[lows]) / fps
    starts = np.arange(count)
    per_second = (magnitude[np.minimum(count, starts + second)] - magnitude[starts]) / fps
    blocked = np.concatenate([[0], np.cumsum(~usable)])
    result = np.zeros(count)
    for index in np.flatnonzero(net >= config.loop_deg):
        low, high = int(lows[index]), int(highs[index])
        if blocked[high] - blocked[low] > 0:
            continue
        if float(per_second[low : max(low + 1, high - second)].max()) < config.loop_fast_deg:
            result[index] = net[index]
    return result


COUNTS = {1: "", 2: "double ", 3: "triple "}
PARTS = {1: "quarter ", 2: "half ", 3: "3/4 "}
FRACTIONS = {1: "1/4", 2: "1/2", 3: "3/4"}


def name_measured(turn: Turn, attitude: Attitude, config: TrickConfig | None = None) -> str:
    """Name a turn that gravity checked, from the quad's axes and its measured angle."""
    config = config or TrickConfig()
    pitch, yaw, roll = attitude.body_axis(turn)
    sizes = sorted(((abs(pitch), "pitch"), (abs(yaw), "yaw"), (abs(roll), "roll")), reverse=True)
    (top, axis), (second, _) = sizes[0], sizes[1]
    if top < config.measured_dominant * second:
        kind = "flip"  # the axes mix: the kind is unclear
    elif axis == "pitch":
        kind = "back flip" if pitch > 0 else "front flip"
    else:
        kind = "roll" if axis == "roll" else "yaw spin"
    if turn.whole:
        return f"{COUNTS.get(turn.whole, f'{turn.whole}x ')}{kind}"
    quarters = round(turn.degrees / 90)
    if quarters % 4 == 0:
        return kind  # nearly whole, but gravity did not quite agree
    whole, part = divmod(quarters, 4)
    return f"{whole} {FRACTIONS[part]} {kind}s" if whole else f"{PARTS[part]}{kind}"
