"""The throttle, read from the Betaflight OSD burned into the DVR picture.

The OSD font is not known in advance (and Betaflight's fonts cannot ship with FPVEdit), so
the digits are learned from the recording itself: the flight timer's last digit counts
0-9 once a second. Steps:

1. Glyph pixels (bright strokes with a dark outline) present in most sampled frames mark
   the OSD elements: the throttle, the timers, the voltage...
2. One pass at `FPS` keeps only a narrow strip around each element, as stroke strength.
3. The running timer is the element whose tens digit changes every 10 s. Averaged over
   each second, its last digit gives ten digit images in counting order; of the ten ways
   to number them, the one under which the tens digit reads steady and steps up by one at
   each rollover is kept.
4. The throttle is the element whose readings (1-3 right-aligned digits) stay within
   0-100 and change the most. Each character is cut at its fixed screen position and
   matched to the nearest digit (with a few pixels of slack); a reading needs every
   character to match clearly, and a 3-frame majority smooths it.

Frames that cannot be read (static, the OSD blanked, analog breakup) stay NaN, never a
guess. A recording whose throttle is not on screen gives None.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from fpvedit.media import NO_WINDOW, MediaInfo, Tools
from fpvedit.picture import Area

FloatArray = NDArray[np.float64]
Strip = NDArray[np.float32]

VERSION = 1  # bump when the readings change (it names the throttle cache)
FPS = 12.0  # throttle samples per second
WIDTH, HEIGHT = 960, 720  # the 4:3 picture is read at this size (analog OSD: 32 px a character)
STRIP_H = 36  # rows of a glyph strip, centred on its element
STRIP_LEFT = 110  # columns kept left of an element's right edge ...
STRIP_RIGHT = 4  # ... and right of it
CELL = 22  # width of a centred character cut
SLACK = 6  # pixels a character may sit off its expected place
SAMPLES = 150  # frames sampled to find the OSD elements


@dataclass(frozen=True)
class OsdConfig:
    glyph_level: int = 170  # a glyph stroke is at least this bright ...
    glyph_outline: int = 90  # ... and this much brighter than its darkest 7x7 neighbour
    presence: float = 0.25  # share of sampled frames a glyph pixel shows in
    min_score: float = 0.5  # a character must match its digit at least this well ...
    min_margin: float = 0.03  # ... and this much better than any other digit
    blank_ink: float = 0.3  # a character with less ink than this share of usual is blank


@dataclass(frozen=True)
class Throttle:
    """Throttle in percent, sampled at `fps` from the first video frame; NaN = unread."""

    fps: float
    values: FloatArray

    def between(self, start: float, end: float) -> FloatArray:
        """The samples from ``start`` to ``end`` (seconds), NaN where unread."""
        a, b = max(0, round(start * self.fps)), max(0, round(end * self.fps))
        return self.values[a:b]

    @property
    def coverage(self) -> float:
        return float(np.isfinite(self.values).mean()) if len(self.values) else 0.0


Box = tuple[int, int, int, int]  # x, y, width, height in the 960x720 picture


def glyph_pixels(frame: NDArray[np.uint8], config: OsdConfig | None = None) -> NDArray[np.bool_]:
    config = config or OsdConfig()
    darkest = cv2.erode(frame, np.ones((7, 7), np.uint8)).astype(np.int16)
    return (frame >= config.glyph_level) & (darkest < frame.astype(np.int16) - config.glyph_outline)


def strokes(img: NDArray[np.uint8]) -> Strip:
    """Stroke strength of OSD glyphs (bright, outlined), softened; the scenery drops out."""
    a = img.astype(np.float32)
    darkest = cv2.erode(img, np.ones((7, 7), np.uint8)).astype(np.float32)
    s = np.clip((a - 150) / 60, 0, 1) * np.clip((a - darkest - 60) / 60, 0, 1)
    return np.asarray(cv2.GaussianBlur(s, (5, 5), 1.2), dtype=np.float32)


def find_elements(frames: list[NDArray[np.uint8]], config: OsdConfig | None = None) -> list[Box]:
    """Boxes of the OSD elements (runs of glyphs on one line) in sampled frames."""
    config = config or OsdConfig()
    if not frames:
        return []
    presence = np.mean([glyph_pixels(f, config) for f in frames], axis=0)
    mask = (presence >= config.presence).astype(np.uint8)
    joined = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 9))
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if 12 <= h <= 80 and w >= 12 and area >= 60:
            boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def characters(width: int) -> int:
    """Characters in an element this many pixels wide."""
    return max(1, round((width + 18) / 32))


def strip_of(frame: NDArray[np.uint8], box: Box) -> Strip:
    """The stroke strip of one element: STRIP_H rows around its middle, STRIP_LEFT columns
    left of its right edge (zero beyond the picture)."""
    x, y, w, h = box
    mid, right = y + h // 2, x + w
    pad = np.pad(frame, ((STRIP_H, STRIP_H), (STRIP_LEFT, STRIP_RIGHT)))
    rows = slice(mid + STRIP_H - STRIP_H // 2, mid + STRIP_H + STRIP_H // 2)
    cols = slice(right, right + STRIP_LEFT + STRIP_RIGHT)
    return strokes(np.ascontiguousarray(pad[rows, cols]))


def _norm(a: NDArray[np.float32]) -> NDArray[np.float32]:
    v = a - a.mean(axis=(-2, -1), keepdims=True)
    return np.asarray(v / (np.linalg.norm(v, axis=(-2, -1), keepdims=True) + 1e-6), np.float32)


def _centres(profile: FloatArray, apart: int = 18) -> list[int]:
    """Column peaks of a smoothed ink profile, at least ``apart`` apart."""
    p = np.convolve(profile, np.ones(9) / 9, mode="same")
    if not p.max() > 0:
        return []
    picked: list[int] = []
    for i in np.argsort(-p):
        if p[i] < 0.25 * p.max():
            break
        if all(abs(int(i) - j) >= apart for j in picked):
            picked.append(int(i))
    return sorted(picked)


def _cut(s: Strip, centre: int, slack: int = 0) -> Strip:
    pad = np.pad(s, ((slack, slack), (CELL + slack, CELL + slack)))
    c = centre + CELL + slack
    return pad[:, c - CELL // 2 - slack : c + CELL // 2 + slack]


def _per_second(cells: NDArray[np.float32], fps: int) -> NDArray[np.float32]:
    """The cell averaged over each second of a 1 Hz counter, aligned to its flips."""
    n = _norm(cells).reshape(len(cells), -1)
    change = 1 - (n[1:] * n[:-1]).sum(axis=1)
    phase = int(np.argmax([change[p::fps].mean() for p in range(fps)])) + 1
    starts = np.arange(phase, len(cells) - fps + 1, fps)
    if not len(starts):
        return np.zeros((0, *cells.shape[1:]), np.float32)
    return np.stack([cells[s + 1 : s + fps - 1].mean(axis=0) for s in starts]).astype(np.float32)


@dataclass(frozen=True)
class _Timer:
    last: NDArray[np.float32]  # per second
    tens: NDArray[np.float32]
    rollovers: NDArray[np.int64]  # seconds at which the tens digit changed
    pitch: int


def _timer(strips: NDArray[np.float32], fps: int) -> _Timer | None:
    cs = _centres(strips.mean(axis=0).sum(axis=0).astype(np.float64))
    if len(cs) < 2:
        return None
    last = _per_second(np.stack([_cut(s, cs[-1]) for s in strips]), fps)
    tens = _per_second(np.stack([_cut(s, cs[-2]) for s in strips]), fps)
    if len(tens) < 12:
        return None
    nt = _norm(tens)
    same = (nt[1:] * nt[:-1]).sum(axis=(1, 2))
    # A rollover is a clear drop below how steady the tens digit usually is.
    rollovers = np.flatnonzero(same < float(np.median(same)) - 0.2) + 1
    return _Timer(last, tens, rollovers, cs[-1] - cs[-2])


def _regular(rollovers: NDArray[np.int64]) -> list[int]:
    """Rollovers with another exactly 10 s later or earlier (a running timer)."""
    found = set(int(r) for r in rollovers)
    return sorted(r for r in found if r + 10 in found or r - 10 in found)


def learn_digits(timer: _Timer) -> NDArray[np.float32] | None:
    """Templates for 0-9 from a running timer, numbered by tens-digit consistency."""
    regular = _regular(timer.rollovers)
    if not regular:
        return None
    r0 = regular[0]
    by_phase = np.zeros((10, *timer.last.shape[1:]), np.float32)
    counts = np.zeros(10)
    for k in range(len(timer.last)):
        by_phase[(k - r0) % 10] += timer.last[k]
        counts[(k - r0) % 10] += 1
    if counts.min() == 0:
        return None
    by_phase /= counts[:, None, None]
    best: tuple[float, NDArray[np.float32]] | None = None
    tens = _norm(timer.tens).reshape(len(timer.tens), -1)
    rolled = np.isin(np.arange(1, len(timer.tens)), timer.rollovers)
    for offset in range(10):
        temps = np.roll(by_phase, offset, axis=0)
        read = (tens @ _norm(temps).reshape(10, -1).T).argmax(axis=1)
        steps = np.diff(read)
        fit = float(np.where(rolled, steps % 6 == 1, steps == 0).mean() + (read <= 5).mean())
        if best is None or fit > best[0]:
            best = (fit, temps)
    return best[1] if best is not None else None


def read_values(
    strips: NDArray[np.float32],
    digits: NDArray[np.float32],
    pitch: int,
    config: OsdConfig | None = None,
) -> FloatArray:
    """Read a right-aligned number of 1-3 digits in every strip; NaN where unclear."""
    config = config or OsdConfig()
    cs = _centres(strips.mean(axis=0).sum(axis=0).astype(np.float64))
    out = np.full(len(strips), np.nan)
    if not cs:
        return out
    right = cs[-1]
    places = [c for c in (right - 2 * pitch, right - pitch, right) if c >= 0]
    core = digits[:, SLACK : digits.shape[1] - SLACK, 3 : CELL - 3]
    usual = float(np.median([_cut(s, right).sum() for s in strips]))
    for i, s in enumerate(strips):
        if _cut(s, right).sum() < config.blank_ink * usual:
            continue  # right-aligned: the last character always has ink
        text, ok = "", True
        for c in places:
            cell = _cut(s, c)
            if cell.sum() < config.blank_ink * usual:
                if text:
                    ok = False  # a gap inside the number
                continue
            big = _cut(s, c, SLACK)
            scores = np.array([cv2.matchTemplate(big, t, cv2.TM_CCOEFF_NORMED).max() for t in core])
            top = np.sort(scores)
            if top[-1] < config.min_score or top[-1] - top[-2] < config.min_margin:
                ok = False
                break
            text += str(int(scores.argmax()))
        if ok and text and int(text) <= 100:
            out[i] = float(int(text))
    return _majority(out)


def _majority(values: FloatArray) -> FloatArray:
    """Each sample becomes the value at least two of its three neighbours agree on."""
    out = values.copy()
    for i in range(1, len(values) - 1):
        window = values[i - 1 : i + 2]
        window = window[np.isfinite(window)]
        if len(window) >= 2:
            seen, counts = np.unique(window, return_counts=True)
            if counts.max() >= 2:
                out[i] = seen[counts.argmax()]
    return out


def find_throttle(
    elements: list[Box],
    strips: dict[Box, NDArray[np.float32]],
    fps: int,
    config: OsdConfig | None = None,
) -> tuple[Box, FloatArray] | None:
    """The throttle element and its readings, given every element's strips."""
    timers = []
    for box in elements:
        if characters(box[2]) in (5, 6):
            t = _timer(strips[box], fps)
            if t is not None:
                timers.append((len(_regular(t.rollovers)), box, t))
    timers.sort(key=lambda item: -item[0])
    if not timers or timers[0][0] < 2:
        return None
    _, timer_box, timer = timers[0]
    digits = learn_digits(timer)
    if digits is None:
        return None
    best: tuple[float, Box, FloatArray] | None = None
    for box in elements:
        if box == timer_box or not 1 <= characters(box[2]) <= 3:
            continue
        values = read_values(strips[box], digits, timer.pitch, config)
        read = np.isfinite(values)
        if read.mean() < 0.2 or float(np.nanstd(values)) <= 5:
            continue
        changes = float(np.mean(np.diff(values[read]) != 0))
        score = float(read.mean()) * changes
        if best is None or score > best[0]:
            best = (score, box, values)
    return (best[1], best[2]) if best is not None else None


def _decode(tools: Tools, info: MediaInfo, area: Area, rate: str) -> Iterator[NDArray[np.uint8]]:
    """Gray frames of the active picture at WIDTH x HEIGHT, at the FFmpeg ``fps`` rate."""
    args = [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error"),
        *("-i", str(info.path), "-map", f"0:{info.video.index}", "-an", "-sn", "-dn"),
        *("-vf", f"{area.crop_filter},scale={WIDTH}:{HEIGHT},fps={rate},format=gray"),
        *("-f", "rawvideo", "-pix_fmt", "gray", "-"),
    ]
    size = WIDTH * HEIGHT
    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW
    ) as process:
        stdout = process.stdout
        if stdout is None:
            return
        while True:
            raw = stdout.read(size)
            if len(raw) < size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH)


def read_throttle(
    tools: Tools,
    info: MediaInfo,
    area: Area,
    *,
    progress: Callable[[float], None] | None = None,
    config: OsdConfig | None = None,
) -> Throttle | None:
    """Read the throttle of a recording (None when it is not on screen)."""
    duration = info.duration or 0.0
    if duration <= 0:
        return None
    sample_rate = f"{min(FPS, SAMPLES / duration):.4f}"
    elements = find_elements(list(_decode(tools, info, area, sample_rate)), config)
    if not elements:
        return None
    strips: dict[Box, list[Strip]] = {box: [] for box in elements}
    expected = max(1, int(duration * FPS))
    for count, frame in enumerate(_decode(tools, info, area, f"{FPS:g}")):
        for box in elements:
            strips[box].append(strip_of(frame, box))
        if progress is not None and count % 60 == 0:
            progress(min(1.0, count / expected))
    stacked = {box: np.stack(s) for box, s in strips.items() if s}
    if len(stacked) != len(elements):
        return None
    found = find_throttle(elements, stacked, int(FPS), config)
    if found is None:
        return None
    return Throttle(FPS, found[1])
