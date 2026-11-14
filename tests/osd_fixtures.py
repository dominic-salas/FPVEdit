"""Synthetic recordings of a digital (HD) on-screen display, drawn with OpenCV's own font.

Betaflight's HD OSD (MSP DisplayPort, as HDZero goggles record it) is a grid of about 50
columns by 18 rows over the whole 16:9 frame: narrow, tall characters, white with a dark
outline, with items spread out to the edges. No Betaflight or HDZero font is shipped:
every glyph is a tall, narrow rendering of OpenCV's Hershey font. The picture behind it is
clean (no analog noise) and moves steadily.

An `HdClip` is a short flight with known ground truth: a boot logo at the start, the
disarmed layout, an ARMED message, a flight timer counting from there with a throttle that
changes twice a second, then the statistics screen of the disarm.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

Frame = np.ndarray
COLUMNS, ROWS = 50, 18

BOOT_LOGO = ("BETAFLIGHT", "4.4.2", "TARGET", "BOOTING")
STATS = (
    "FLIGHT TIME",
    "00:24",
    "MAX SPEED",
    "62",
    "MAX CURRENT",
    "31",
    "USED MAH",
    "612",
    "MIN VOLTAGE",
    "14.2V",
)


@cache
def _glyph(char: str, ink_w: int, ink_h: int, outline: int, weight: int) -> tuple[Frame, Frame]:
    """(ink mask, ink plus outline mask) of one character, ``ink_h`` tall.

    Characters keep their natural widths relative to a digit's ``ink_w``; narrow ones
    ('.', ':', '1') stay narrow.
    """
    canvas = np.zeros((120, 120), np.uint8)
    base = 90
    cv2.putText(canvas, char, (30, base), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, weight)
    zero = np.zeros_like(canvas)
    cv2.putText(zero, "0", (30, base), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, weight)
    rows = np.flatnonzero(zero.any(axis=1))
    zero_cols = np.flatnonzero(zero.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    x_scale = ink_w / (zero_cols[-1] - zero_cols[0] + 1)
    cols = np.flatnonzero(canvas.any(axis=0))
    if not len(cols):
        empty = np.zeros((ink_h + 2 * outline, ink_w + 2 * outline), bool)
        return empty, empty
    crop = canvas[top:bottom, cols[0] : cols[-1] + 1]
    width = max(2, round(crop.shape[1] * x_scale))
    ink = cv2.resize(crop, (width, ink_h), interpolation=cv2.INTER_AREA) > 90
    padded = np.pad(ink, outline)
    grown = cv2.dilate(
        padded.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1,) * 2),
    )
    return padded, grown > 0


@dataclass(frozen=True)
class Grid:
    """Where the characters go: ``pitch`` apart, in a COLUMNS x ROWS grid centred in the frame."""

    width: int
    height: int
    pitch: int
    weight: int = 3  # stroke weight: 3 is bold, 2 is thin

    @property
    def cell_h(self) -> int:
        return round(1.5 * self.pitch)

    @property
    def ink_w(self) -> int:
        return round(0.66 * self.pitch)

    @property
    def ink_h(self) -> int:
        return round(1.17 * self.pitch)

    @property
    def outline(self) -> int:
        return max(1, round(self.pitch / 12))

    def draw(self, frame: Frame, text: str, column: int, row: int, *, right: bool = False) -> None:
        """Draw ``text`` from ``column`` (or ending at it, if ``right``) on ``row``."""
        if right:
            column -= len(text) - 1
        x0 = (self.width - COLUMNS * self.pitch) // 2 + column * self.pitch
        y0 = (
            (self.height - ROWS * self.cell_h) // 2
            + row * self.cell_h
            + (self.cell_h - self.ink_h) // 2
        )
        for k, char in enumerate(text):
            if char == " ":
                continue
            ink, grown = _glyph(char, self.ink_w, self.ink_h, self.outline, self.weight)
            x = x0 + k * self.pitch + (self.pitch - ink.shape[1]) // 2
            y = y0 - self.outline
            region = frame[y : y + ink.shape[0], x : x + ink.shape[1]]
            h, w = region.shape
            region[grown[:h, :w]] = 0
            region[ink[:h, :w]] = 255


class HdClip:
    """A flight in HD OSD: ``frames()`` at ``fps``, and the truth it should be read as."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        pitch: int | None = None,
        weight: int = 3,
        seconds: float = 32.0,
        boot: float = 1.5,
        arm: float = 3.0,
        disarm: float = 28.0,
        stats_s: float = 3.0,
        goggle: tuple[str, ...] = (),
        seed: int = 0,
    ) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.goggle = goggle  # the goggles' own status text: on every screen, always
        self.grid = Grid(width, height, pitch or round(24 * width / 1280), weight)
        self.seconds, self.boot, self.arm, self.disarm, self.stats_s = (
            seconds,
            boot,
            arm,
            disarm,
            stats_s,
        )
        rng = np.random.default_rng(seed)
        self._levels = np.clip(30 + rng.normal(0, 22, int(seconds * 2) + 2), 0, 100).round()
        noise = cv2.GaussianBlur(
            rng.normal(0, 1, (height, 3 * width)).astype(np.float32), (0, 0), 6
        )
        texture = noise[:, width : 2 * width] / noise.std() * 32 + 100
        self._scenery = np.tile(np.clip(texture, 30, 160), (1, 2)).astype(np.uint8)

    @property
    def armed_from(self) -> float:
        """When the ARMED message ends: the flight starts."""
        return self.arm + 0.5

    def throttle_at(self, t: float) -> int:
        if self.armed_from <= t < self.disarm:
            return int(self._levels[int(t * 2)])
        return 0

    def throttle_series(self, count: int, rate: float = 12.0) -> np.ndarray:
        return np.array([self.throttle_at(i / rate) for i in range(count)], dtype=np.float64)

    def screen_at(self, t: float) -> str:
        if t < self.boot:
            return "boot"
        if self.arm <= t < self.armed_from:
            return "armed"
        if self.disarm <= t < self.disarm + self.stats_s:
            return "stats"
        return "layout"

    def frame(self, t: float) -> Frame:
        shift = int(t * 240) % self.width
        frame = self._scenery[:, shift : shift + self.width].copy()
        grid, screen = self.grid, self.screen_at(t)
        if screen == "boot":
            for k, line in enumerate(BOOT_LOGO):
                grid.draw(frame, line, 20 - len(line) // 2 + 5, 6 + k)
        elif screen == "armed":
            grid.draw(frame, "ARMED", 22, 9)
        elif screen == "stats":
            for k, line in enumerate(STATS):
                grid.draw(frame, line, 20 - len(line) // 2 + 5, 4 + k)
        else:
            flown = min(max(t - self.armed_from, 0.0), self.disarm - self.armed_from)
            grid.draw(frame, f"{self.throttle_at(t):3d}", 47, 14, right=True)
            grid.draw(frame, f"{int(flown) // 60:02d}:{int(flown) % 60:02d}", 47, 16, right=True)
            grid.draw(frame, "16.8V", 1, 17)
            grid.draw(frame, "RSSI 99", 1, 1)
            grid.draw(frame, "LQ 100", COLUMNS - 1, 0, right=True)
        for k, item in enumerate(self.goggle):
            grid.draw(frame, item, 21, k)
        return frame

    def frames(self) -> Iterator[Frame]:
        for i in range(int(self.seconds * self.fps)):
            yield self.frame(i / self.fps)

    def write(self, path: Path, *, container: str = "mp4", crf: int = 16) -> Path:
        """Encode as H.264 (what the goggles' DVR files hold, for the OSD's purposes)."""
        args = [
            *("ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y"),
            *("-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{self.width}x{self.height}"),
            *("-r", str(self.fps), "-i", "-"),
            *("-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf), "-pix_fmt", "yuv420p"),
            *("-f", container, str(path)),
        ]
        with subprocess.Popen(args, stdin=subprocess.PIPE) as process:
            assert process.stdin is not None
            for frame in self.frames():
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg could not encode the OSD fixture")
        return path
