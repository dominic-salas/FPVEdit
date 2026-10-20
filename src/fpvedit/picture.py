"""Find the part of the frame that shows the flight: the active picture and the OSD mask.

Goggle DVRs often record a 4:3 picture pillarboxed inside 16:9, and burn the flight
controller's on-screen display (OSD: timers, voltage, warnings) into the image. Black
bars halve measured motion, and OSD text is static high-contrast detail that makes a
moving scene look still, so analysis crops the bars and ignores the OSD.

Both are estimated from a few frames spread over the whole recording. Only keyframes
are decoded for this: they are intra-coded, so they are clean even in HEVC MPEG-TS
files whose frames right after a seek decode as grey garbage, and decoding them is
cheap.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from fpvedit.media import NO_WINDOW, MediaInfo, Tools

GrayFrame = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]

SAMPLE_FRAMES = 24
MIN_FRAMES = 6


@dataclass(frozen=True)
class Area:
    """A rectangle in decoded (post-rotation) pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def crop_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


@dataclass(frozen=True)
class PictureConfig:
    # A column/row belongs to the picture if, in most frames, it is clearly brighter than
    # the bars or carries noise (analog video is never perfectly flat).
    level_above_black: float = 8.0
    noise_std: float = 2.0
    min_active_fraction: float = 0.5  # of the frame, per axis; otherwise keep it all
    edge_inset_fraction: float = 0.004  # skip the soft transition into the bars
    # OSD glyphs: bright strokes with a dark outline, present in most sampled frames.
    glyph_level: int = 170
    glyph_outline: int = 90
    glyph_presence: float = 0.3
    # Other burned-in overlays (frame arms, logos): unchanging and detailed.
    static_stability: float = 0.8
    static_change: int = 10
    static_edge: int = 25
    max_overlay_box: float = 0.02  # larger static regions are scenery, not overlay


def decoded_size(info: MediaInfo) -> tuple[int, int]:
    width, height = info.video.width, info.video.height
    return (height, width) if info.video.rotation % 180 == 90 else (width, height)


def sample_frames(tools: Tools, info: MediaInfo, count: int = SAMPLE_FRAMES) -> list[GrayFrame]:
    """Up to ``count`` full-resolution grey frames spread over the recording.

    Keyframes only, unless the file has too few of them (long-GOP or intra-refresh
    streams); then every frame is decoded once and frames are picked by time.
    """
    duration = info.duration or 0.0
    step = max(duration / count, 0.5) if duration > 0 else 1.0
    frames = _grab(tools, info, step, keyframes_only=True)
    if len(frames) < min(MIN_FRAMES, count) and duration > step * MIN_FRAMES:
        frames = _grab(tools, info, step, keyframes_only=False)
    return frames[:count]


def _grab(tools: Tools, info: MediaInfo, step: float, *, keyframes_only: bool) -> list[GrayFrame]:
    width, height = decoded_size(info)
    pick = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step:.3f})'"
    args = [tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "quiet"]
    if keyframes_only:
        args += ["-skip_frame", "nokey"]
    args += [
        *("-i", str(info.path), "-map", f"0:{info.video.index}", "-an", "-sn", "-dn"),
        *("-vf", f"{pick},format=gray", "-fps_mode", "passthrough"),
        *("-f", "rawvideo", "-pix_fmt", "gray", "-"),
    ]
    # Errors are not fatal here: a damaged recording simply yields fewer frames, and
    # the main analysis pass reports decoding problems.
    raw = subprocess.run(args, capture_output=True, check=False, creationflags=NO_WINDOW).stdout
    size = width * height
    usable = len(raw) // size
    frames = np.frombuffer(raw[: usable * size], dtype=np.uint8).reshape(usable, height, width)
    return [frame for frame in frames if float(frame.std()) > 3.0]  # drop blank/grey frames


def detect_active_area(
    frames: list[GrayFrame], full: tuple[int, int], config: PictureConfig | None = None
) -> Area:
    """Bounding box of the picture inside black bars; the whole frame if unsure."""
    config = config or PictureConfig()
    width, height = full
    whole = Area(0, 0, width, height)
    if len(frames) < 2:
        return whole
    stack = np.stack(frames).astype(np.float32)
    columns = _active_profile(stack, axis=1, config=config)
    rows = _active_profile(stack, axis=2, config=config)
    xs, ys = np.flatnonzero(columns), np.flatnonzero(rows)
    if len(xs) == 0 or len(ys) == 0:
        return whole
    x0, x1 = _inset(int(xs[0]), int(xs[-1]) + 1, width, config)
    y0, y1 = _inset(int(ys[0]), int(ys[-1]) + 1, height, config)
    if (
        x1 - x0 < config.min_active_fraction * width
        or y1 - y0 < config.min_active_fraction * height
    ):
        return whole
    return Area(x0, y0, x1 - x0, y1 - y0)


def _active_profile(stack: NDArray[np.float32], *, axis: int, config: PictureConfig) -> BoolMask:
    """Per column (axis=1) or row (axis=2): does it show picture in most frames?"""
    means = stack.mean(axis=axis)
    noise = stack.std(axis=axis)
    black = float(np.median(np.percentile(means, 1, axis=1)))
    lit = (means > black + config.level_above_black) | (noise > config.noise_std)
    return np.asarray(lit.mean(axis=0) >= 0.5)


def _inset(start: int, end: int, size: int, config: PictureConfig) -> tuple[int, int]:
    """Trim the soft edge next to a bar and keep even offsets and sizes for FFmpeg."""
    inset = max(1, round(config.edge_inset_fraction * size))
    if start > 0:
        start += inset
    if end < size:
        end -= inset
    start += start % 2
    end -= (end - start) % 2
    return start, max(start + 2, end)


def detect_overlay(frames: list[GrayFrame], config: PictureConfig | None = None) -> BoolMask | None:
    """Mask of burned-in overlay pixels (OSD text and icons) at the frames' size.

    ``frames`` must already be cropped and scaled like the analysis frames. Returns
    None when there are too few frames to tell overlay from scenery.
    """
    config = config or PictureConfig()
    if len(frames) < MIN_FRAMES:
        return None
    stack = np.stack(frames)
    kernel = np.ones((3, 3), np.uint8)
    glyphs = np.zeros(stack.shape[1:], dtype=np.float64)
    for frame in stack:
        darkest_nearby = np.asarray(cv2.erode(frame, kernel), dtype=np.int16)
        bright = frame >= config.glyph_level
        outlined = darkest_nearby < frame.astype(np.int16) - config.glyph_outline
        glyphs += bright & outlined
    glyph_mask = glyphs / len(stack) >= config.glyph_presence

    changes = np.abs(np.diff(stack.astype(np.int16), axis=0))
    stable = (changes < config.static_change).mean(axis=0) >= config.static_stability
    median = np.median(stack, axis=0).astype(np.uint8)
    edges = np.asarray(cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel)) > config.static_edge
    mask = (glyph_mask | (stable & edges)).astype(np.uint8)

    # Join the strokes of each OSD element and cover its whole box, so digits that
    # change between samples are masked too. Large static regions are scenery.
    mask = np.asarray(
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filled = np.zeros_like(mask)
    for index in range(1, count):
        x, y, w, h, _ = (int(v) for v in stats[index])
        if w * h <= config.max_overlay_box * mask.size:
            filled[y : y + h, x : x + w] = 1
    grown = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return np.asarray(grown, dtype=bool)


def fit_frames(frames: list[GrayFrame], area: Area, size: tuple[int, int]) -> list[GrayFrame]:
    """Crop to ``area`` and scale to the analysis ``size`` (width, height)."""
    return [
        np.asarray(
            cv2.resize(
                frame[area.y : area.y + area.height, area.x : area.x + area.width],
                size,
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )
        for frame in frames
    ]
