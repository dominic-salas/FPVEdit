"""Per-frame measurements of a recording, from one FFmpeg decode of every frame.

FFmpeg crops the active picture, resamples to the nominal frame rate (a gap from signal
loss becomes repeated frames) and scales to a fixed analysis height. For each frame:

- picture: brightness, contrast, fine-grain noise and change from the previous frame,
  all outside the OSD, so "no signal", static and a still picture can be told apart;
- motion (`motion.py`): pitch, yaw and roll rates, a confidence, flow speed and the
  expansion rate, measured at 240 px wide;
- OSD: how much of the usual on-screen display layout is showing, and where text-like
  pixels appear outside it. Betaflight hides the normal layout while it shows the
  "ARMED" message or the post-flight statistics, which marks arming and landing.
- a small thumbnail every few frames, for scene comparisons.

Frame ``i`` is ``i / fps`` seconds after the first video frame (the analysis timeline,
which rendering also uses).
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np
from numpy.typing import NDArray

from fpvedit.errors import MediaToolError
from fpvedit.media import NO_WINDOW, MediaInfo, Tools
from fpvedit.motion import FIELDS, MotionConfig, MotionEstimator
from fpvedit.picture import Area, PictureConfig, decoded_size, detect_overlay, fit_frames

FloatArray = NDArray[np.float64]
GrayFrame = NDArray[np.uint8]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class MeasureConfig:
    height: int = 360  # analysis height for the picture and OSD (OSD text stays legible)
    flow_width: int = 240  # motion is measured at this width
    text_bands: int = 16  # horizontal bands for locating text outside the OSD layout
    glyph_level: int = 190  # OSD glyphs: bright strokes ...
    glyph_drop: int = 120  # ... at least this much brighter than their darkest neighbour
    thumbnail: tuple[int, int] = (48, 36)
    thumbnail_every: int = 6  # frames
    layout_samples: int = 48  # frames spread over the recording for the OSD layout
    # A recording that shows the stats screen for a long time (goggles left recording
    # after landing) has two sets of steady text: the layout and the stats. They never
    # show together, and the stats text sits in the middle rows.
    screen_overlap: float = 0.1  # the two sets share at most this share of pixels ...
    screen_min_share: float = 0.2  # ... each holds at least this share of the candidates ...
    screen_centre_share: float = 0.75  # ... and the stats set has this share in the middle
    motion: MotionConfig = field(default_factory=MotionConfig)


@dataclass(frozen=True)
class Measurements:
    fps: float
    brightness: FloatArray  # mean luma outside the OSD, 0..1
    contrast: FloatArray  # luma standard deviation outside the OSD, 0..1
    noise: FloatArray  # mean |pixel - 3x3 median| at 240 px, 0..1 (static is high)
    change: FloatArray  # mean |pixel change| vs the previous frame at 240 px, 0..1
    motion: FloatArray  # (frames, 6): see `motion.py`; row 0 is all NaN
    layout: FloatArray  # share of the OSD layout's glyph pixels showing (NaN: no OSD)
    text: NDArray[np.float32]  # (frames, bands): glyph pixels outside the layout, share
    thumbnails: NDArray[np.uint8]  # (frames // thumbnail_every, h, w)
    thumbnail_every: int

    def __len__(self) -> int:
        return len(self.brightness)


@dataclass(frozen=True)
class Layout:
    """Where things are in the analysis frame."""

    size: tuple[int, int]  # analysis width, height
    overlay: Mask | None  # OSD and other burned-in overlay pixels
    glyphs: Mask | None  # pixels that usually show OSD glyphs (the normal layout)
    usable: Mask  # pixels at flow size whose motion can be trusted
    text_area: Mask  # where text other than the layout (stats, logo, messages) is counted


def analysis_frame_size(info: MediaInfo, area: Area, height: int) -> tuple[int, int]:
    """Even width x ``height`` that keeps the display aspect ratio of ``area``."""
    full_width, full_height = decoded_size(info)
    aspect = info.display_aspect * (area.width / full_width) / (area.height / full_height)
    return max(2, round(height * aspect / 2) * 2), height


def plan_layout(
    samples: list[GrayFrame],
    area: Area,
    size: tuple[int, int],
    config: MeasureConfig,
    picture: PictureConfig,
) -> Layout:
    """OSD masks and the usable-for-motion mask, from frames spread over the recording."""
    fitted = fit_frames(samples, area, size)
    overlay = detect_overlay(fitted, picture)
    glyphs = None
    text_area = ~overlay if overlay is not None else np.ones(size[::-1], dtype=bool)
    if overlay is not None and len(fitted) >= 6:
        stack = np.stack([_glyph_pixels(f, config) & overlay for f in fitted])
        typical = stack.mean(axis=0)
        stats = _stats_screen(stack, typical >= 0.3, config)
        if stats is not None:
            stats_pixels, layout_samples = stats
            typical = stack[layout_samples].mean(axis=0)
            nearby = cv2.dilate(stats_pixels.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            text_area |= nearby & (typical < 0.3)
        glyphs = overlay & (typical >= 0.3)
        if glyphs.sum() < 50:  # too little to judge the layout by
            glyphs = None
    flow_size = _flow_size(size, config)
    usable = np.ones((flow_size[1], flow_size[0]), dtype=bool)
    if overlay is not None:
        small = cv2.resize(overlay.astype(np.uint8), flow_size, interpolation=cv2.INTER_NEAREST)
        usable &= small == 0
    if len(fitted) >= 6:
        # Props, frame arms and dead corners barely change while the scenery moves.
        small_frames = np.stack(
            [cv2.resize(f, flow_size, interpolation=cv2.INTER_AREA) for f in fitted]
        ).astype(np.float32)
        spread = small_frames.std(axis=0)
        usable &= spread >= 0.6 * float(np.median(spread))
    border = max(3, round(3 * flow_size[0] / 240))
    usable[:border] = usable[-border:] = False
    usable[:, :border] = usable[:, -border:] = False
    return Layout(size=size, overlay=overlay, glyphs=glyphs, usable=usable, text_area=text_area)


def _stats_screen(
    stack: NDArray[np.bool_], candidates: Mask, config: MeasureConfig
) -> tuple[Mask, NDArray[np.bool_]] | None:
    """Pixels of a long-shown stats screen, and the samples showing the layout instead.

    The samples are split in two by their steady glyph pixels (2-means, seeded with the
    two most different samples). Usually one group is the layout and the other is frames
    where it is hard to see; only two large, disjoint sets of text, one of them in the
    middle rows, are taken as layout and stats screen.
    """
    pixels = stack[:, candidates].astype(np.float64)
    if len(pixels) < 2 or pixels.shape[1] == 0:
        return None
    distance = np.abs(pixels[:, None, :] - pixels[None, :, :]).sum(axis=2)
    first, second = np.unravel_index(int(np.argmax(distance)), distance.shape)
    centres = pixels[[first, second]]
    labels = np.zeros(len(pixels), dtype=int)
    for _ in range(20):
        labels = ((pixels[:, None, :] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
        updated = np.stack(
            [
                pixels[labels == k].mean(axis=0) if (labels == k).any() else centres[k]
                for k in (0, 1)
            ]
        )
        if np.array_equal(updated, centres):
            break
        centres = updated
    sets = centres >= 0.5
    total = float(candidates.sum())
    shared = float((sets[0] & sets[1]).sum()) / max(1.0, float((sets[0] | sets[1]).sum()))
    if shared > config.screen_overlap or (sets.sum(axis=1) < config.screen_min_share * total).any():
        return None
    rows = np.nonzero(candidates)[0]
    height = candidates.shape[0]
    middle = (rows > 0.25 * height) & (rows < 0.75 * height)
    centre_share = [float(middle[group].mean()) for group in sets]
    stats = int(np.argmax(centre_share))
    if (
        centre_share[stats] < config.screen_centre_share
        or centre_share[1 - stats] >= centre_share[stats]
    ):
        return None
    mask = np.zeros_like(candidates)
    mask[candidates] = sets[stats]
    return mask, labels != stats


def _flow_size(size: tuple[int, int], config: MeasureConfig) -> tuple[int, int]:
    width, height = size
    flow_height = max(2, round(config.flow_width * height / width / 2) * 2)
    return config.flow_width, flow_height


def _glyph_pixels(frame: GrayFrame, config: MeasureConfig) -> Mask:
    """Bright pixels with a much darker pixel within 2 px: how OSD characters are drawn
    (white with a black outline). Relative to the neighbourhood, so the outline still
    counts after analog blur and compression has lifted it from black."""
    darkest = cv2.erode(frame, np.ones((5, 5), np.uint8))
    outlined = frame.astype(np.int16) - darkest >= config.glyph_drop
    return np.asarray((frame >= config.glyph_level) & outlined)


def measure_frames(
    frames: Iterator[GrayFrame], fps: float, layout: Layout, config: MeasureConfig
) -> Measurements:
    flow_size = _flow_size(layout.size, config)
    estimator = MotionEstimator(flow_size[0], flow_size[1], layout.usable, fps, config.motion)
    clean = ~layout.overlay if layout.overlay is not None else None
    clean_small = layout.usable
    glyph_count = float(layout.glyphs.sum()) if layout.glyphs is not None else 0.0
    outside = layout.text_area
    band_edges = np.linspace(0, layout.size[1], config.text_bands + 1).astype(int)
    band_area = np.array(
        [outside[a:b].sum() for a, b in pairwise(band_edges)],
        dtype=np.float64,
    )
    rows: dict[str, list[float]] = {k: [] for k in ("brightness", "contrast", "noise", "change")}
    motion: list[FloatArray] = []
    layout_share: list[float] = []
    text: list[FloatArray] = []
    thumbnails: list[GrayFrame] = []
    previous: GrayFrame | None = None
    for index, frame in enumerate(frames):
        pixels = frame[clean] if clean is not None else frame.ravel()
        rows["brightness"].append(float(pixels.mean()) / 255)
        rows["contrast"].append(float(pixels.std()) / 255)
        small = np.asarray(cv2.resize(frame, flow_size, interpolation=cv2.INTER_AREA), np.uint8)
        detail = np.abs(small.astype(np.int16) - cv2.medianBlur(small, 3))
        rows["noise"].append(float(detail[clean_small].mean()) / 255)
        if previous is None:
            rows["change"].append(np.nan)
            motion.append(np.full(FIELDS, np.nan))
        else:
            difference = np.abs(small.astype(np.int16) - previous)
            rows["change"].append(float(difference[clean_small].mean()) / 255)
            motion.append(estimator(previous, small))
        previous = small
        glyphs = _glyph_pixels(frame, config)
        if layout.glyphs is not None:
            layout_share.append(float(glyphs[layout.glyphs].sum()) / glyph_count)
        else:
            layout_share.append(np.nan)
        loose = glyphs & outside
        counts = np.add.reduceat(loose.sum(axis=1), band_edges[:-1])
        text.append(counts / np.maximum(band_area, 1))
        if index % config.thumbnail_every == 0:
            thumbnail = cv2.resize(frame, config.thumbnail, interpolation=cv2.INTER_AREA)
            thumbnails.append(np.asarray(thumbnail, dtype=np.uint8))
    count = len(rows["brightness"])
    return Measurements(
        fps=fps,
        brightness=np.array(rows["brightness"]),
        contrast=np.array(rows["contrast"]),
        noise=np.array(rows["noise"]),
        change=np.array(rows["change"]),
        motion=np.array(motion).reshape(count, FIELDS),
        layout=np.array(layout_share),
        text=np.array(text, dtype=np.float32).reshape(count, config.text_bands),
        thumbnails=np.array(thumbnails, dtype=np.uint8).reshape(
            -1, config.thumbnail[1], config.thumbnail[0]
        ),
        thumbnail_every=config.thumbnail_every,
    )


@dataclass
class DecodeReport:
    frames: int = 0
    warnings: list[str] = field(default_factory=list)


def decode_frames(
    tools: Tools,
    info: MediaInfo,
    area: Area,
    size: tuple[int, int],
    fps: float,
    report: DecodeReport,
) -> Generator[GrayFrame, None, None]:
    """Every frame of the active picture at ``fps``, grey, at ``size``."""
    width, height = size
    frame_bytes = width * height
    args = [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-i", str(info.path)),
        *("-map", f"0:{info.video.index}", "-an", "-sn", "-dn"),
        "-vf",
        f"{area.crop_filter},fps={fps}:round=near,scale={width}:{height}:flags=area,format=gray",
        *("-f", "rawvideo", "-pix_fmt", "gray", "-"),
    ]
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=stderr_file, creationflags=NO_WINDOW
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
            raise RuntimeError("ffmpeg stdout pipe was not created")
        try:
            while True:
                chunk = process.stdout.read(frame_bytes)
                if len(chunk) < frame_bytes:
                    break
                report.frames += 1
                yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
            returncode = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace")
    if report.frames == 0:
        raise MediaToolError(f"decoding {info.path.name}", returncode or 1, stderr)
    if returncode != 0 or stderr.strip():
        first = stderr.strip().splitlines()[0] if stderr.strip() else f"exit code {returncode}"
        report.warnings.append(
            f"decoder reported problems (damaged or truncated recording?): {first}"
        )
