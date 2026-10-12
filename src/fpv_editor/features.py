"""Conventional per-sample visual features computed from grayscale analysis frames.

Every feature is in units that do not depend on the source resolution, so the same
thresholds apply to 720p analog DVR and 4K digital recordings alike.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from fpv_editor.sampling import GrayFrame

FloatArray = NDArray[np.float64]

HISTOGRAM_BINS = 32


@dataclass(frozen=True)
class FeatureConfig:
    novelty_window_s: float = 4.0


@dataclass(frozen=True)
class FeatureSeries:
    """One value per analysis sample; sample ``i`` is at ``i / fps`` seconds."""

    fps: float
    motion: FloatArray  # median optical-flow speed, frame widths per second
    change: FloatArray  # mean absolute pixel change vs previous sample, 0..1
    novelty: FloatArray  # histogram distance to the recent past, 0..1
    brightness: FloatArray  # mean luma, 0..1
    contrast: FloatArray  # luma standard deviation, 0..1

    def __len__(self) -> int:
        return len(self.motion)

    @property
    def times(self) -> FloatArray:
        return np.arange(len(self), dtype=np.float64) / self.fps


def compute_features(
    frames: Iterable[GrayFrame], fps: float, config: FeatureConfig | None = None
) -> FeatureSeries:
    config = config or FeatureConfig()
    window = max(1, round(config.novelty_window_s * fps))
    recent: deque[NDArray[np.float32]] = deque(maxlen=window)
    motion: list[float] = []
    change: list[float] = []
    novelty: list[float] = []
    brightness: list[float] = []
    contrast: list[float] = []
    previous: NDArray[np.uint8] | None = None

    for frame in frames:
        smooth = np.asarray(cv2.GaussianBlur(frame, (5, 5), 0), dtype=np.uint8)
        histogram = _histogram(smooth)
        brightness.append(float(frame.mean()) / 255.0)
        contrast.append(float(frame.std()) / 255.0)
        if previous is None:
            motion.append(0.0)
            change.append(0.0)
            novelty.append(0.0)
        else:
            motion.append(_flow_speed(previous, smooth) * fps)
            change.append(float(cv2.absdiff(previous, smooth).mean()) / 255.0)
            reference = np.mean(np.stack(recent), axis=0)
            novelty.append(0.5 * float(np.abs(histogram - reference).sum()))
        recent.append(histogram)
        previous = smooth

    series = [np.asarray(values, dtype=np.float64) for values in (motion, change, novelty)]
    # The first sample has no predecessor; give it its neighbour's value instead of 0.
    for values in series:
        if len(values) > 1:
            values[0] = values[1]
    return FeatureSeries(
        fps=fps,
        motion=series[0],
        change=series[1],
        novelty=series[2],
        brightness=np.asarray(brightness, dtype=np.float64),
        contrast=np.asarray(contrast, dtype=np.float64),
    )


def _flow_speed(previous: NDArray[np.uint8], current: NDArray[np.uint8]) -> float:
    """Median dense optical-flow magnitude, in frame widths per sample step.

    Flow is computed at half the analysis resolution: the median global speed is
    what matters here, and this makes the dominant cost about four times cheaper.
    """
    size = (int(previous.shape[1]) // 2, int(previous.shape[0]) // 2)
    small_previous = np.asarray(cv2.resize(previous, size, interpolation=cv2.INTER_AREA))
    small_current = np.asarray(cv2.resize(current, size, interpolation=cv2.INTER_AREA))
    flow = cv2.calcOpticalFlowFarneback(
        small_previous,
        small_current,
        np.zeros((size[1], size[0], 2), dtype=np.float32),
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    vectors = np.asarray(flow, dtype=np.float64)
    magnitude = np.hypot(vectors[..., 0], vectors[..., 1])
    return float(np.median(magnitude)) / size[0]


def _histogram(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    histogram = cv2.calcHist([frame], [0], None, [HISTOGRAM_BINS], [0, 256]).ravel()
    total = float(histogram.sum())
    return (histogram / total if total > 0 else histogram).astype(np.float32)
