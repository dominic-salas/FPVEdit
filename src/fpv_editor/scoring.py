"""Turn features into an idle mask and a smoothed, bounded interest score.

The baseline is deliberately transparent: each feature is divided by a fixed
reference value and clipped to 0..1, then combined with fixed weights. Because every
term is capped, no single feature (in particular raw motion) can dominate the score.
The references are first guesses that must be calibrated on annotated flights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fpv_editor.features import FeatureSeries

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ScoringConfig:
    # Feature value treated as "fully interesting" (score term saturates at 1).
    motion_ref: float = 1.0  # frame widths per second
    change_ref: float = 0.15
    novelty_ref: float = 0.30
    motion_weight: float = 0.4
    change_weight: float = 0.3
    novelty_weight: float = 0.3
    smooth_sigma_s: float = 1.0
    # Idle: almost no motion and almost no pixel change (on the ground, waiting).
    idle_motion: float = 0.02
    idle_change: float = 0.01
    # Blank: dark and flat (goggles not receiving, lens covered, black frames).
    blank_brightness: float = 0.06
    blank_contrast: float = 0.03

    def __post_init__(self) -> None:
        weights = (self.motion_weight, self.change_weight, self.novelty_weight)
        if min(weights) < 0:
            raise ValueError("score weights must not be negative")
        total = sum(weights)
        if not np.isclose(total, 1.0):
            raise ValueError(f"score weights must sum to 1, got {total}")


def normalize(values: FloatArray, reference: float) -> FloatArray:
    """Scale by a reference value and clip to 0..1."""
    if reference <= 0:
        raise ValueError("reference must be positive")
    return np.clip(values / reference, 0.0, 1.0)


def gaussian_smooth(values: FloatArray, sigma_samples: float) -> FloatArray:
    """Gaussian smoothing that repeats edge values; returns a copy for sigma <= 0."""
    if sigma_samples <= 0 or len(values) == 0:
        return values.astype(np.float64, copy=True)
    radius = max(1, int(np.ceil(3 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def interest_score(features: FeatureSeries, config: ScoringConfig) -> FloatArray:
    raw = (
        config.motion_weight * normalize(features.motion, config.motion_ref)
        + config.change_weight * normalize(features.change, config.change_ref)
        + config.novelty_weight * normalize(features.novelty, config.novelty_ref)
    )
    return gaussian_smooth(raw, config.smooth_sigma_s * features.fps)


def idle_mask(features: FeatureSeries, config: ScoringConfig) -> BoolArray:
    still = (features.motion < config.idle_motion) & (features.change < config.idle_change)
    blank = (features.brightness < config.blank_brightness) & (
        features.contrast < config.blank_contrast
    )
    return np.asarray(still | blank, dtype=np.bool_)
