"""Learn one pilot's taste from their own star ratings and predict the stars of new shots.

The app's own rating (`rating.py`) is the starting point. The pilot's ratings teach a
correction to it: a ridge regression of (their stars - the app's stars) on what the app
measured about each shot (its four rating parts, its tricks, its notes such as "crash
right after" or "fast", and its length). Nothing is learned from the picture itself, so
a taste that depends on what is in view (a gap, a person, scenery) is only learned as
far as these measurements follow it.

How strongly the correction counts is chosen by leave-one-out over the ratings: each
rating is predicted from all the others, for several strengths, including none at all
(the app's own stars). The strength that predicted best is used, so with few or
inconsistent ratings the prediction stays the app's own rating, and it moves towards the
pilot's taste only as far as their ratings show it can. Everything runs on this computer
in a fraction of a second, so it is fitted again whenever shots are found.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fpv_editor.rating import Quality

FloatArray = NDArray[np.float64]

MIN_RATINGS = 8  # fewer than this and the app's own rating is used as it is
STRENGTHS = (1.0, 3.0, 10.0, 30.0, 100.0)  # ridge penalties tried (plus "no correction")
FLIPS = ("flip", "roll", "power loop")
NOTES = (
    "crash right after",
    "wobbly recovery",
    "smooth recovery",
    "loses speed",
    "keeps its speed",
    "very close",
    "fast",
    "dark",
)


@dataclass(frozen=True)
class RatedShot:
    quality: Quality  # the app's rating when it was rated
    tricks: tuple[str, ...]
    duration: float
    stars: int  # the pilot's own, 1-5


def features(quality: Quality, tricks: Sequence[str], duration: float) -> FloatArray:
    """What the correction is learned from, for one shot."""
    names = [t.lower() for t in tricks]
    reasons = " | ".join(quality.reasons)
    values = [
        quality.trick,
        quality.clean,
        quality.impressive,
        quality.watchable,
        duration / 15.0,
        min(len(names), 6) / 6.0,
        float(any(any(f in n for f in FLIPS) for n in names)),
        float(any("double" in n for n in names)),
        float(any("spin" in n for n in names)),
        float(bool(names) and all(n == "close pass" for n in names)),
        float("linked" in reasons),
        *(float(note in quality.reasons) for note in NOTES),
    ]
    return np.asarray(values, dtype=np.float64)


def _base(quality: Quality) -> float:
    """The app's own stars, without any learned taste."""
    return float(
        Quality(
            quality.trick,
            quality.clean,
            quality.impressive,
            quality.watchable,
            quality.reasons,
            quality.indoorness,
        ).stars
    )


@dataclass(frozen=True)
class Taste:
    mean: FloatArray
    scale: FloatArray
    weights: FloatArray | None  # None: the app's own rating is the prediction
    strength: float | None
    ratings: int
    error: float  # leave-one-out mean absolute error in stars, with the chosen strength
    base_error: float  # ... and of the app's own rating

    @property
    def learned(self) -> bool:
        return self.weights is not None

    @classmethod
    def fit(cls, rated: Sequence[RatedShot]) -> Taste:
        count = len(rated)
        x = np.array([features(r.quality, r.tricks, r.duration) for r in rated])
        base = np.array([_base(r.quality) for r in rated])
        y = np.array([float(r.stars) for r in rated])
        width = len(features(Quality(0, 0, 0, 0, ()), (), 0.0))
        if count < MIN_RATINGS:
            empty = np.zeros(width)
            error = float(np.mean(np.abs(y - base))) if count else 0.0
            return cls(empty, np.ones(width), None, None, count, error, error)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        z = np.hstack([(x - mean) / scale, np.ones((count, 1))])
        residual = y - base
        base_error = float(np.mean(np.abs(y - base)))
        best: tuple[float, float | None] = (base_error, None)
        for tried in STRENGTHS:
            loo = _leave_one_out(z, residual, base, y, tried)
            if loo < best[0] - 1e-9:
                best = (loo, tried)
        error, strength = best
        if strength is None:
            return cls(mean, scale, None, None, count, error, base_error)
        weights = _ridge(z, residual, strength)
        return cls(mean, scale, weights, strength, count, error, base_error)

    def predict(self, quality: Quality, tricks: Sequence[str], duration: float) -> float:
        """Predicted stars (1-5, not rounded) for a shot the pilot has not rated."""
        base = _base(quality)
        if self.weights is None:
            return base
        z = np.append((features(quality, tricks, duration) - self.mean) / self.scale, 1.0)
        return float(np.clip(base + z @ self.weights, 1.0, 5.0))


def _ridge(z: FloatArray, target: FloatArray, strength: float) -> FloatArray:
    gram = z.T @ z + strength * np.eye(z.shape[1])
    return np.asarray(np.linalg.solve(gram, z.T @ target), dtype=np.float64)


def _leave_one_out(
    z: FloatArray, residual: FloatArray, base: FloatArray, y: FloatArray, strength: float
) -> float:
    errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        weights = _ridge(z[keep], residual[keep], strength)
        predicted = float(np.clip(base[i] + z[i] @ weights, 1.0, 5.0))
        errors.append(abs(round(predicted) - y[i]))
    return float(np.mean(errors))
