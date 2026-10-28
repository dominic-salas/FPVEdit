"""Learn one pilot's taste from their own star ratings and predict the stars of new shots.

A ridge regression predicts the pilot's stars from what the app knows about a shot:

- what it measured: the four rating parts (`rating.py`), the kinds of tricks, notes such
  as "crash right after" or "fast", and the length;
- what is in the picture, when the image model is installed (`scenery.py`): how much the
  shot's frames look like each of a few plain descriptions (a person, a narrow gap, trees,
  a car close up, the sky, night...). These are similarities, not probabilities, and they
  name what is in view, never a trick.

How strongly it is regularised is chosen by leave-one-out over the ratings (each rating
predicted from all the others), against the app's own stars as the alternative: with few
or inconsistent ratings the app's own rating stays, and the learned taste takes over only
when it predicts the ratings better. Fitting takes well under a second, so it is redone
whenever shots are found.

The pilot's edits teach it too, more weakly than a star (`EDITS`): a shot of the app's that
they removed counts as a low rating, one they exported as it was or added as a high one.
They are weighted in the fit and in the leave-one-out comparison alike, so edits alone can
also teach a taste, and the app's own rating still stays unless the taste predicts them
better.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from fpvedit.montage import Shot
from fpvedit.rating import Quality

FloatArray = NDArray[np.float64]

MIN_RATINGS = 8  # fewer than this and the app's own rating is used as it is
# What an edit says, as (stars, weight against a star rating's 1). Trims are only kept.
EDITS = {
    "removed": (2, 0.3),
    "exported": (4, 0.2),
    "added": (4, 0.6),
    "marked": (4, 0.6),
}
EXACT_UP_TO = 80  # more samples than this: leave-one-out from the closed form
STRENGTHS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)  # ridge penalties tried
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
    quality: Quality  # the app's rating when it was rated (with its picture, if known)
    tricks: tuple[str, ...]
    duration: float
    stars: int  # the pilot's own, 1-5
    recording: Path | None = None  # where it came from, to see its picture later
    start: float = 0.0
    weight: float = 1.0  # a star rating's 1; an edit's is lower (`EDITS`)


class Ratings(Protocol):
    """A pilot's own ratings (`app/myratings.py` keeps them on disk)."""

    def stars(self, path: Path, shot: Shot) -> int | None: ...

    def rated_shots(self) -> list[RatedShot]: ...


def features(
    quality: Quality, tricks: Sequence[str], duration: float, *, picture: bool = False
) -> FloatArray:
    """What the taste is learned from, for one shot (with ``picture``, its picture too)."""
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
    if picture:
        if quality.picture is None:
            raise ValueError("this shot has no picture features")
        values.extend(quality.picture)
    return np.asarray(values, dtype=np.float64)


def _base(quality: Quality) -> float:
    """The app's own stars, without any learned taste."""
    return float(dataclasses.replace(quality, taste=None, picture=None).stars)


@dataclass(frozen=True)
class _Model:
    mean: FloatArray
    scale: FloatArray
    offset: float
    weights: FloatArray

    def predict(self, x: FloatArray) -> float:
        return float(np.clip(self.offset + ((x - self.mean) / self.scale) @ self.weights, 1, 5))


@dataclass(frozen=True)
class Taste:
    plain: _Model | None  # from the measurements; None: the app's own stars are better
    pictured: _Model | None  # from the measurements and the picture
    ratings: int
    error: float  # leave-one-out mean absolute error in stars of what is used
    base_error: float  # ... and of the app's own stars
    uses_picture: bool

    @property
    def learned(self) -> bool:
        return self.plain is not None or self.pictured is not None

    @classmethod
    def fit(cls, rated: Sequence[RatedShot]) -> Taste:
        y = np.array([float(r.stars) for r in rated])
        w = np.array([r.weight for r in rated])
        base = np.array([_base(r.quality) for r in rated])
        base_error = _mean(np.abs(y - base), w) if len(rated) else 0.0
        if len(rated) < MIN_RATINGS:
            return cls(None, None, len(rated), base_error, base_error, False)
        x = np.array([features(r.quality, r.tricks, r.duration) for r in rated])
        plain, plain_error = _choose(x, y, w, base_error)
        seen = [r for r in rated if r.quality.picture is not None]
        pictured, pictured_error = None, base_error
        if len(seen) >= MIN_RATINGS:
            xp = np.array([features(r.quality, r.tricks, r.duration, picture=True) for r in seen])
            yp = np.array([float(r.stars) for r in seen])
            wp = np.array([r.weight for r in seen])
            seen_base = _mean(np.abs(yp - np.array([_base(r.quality) for r in seen])), wp)
            pictured, pictured_error = _choose(xp, yp, wp, min(seen_base, plain_error))
        uses_picture = pictured is not None
        error = pictured_error if uses_picture else plain_error
        return cls(plain, pictured, len(rated), error, base_error, uses_picture)

    def predict(self, quality: Quality, tricks: Sequence[str], duration: float) -> float:
        """Predicted stars (1-5, not rounded) for a shot the pilot has not rated."""
        if self.pictured is not None and quality.picture is not None:
            return self.pictured.predict(features(quality, tricks, duration, picture=True))
        if self.plain is not None:
            return self.plain.predict(features(quality, tricks, duration))
        return _base(quality)


def _mean(values: FloatArray, weights: FloatArray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _fit(x: FloatArray, y: FloatArray, w: FloatArray, strength: float) -> _Model:
    """Ridge regression with each sample weighted by ``w``."""
    mean = (w @ x) / w.sum()
    scale = np.sqrt((w @ (x - mean) ** 2) / w.sum())
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    offset = float(w @ y / w.sum())
    # Minimising sum(w (y - z b)^2) + strength |b|^2. With fewer samples than features (a
    # few dozen ratings against ~35) the dual form is smaller: b = z.T a with
    # (W z z.T + s I) a = W y; with months of edits, the primal (z.T W z + s I) b = z.T W y.
    if len(z) <= z.shape[1]:
        gram = w[:, None] * (z @ z.T) + strength * np.eye(len(z))
        weights = z.T @ np.linalg.solve(gram, w * (y - offset))
    else:
        inner = z.T @ (w[:, None] * z) + strength * np.eye(z.shape[1])
        weights = np.linalg.solve(inner, z.T @ (w * (y - offset)))
    return _Model(mean, scale, offset, np.asarray(weights, dtype=np.float64))


def _left_out(x: FloatArray, y: FloatArray, w: FloatArray, strength: float) -> FloatArray:
    """Each sample's prediction from a fit to all the others."""
    if len(y) <= EXACT_UP_TO:
        keep = ~np.eye(len(y), dtype=bool)
        return np.array(
            [
                _fit(x[keep[i]], y[keep[i]], w[keep[i]], strength).predict(x[i])
                for i in range(len(y))
            ]
        )
    # Closed form (centring kept from the full fit): residual / (1 - leverage).
    model = _fit(x, y, w, strength)
    z = (x - model.mean) / model.scale
    inner = z.T @ (w[:, None] * z) + strength * np.eye(z.shape[1])
    leverage = w * np.einsum("ij,ji->i", z, np.linalg.solve(inner, z.T))
    residual = y - (model.offset + z @ model.weights)
    return np.asarray(
        np.clip(y - residual / np.maximum(1.0 - leverage, 1e-6), 1, 5), dtype=np.float64
    )


def _choose(
    x: FloatArray, y: FloatArray, w: FloatArray, to_beat: float
) -> tuple[_Model | None, float]:
    """The ridge strength that predicts left-out ratings best, if it beats ``to_beat``."""
    best: tuple[float, float | None] = (to_beat, None)
    for tried in STRENGTHS:
        loo = _mean(np.abs(_left_out(x, y, w, tried) - y), w)
        if loo < best[0] - 1e-9:
            best = (loo, tried)
    error, strength = best
    return (_fit(x, y, w, strength) if strength is not None else None), error
