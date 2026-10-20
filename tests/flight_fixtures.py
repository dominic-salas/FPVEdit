"""Synthetic per-frame measurements for flight-state and selection tests.

A `Timeline` starts as calm level flight with the OSD showing, and spans of it are then
turned into other things: the OSD cleared (with or without text), static, a still
camera, fast tumbling, a flip.
"""

from __future__ import annotations

import numpy as np

from fpvedit.measure import Measurements
from fpvedit.motion import CONFIDENCE, DIVERGENCE, FIELDS, PITCH, ROLL, SPEED, YAW

FPS = 20.0
BANDS = 16
STATS_BANDS = range(4, 12)  # rows the Betaflight stats screen fills
LOGO_BANDS = range(1, 4)


class Timeline:
    def __init__(self, seconds: float, *, osd: bool = True, seed: int = 0) -> None:
        self.count = round(seconds * FPS)
        rng = np.random.default_rng(seed)
        self.brightness = np.full(self.count, 0.40) + rng.normal(0, 0.01, self.count)
        self.contrast = np.full(self.count, 0.12)
        self.noise = np.full(self.count, 0.004)
        self.change = np.full(self.count, 0.012)
        self.motion = np.zeros((self.count, FIELDS))
        # Calm cruising: gentle turns with some jitter, moving forward.
        self.motion[:, YAW] = 20 + rng.normal(0, 8, self.count)
        self.motion[:, PITCH] = rng.normal(0, 8, self.count)
        self.motion[:, SPEED] = 0.3
        self.motion[:, CONFIDENCE] = 0.9
        self.motion[:, DIVERGENCE] = rng.normal(0.5, 0.3, self.count)
        self.layout = np.full(self.count, 0.6) if osd else np.full(self.count, np.nan)
        self.text = np.zeros((self.count, BANDS), dtype=np.float32)

    def _span(self, start: float, end: float) -> slice:
        return slice(round(start * FPS), round(end * FPS))

    def clear(self, start: float, end: float, bands: range | None = None) -> Timeline:
        """The normal OSD layout disappears; text fills ``bands`` if given."""
        span = self._span(start, end)
        self.layout[span] = 0.0
        if bands is not None:
            self.text[span, list(bands)] = 0.01
        return self

    def armed_message(self, at: float) -> Timeline:
        return self.clear(at, at + 0.5)

    def stats(self, start: float, end: float) -> Timeline:
        return self.clear(start, end, STATS_BANDS)

    def dip(self, start: float, end: float, share: float = 0.3) -> Timeline:
        """A partial drop of the layout (bright scenery behind the text)."""
        self.layout[self._span(start, end)] = share
        return self

    def static(self, start: float, end: float) -> Timeline:
        self.noise[self._span(start, end)] = 0.05
        return self

    def grey(self, start: float, end: float) -> Timeline:
        span = self._span(start, end)
        self.brightness[span] = 0.34
        self.contrast[span] = 0.034
        return self

    def still(self, start: float, end: float) -> Timeline:
        span = self._span(start, end)
        self.motion[span, PITCH : ROLL + 1] = 0.0
        self.motion[span, SPEED] = 0.01
        self.motion[span, DIVERGENCE] = 0.0
        self.change[span] = 0.0005
        return self

    def tumble(self, start: float, end: float) -> Timeline:
        """Fast, chaotic motion: what the second before a crash looks like."""
        span = self._span(start, end)
        self.motion[span, ROLL] = 600.0
        self.motion[span, SPEED] = 3.0
        return self

    def flip(self, at: float) -> Timeline:
        """A 360 degree flip in half a second."""
        self.motion[self._span(at, at + 0.5), PITCH] = 720.0
        return self

    def cruise_straight(self, start: float, end: float) -> Timeline:
        """Flying straight and level: nothing happening."""
        span = self._span(start, end)
        self.motion[span, PITCH : ROLL + 1] = 5.0
        self.motion[span, DIVERGENCE] = 0.0
        return self

    def measurements(self) -> Measurements:
        return Measurements(
            fps=FPS,
            brightness=self.brightness,
            contrast=self.contrast,
            noise=self.noise,
            change=self.change,
            motion=self.motion,
            layout=self.layout,
            text=self.text,
            thumbnails=np.zeros((self.count // 6, 36, 48), dtype=np.uint8),
            thumbnail_every=6,
        )
