"""Flight state from the burned-in Betaflight OSD: when the quad was armed, and what else
was on screen (boot logo, stats after a disarm, static).

Betaflight clears its normal on-screen layout for three things: the "ARMED" message
(about half a second when arming), the post-flight statistics after a disarm, and the
boot logo (at power-up, and on some setups again when arming). `measure.py` records, for
every frame, how much of the normal layout is showing and where other text appears.
From those:

- a *clear* run is a stretch where the normal layout is fully gone;
- a clear run with text across several rows is the stats screen (a disarm), unless its
  text matches the boot logo seen at the start of the recording: then it is the logo
  shown when arming, or, straight after the boot screen, the rest of the boot;
- a short clear run without such text is the ARMED message;
- picture noise far above the recording's usual level is static or no signal.

Armed time runs from the end of an ARMED message to the next stats screen. Stats only
appear at a disarm, so a stats screen while the state is "disarmed" means an arm was
missed, and the quad counts as armed from when the OSD came back. Partial dropouts of
the layout (bright scenery behind the text, analog breakup) are not clear runs and are
ignored, so they cannot end a flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from fpv_editor.measure import Measurements

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class Screen(Enum):
    BOOT = "boot"  # clear screen at the very start: goggle/flight-controller boot
    LOGO = "logo"  # the boot logo shown again when arming
    ARMED = "armed"  # the ARMED message (or a short clear screen while disarmed)
    STATS = "stats"  # post-flight statistics: a disarm
    STATIC = "static"  # static (noise) or no signal (flat grey screen)
    BLANK = "blank"  # clear screen without recognisable text


@dataclass(frozen=True)
class ScreenEvent:
    kind: Screen
    start: float
    end: float


@dataclass(frozen=True)
class FlightConfig:
    clear_level: float = 0.15  # layout share (of the usual level) below which it is gone
    layout_level_percentile: float = 75.0  # "usual" layout share
    text_margin: float = 0.002  # glyph share above a band's usual level that counts as text
    text_rows: int = 3  # bands with text needed for a stats screen
    text_window_s: float = 0.15  # majority vote window for the stats text
    stats_text_share: float = 0.3  # share of a clear run with stats text
    min_stats_s: float = 0.25
    armed_message_s: tuple[float, float] = (0.25, 2.0)
    static_noise: float = 0.015  # static: noise above this ...
    static_noise_factor: float = 4.0  # ... and this many times the recording's median
    static_window_s: float = 0.25
    min_static_s: float = 0.5  # static while the OSD layout stays on screen
    # No signal: the goggles show a flat grey screen (OSD may still be drawn on top).
    grey_contrast: float = 0.045  # luma standard deviation outside the OSD below this ...
    grey_brightness: tuple[float, float] = (0.28, 0.42)  # ... at mid grey ...
    grey_steady: float = 0.004  # ... with brightness this steady over a second
    logo_similarity: float = 0.85  # text profile cosine similarity to the boot logo
    boot_gap_s: float = 1.0  # a logo this soon after the boot screen is still booting
    min_boot_s: float = 0.5  # a shorter clear screen at the start is not a boot screen
    missed_arm_gap_s: float = 1.0


@dataclass(frozen=True)
class FlightState:
    events: tuple[ScreenEvent, ...]
    armed: tuple[tuple[float, float], ...]
    has_osd: bool  # an OSD layout was found, so arm/disarm events could be detected
    clear: BoolArray  # per frame: the normal layout is gone (stats, logo, message ...)
    static: BoolArray  # per frame: static or no signal (grey screen)

    @property
    def disarms(self) -> tuple[ScreenEvent, ...]:
        return tuple(e for e in self.events if e.kind is Screen.STATS)


def runs(mask: BoolArray) -> list[tuple[int, int]]:
    """Half-open index ranges [start, end) of consecutive True values."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def rolling_median(values: NDArray[np.floating] | BoolArray, window: int) -> FloatArray:
    """Median over a centred window, repeating edge values."""
    data = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(data) == 0:
        return data.copy()
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
    return np.asarray(
        np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1),
        dtype=np.float64,
    )


def rolling_mean(values: FloatArray, window: int) -> FloatArray:
    """Mean over a centred window, repeating edge values."""
    if window <= 1 or len(values) == 0:
        return values.copy()
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.asarray(np.convolve(padded, np.ones(window) / window, mode="valid"))


def detect_flight(measurements: Measurements, config: FlightConfig | None = None) -> FlightState:
    config = config or FlightConfig()
    fps = measurements.fps
    count = len(measurements)
    duration = count / fps
    noise = measurements.noise
    static = rolling_median(noise, max(1, round(config.static_window_s * fps))) > max(
        config.static_noise, config.static_noise_factor * float(np.median(noise))
    )
    static |= _grey_screen(measurements, config)
    layout = measurements.layout
    has_osd = count > 0 and bool(np.isfinite(layout).any())
    events: list[ScreenEvent] = []
    clear = np.zeros(count, dtype=bool)
    if has_osd:
        clear, events = _screen_events(measurements, static, config)
    events.extend(_static_events(static & ~clear, fps, config))
    events.sort(key=lambda e: e.start)
    armed = armed_intervals(events, duration, config) if has_osd else ((0.0, duration),)
    return FlightState(
        events=tuple(events), armed=tuple(armed), has_osd=has_osd, clear=clear, static=static
    )


def _grey_screen(measurements: Measurements, config: FlightConfig) -> BoolArray:
    """Frames of the flat mid-grey screen goggles show without a video signal."""
    brightness = np.asarray(measurements.brightness, dtype=np.float64)
    window = max(1, round(measurements.fps))
    mean = rolling_mean(brightness, window)
    spread = np.sqrt(np.maximum(rolling_mean(brightness**2, window) - mean**2, 0.0))
    low, high = config.grey_brightness
    return np.asarray(
        (measurements.contrast < config.grey_contrast)
        & (brightness > low)
        & (brightness < high)
        & (spread < config.grey_steady)
    )


def _screen_events(
    measurements: Measurements, static: BoolArray, config: FlightConfig
) -> tuple[BoolArray, list[ScreenEvent]]:
    fps = measurements.fps
    count = len(measurements)
    share = np.nan_to_num(measurements.layout)
    # The OSD flickers on alternate frames on some setups: take the max of neighbours.
    share = np.maximum(
        share, np.maximum(np.r_[share[1:], share[-1:]], np.r_[share[:1], share[:-1]])
    )
    level = float(np.percentile(share, config.layout_level_percentile))
    clear = share < config.clear_level * level
    for start, end in runs(~clear):  # close one- and two-frame holes
        if end - start <= 2 and start > 0 and end < count:
            clear[start:end] = True
    text = measurements.text.astype(np.float64)
    usual = np.median(text[share > 0.5 * level], axis=0) if (share > 0.5 * level).any() else 0.0
    rows = ((text - usual) > config.text_margin).sum(axis=1) >= config.text_rows
    window = max(1, round(config.text_window_s * fps)) | 1
    textual = rolling_median(rows, window) > 0.5

    events: list[ScreenEvent] = []
    low, high = config.armed_message_s
    for start, end in runs(clear):
        length = (end - start) / fps
        has_text = textual[start:end]
        if start == 0:
            kind = Screen.BOOT
        elif static[start:end].mean() > 0.5:
            kind = Screen.STATIC
        elif has_text.mean() > config.stats_text_share and length >= config.min_stats_s:
            # The stats screen, sometimes followed by the ARMED message once the text goes.
            last = start + int(np.flatnonzero(has_text)[-1]) + 1
            events.append(ScreenEvent(Screen.STATS, start / fps, last / fps))
            if (end - last) / fps >= 0.2 and end < count:
                events.append(ScreenEvent(Screen.ARMED, last / fps, end / fps))
            continue
        elif low <= length <= high and end < count:
            kind = Screen.ARMED
        else:
            kind = Screen.BLANK
        events.append(ScreenEvent(kind, start / fps, end / fps))
    return clear, _relabel_logo(events, text - usual, fps, config)


def _relabel_logo(
    events: list[ScreenEvent], text: FloatArray, fps: float, config: FlightConfig
) -> list[ScreenEvent]:
    """A 'stats' screen whose text matches the boot logo is the arming logo."""
    boot = next((e for e in events if e.kind is Screen.BOOT), None)
    if boot is None:
        return events

    def profile(event: ScreenEvent) -> FloatArray:
        start = int(event.start * fps)
        return np.asarray(text[start : max(start + 1, int(event.end * fps))].mean(axis=0))

    reference = profile(boot)
    booting_until = boot.end
    result = []
    for event in events:
        if event.kind is Screen.STATS:
            candidate = profile(event)
            norm = float(np.linalg.norm(candidate) * np.linalg.norm(reference))
            if norm > 0 and float(candidate @ reference) / norm > config.logo_similarity:
                booting = event.start - booting_until <= config.boot_gap_s
                event = ScreenEvent(Screen.BOOT if booting else Screen.LOGO, event.start, event.end)
                if booting:
                    booting_until = event.end
        result.append(event)
    return result


def _static_events(mask: BoolArray, fps: float, config: FlightConfig) -> list[ScreenEvent]:
    return [
        ScreenEvent(Screen.STATIC, start / fps, end / fps)
        for start, end in runs(mask)
        if (end - start) / fps >= config.min_static_s
    ]


def armed_intervals(
    events: list[ScreenEvent], duration: float, config: FlightConfig | None = None
) -> list[tuple[float, float]]:
    """Armed from the end of an ARMED message (or arming logo, and the ARMED message that
    may follow it) to the next stats screen.

    Static does not change the state; it is excluded from the edit separately.
    """
    config = config or FlightConfig()
    first = next(
        (e.kind for e in events if e.kind in (Screen.ARMED, Screen.STATS, Screen.LOGO)), None
    )
    # A flight controller always boots disarmed. Without a boot screen the recording may
    # have started mid-flight, which the first event being a disarm (or none) shows.
    booted = any(e.kind is Screen.BOOT and e.end - e.start >= config.min_boot_s for e in events)
    armed = not booted and (first is Screen.STATS or first is None)
    start = 0.0
    last_clear = 0.0
    logo_end: float | None = None
    spans: list[tuple[float, float]] = []
    for event in events:
        if event.kind in (Screen.ARMED, Screen.LOGO):
            if not armed:
                start, armed = event.end, True
                logo_end = event.end if event.kind is Screen.LOGO else None
            elif (
                event.kind is Screen.ARMED and logo_end is not None and event.start - logo_end < 1.5
            ):
                start = event.end  # the ARMED message that follows the arming logo
            if event.kind is Screen.ARMED:
                logo_end = None
        elif event.kind is Screen.STATS:
            if not armed and event.start - last_clear > config.missed_arm_gap_s:
                start, armed = last_clear, True  # an arm was missed
            if armed:
                spans.append((start, event.start))
                armed = False
        if event.kind in (Screen.STATS, Screen.BOOT, Screen.ARMED, Screen.LOGO):
            last_clear = event.end
    if armed:
        spans.append((start, duration))
    return [(a, b) for a, b in spans if b > a]
