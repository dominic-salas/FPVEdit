"""What kind of recording a file is: an analog goggle DVR, an HDZero (digital) goggle DVR
or an action camera.

The kind decides the rules. Both goggle kinds carry the flight controller's on-screen
display (OSD), so the arm and disarm screens mark the flight. An action camera has no OSD:
bright static detail in its picture is scenery or a prop guard, so it gets the rules for
recordings without an OSD (motion decides what is flying).

The file alone cannot tell the two goggle kinds apart, so the picture does: analog video
has black bars around its picture (or is plainly 4:3) and carries analog grain; digital
video fills the frame. An action camera shows in the file: a frame size or rate the goggles
never write, a maker in the container tags, or a gyro log (``.gcsv``) beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fpvedit.media import MediaInfo


class SourceKind(Enum):
    ANALOG = "analog"
    DIGITAL = "hdzero"
    ACTION_CAM = "action"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def has_osd(self) -> bool:
        """Whether recordings of this kind carry the flight controller's OSD."""
        return self is not SourceKind.ACTION_CAM


_LABELS = {
    SourceKind.ANALOG: "Analog",
    SourceKind.DIGITAL: "HDZero",
    SourceKind.ACTION_CAM: "Action cam",
}

AUTO = "auto"
CHOICES = (AUTO, *(kind.value for kind in SourceKind))


def parse_choice(choice: str) -> SourceKind | None:
    """The kind a ``--source`` value forces; None for ``auto``."""
    if choice == AUTO:
        return None
    try:
        return SourceKind(choice)
    except ValueError:
        raise ValueError(
            f"unknown source kind {choice!r}; use one of {', '.join(CHOICES)}"
        ) from None


@dataclass(frozen=True)
class Source:
    kind: SourceKind
    reason: str  # one short sentence saying what decided it
    forced: bool = False  # chosen by the user, not found out


@dataclass(frozen=True)
class SourceConfig:
    # Bars around the picture: the share of the frame that is not picture.
    bars_share: float = 0.03
    # A full-frame recording this close to 4:3 is analog (digital FPV records 16:9).
    four_three: tuple[float, float] = (1.25, 1.45)
    # Analog grain when there are no bars: luma detail (measure.py's `noise`) above this.
    analog_noise: float = 0.008
    # Goggle DVRs write 1280x720 at 50, 60 and 90 fps; HDZero may write 1080p30.
    goggle_max_rate: float = 90.5
    camera_min_height: int = 1440  # 2.7K, 4K: more than any goggle writes
    full_hd_goggle_rates: tuple[float, float] = (29.0, 30.1)


@dataclass(frozen=True)
class PictureEvidence:
    """What the picture shows, measured while analysing the recording."""

    active_aspect: float  # display aspect of the picture inside any bars
    bars_share: float  # share of the frame outside the picture, 0..1
    noise: float | None  # median luma detail of the picture (None: not measured)
    osd_found: bool  # an OSD layout was found in the picture


CAMERA_MAKERS = (
    "runcam", "gopro", "dji", "osmo", "insta360", "sony", "xiaomi", "akaso", "caddx", "sjcam",
)  # fmt: skip

GYRO_SUFFIXES = (".gcsv",)


def gyro_log_beside(path: Path) -> bool:
    """Whether a gyro log with the recording's name sits next to it (its existence only)."""
    try:
        names = {entry.name.lower() for entry in path.parent.iterdir()}
    except OSError:
        return False
    return any(f"{path.stem}{suffix}".lower() in names for suffix in GYRO_SUFFIXES)


def camera_tags(info: MediaInfo) -> list[str]:
    """Container or stream tags that name a camera maker, as ``key=value``."""
    return [
        f"{key}={value}"
        for key, value in info.tags
        if any(maker in value.lower() for maker in CAMERA_MAKERS)
    ]


def frame_evidence(info: MediaInfo, config: SourceConfig) -> list[str]:
    """Frame size or rate a goggle DVR never writes (empty when the file could be one's)."""
    video = info.video
    rate = float(video.nominal_rate) if video.nominal_rate is not None else None
    height = min(video.width, video.height) if video.rotation % 180 == 90 else video.height
    if height >= config.camera_min_height:
        return [f"{video.width}x{video.height} is larger than goggles record"]
    if rate is not None and rate > config.goggle_max_rate:
        return [f"{_rate_text(rate)} fps is faster than goggles record"]
    low, high = config.full_hd_goggle_rates
    if rate is not None and height >= 1080 and not low <= rate <= high:
        return [f"{video.width}x{video.height} at {_rate_text(rate)} fps is not a goggle mode"]
    return []  # 1080p30 could be HDZero's own mode, so size alone does not decide


def _rate_text(rate: float) -> str:
    return f"{rate:.0f}" if abs(rate - round(rate)) < 0.01 else f"{rate:.2f}"


def looks_analog(evidence: PictureEvidence, config: SourceConfig) -> str | None:
    """Why the picture looks like analog video, or None."""
    if evidence.bars_share >= config.bars_share:
        return "black bars around a picture that does not fill the frame"
    low, high = config.four_three
    if low <= evidence.active_aspect <= high:
        return "a 4:3 picture"
    if evidence.noise is not None and evidence.noise >= config.analog_noise:
        return "analog grain in the picture"
    return None


def classify(
    info: MediaInfo,
    evidence: PictureEvidence,
    config: SourceConfig | None = None,
    *,
    gyro_log: bool | None = None,
) -> Source:
    """The kind of ``info`` from its file and its picture, with the reason.

    Evidence for a camera, in order of how little it can be mistaken: a gyro log beside the
    file, a maker in the tags, a frame size or rate no goggle writes. A camera is only
    overruled by the picture when an OSD was found in a picture that looks analog, since
    then editing by the OSD is the safe choice: an OSD is only assumed to be missing when
    none was found.
    """
    config = config or SourceConfig()
    analog = looks_analog(evidence, config)
    camera: list[str] = []
    if gyro_log if gyro_log is not None else gyro_log_beside(info.path):
        camera.append("a gyro log (.gcsv) sits beside it")
    camera.extend(f"its tags name a camera ({tag})" for tag in camera_tags(info)[:1])
    camera.extend(frame_evidence(info, config))
    if camera and evidence.osd_found and analog is not None:
        return Source(
            SourceKind.ANALOG,
            f"{analog}, and an OSD was found, although {camera[0]}; edited by its OSD",
        )
    if camera:
        found = " (glyph-like detail ignored: scenery, not an OSD)" if evidence.osd_found else ""
        return Source(SourceKind.ACTION_CAM, f"{camera[0]}{found}")
    if analog is not None:
        return Source(SourceKind.ANALOG, analog)
    note = "" if evidence.osd_found else "; no OSD was found, so it is edited by its motion"
    return Source(SourceKind.DIGITAL, f"the picture fills the frame and is clean{note}")


def active_aspect(info: MediaInfo, width: int, height: int, full: tuple[int, int]) -> float:
    """Display aspect ratio of a ``width`` x ``height`` picture inside the decoded frame."""
    full_width, full_height = full
    return info.display_aspect * (width / full_width) / (height / full_height)


def median_noise(noise: NDArray[np.float64]) -> float | None:
    values = noise[np.isfinite(noise)]
    return float(np.median(values)) if len(values) else None
