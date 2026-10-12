"""Safe ffprobe/ffmpeg integration: tool discovery, probing and process execution.

All commands are argument lists (never shell strings), and input paths are made
absolute so a file name can never be parsed as an option.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from fpv_editor.errors import InvalidInputError, MediaToolError, ToolNotFoundError

INSTALL_HINT = (
    "Install FFmpeg 6 or newer and make sure `ffmpeg` and `ffprobe` are on PATH "
    "(Windows: `winget install Gyan.FFmpeg`; Debian/Ubuntu: `sudo apt install ffmpeg`)."
)


@dataclass(frozen=True)
class Tools:
    ffmpeg: str
    ffprobe: str
    ffmpeg_major: int | None

    @property
    def filter_script_option(self) -> str:
        """Option that reads a filter graph from a file (renamed in FFmpeg 7)."""
        if self.ffmpeg_major is not None and self.ffmpeg_major < 7:
            return "-filter_complex_script"
        return "-/filter_complex"


@dataclass(frozen=True)
class VideoStream:
    index: int
    codec: str
    width: int
    height: int
    sample_aspect_ratio: Fraction
    frame_rate: float | None
    start_time: float | None = None
    rotation: int = 0  # degrees from the display matrix, e.g. 90 for portrait phone video


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str
    sample_rate: int | None
    channels: int | None
    start_time: float | None = None


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    format_name: str
    duration: float | None
    start_time: float
    video: VideoStream
    audio: AudioStream | None

    @property
    def audio_lead(self) -> float:
        """Seconds the audio starts before the first video frame (negative if after).

        MPEG-TS DVR files often start audio and video at different times; the analysis
        timeline begins at the first video frame, so audio cuts are shifted by this.
        """
        if self.audio is None or self.audio.start_time is None or self.video.start_time is None:
            return 0.0
        return self.video.start_time - self.audio.start_time

    @property
    def display_aspect(self) -> float:
        """Width/height as displayed, after sample aspect ratio and rotation."""
        aspect = self.video.width * float(self.video.sample_aspect_ratio) / self.video.height
        return 1 / aspect if self.video.rotation % 180 == 90 else aspect


def find_tools() -> Tools:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    missing = [name for name, found in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not found]
    if ffmpeg is None or ffprobe is None:
        raise ToolNotFoundError(f"{' and '.join(missing)} not found. {INSTALL_HINT}")
    version = run_tool([ffmpeg, "-hide_banner", "-version"], action="ffmpeg -version").stdout
    return Tools(ffmpeg=ffmpeg, ffprobe=ffprobe, ffmpeg_major=parse_ffmpeg_major(version))


def parse_ffmpeg_major(version_output: str) -> int | None:
    """Major version from `ffmpeg -version`; None for git builds without a release number."""
    match = re.search(r"version n?(\d+)\.", version_output)
    return int(match.group(1)) if match else None


def run_tool(args: list[str], *, action: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        raise ToolNotFoundError(f"could not run {args[0]}: {exc}. {INSTALL_HINT}") from exc
    if result.returncode != 0:
        raise MediaToolError(action, result.returncode, result.stderr)
    return result


def probe(tools: Tools, path: Path) -> MediaInfo:
    path = path.resolve()
    if not path.is_file():
        raise InvalidInputError(f"input not found or not a file: {path}")
    try:
        result = run_tool(
            [
                tools.ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-i",
                str(path),
            ],
            action=f"ffprobe {path.name}",
        )
    except MediaToolError as exc:
        raise InvalidInputError(f"{path.name} is not a readable video file. {exc}") from exc
    return parse_probe(path, json.loads(result.stdout))


def parse_probe(path: Path, data: dict[str, Any]) -> MediaInfo:
    streams: list[dict[str, Any]] = data.get("streams", [])
    fmt: dict[str, Any] = data.get("format", {})
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")
            and s.get("width")
            and s.get("height")
        ),
        None,
    )
    if video is None:
        raise InvalidInputError(f"{path.name} has no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _positive_float(fmt.get("duration")) or _positive_float(video.get("duration"))
    return MediaInfo(
        path=path,
        format_name=str(fmt.get("format_name", "unknown")),
        duration=duration,
        start_time=_float(fmt.get("start_time")) or 0.0,
        video=VideoStream(
            index=int(video.get("index", 0)),
            codec=str(video.get("codec_name", "unknown")),
            width=int(video["width"]),
            height=int(video["height"]),
            sample_aspect_ratio=parse_ratio(video.get("sample_aspect_ratio")) or Fraction(1),
            frame_rate=_fraction_to_float(video.get("avg_frame_rate"))
            or _fraction_to_float(video.get("r_frame_rate")),
            start_time=_float(video.get("start_time")),
            rotation=_rotation(video),
        ),
        audio=None
        if audio is None
        else AudioStream(
            index=int(audio.get("index", 0)),
            codec=str(audio.get("codec_name", "unknown")),
            sample_rate=_int(audio.get("sample_rate")),
            channels=_int(audio.get("channels")),
            start_time=_float(audio.get("start_time")),
        ),
    )


def _rotation(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list", []):
        rotation = _float(side_data.get("rotation"))
        if rotation is not None:
            return round(rotation) % 360
    rotate_tag = _float(stream.get("tags", {}).get("rotate"))
    return round(rotate_tag) % 360 if rotate_tag is not None else 0


def parse_ratio(value: object) -> Fraction | None:
    """Parse ffprobe ratios such as '16:9', '1/1' or '30000/1001'; None if absent or zero."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d+)[:/](\d+)", value.strip())
    if not match or int(match.group(1)) == 0 or int(match.group(2)) == 0:
        return None
    return Fraction(int(match.group(1)), int(match.group(2)))


def _fraction_to_float(value: object) -> float | None:
    ratio = parse_ratio(value)
    return float(ratio) if ratio is not None else None


def _float(value: object) -> float | None:
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _positive_float(value: object) -> float | None:
    number = _float(value)
    return number if number is not None and number > 0 else None


def _int(value: object) -> int | None:
    if not isinstance(value, str | int):
        return None
    try:
        return int(value)
    except ValueError:
        return None
