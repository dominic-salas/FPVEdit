"""Safe ffprobe/ffmpeg integration: tool discovery, probing and process execution.

All commands are argument lists (never shell strings), and input paths are made
absolute so a file name can never be parsed as an option.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpv_editor.errors import InvalidInputError, MediaToolError, ToolNotFoundError

# Without this, every ffmpeg/ffprobe run from the desktop app (a GUI program on Windows)
# flashes a console window. 0 elsewhere.
NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)

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
    nominal_rate: Fraction | None = None  # e.g. 60/1 or 30000/1001; None if unknown
    pix_fmt: str | None = None
    color_range: str | None = None  # "tv" (limited), "pc" (full) or None if unspecified
    frame_count: int | None = None  # only filled when probing with count_frames=True


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
    def display_aspect(self) -> float:
        """Width/height as displayed, after sample aspect ratio and rotation."""
        aspect = self.video.width * float(self.video.sample_aspect_ratio) / self.video.height
        return 1 / aspect if self.video.rotation % 180 == 90 else aspect


def bundled_ffmpeg() -> Path | None:
    """The FFmpeg folder shipped next to the standalone app, if this is one."""
    if not getattr(sys, "frozen", False):
        return None
    folder = Path(sys.executable).parent / "ffmpeg"
    return folder if folder.is_dir() else None


def find_tools() -> Tools:
    """FFmpeg and ffprobe: the bundled copy of a standalone build first, then PATH."""
    bundled = bundled_ffmpeg()
    here = str(bundled) if bundled is not None else None
    ffmpeg = (here and shutil.which("ffmpeg", path=here)) or shutil.which("ffmpeg")
    ffprobe = (here and shutil.which("ffprobe", path=here)) or shutil.which("ffprobe")
    missing = [name for name, found in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not found]
    if ffmpeg is None or ffprobe is None:
        raise ToolNotFoundError(f"{' and '.join(missing)} not found. {INSTALL_HINT}")
    version = run_tool([ffmpeg, "-hide_banner", "-version"], action="ffmpeg -version").stdout
    return Tools(ffmpeg=ffmpeg, ffprobe=ffprobe, ffmpeg_major=parse_ffmpeg_major(version))


def parse_ffmpeg_major(version_output: str) -> int | None:
    """Major version from `ffmpeg -version`.

    Git builds have no release number, so their libavfilter version decides
    (libavfilter 9 shipped with FFmpeg 6, 10 with FFmpeg 7, and so on).
    """
    match = re.search(r"version n?(\d+)\.", version_output)
    if match:
        return int(match.group(1))
    libavfilter = re.search(r"libavfilter\s+(\d+)\.", version_output)
    return int(libavfilter.group(1)) - 3 if libavfilter else None


def run_tool(args: list[str], *, action: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
        )
    except OSError as exc:
        raise ToolNotFoundError(f"could not run {args[0]}: {exc}. {INSTALL_HINT}") from exc
    if result.returncode != 0:
        raise MediaToolError(action, result.returncode, result.stderr)
    return result


def run_tool_with_progress(
    args: list[str], *, action: str, on_frames: Callable[[int], None]
) -> None:
    """Run an ffmpeg command, calling ``on_frames`` with the frames written so far.

    ``args`` must not contain ``-progress``; it is added here. If ``on_frames`` raises
    (for example to cancel), ffmpeg is stopped and the exception propagates.
    """
    args = [args[0], "-progress", "pipe:1", "-nostats", *args[1:]]
    with tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                creationflags=NO_WINDOW,
            )
        except OSError as exc:
            raise ToolNotFoundError(f"could not run {args[0]}: {exc}. {INSTALL_HINT}") from exc
        if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
            raise RuntimeError("ffmpeg stdout pipe was not created")
        finished = False
        try:
            for line in process.stdout:
                key, _, value = line.strip().partition("=")
                if key == "frame" and value.isdigit():
                    on_frames(int(value))
            finished = True
        finally:
            if not finished and process.poll() is None:
                process.kill()  # on_frames raised: stop ffmpeg
            process.stdout.close()
            returncode = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace")
    if returncode != 0:
        raise MediaToolError(action, returncode, stderr)


def probe(tools: Tools, path: Path, *, count_frames: bool = False) -> MediaInfo:
    """Describe ``path``; with ``count_frames`` also count video packets (reads the file)."""
    path = path.resolve()
    if not path.is_file():
        raise InvalidInputError(f"input not found or not a file: {path}")
    args = [tools.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams"]
    if count_frames:
        # One packet per frame for the H.264 MP4 files this tool writes; far cheaper
        # than decoding, which validation does separately.
        args.append("-count_packets")
    try:
        result = run_tool(
            [*args, "-i", str(path)],
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
            nominal_rate=_nominal_rate(video),
            pix_fmt=_str(video.get("pix_fmt")),
            color_range=_color_range(video),
            frame_count=_int(video.get("nb_read_packets")),
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


def _nominal_rate(stream: dict[str, Any]) -> Fraction | None:
    """The stream's regular frame rate, preferring the container's base rate.

    ``r_frame_rate`` is exact for constant-rate DVR footage (60/1). ``avg_frame_rate``
    drifts by a few parts per million when a frame is missing, so it is the fallback.
    Absurd values (some MPEG-TS files report 90000/1) fall through to the average.
    """
    for key in ("r_frame_rate", "avg_frame_rate"):
        rate = parse_ratio(stream.get(key))
        if rate is not None and 1 <= rate <= 240:
            return _standard_rate(rate)
    return None


STANDARD_RATES = tuple(
    Fraction(n, d)
    for n, d in (
        (24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1), (48, 1), (50, 1),
        (60000, 1001), (60, 1), (90, 1), (100, 1), (120000, 1001), (120, 1), (240, 1),
    )
)  # fmt: skip


def _standard_rate(rate: Fraction) -> Fraction:
    """Snap to a standard rate within 0.5 % (an averaged rate drifts slightly)."""
    nearest = min(STANDARD_RATES, key=lambda standard: abs(standard - rate))
    if abs(nearest - rate) <= nearest * Fraction(5, 1000):
        return nearest
    return rate.limit_denominator(1001)


# A recording counts as constant-rate when this share of its frame steps is one frame
# long. Signal loss on a DVR leaves a gap and sometimes a few frames at the receiver's
# no-signal rate; phones and screen captures vary throughout.
MIN_REGULAR_SHARE = 0.9
GAP_FRAMES = 1.5  # a step longer than this many frames means frames are missing


def frame_times(tools: Tools, info: MediaInfo) -> NDArray[np.float64] | None:
    """Presentation time of every video frame, in seconds since the first one.

    Read from the packets (no decoding), so it is cheap. None if ffprobe cannot list them.
    """
    args = [tools.ffprobe, "-v", "error", "-select_streams", f"{info.video.index}"]
    args += ["-show_entries", "packet=pts_time", "-of", "default=nw=1:nk=1", "-i", str(info.path)]
    try:
        output = run_tool(args, action=f"listing the frames of {info.path.name}").stdout
    except MediaToolError:
        return None
    values = []
    for line in output.split():
        try:
            values.append(float(line))
        except ValueError:
            continue  # "N/A": a packet without a timestamp
    if len(values) < 2:
        return None
    times = np.sort(np.array(values, dtype=np.float64))
    return np.asarray(times - times[0], dtype=np.float64)


def regular_step_share(times: NDArray[np.float64], rate: Fraction) -> float:
    """Share of frame-to-frame steps within 10 % of one frame at ``rate``."""
    steps = np.diff(times)
    if len(steps) == 0:
        return 1.0
    frame = 1 / float(rate)
    return float(np.mean(np.abs(steps - frame) <= 0.1 * frame))


def frame_gaps(times: NDArray[np.float64], rate: Fraction) -> list[tuple[float, float]]:
    """(start, length) of every stretch with missing frames, e.g. signal loss."""
    steps = np.diff(times)
    where = np.flatnonzero(steps > GAP_FRAMES / float(rate))
    return [(float(times[i]), float(steps[i])) for i in where]


def is_variable_rate(stream: VideoStream, times: NDArray[np.float64] | None = None) -> bool:
    """True for footage whose frames do not come at a steady rate (phones, screen captures).

    With the frame ``times`` the steps themselves decide, so a steady recording with a
    gap is accepted. Without them the average rate must be within 2 % of the nominal one.
    """
    if stream.nominal_rate is None:
        return False
    if times is not None and len(times) > 2:
        return regular_step_share(times, stream.nominal_rate) < MIN_REGULAR_SHARE
    if stream.frame_rate is None:
        return False
    return abs(stream.frame_rate - float(stream.nominal_rate)) > 0.02 * float(stream.nominal_rate)


def _color_range(stream: dict[str, Any]) -> str | None:
    value = _str(stream.get("color_range"))
    if value in ("tv", "pc"):
        return value
    # Older FFmpeg builds report JPEG-range pixel formats instead of a range tag.
    pix_fmt = _str(stream.get("pix_fmt")) or ""
    return "pc" if pix_fmt.startswith("yuvj") else None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value and value != "unknown" else None


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


MUSIC_SUFFIXES = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma")


@dataclass(frozen=True)
class AudioFile:
    """A music track for a montage's soundtrack."""

    path: Path
    duration: float


def probe_audio(tools: Tools, path: Path) -> AudioFile:
    """Describe a music file; raises InvalidInputError without a readable audio stream."""
    path = path.resolve()
    if not path.is_file():
        raise InvalidInputError(f"music file not found: {path}")
    args = [tools.ffprobe, "-v", "error", "-print_format", "json", "-show_format"]
    args += ["-show_streams", "-select_streams", "a:0", "-i", str(path)]
    try:
        result = run_tool(args, action=f"ffprobe {path.name}")
    except MediaToolError as exc:
        raise InvalidInputError(f"{path.name} is not a readable audio file. {exc}") from exc
    data: dict[str, Any] = json.loads(result.stdout)
    streams: list[dict[str, Any]] = data.get("streams", [])
    if not streams:
        raise InvalidInputError(f"{path.name} has no audio")
    duration = _positive_float(data.get("format", {}).get("duration")) or _positive_float(
        streams[0].get("duration")
    )
    if duration is None:
        raise InvalidInputError(f"{path.name}: the length of its audio is unknown")
    return AudioFile(path, duration)
