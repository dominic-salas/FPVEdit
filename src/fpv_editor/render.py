"""Render selected intervals to a video-only H.264 MP4, validate it, then move it into place.

The recording's audio is not used: on FPV DVRs it is goggle audio (receiver noise and
beeps) that has nothing to do with the flight, and pilots add their own music.

The source is decoded once and a ``select`` filter keeps the frames inside the
intervals, so memory use does not grow with the number of cuts (separate trim chains
per segment made FFmpeg 6 buffer gigabytes). Frames get new
timestamps at the source's own frame rate, which keeps the rate identical on FFmpeg 6,
7 and 9 (FFmpeg 7's concat defaulted to 25 fps). Cuts are frame-accurate; stream copy
is not used because H.264 can only be cut cleanly at keyframes.

Output is standard limited-range BT.709 H.264, whatever range the source used (analog
DVRs often record full-range "yuvj" video), so every player shows the same levels.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from fpv_editor.errors import (
    InvalidInputError,
    MediaToolError,
    OutputExistsError,
    ToolNotFoundError,
    ValidationError,
)
from fpv_editor.highlights import Interval
from fpv_editor.media import MediaInfo, Tools, probe, run_tool, run_tool_with_progress

OUTPUT_SUFFIX = "_highlight"
DEFAULT_RATE = Fraction(30)  # only for sources that report no usable frame rate
# A boundary belongs to the frame nearest to it: frames are selected from half a frame
# before each boundary, which absorbs timestamp rounding (1 ms timebases in MKV/MP4).
BOUNDARY_FRAMES = 0.5
READ_MARGIN_S = 2.0  # decoding stops this long after the last kept frame
RENDER_ACTION = "rendering the highlight video"  # MediaToolError.action of a failed encode
# A part whose first kept frame is this far into its clip seeks instead of decoding the
# clip from the start; decoding then begins at the keyframe before SEEK_LEAD_S earlier.
SEEK_AFTER_S = 10.0
SEEK_LEAD_S = 5.0
KEYFRAME_MARGIN_S = 10.0  # extra reading after a seek, for a keyframe far before it


@dataclass(frozen=True)
class EncodeConfig:
    video_codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18  # constant quality; for NVENC its -cq level

    def codec_args(self) -> list[str]:
        if self.video_codec == "h264_nvenc":
            # Constant-quality VBR, the closest NVENC match to x264's CRF.
            return [
                *("-c:v", "h264_nvenc", "-preset", self.preset, "-tune", "hq"),
                *("-rc", "vbr", "-cq", str(self.crf), "-b:v", "0"),
            ]
        return ["-c:v", self.video_codec, "-preset", self.preset, "-crf", str(self.crf)]


# NVIDIA GPU encoding: much faster than libx264 at a similar look for this footage.
NVENC = EncodeConfig(video_codec="h264_nvenc", preset="p5", crf=19)


DEVICES = ("auto", "cpu", "cuda")


def encoders(tools: Tools, device: str, cpu: EncodeConfig) -> list[EncodeConfig]:
    """Encoders to try in order: ``auto`` is NVENC when it works here, then ``cpu``;
    ``cuda`` is NVENC only and fails if it is unavailable."""
    if device not in DEVICES:
        raise InvalidInputError(f"device must be one of {', '.join(DEVICES)}, not {device!r}")
    if device == "cpu":
        return [cpu]
    if nvenc_works(tools):
        return [NVENC] if device == "cuda" else [NVENC, cpu]
    if device == "cuda":
        raise ToolNotFoundError(
            "--device cuda: this FFmpeg cannot encode on an NVIDIA GPU (h264_nvenc) here; "
            "check the NVIDIA driver, or use --device cpu"
        )
    return [cpu]


Result = TypeVar("Result")


def with_fallback(
    choices: Sequence[EncodeConfig], attempt: Callable[[EncodeConfig], Result]
) -> tuple[Result, EncodeConfig]:
    """``attempt(encode)`` with each encoder in turn until one works; only a failed encode
    moves on to the next (other failures are real problems and are raised)."""
    for number, encode in enumerate(choices):
        try:
            return attempt(encode), encode
        except MediaToolError as exc:
            if exc.action != RENDER_ACTION or number == len(choices) - 1:
                raise
    raise ValueError("no encoder to try")


def nvenc_works(tools: Tools) -> bool:
    """True if this FFmpeg build can encode H.264 on an NVIDIA GPU here (a tiny test encode)."""
    args = [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error"),
        *("-f", "lavfi", "-i", "color=black:size=256x144:rate=30", "-frames:v", "5"),
        *NVENC.codec_args(),
        *("-f", "null", "-"),
    ]
    try:
        run_tool(args, action="testing NVENC")
    except (MediaToolError, ToolNotFoundError):
        return False
    return True


def default_output_path(source: Path, suffix: str = OUTPUT_SUFFIX) -> Path:
    """`<stem>_highlight.mp4` beside the source, or `_highlight-2` etc. if taken."""
    first = source.with_name(f"{source.stem}{suffix}.mp4")
    if not first.exists():
        return first
    for index in range(2, 1000):
        candidate = source.with_name(f"{source.stem}{suffix}-{index}.mp4")
        if not candidate.exists():
            return candidate
    raise OutputExistsError(f"too many existing outputs next to {source.name}; use --output")


def resolve_output_path(
    source: Path | Sequence[Path],
    requested: Path | None,
    force: bool,
    suffix: str = OUTPUT_SUFFIX,
) -> Path:
    sources = [source] if isinstance(source, Path) else list(source)
    output = (requested or default_output_path(sources[0], suffix)).resolve()
    if any(_same_file(output, path) for path in sources):
        raise InvalidInputError("the output path is the same as an input; refusing to overwrite it")
    if output.suffix.lower() != ".mp4":
        raise InvalidInputError(f"output must be an .mp4 file: {output}")
    if output.exists() and not force:
        raise OutputExistsError(f"{output} already exists; use --force to replace it")
    if output.exists() and not output.is_file():
        raise InvalidInputError(f"output path exists and is not a file: {output}")
    if not output.parent.is_dir():
        raise InvalidInputError(f"output folder does not exist: {output.parent}")
    return output


def _same_file(a: Path, b: Path) -> bool:
    """True if both paths name one file, including on case-insensitive filesystems."""
    if a.exists() and b.exists():
        return a.samefile(b)
    return os.path.normcase(a) == os.path.normcase(b.resolve())


def output_rate(info: MediaInfo) -> Fraction:
    return info.video.nominal_rate or DEFAULT_RATE


def snap_to_frames(intervals: Sequence[Interval], rate: Fraction) -> tuple[Interval, ...]:
    """Move each boundary to the nearest frame time.

    The edit's duration and frame count are then exact, which validation relies on.
    """
    snapped = []
    for interval in intervals:
        start = _next_frame(interval.start, rate)
        end = _next_frame(interval.end, rate)
        if end > start:
            snapped.append(Interval(float(start), float(end)))
    return tuple(snapped)


def _next_frame(seconds: float, rate: Fraction) -> Fraction:
    index = math.ceil(seconds * float(rate) - BOUNDARY_FRAMES)
    return Fraction(max(0, index)) / rate


def expected_frame_count(
    intervals: Sequence[Interval], rate: Fraction, times: NDArray[np.float64] | None = None
) -> int:
    """Frames the render keeps from frame-snapped ``intervals``.

    With the source's frame ``times`` (`media.frame_times`) the frames are counted the way
    the render selects them, so gaps from signal loss are allowed for. Without them the
    source is assumed to be perfectly constant-rate.
    """
    if times is None:
        return sum(
            round(Fraction(i.end) * rate) - round(Fraction(i.start) * rate) for i in intervals
        )
    margin = BOUNDARY_FRAMES / float(rate)
    return sum(
        int(np.searchsorted(times, i.end - margin) - np.searchsorted(times, i.start - margin))
        for i in intervals
    )


@dataclass(frozen=True)
class Fit:
    """Output frame size and rate for a montage of clips that may differ in both."""

    width: int
    height: int
    rate: Fraction


def build_filter_graph(
    intervals: Sequence[Interval],
    *,
    video_index: int,
    rate: Fraction,
    fit: Fit | None = None,
    origin: float | None = None,
) -> str:
    """Filter graph that keeps ``intervals`` of input 0 as one continuous ``[v]``.

    Interval times are on the analysis timeline (seconds since the first video frame)
    and should already be snapped to frame times (`snap_to_frames`). With ``fit`` the
    frames are converted to its rate and scaled to fit its size, centred on black.
    ``origin`` is the first frame's timestamp in seconds, for input that was seeked with
    ``-copyts`` (its first frame is then not the recording's).
    """
    if not intervals:
        raise ValueError("at least one interval is required")
    margin = BOUNDARY_FRAMES / float(rate)
    keep = "+".join(f"between(t,{i.start - margin:.6f},{i.end - margin:.6f})" for i in intervals)
    if fit is None:
        # yuv420p H.264 needs even dimensions; uses the post-rotation size, so it is
        # safe for footage with rotation metadata.
        crop = "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0"
    else:
        crop = (
            "scale=trunc(iw*sar/2)*2:ih,setsar=1,"  # square pixels first
            f"fps={fit.rate.numerator}/{fit.rate.denominator},"
            f"scale={fit.width}:{fit.height}:force_original_aspect_ratio=decrease,"
            f"pad={fit.width}:{fit.height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
    colour = (
        "scale=in_range=auto:out_range=tv:in_color_matrix=auto:out_color_matrix=bt709,"
        "format=yuv420p"
    )
    # setpts renumbers the frames but keeps each one's duration: the frame before a
    # signal-loss gap would still last ~1 s and the constant-rate encoder would repeat
    # it. The fps filter resets every duration to one frame.
    retime = (
        f"setpts=N*{rate.denominator}/{rate.numerator}/TB,fps={rate.numerator}/{rate.denominator}"
    )
    start = "PTS-STARTPTS" if origin is None else f"PTS-{origin:.6f}/TB"
    return f"[0:{video_index}]setpts={start},select='{keep}',{retime},{crop},{colour}[v]"


def render(
    tools: Tools,
    info: MediaInfo,
    intervals: Sequence[Interval],
    destination: Path,
    *,
    encode: EncodeConfig,
    work_dir: Path,
    fit: Fit | None = None,
    on_frames: Callable[[int], None] | None = None,
    seek: bool = True,
) -> None:
    """Encode frame-snapped ``intervals`` to ``destination``, a temporary path the caller owns.

    ``on_frames`` is called with the number of frames written so far; if it raises, the
    encode stops. With ``seek``, intervals late in the recording are reached by seeking
    instead of decoding everything before them; the frames kept are the same.
    """
    rate = output_rate(info)
    first, last = min(i.start for i in intervals), max(i.end for i in intervals)
    origin = first_frame_time(tools, info) if seek and first > SEEK_AFTER_S else None
    graph = build_filter_graph(
        intervals, video_index=info.video.index, rate=rate, fit=fit, origin=origin
    )
    script = work_dir / f"{destination.stem}.filter_graph.txt"
    script.write_text(graph, encoding="utf-8")
    out_rate = fit.rate if fit is not None else rate
    # Stop reading a little after the last kept frame instead of decoding the whole file.
    # Input times count from the start of the file, analysis times from the first frame.
    video_start = origin if origin is not None else _video_start(info)
    offset = video_start - info.start_time
    if origin is None:
        reading = ["-t", f"{last + READ_MARGIN_S + max(0.0, offset):.3f}"]
    else:
        # Keep the file's timestamps (the graph subtracts ``origin``) and start decoding at
        # the keyframe before the seek point; ``select`` drops the frames before ``first``.
        begin = first - SEEK_LEAD_S
        length = last - begin + READ_MARGIN_S + KEYFRAME_MARGIN_S
        reading = [
            *("-copyts", "-noaccurate_seek"),
            *("-ss", f"{begin + offset:.3f}", "-t", f"{length:.3f}"),
        ]
    args = [
        tools.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        *reading,
        "-i",
        str(info.path),
        tools.filter_script_option,
        str(script),
        "-map",
        "[v]",
        "-r",
        f"{out_rate.numerator}/{out_rate.denominator}",
        *encode.codec_args(),
        "-pix_fmt",
        "yuv420p",
        *("-color_range", "tv", "-colorspace", "bt709"),
        *("-color_primaries", "bt709", "-color_trc", "bt709"),
        *("-an", "-sn", "-dn"),
        *("-movflags", "+faststart", "-f", "mp4", str(destination)),
    ]
    if on_frames is None:
        run_tool(args, action=RENDER_ACTION)
    else:
        run_tool_with_progress(args, action=RENDER_ACTION, on_frames=on_frames)


def _video_start(info: MediaInfo) -> float:
    return info.video.start_time if info.video.start_time is not None else info.start_time


def first_frame_time(tools: Tools, info: MediaInfo) -> float | None:
    """Timestamp (s) of the first frame the decoder outputs, where the render's and the
    analysis's timeline starts; None if it cannot be read."""
    args = [
        *(tools.ffprobe, "-v", "error", "-select_streams", str(info.video.index)),
        *("-read_intervals", "%+#120", "-show_entries", "frame=best_effort_timestamp_time"),
        *("-of", "csv=p=0", "-i", str(info.path)),
    ]
    try:
        output = run_tool(args, action=f"reading the first frame of {info.path.name}").stdout
    except MediaToolError:
        return None
    for line in output.split():
        try:
            return float(line.strip().strip(","))
        except ValueError:
            continue
    return None


def join_parts(tools: Tools, parts: Sequence[Path], destination: Path, work_dir: Path) -> None:
    """Join rendered parts (same codec, size and rate) without encoding them again."""
    if not parts:
        raise ValueError("at least one part is required")
    listing = work_dir / "parts.txt"
    # Paths in the list are relative to it; the parts are named by us, so need no quoting.
    listing.write_text(
        "".join(f"file '{part.relative_to(work_dir).as_posix()}'\n" for part in parts),
        encoding="utf-8",
    )
    args = [
        tools.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        *("-f", "concat", "-safe", "0", "-i", str(listing)),
        *("-map", "0:v:0", "-c", "copy", "-an", "-sn", "-dn"),
        *("-movflags", "+faststart", "-f", "mp4", str(destination)),
    ]
    run_tool(args, action="joining the montage")


def validate_output(
    tools: Tools,
    path: Path,
    *,
    expected_duration: float,
    rate: Fraction | None = None,
    expected_frames: int | None = None,
    segments: int = 1,
) -> MediaInfo:
    """Check the stream, frame rate, frame count, colour range and duration, then decode it all."""
    try:
        info = probe(tools, path, count_frames=True)
    except InvalidInputError as exc:
        raise ValidationError(f"rendered file is not readable: {exc}") from exc
    video = info.video
    if video.codec != "h264":
        raise ValidationError(f"rendered video codec is {video.codec}, expected h264")
    if rate is not None and video.nominal_rate != rate:
        raise ValidationError(
            f"rendered frame rate is {video.nominal_rate} fps, expected {rate} fps"
        )
    if video.pix_fmt != "yuv420p" or video.color_range != "tv":
        raise ValidationError(
            f"rendered video is {video.pix_fmt} / {video.color_range} range, "
            "expected yuv420p limited (tv) range"
        )
    if info.audio is not None:
        raise ValidationError("rendered file has an audio track; expected video only")
    if expected_frames is not None and video.frame_count is not None:
        # Recordings with dropped frames have a few fewer frames than their timestamps
        # promise; a wrong output rate changes the count by far more.
        tolerance = max(2 * segments, round(0.01 * expected_frames))
        if abs(video.frame_count - expected_frames) > tolerance:
            raise ValidationError(
                f"rendered video has {video.frame_count} frames, expected about {expected_frames}"
            )
    if info.duration is None:
        raise ValidationError("rendered file has no duration")
    tolerance_s = max(0.5, 0.02 * expected_duration)
    if abs(info.duration - expected_duration) > tolerance_s:
        raise ValidationError(
            f"rendered duration {info.duration:.2f} s differs from the expected "
            f"{expected_duration:.2f} s"
        )
    decode = run_tool(
        [
            tools.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        action="decoding the rendered video",
    )
    if decode.stderr.strip():
        raise ValidationError(f"decoding the rendered video reported errors: {decode.stderr}")
    return info


def temporary_output(final: Path) -> Path:
    """Reserve a unique temporary file next to the final output (same filesystem)."""
    handle, name = tempfile.mkstemp(
        prefix=f".{final.stem}.", suffix=".partial.mp4", dir=final.parent
    )
    os.close(handle)
    return Path(name)


def finalize_output(temporary: Path, final: Path, *, force: bool) -> None:
    """Move the validated file into place; without ``force`` never replace anything."""
    if force:
        temporary.replace(final)
        return
    try:
        # A hard link fails atomically if ``final`` exists, closing the check/replace race.
        os.link(temporary, final)
    except FileExistsError as exc:
        raise OutputExistsError(f"{final} appeared while rendering; not replacing it") from exc
    except OSError:
        # Filesystems without hard links (e.g. exFAT): fall back to check-then-rename.
        if final.exists():
            raise OutputExistsError(f"{final} appeared while rendering; not replacing it") from None
        temporary.replace(final)
        return
    temporary.unlink()
