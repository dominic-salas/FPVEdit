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
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from fpv_editor.errors import InvalidInputError, OutputExistsError, ValidationError
from fpv_editor.highlights import Interval
from fpv_editor.media import MediaInfo, Tools, probe, run_tool

OUTPUT_SUFFIX = "_highlight"
DEFAULT_RATE = Fraction(30)  # only for sources that report no usable frame rate
# A boundary belongs to the frame nearest to it: frames are selected from half a frame
# before each boundary, which absorbs timestamp rounding (1 ms timebases in MKV/MP4).
BOUNDARY_FRAMES = 0.5


@dataclass(frozen=True)
class EncodeConfig:
    video_codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18


def default_output_path(source: Path) -> Path:
    """`<stem>_highlight.mp4` beside the source, or `_highlight-2` etc. if taken."""
    first = source.with_name(f"{source.stem}{OUTPUT_SUFFIX}.mp4")
    if not first.exists():
        return first
    for index in range(2, 1000):
        candidate = source.with_name(f"{source.stem}{OUTPUT_SUFFIX}-{index}.mp4")
        if not candidate.exists():
            return candidate
    raise OutputExistsError(f"too many existing outputs next to {source.name}; use --output")


def resolve_output_path(source: Path, requested: Path | None, force: bool) -> Path:
    output = (requested or default_output_path(source)).resolve()
    if _same_file(output, source):
        raise InvalidInputError(
            "the output path is the same as the input; refusing to overwrite it"
        )
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


def expected_frame_count(intervals: Sequence[Interval], rate: Fraction) -> int:
    """Frames a constant-rate source has inside frame-snapped ``intervals``."""
    return sum(round(Fraction(i.end) * rate) - round(Fraction(i.start) * rate) for i in intervals)


def build_filter_graph(intervals: Sequence[Interval], *, video_index: int, rate: Fraction) -> str:
    """Filter graph that keeps ``intervals`` of input 0 as one continuous ``[v]``.

    Interval times are on the analysis timeline (seconds since the first video frame)
    and should already be snapped to frame times (`snap_to_frames`).
    """
    if not intervals:
        raise ValueError("at least one interval is required")
    margin = BOUNDARY_FRAMES / float(rate)
    keep = "+".join(f"between(t,{i.start - margin:.6f},{i.end - margin:.6f})" for i in intervals)
    # yuv420p H.264 needs even dimensions; uses the post-rotation size, so it is safe
    # for footage with rotation metadata.
    crop = "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0"
    colour = (
        "scale=in_range=auto:out_range=tv:in_color_matrix=auto:out_color_matrix=bt709,"
        "format=yuv420p"
    )
    return (
        f"[0:{video_index}]setpts=PTS-STARTPTS,select='{keep}',"
        f"setpts=N*{rate.denominator}/{rate.numerator}/TB,{crop},{colour}[v]"
    )


def render(
    tools: Tools,
    info: MediaInfo,
    intervals: Sequence[Interval],
    destination: Path,
    *,
    encode: EncodeConfig,
    work_dir: Path,
) -> None:
    """Encode frame-snapped ``intervals`` to ``destination``, a temporary path the caller owns."""
    rate = output_rate(info)
    graph = build_filter_graph(intervals, video_index=info.video.index, rate=rate)
    script = work_dir / "filter_graph.txt"
    script.write_text(graph, encoding="utf-8")
    args = [
        tools.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-i",
        str(info.path),
        tools.filter_script_option,
        str(script),
        "-map",
        "[v]",
        "-r",
        f"{rate.numerator}/{rate.denominator}",
        "-c:v",
        encode.video_codec,
        "-preset",
        encode.preset,
        "-crf",
        str(encode.crf),
        "-pix_fmt",
        "yuv420p",
        *("-color_range", "tv", "-colorspace", "bt709"),
        *("-color_primaries", "bt709", "-color_trc", "bt709"),
        *("-an", "-sn", "-dn"),
        *("-movflags", "+faststart", "-f", "mp4", str(destination)),
    ]
    run_tool(args, action="rendering the highlight video")


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
