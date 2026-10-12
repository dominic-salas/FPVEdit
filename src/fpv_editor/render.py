"""Render selected intervals to an H.264/AAC MP4, validate it, then move it into place.

Cuts are made with FFmpeg's trim/atrim + concat filters and re-encoded, which is
frame-accurate at any cut point. Stream copy is not used because H.264 can only be
cut cleanly at keyframes, which rarely coincide with the chosen boundaries.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fpv_editor.errors import InvalidInputError, OutputExistsError, ValidationError
from fpv_editor.highlights import Interval
from fpv_editor.media import MediaInfo, Tools, probe, run_tool

AUDIO_FADE_S = 0.01  # prevents clicks at cut points; not an audible effect
OUTPUT_SUFFIX = "_highlight"


@dataclass(frozen=True)
class EncodeConfig:
    video_codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18
    audio_bitrate: str = "192k"


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


def build_filter_graph(
    intervals: Sequence[Interval],
    *,
    video_index: int,
    audio_index: int | None = None,
    audio_lead: float = 0.0,
) -> str:
    """Filter graph that cuts ``intervals`` from input 0 and joins them.

    Interval times are on the analysis timeline: seconds since the first video frame.
    Each stream is rebased to its own first timestamp before trimming, so the result
    does not depend on how FFmpeg picks the zero point when streams start at different
    times. ``audio_lead`` is how many seconds the audio starts before the video
    (negative if it starts after), so audio is cut at the matching moments.
    """
    if not intervals:
        raise ValueError("at least one interval is required")
    # yuv420p H.264 needs even dimensions; uses the post-rotation size, so it is safe
    # for footage with rotation metadata.
    crop = "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0"
    chains = []
    pads = []
    for index, interval in enumerate(intervals):
        chains.append(
            f"[0:{video_index}]setpts=PTS-STARTPTS,"
            f"trim=start={interval.start:.6f}:end={interval.end:.6f},"
            f"setpts=PTS-STARTPTS,{crop}[v{index}]"
        )
        pads.append(f"[v{index}]")
        if audio_index is not None:
            audio_start = max(0.0, interval.start + audio_lead)
            audio_end = max(audio_start, interval.end + audio_lead)
            fade_out = max(0.0, interval.duration - AUDIO_FADE_S)
            chains.append(
                f"[0:{audio_index}]asetpts=PTS-STARTPTS,"
                f"atrim=start={audio_start:.6f}:end={audio_end:.6f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={AUDIO_FADE_S},"
                f"afade=t=out:st={fade_out:.6f}:d={AUDIO_FADE_S}[a{index}]"
            )
            pads.append(f"[a{index}]")
    has_audio = audio_index is not None
    outputs = "[v][a]" if has_audio else "[v]"
    chains.append(
        f"{''.join(pads)}concat=n={len(intervals)}:v=1:a={1 if has_audio else 0}{outputs}"
    )
    return ";\n".join(chains)


def render(
    tools: Tools,
    info: MediaInfo,
    intervals: Sequence[Interval],
    destination: Path,
    *,
    encode: EncodeConfig,
    work_dir: Path,
) -> None:
    """Encode the intervals to ``destination`` (a temporary path owned by the caller)."""
    has_audio = info.audio is not None
    graph = build_filter_graph(
        intervals,
        video_index=info.video.index,
        audio_index=info.audio.index if info.audio is not None else None,
        audio_lead=info.audio_lead,
    )
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
        "-c:v",
        encode.video_codec,
        "-preset",
        encode.preset,
        "-crf",
        str(encode.crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", encode.audio_bitrate]
    args += ["-movflags", "+faststart", "-f", "mp4", str(destination)]
    run_tool(args, action="rendering the highlight video")


def validate_output(
    tools: Tools, path: Path, *, expected_duration: float, expect_audio: bool
) -> MediaInfo:
    """Check streams and duration with ffprobe, then decode the whole file."""
    try:
        info = probe(tools, path)
    except InvalidInputError as exc:
        raise ValidationError(f"rendered file is not readable: {exc}") from exc
    if info.video.codec != "h264":
        raise ValidationError(f"rendered video codec is {info.video.codec}, expected h264")
    if expect_audio and (info.audio is None or info.audio.codec != "aac"):
        raise ValidationError("rendered file is missing its AAC audio track")
    if info.duration is None:
        raise ValidationError("rendered file has no duration")
    tolerance = max(0.5, 0.02 * expected_duration)
    if abs(info.duration - expected_duration) > tolerance:
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
