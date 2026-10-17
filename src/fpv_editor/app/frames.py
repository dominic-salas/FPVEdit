"""Frames from a recording for the app: shot thumbnails and the looping shot preview.

FFmpeg decodes both, so the preview shows exactly the frames the render will use, from
any recording the render can read (HEVC MPEG-TS included). Shot times are on the
analysis timeline (seconds since the first video frame); FFmpeg's ``-ss`` counts from
the start of the file, so the video's own start is added.
"""

from __future__ import annotations

import subprocess

from fpv_editor.media import NO_WINDOW, MediaInfo, Tools


def file_seconds(info: MediaInfo, seconds: float) -> float:
    """``seconds`` on the analysis timeline as a position for FFmpeg's ``-ss``."""
    video_start = info.video.start_time if info.video.start_time is not None else info.start_time
    quarter_frame = 0.25 / float(info.video.nominal_rate or 30)
    return max(0.0, seconds + video_start - info.start_time - quarter_frame)


def preview_size(info: MediaInfo, height: int) -> tuple[int, int]:
    """Even frame size ``height`` pixels high with the recording's display aspect."""
    width = round(height * info.display_aspect / 2) * 2
    return max(2, width), height - height % 2


def grab_frame(tools: Tools, info: MediaInfo, seconds: float, height: int = 90) -> bytes | None:
    """The frame at ``seconds`` as PNG bytes, ``height`` pixels high; None if unreadable."""
    width, height = preview_size(info, height)
    args = [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error"),
        *("-ss", f"{file_seconds(info, seconds):.3f}", "-i", str(info.path)),
        *("-map", f"0:{info.video.index}", "-frames:v", "1", "-vf", f"scale={width}:{height}"),
        *("-an", "-sn", "-dn", "-f", "image2pipe", "-c:v", "png", "-"),
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False, creationflags=NO_WINDOW)
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def preview_args(
    tools: Tools, info: MediaInfo, start: float, end: float, size: tuple[int, int], fps: int
) -> list[str]:
    """FFmpeg command writing ``start``-``end`` as raw RGB frames of ``size`` to stdout."""
    width, height = size
    return [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error"),
        *("-ss", f"{file_seconds(info, start):.3f}", "-t", f"{max(0.1, end - start):.3f}"),
        *("-i", str(info.path), "-map", f"0:{info.video.index}", "-an", "-sn", "-dn"),
        *("-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"),
    ]
