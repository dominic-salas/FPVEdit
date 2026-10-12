"""Decode small grayscale analysis frames straight from the source with FFmpeg.

FFmpeg resamples to a fixed rate (which also handles variable frame rate input) and
downscales before Python sees a frame, so no intermediate video file is written.
Sample ``i`` is ``i / fps`` seconds after the first video frame (the analysis
timeline). Rendering rebases each stream to its own start, so it uses the same timeline.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fpv_editor.errors import MediaToolError
from fpv_editor.media import MediaInfo, Tools

GrayFrame = NDArray[np.uint8]


@dataclass(frozen=True)
class SamplingConfig:
    fps: float = 8.0
    width: int = 320


@dataclass
class SamplingReport:
    """Filled in while frames are read; describes how decoding ended."""

    frames: int = 0
    warnings: list[str] = field(default_factory=list)


def analysis_size(info: MediaInfo, width: int) -> tuple[int, int]:
    """Even-sized analysis resolution that keeps the display aspect ratio."""
    height = max(2, round(width / info.display_aspect / 2) * 2)
    return width, height


def iter_gray_frames(
    tools: Tools, info: MediaInfo, config: SamplingConfig, report: SamplingReport
) -> Generator[GrayFrame, None, None]:
    width, height = analysis_size(info, config.width)
    frame_bytes = width * height
    args = [
        tools.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(info.path),
        "-map",
        f"0:{info.video.index}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"fps={config.fps},scale={width}:{height}:flags=area,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr_file)
        if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
            raise RuntimeError("ffmpeg stdout pipe was not created")
        try:
            while True:
                chunk = process.stdout.read(frame_bytes)
                if len(chunk) < frame_bytes:
                    break
                report.frames += 1
                yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
            returncode = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace")

    if report.frames == 0:
        raise MediaToolError(f"decoding {info.path.name}", returncode or 1, stderr)
    if returncode != 0 or stderr.strip():
        first = stderr.strip().splitlines()[0] if stderr.strip() else f"exit code {returncode}"
        report.warnings.append(
            f"decoder reported problems (damaged or truncated recording?): {first}"
        )


def sample_duration(report: SamplingReport, config: SamplingConfig) -> float:
    return report.frames / config.fps
