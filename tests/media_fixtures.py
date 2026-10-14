"""Generate tiny synthetic "flights" with FFmpeg so tests need no real footage.

Each flight is a sequence of labelled spans with known ground truth:
- ``idle``: a static textured scene with sensor noise (drone sitting on the ground)
- ``active``: a large test pattern panned and rolled across the frame (flying)
- ``flip``: like ``active``, with a full 360-degree roll in one second at the middle
- ``black``: black frames (no video signal)
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SpanKind = Literal["idle", "active", "flip", "black"]


@dataclass(frozen=True)
class Span:
    kind: SpanKind
    seconds: float


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _source(kind: SpanKind, seconds: float, width: int, height: int, fps: int) -> str:
    if kind == "idle":
        return (
            f"color=c=gray:s={width}x{height}:r={fps}:d={seconds},"
            "geq=lum='128+60*sin(X/7)*cos(Y/5)':cb=128:cr=128,"
            "noise=alls=6:allf=t"
        )
    if kind in ("active", "flip"):
        roll = f"+2*PI*clip(t-{seconds / 2 - 0.5},0,1)" if kind == "flip" else ""
        return (
            f"testsrc2=s={width * 4}x{height * 4}:r={fps}:d={seconds},"
            f"crop={width}:{height}:x='{width * 1.5}+{width * 1.2}*sin(t*0.8)'"
            f":y='{height * 1.5}+{height}*sin(t*1.3)',"
            f"rotate=a='0.6*sin(t*1.1){roll}':ow={width}:oh={height}"
        )
    return f"color=c=black:s={width}x{height}:r={fps}:d={seconds}"


def make_flight(
    path: Path,
    spans: Sequence[Span],
    *,
    audio: bool = True,
    width: int = 320,
    height: int = 180,
    fps: int = 30,
    full_range: bool = False,
    pillarbox: bool = False,
    osd: bool = False,
) -> Path:
    """Write an H.264 (+ AAC tone) MPEG-TS or MP4 file, chosen by ``path``'s suffix.

    ``full_range`` writes full-range ("yuvj") video, as analog DVR goggles record it.
    ``pillarbox`` shrinks the picture to 4:3 between black bars, and ``osd`` burns in
    static white-on-black boxes like a flight controller's on-screen display.
    """
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y"]
    for span in spans:
        args += ["-f", "lavfi", "-i", _source(span.kind, span.seconds, width, height, fps)]
    total = sum(span.seconds for span in spans)
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:d={total}"]
    overlay = ""
    if osd:
        boxes = [(0.05, 0.05), (0.75, 0.05), (0.05, 0.85), (0.70, 0.85), (0.40, 0.08)]
        overlay = "".join(
            f"drawbox=x=iw*{x}:y=ih*{y}:w=iw*0.18:h=ih*0.07:color=black:t=fill,"
            f"drawbox=x=iw*{x}+2:y=ih*{y}+2:w=iw*0.18-4:h=ih*0.07-4:color=white:t=fill,"
            for x, y in boxes
        )
    if pillarbox:
        overlay += f"scale={height * 4 // 3}:{height},pad={width}:{height}:(ow-iw)/2:0:black,"
    pixel_format = "scale=out_range=pc,format=yuvj420p" if full_range else "format=yuv420p"
    inputs = "".join(f"[{index}:v]" for index in range(len(spans)))
    args += [
        "-filter_complex",
        f"{inputs}concat=n={len(spans)}:v=1:a=0,{overlay}{pixel_format}[v]",
        "-map",
        "[v]",
    ]
    if audio:
        args += ["-map", f"{len(spans)}:a", "-c:a", "aac", "-b:a", "96k"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-g", str(fps)]
    args += ["-f", "mpegts" if path.suffix == ".ts" else "mp4", str(path)]
    subprocess.run(args, check=True, capture_output=True)
    return path


def ground_truth(spans: Sequence[Span]) -> list[tuple[SpanKind, float, float]]:
    result = []
    start = 0.0
    for span in spans:
        result.append((span.kind, start, start + span.seconds))
        start += span.seconds
    return result
