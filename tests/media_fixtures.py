"""Generate tiny synthetic "flights" with FFmpeg so tests need no real footage.

Each flight is a sequence of labelled spans with known ground truth:
- ``idle``: a static textured scene with sensor noise (drone sitting on the ground)
- ``active``: a large test pattern panned and rolled across the frame (flying)
- ``black``: black frames (no video signal)
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SpanKind = Literal["idle", "active", "black"]


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
    if kind == "active":
        return (
            f"testsrc2=s={width * 4}x{height * 4}:r={fps}:d={seconds},"
            f"crop={width}:{height}:x='{width * 1.5}+{width * 1.2}*sin(t*0.8)'"
            f":y='{height * 1.5}+{height}*sin(t*1.3)',"
            f"rotate=a='0.6*sin(t*1.1)':ow={width}:oh={height}"
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
    audio_lead: float = 0.0,
    gated_audio: bool = False,
) -> Path:
    """Write an H.264 (+ AAC) MPEG-TS or MP4 file, chosen by ``path``'s suffix.

    ``audio_lead`` starts the audio that many seconds before the first video frame (as
    in many DVR recordings) while keeping sound and picture in sync. With
    ``gated_audio`` the tone only plays during ``active`` spans, so sync can be checked.
    """
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y"]
    for span in spans:
        args += ["-f", "lavfi", "-i", _source(span.kind, span.seconds, width, height, fps)]
    total = sum(span.seconds for span in spans)
    if audio:
        gate = "1"
        if gated_audio:
            windows = [
                f"between(t-{audio_lead},{start},{end})"
                for kind, start, end in ground_truth(spans)
                if kind == "active"
            ]
            gate = "(" + "+".join(windows or ["0"]) + ")"
        source = f"aevalsrc=exprs='0.5*sin(2*PI*440*t)*{gate}':s=48000:d={total + audio_lead}"
        args += ["-f", "lavfi", "-i", source]
    inputs = "".join(f"[{index}:v]" for index in range(len(spans)))
    args += [
        "-filter_complex",
        f"{inputs}concat=n={len(spans)}:v=1:a=0,format=yuv420p,setpts=PTS+{audio_lead}/TB[v]",
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
