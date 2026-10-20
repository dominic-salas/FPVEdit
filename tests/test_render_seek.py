"""Seeking to late shots keeps exactly the frames decoding from the start keeps."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fpvedit.highlights import Interval
from fpvedit.media import find_tools, frame_times, probe
from fpvedit.render import (
    EncodeConfig,
    expected_frame_count,
    first_frame_time,
    output_rate,
    render,
    snap_to_frames,
    validate_output,
)
from media_fixtures import Span, ffmpeg_available, make_flight

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH"),
]

FAST = EncodeConfig(preset="ultrafast")


def _digests(path: Path) -> list[str]:
    lines = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "framemd5", "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [line.split(",")[-1].strip() for line in lines if not line.startswith("#")]


def _gap_clip(path: Path) -> Path:
    """60 fps with 1.13 s of frames missing at 20 s, like a DVR recording with signal loss."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=60:d=30"),
            *("-vf", "select='not(between(t,20.0,21.13))'", "-fps_mode", "passthrough"),
            *("-c:v", "libx264", "-preset", "ultrafast", "-g", "120", str(path)),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize("kind", ["ts", "mp4", "gap"])
def test_seeking_keeps_the_same_frames(kind: str, tmp_path: Path) -> None:
    if kind == "gap":
        source = _gap_clip(tmp_path / "gap.mp4")
    else:
        # MPEG-TS starts at a nonzero timestamp and has audio starting with the video.
        source = make_flight(
            tmp_path / f"flight.{kind}",
            [Span("active", 14), Span("flip", 10), Span("active", 14)],
            fps=60 if kind == "mp4" else 30,
        )
    tools = find_tools()
    info = probe(tools, source)
    times = frame_times(tools, info)
    rate = output_rate(info)
    intervals = snap_to_frames([Interval(12.3, 15.0), Interval(19.5, 23.25)], rate)
    expected = expected_frame_count(intervals, rate, times)
    digests = []
    for seek in (False, True):
        output = tmp_path / f"seek{seek}.mp4"
        render(tools, info, intervals, output, encode=FAST, work_dir=tmp_path, seek=seek)
        validate_output(
            tools,
            output,
            expected_duration=expected / float(rate),
            rate=rate,
            expected_frames=expected,
            segments=len(intervals),
        )
        digests.append(_digests(output))
    assert first_frame_time(tools, info) is not None
    assert len(digests[0]) == expected
    assert digests[0] == digests[1]
