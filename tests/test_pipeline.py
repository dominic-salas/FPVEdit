"""End-to-end runs of the real pipeline on generated synthetic flights."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fpv_editor.cli import PipelineConfig, main, run
from fpv_editor.errors import InvalidInputError, NoFlightContentError, OutputExistsError
from fpv_editor.media import find_tools, probe
from fpv_editor.render import EncodeConfig
from media_fixtures import Span, ffmpeg_available, ground_truth, make_flight

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH"),
]

FAST = PipelineConfig(encode=EncodeConfig(preset="ultrafast"))
FLIGHT = [Span("idle", 8), Span("active", 8), Span("idle", 8), Span("active", 6), Span("idle", 6)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


@pytest.fixture(scope="session")
def flight_ts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_flight(tmp_path_factory.mktemp("src") / "flight.ts", FLIGHT)


@pytest.fixture
def flight(flight_ts: Path, tmp_path: Path) -> Path:
    copy = tmp_path / flight_ts.name
    copy.write_bytes(flight_ts.read_bytes())
    return copy


def test_end_to_end_keeps_flying_and_cuts_idle(flight: Path) -> None:
    before = sha256(flight)
    result = run(flight, config=FAST)

    assert result.output == flight.with_name("flight_highlight.mp4")
    assert result.output.is_file()
    assert sha256(flight) == before, "source must never be modified"
    assert not list(flight.parent.glob("*.partial.mp4"))

    kept = [(i.start, i.end) for i in result.selection.intervals]
    for kind, start, end in ground_truth(FLIGHT):
        fraction = overlap(kept, start, end) / (end - start)
        if kind == "active":
            assert fraction >= 0.9, f"active span {start}-{end} only {fraction:.0%} kept"
    idle_total = sum(e - s for k, s, e in ground_truth(FLIGHT) if k == "idle")
    idle_kept = sum(overlap(kept, s, e) for k, s, e in ground_truth(FLIGHT) if k == "idle")
    assert idle_kept / idle_total <= 0.2

    info = probe(find_tools(), result.output)
    assert info.video.codec == "h264" and info.audio is not None and info.audio.codec == "aac"
    assert (info.video.width, info.video.height) == (320, 180)
    assert info.duration == pytest.approx(result.edited_duration, abs=0.5)


def test_cli_prints_summary_and_writes_analysis(
    flight: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(flight), "--keep-intermediates"]) == 0
    out = capsys.readouterr().out
    for label in ("Output:", "Original:", "Edited:", "retained", "Time:"):
        assert label in out
    analysis = json.loads(flight.with_name("flight_highlight.analysis.json").read_text())
    assert analysis["selection"]["intervals"]
    assert len(analysis["samples"]["score"]) == pytest.approx(36 * 8, abs=2)


def test_existing_output_gets_a_new_name_and_is_untouched(flight: Path) -> None:
    taken = flight.with_name("flight_highlight.mp4")
    taken.write_bytes(b"previous edit")
    result = run(flight, config=FAST)
    assert result.output == flight.with_name("flight_highlight-2.mp4")
    assert taken.read_bytes() == b"previous edit"


def test_explicit_existing_output_needs_force(flight: Path) -> None:
    target = flight.with_name("mine.mp4")
    target.write_bytes(b"previous edit")
    with pytest.raises(OutputExistsError):
        run(flight, output=target, config=FAST)
    assert target.read_bytes() == b"previous edit"
    run(flight, output=target, force=True, config=FAST)
    assert target.stat().st_size > 1000


def test_input_without_audio(tmp_path: Path) -> None:
    source = make_flight(tmp_path / "silent.mp4", FLIGHT, audio=False)
    result = run(source, config=FAST)
    info = probe(find_tools(), result.output)
    assert info.audio is None


def test_very_short_input_is_kept_whole(tmp_path: Path) -> None:
    source = make_flight(tmp_path / "short.ts", [Span("active", 2)])
    result = run(source, config=FAST)
    assert result.edited_duration == pytest.approx(2.0, abs=0.2)
    assert any("shorter than" in w for w in result.warnings)


def test_all_idle_recording_is_refused_without_output(tmp_path: Path) -> None:
    source = make_flight(tmp_path / "ground.ts", [Span("idle", 10), Span("black", 5)])
    with pytest.raises(NoFlightContentError):
        run(source, config=FAST)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ground.ts"]


def test_truncated_recording_is_still_edited(flight_ts: Path, tmp_path: Path) -> None:
    # MPEG-TS cut mid-packet decodes up to the cut without errors in FFmpeg 6, so no
    # warning is required; the pipeline must simply finish on the readable part.
    data = flight_ts.read_bytes()
    source = tmp_path / "cut.ts"
    source.write_bytes(data[: int(len(data) * 0.6) + 77])  # ends mid-packet
    before = sha256(source)
    result = run(source, config=FAST)
    assert result.output.is_file()
    assert sha256(source) == before
    assert result.original_duration < 36 * 0.7


def test_garbage_input_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "notes.ts"
    source.write_bytes(b"this is not a video" * 100)
    with pytest.raises(InvalidInputError):
        run(source, config=FAST)
    assert main([str(source)]) == 1
    assert "error:" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.ts"]


def audio_envelope(path: Path, rate: int = 8000) -> np.ndarray:
    """RMS of the first audio stream in 50 ms windows."""
    raw = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0"),
            *("-ac", "1", "-ar", str(rate), "-f", "f32le", "-"),
        ],
        check=True,
        capture_output=True,
    ).stdout
    samples = np.frombuffer(raw, dtype=np.float32)
    window = rate // 20
    usable = len(samples) // window * window
    return np.sqrt((samples[:usable].reshape(-1, window) ** 2).mean(axis=1))


def test_audio_starting_before_video_stays_in_sync(tmp_path: Path) -> None:
    # DVR-style file: audio starts 0.5 s before the first video frame. The tone plays
    # exactly while the picture is "flying", so a timeline mismatch moves the tone.
    spans = [Span("idle", 8), Span("active", 8), Span("idle", 8)]
    source = make_flight(tmp_path / "dvr.ts", spans, audio_lead=0.5, gated_audio=True)
    info = probe(find_tools(), source)
    assert info.audio_lead == pytest.approx(0.5, abs=0.05)

    result = run(source, config=FAST)
    first = result.selection.intervals[0]
    expected_onset = 8.0 - first.start  # active span starts at 8 s of video
    envelope = audio_envelope(result.output)
    onset = float(np.argmax(envelope > 0.1)) * 0.05
    assert onset == pytest.approx(expected_onset, abs=0.1)


def test_rotated_odd_sized_input_renders(tmp_path: Path) -> None:
    plain = tmp_path / "plain.mp4"
    rotated = tmp_path / "phone.mp4"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=8"),
            *("-vf", "scale=321:181,format=yuv444p", "-c:v", "libx264", "-preset", "ultrafast"),
            str(plain),
        ],
        check=True,
    )
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(plain)),
            *("-c", "copy", str(rotated)),
        ],
        check=True,
    )
    result = run(rotated, config=FAST)
    info = probe(find_tools(), result.output)
    assert (info.video.width, info.video.height) == (180, 320)
