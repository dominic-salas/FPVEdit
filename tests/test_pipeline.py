"""End-to-end runs of the real pipeline on generated synthetic flights."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from fpv_editor.cli import PipelineConfig, _render_and_validate, main, run
from fpv_editor.errors import (
    InvalidInputError,
    NoFlightContentError,
    OutputExistsError,
    ValidationError,
)
from fpv_editor.highlights import Interval
from fpv_editor.measure import MeasureConfig
from fpv_editor.media import find_tools, frame_times, is_variable_rate, probe
from fpv_editor.render import EncodeConfig, validate_output
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
    assert info.video.codec == "h264"
    assert info.audio is None, "goggle audio is never carried into the edit"
    assert (info.video.width, info.video.height) == (320, 180)
    assert info.duration == pytest.approx(result.edited_duration, abs=0.5)


def test_cli_prints_summary_and_writes_analysis(
    flight: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(flight), "--trim", "--keep-intermediates"]) == 0
    out = capsys.readouterr().out
    lines = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1]
        for line in out.splitlines()
        if ":" in line
    }
    for label in ("Output", "Original", "Edited", "Time"):
        assert label in lines, out
    assert "flight_highlight.mp4" in lines["Output"]
    info = probe(find_tools(), flight.with_name("flight_highlight.mp4"))
    original = probe(find_tools(), flight).duration
    assert original is not None and info.duration is not None
    numbers = re.findall(r"(\d+):(\d+(?:\.\d+)?)", lines["Original"] + " " + lines["Edited"])
    shown = [int(m) * 60 + float(s) for m, s in numbers]
    assert shown[0] == pytest.approx(original, abs=1.0)
    assert shown[1] == pytest.approx(info.duration, abs=1.0)
    percent = float(re.findall(r"(\d+(?:\.\d+)?)\s*%", lines["Edited"])[0])
    assert percent == pytest.approx(100 * info.duration / original, abs=1.5)
    analysis = json.loads(flight.with_name("flight_highlight.analysis.json").read_text())
    assert analysis["selection"]["intervals"]
    assert len(analysis["signals"]["image_speed"]) == pytest.approx(36 * 10, abs=2)
    assert analysis["selection"]["mode"] == "no-osd"


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


def test_full_range_60fps_dvr_footage_renders_limited_range_at_60fps(tmp_path: Path) -> None:
    # Analog goggles record full-range ("yuvj") 60 fps MPEG-TS. FFmpeg 7's concat
    # filter used to write 25 fps, and the range tag differed between versions.
    # Without an OSD, only pieces of at least 6 s of clear flying are kept.
    spans = [
        Span("idle", 4),
        Span("active", 7),
        Span("idle", 8),
        Span("active", 7),
        Span("idle", 3),
    ]
    source = make_flight(tmp_path / "dvr.ts", spans, fps=60, full_range=True)
    tools = find_tools()
    assert probe(tools, source).video.color_range == "pc"

    result = run(source, config=FAST)
    info = probe(tools, result.output, count_frames=True)
    assert info.video.nominal_rate == 60
    assert (info.video.pix_fmt, info.video.color_range) == ("yuv420p", "tv")
    assert info.video.frame_count == round(result.edited_duration * 60)
    assert len(result.selection.intervals) >= 2


def test_validation_rejects_a_wrong_frame_rate(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=25:d=2"),
            *("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
            *("-color_range", "tv", str(clip)),
        ],
        check=True,
    )
    with pytest.raises(ValidationError, match="frame rate"):
        validate_output(find_tools(), clip, expected_duration=2.0, rate=Fraction(60))


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


def test_pillarboxed_osd_flight_is_analysed_inside_the_picture(tmp_path: Path) -> None:
    source = make_flight(
        tmp_path / "goggles.ts", FLIGHT, width=320, height=180, pillarbox=True, osd=True
    )
    result = run(source, analyze_only=True, config=FAST)
    assert result.output == tmp_path / "goggles.analysis.json"
    assert not (tmp_path / "goggles_highlight.mp4").exists(), "analyze-only writes no video"
    document = json.loads(result.output.read_text())
    x, y, width, height = document["picture"]["active_area"]
    assert abs(x - 40) <= 3 and abs(width - 240) <= 6 and (y, height) == (0, 180)
    assert 0.02 < document["picture"]["overlay_fraction"] < 0.3

    kept = [(i.start, i.end) for i in result.selection.intervals]
    for kind, start, end in ground_truth(FLIGHT):
        if kind == "active":
            assert overlap(kept, start, end) / (end - start) >= 0.9


def test_cache_reuses_measurements_and_gives_the_same_edit(flight: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = run(flight, analyze_only=True, cache_dir=cache, config=FAST)
    assert not first.from_cache
    assert len(list(cache.glob("*.measurements.npz"))) == 1
    second = run(flight, analyze_only=True, force=True, cache_dir=cache, config=FAST)
    assert second.from_cache
    assert second.selection == first.selection

    # The same recording copied to another folder (times kept) reuses them.
    moved = tmp_path / "elsewhere" / flight.name
    moved.parent.mkdir()
    shutil.copy2(flight, moved)
    assert run(moved, analyze_only=True, cache_dir=cache, config=FAST).from_cache

    # A changed recording (new modification time) is measured again.
    os.utime(flight, ns=(flight.stat().st_atime_ns, flight.stat().st_mtime_ns + 10**9))
    assert not run(flight, analyze_only=True, force=True, cache_dir=cache, config=FAST).from_cache

    # A different measurement setting must not reuse the old file.
    other = PipelineConfig(
        measure=MeasureConfig(height=240), encode=EncodeConfig(preset="ultrafast")
    )
    run(flight, analyze_only=True, force=True, cache_dir=cache, config=other)
    assert len(list(cache.glob("*.measurements.npz"))) == 3


def test_analyze_only_refuses_to_replace_an_analysis_without_force(flight: Path) -> None:
    run(flight, analyze_only=True, config=FAST)
    with pytest.raises(OutputExistsError):
        run(flight, analyze_only=True, config=FAST)


def test_damaged_cache_file_is_measured_again(flight: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run(flight, analyze_only=True, cache_dir=cache, config=FAST)
    (cached,) = cache.glob("*.measurements.npz")
    cached.write_bytes(cached.read_bytes()[:100])  # truncated, e.g. by a full disk
    result = run(flight, analyze_only=True, force=True, cache_dir=cache, config=FAST)
    assert not result.from_cache
    assert run(flight, analyze_only=True, force=True, cache_dir=cache, config=FAST).from_cache


def test_unwritable_cache_only_warns(flight: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    result = run(flight, analyze_only=True, cache_dir=blocker, config=FAST)
    assert any("measurement cache" in warning for warning in result.warnings)


def test_analyze_only_name_does_not_depend_on_existing_videos(flight: Path) -> None:
    flight.with_name("flight_highlight.mp4").write_bytes(b"earlier edit")
    result = run(flight, analyze_only=True, config=FAST)
    assert result.output == flight.with_name("flight.analysis.json")
    assert flight.with_name("flight_highlight.mp4").read_bytes() == b"earlier edit"


LIMITED = ("-vf", "scale=out_range=tv,format=yuv420p", "-color_range", "tv", "-colorspace", "bt709")


def _encode(path: Path, *extra: str) -> Path:
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=60:d=2"),
            *extra,
            *("-c:v", "libx264", "-preset", "ultrafast", str(path)),
        ],
        check=True,
    )
    return path


def test_recording_with_signal_loss_renders_and_validates(tmp_path: Path) -> None:
    # Like Dom's converted DVR clips: steady 60 fps, 1.13 s of frames missing (signal
    # loss), then frames at the 50 fps no-signal rate. In MP4 the frame before the gap
    # carries the gap as its duration, which once made the encoder repeat it ~60 times.
    source = tmp_path / "gap.mp4"
    retime = "setpts='if(gte(T,7),7+(T-7)*1.2,T)/TB'"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=60:d=8"),
            *("-vf", f"select='not(between(t,4.0,5.13))',{retime}", "-fps_mode", "passthrough"),
            *("-c:v", "libx264", "-preset", "ultrafast", str(source)),
        ],
        check=True,
    )
    tools = find_tools()
    info = probe(tools, source)
    times = frame_times(tools, info)
    assert times is not None and not is_variable_rate(info.video, times)
    for end in (5.5, 7.9):  # just after the gap, and into the 50 fps tail
        output = tmp_path / f"out{end}.mp4"
        _render_and_validate(
            tools, info, (Interval(1.0, end),), output, force=False, config=FAST, times=times
        )
        digests = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-f", "framemd5", "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        frames = [line.split(",")[-1] for line in digests if not line.startswith("#")]
        assert all(a != b for a, b in itertools.pairwise(frames)), "a frame was repeated"


def test_validation_rejects_full_range_output(tmp_path: Path) -> None:
    clip = _encode(tmp_path / "full.mp4", "-vf", "scale=out_range=pc,format=yuvj420p")
    with pytest.raises(ValidationError, match="range"):
        validate_output(find_tools(), clip, expected_duration=2.0, rate=Fraction(60))


def test_validation_rejects_an_audio_track(tmp_path: Path) -> None:
    clip = _encode(
        tmp_path / "sound.mp4",
        *("-f", "lavfi", "-i", "sine=d=2", *LIMITED, "-c:a", "aac"),
    )
    with pytest.raises(ValidationError, match="audio"):
        validate_output(find_tools(), clip, expected_duration=2.0, rate=Fraction(60))


def test_validation_rejects_a_wrong_frame_count(tmp_path: Path) -> None:
    clip = _encode(tmp_path / "ok.mp4", *LIMITED)
    tools = find_tools()
    validate_output(tools, clip, expected_duration=2.0, rate=Fraction(60), expected_frames=120)
    with pytest.raises(ValidationError, match="frames"):
        validate_output(tools, clip, expected_duration=2.0, rate=Fraction(60), expected_frames=200)


def test_variable_frame_rate_input_is_refused_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "phone.mp4"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=60:d=12"),
            # Every frame lands up to 12 ms off the 60 fps grid, like a phone recording.
            *("-vf", "setpts='(N/60+0.012*sin(N*1.7))/TB'", "-fps_mode", "passthrough"),
            *("-c:v", "libx264", "-preset", "ultrafast", str(source)),
        ],
        check=True,
    )
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(InvalidInputError, match="fps_mode cfr"):
        run(source, config=FAST)
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # nothing written


def _analyse_into(args: tuple[str, str]) -> bool:
    from fpv_editor.analysis import analyze

    source, cache = args
    tools = find_tools()
    return analyze(
        tools, probe(tools, Path(source)), FAST.analysis, cache_dir=Path(cache)
    ).from_cache


def test_two_processes_measuring_one_clip_share_the_cache_safely(
    flight: Path, tmp_path: Path
) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    cache = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(2, mp_context=context) as pool:
        results = list(pool.map(_analyse_into, [(str(flight), str(cache))] * 2))
    assert results == [False, False]  # both measured, neither failed
    assert [p.name.endswith(".measurements.npz") for p in cache.iterdir()] == [True]
    assert _analyse_into((str(flight), str(cache)))  # the one file left is whole
