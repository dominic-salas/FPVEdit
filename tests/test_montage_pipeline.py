"""End-to-end montages from generated synthetic clips."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from fpv_editor.cli import PipelineConfig, main, run_montage
from fpv_editor.errors import InvalidInputError, NoFlightContentError, OutputExistsError
from fpv_editor.media import find_tools, probe
from fpv_editor.render import EncodeConfig
from media_fixtures import Span, ffmpeg_available, make_flight

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH"),
]

FAST = PipelineConfig(encode=EncodeConfig(preset="ultrafast"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def clips(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A small 30 fps .ts and a larger 60 fps .mp4, each with one flip in clean flying."""
    folder = tmp_path_factory.mktemp("clips")
    small = make_flight(folder / "small.ts", [Span("idle", 4), Span("flip", 10), Span("idle", 4)])
    large = make_flight(
        folder / "large.mp4",
        [Span("active", 6), Span("flip", 10), Span("active", 6)],
        width=480,
        height=360,
        fps=60,
    )
    return small, large


@pytest.fixture
def copies(clips: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    small, large = (tmp_path / clip.name for clip in clips)
    for source, copy in zip(clips, (small, large), strict=True):
        copy.write_bytes(source.read_bytes())
    return small, large


def test_montage_of_two_different_clips(copies: tuple[Path, Path]) -> None:
    small, large = copies
    before = [sha256(small), sha256(large)]
    result = run_montage([small, large], keep_intermediates=True, config=FAST)

    assert result.output == small.with_name("small_montage.mp4")
    assert [sha256(small), sha256(large)] == before, "inputs must never change"
    assert {shot.clip for shot in result.shots} == {0, 1}
    assert [shot.clip for shot in result.shots] == sorted(shot.clip for shot in result.shots)
    for shot in result.shots:
        assert 8.0 - 1e-6 <= shot.duration <= 15.0 + 1e-6
        assert any(m.kind.value == "flip" for m in shot.moments)
    # Fitted to the larger clip's size and the rate of the clip with the most screen time.
    info = probe(find_tools(), result.output)
    assert (info.video.width, info.video.height) == (480, 360)
    assert info.video.nominal_rate in (Fraction(30), Fraction(60))
    assert info.audio is None
    assert result.edited_duration == pytest.approx(sum(s.duration for s in result.shots), abs=0.2)

    assert result.plan_file is not None
    plan = json.loads(result.plan_file.read_text())
    assert [s["source"] for s in plan["shots"]] == [
        ("small.ts", "large.mp4")[s.clip] for s in result.shots
    ]
    assert plan["output"]["width"] == 480


def test_analyze_only_writes_the_plan_and_no_video(copies: tuple[Path, Path]) -> None:
    small, large = copies
    result = run_montage([small, large], analyze_only=True, config=FAST)
    assert result.output == small.with_name("small_montage.montage.json")
    assert result.output.exists()
    assert not small.with_name("small_montage.mp4").exists()
    with pytest.raises(OutputExistsError):
        run_montage([small, large], analyze_only=True, config=FAST)


def test_clips_without_moments_are_left_out(copies: tuple[Path, Path], tmp_path: Path) -> None:
    small, _ = copies
    idle = make_flight(tmp_path / "idle.mp4", [Span("idle", 12)])
    result = run_montage([idle, small], output=tmp_path / "out.mp4", config=FAST)
    assert {shot.clip for shot in result.shots} == {1}
    assert any("idle.mp4" in warning for warning in result.warnings)


def test_only_idle_clips_are_refused_without_output(tmp_path: Path) -> None:
    idle = make_flight(tmp_path / "idle.mp4", [Span("idle", 12)])
    with pytest.raises(NoFlightContentError):
        run_montage([idle], output=tmp_path / "out.mp4", config=FAST)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idle.mp4"]


def test_a_clip_given_twice_is_used_once(copies: tuple[Path, Path], tmp_path: Path) -> None:
    small, _ = copies
    result = run_montage([small, small], output=tmp_path / "out.mp4", config=FAST)
    assert {shot.clip for shot in result.shots} == {0}
    assert any("more than once" in warning for warning in result.warnings)


def test_output_may_not_be_one_of_the_inputs(copies: tuple[Path, Path]) -> None:
    small, large = copies
    with pytest.raises(InvalidInputError, match="same as an input"):
        run_montage([small, large], output=large, force=True, config=FAST)


def test_cli_montage_summary_and_option_checks(
    copies: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    small, large = copies
    output = small.with_name("m.mp4")
    assert main([str(small), str(large), "-o", str(output), "--length", "20"]) == 0
    printed = capsys.readouterr().out
    assert "Clips:     2 given, shots from 2" in printed
    assert output.exists()
    with pytest.raises(SystemExit):
        main([str(small), str(large), "--trim"])
    assert main([str(small), "--length", "-5", "-o", str(small.with_name("n.mp4"))]) == 1


def make_tone(path: Path, seconds: float, frequency: int) -> Path:
    tools = find_tools()
    args = [
        *(tools.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "lavfi"),
        *("-i", f"sine=frequency={frequency}:duration={seconds}", "-c:a", "libmp3lame"),
        str(path),
    ]
    subprocess.run(args, check=True)
    return path


def test_music_tracks_become_the_soundtrack(copies: tuple[Path, Path], tmp_path: Path) -> None:
    from fpv_editor.media import probe_audio
    from fpv_editor.render import soundtrack_filter

    first = make_tone(tmp_path / "first.mp3", 2.0, 440)
    second = make_tone(tmp_path / "second song.mp3", 60.0, 660)
    result = run_montage(list(copies), config=FAST, music=[first, second])
    info = probe(find_tools(), result.output)
    assert info.audio is not None and info.audio.codec == "aac"
    assert info.duration == pytest.approx(result.edited_duration, abs=0.1)
    # Tracks longer than the montage are cut and faded; shorter ones end in silence.
    tracks = [probe_audio(find_tools(), first), probe_audio(find_tools(), second)]
    assert tracks[0].duration == pytest.approx(2.0, abs=0.1)
    assert "afade=t=out" in soundtrack_filter(tracks, 20.0)
    assert "afade" not in soundtrack_filter(tracks[:1], 20.0)
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            find_tools().ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:duration=1",
            str(silent),
        ],
        check=True,
    )
    with pytest.raises(InvalidInputError):
        probe_audio(find_tools(), silent)  # a video without audio is no music
