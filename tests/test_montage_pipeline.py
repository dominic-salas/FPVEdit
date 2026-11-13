"""End-to-end montages from generated synthetic clips."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from fpvedit.cli import main
from fpvedit.errors import InvalidInputError, NoFlightContentError, OutputExistsError
from fpvedit.media import find_tools, probe
from fpvedit.pipeline import MontageResult, PipelineConfig, run_lines, run_montage
from fpvedit.render import EncodeConfig
from media_fixtures import Span, ffmpeg_available, make_flight, make_line_clips

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
    info = probe(find_tools(), result.output, count_frames=True)
    assert (info.video.width, info.video.height) == (480, 360)
    screen = {clip: sum(s.duration for s in result.shots if s.clip == clip) for clip in (0, 1)}
    rate = Fraction(30) if screen[0] > screen[1] else Fraction(60)
    assert info.video.nominal_rate == rate
    assert info.audio is not None and info.audio.codec == "aac"  # the recordings' sound
    # Measured in the file: exactly the frames of the shots at the montage's rate.
    assert info.video.frame_count == sum(round(s.duration * rate) for s in result.shots)
    assert result.edited_duration == pytest.approx(float(info.video.frame_count / rate), abs=1e-3)

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
    from fpvedit.media import probe_audio
    from fpvedit.render import soundtrack_filter

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


def test_flac_music_with_cover_art(copies: tuple[Path, Path], tmp_path: Path) -> None:
    """FLAC files from music stores carry the album cover as a picture stream."""
    tools = find_tools()
    cover = tmp_path / "cover.jpg"
    subprocess.run(
        [
            tools.ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=64x64",
            "-frames:v",
            "1",
            str(cover),
        ],
        check=True,
    )
    song = tmp_path / "song.flac"
    subprocess.run(
        [
            tools.ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
            "-i",
            str(cover),
            "-map",
            "0:a",
            "-map",
            "1:v",
            "-c:a",
            "flac",
            "-c:v",
            "mjpeg",
            "-disposition:v",
            "attached_pic",
            str(song),
        ],
        check=True,
    )
    result = run_montage(list(copies), config=FAST, music=[song, song])
    info = probe(tools, result.output)
    assert info.audio is not None and info.audio.codec == "aac"
    assert info.video.codec == "h264"


# ---- longest lines --------------------------------------------------------------------


@pytest.fixture(scope="module")
def line_clips(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return make_line_clips(tmp_path_factory.mktemp("line-clips"))


@pytest.fixture
def line_copies(line_clips: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    copies = tuple(tmp_path / clip.name for clip in line_clips)
    for source, copy in zip(line_clips, copies, strict=True):
        copy.write_bytes(source.read_bytes())
    return copies[0], copies[1]


def stream_seconds(path: Path) -> dict[str, float]:
    """Each stream's own length, read from the file."""
    shown = subprocess.run(
        [
            find_tools().ffprobe,
            *("-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json"),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(shown)["streams"]}


def audio_levels(path: Path, cuts: list[float]) -> list[float]:
    """RMS (0-1) of the sound between consecutive ``cuts`` (seconds)."""
    raw = subprocess.run(
        [
            find_tools().ffmpeg,
            *("-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "8000", "-"),
        ],
        check=True,
        capture_output=True,
    ).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768
    spans = pairwise(cuts)
    return [
        float(np.sqrt(np.mean(samples[round(a * 8000) : round(b * 8000)] ** 2))) for a, b in spans
    ]


def gray_frames(path: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(
        [
            find_tools().ffmpeg,
            *("-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"),
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width)


@pytest.fixture(scope="module")
def lines_montage(
    line_clips: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path, MontageResult]:
    """One lines montage of the 4:3 clip with sound and the silent 16:9 clip."""
    folder = tmp_path_factory.mktemp("lines")
    copies = []
    for source in line_clips:
        copy = folder / source.name
        copy.write_bytes(source.read_bytes())
        copies.append(copy)
    result = run_lines(copies, config=FAST, keep_intermediates=True, cache_dir=folder / "cache")
    return copies[0], copies[1], result


def sound_check(result: MontageResult) -> None:
    """The 4:3 line has the recording's tone; the 16:9 recording had none: silence."""
    seconds = stream_seconds(result.output)
    assert seconds["audio"] == pytest.approx(seconds["video"], abs=0.1)
    joint = result.shots[0].duration
    end = result.edited_duration
    (loud,) = audio_levels(result.output, [0.5, joint - 0.5])
    (quiet,) = audio_levels(result.output, [joint + 0.5, end - 0.5])
    assert loud > 0.05 and quiet < 0.005


def test_lines_montage_of_a_4_3_clip_with_sound_and_a_silent_16_9_clip(
    lines_montage: tuple[Path, Path, MontageResult],
) -> None:
    first, _, result = lines_montage
    assert result.output == first.with_name("four_three_lines.mp4")
    assert [line.clip for line in result.lines] == [0, 1]  # input order, one line each
    assert [shot.clip for shot in result.shots] == [0, 1]
    for line, shot in zip(result.lines, result.shots, strict=True):
        assert (shot.start, shot.end) == (line.start, line.end)  # each line plays whole
        assert line.duration >= 10.0
    info = probe(find_tools(), result.output, count_frames=True)
    # The size of the larger picture (4:3 here); the 16:9 recording is not stretched into it.
    assert (info.video.width, info.video.height) == (320, 240)
    rate = info.video.nominal_rate
    assert rate is not None
    assert info.video.frame_count == sum(round(s.duration * rate) for s in result.shots)
    assert result.edited_duration == pytest.approx(float(info.video.frame_count / rate), abs=1e-3)
    sound_check(result)
    frames = gray_frames(result.output, 320, 240)
    cut = round(result.shots[0].duration * float(rate))
    # A 16:9 picture in a 4:3 frame has black bars above and below (30 px of 240), a 4:3 does not.
    assert frames[cut + 5 :, :20].mean() < 24 < frames[cut + 5 :, 110:130].mean()
    assert frames[: cut - 5, :20].mean() > 24
    assert result.plan_file is not None
    plan = json.loads(result.plan_file.read_text())
    assert [(row["source"], row["ended_by"]) for row in plan["lines"]] == [
        ("four_three.ts", result.lines[0].ended_by.value),
        ("sixteen_nine.mp4", result.lines[1].ended_by.value),
    ]


def test_the_same_lines_vertical(line_copies: tuple[Path, Path], tmp_path: Path) -> None:
    result = run_lines(list(line_copies), vertical=True, config=FAST, cache_dir=tmp_path / "c")
    assert result.output.name == "four_three_lines.mp4"
    info = probe(find_tools(), result.output, count_frames=True)
    assert (info.video.width, info.video.height) == (1080, 1920)
    rate = info.video.nominal_rate
    assert rate is not None
    assert info.video.frame_count == sum(round(s.duration * rate) for s in result.shots)
    sound_check(result)
    frames = gray_frames(result.output, 1080, 1920)
    assert frames[:, :, :20].mean() > 24 and frames[:, :, -20:].mean() > 24  # filled: no bars


def test_music_replaces_the_sound_of_a_lines_montage(
    line_copies: tuple[Path, Path], tmp_path: Path
) -> None:
    song = make_tone(tmp_path / "song.mp3", 90.0, 660)
    result = run_lines(list(line_copies), config=FAST, music=[song], cache_dir=tmp_path / "c")
    seconds = stream_seconds(result.output)
    assert seconds["audio"] == pytest.approx(seconds["video"], abs=0.2)
    # The song plays through the whole montage, the silent recording's line included.
    first = result.shots[0].duration
    assert audio_levels(result.output, [first + 1, result.edited_duration - 1])[0] > 0.05


def test_an_existing_lines_file_stops_everything_without_force(
    line_copies: tuple[Path, Path], tmp_path: Path
) -> None:
    target = tmp_path / "out.mp4"
    target.write_bytes(b"not a video")
    with pytest.raises(OutputExistsError):
        run_lines(list(line_copies), output=target, config=FAST, cache_dir=tmp_path / "c")
    assert target.read_bytes() == b"not a video"
    result = run_lines(
        list(line_copies), output=target, config=FAST, force=True, cache_dir=tmp_path / "c"
    )
    assert probe(find_tools(), result.output).video.codec == "h264"


def test_default_names_are_numbered_like_the_montage_s(
    line_copies: tuple[Path, Path], tmp_path: Path
) -> None:
    first = line_copies[0].with_name("four_three_lines.mp4")
    first.write_bytes(b"an earlier export")
    result = run_lines(list(line_copies), config=FAST, cache_dir=tmp_path / "c")
    assert result.output == line_copies[0].with_name("four_three_lines-2.mp4")
    assert first.read_bytes() == b"an earlier export"


def test_analyze_only_writes_the_lines_plan_and_no_video(
    line_copies: tuple[Path, Path], tmp_path: Path
) -> None:
    result = run_lines(list(line_copies), analyze_only=True, config=FAST, cache_dir=tmp_path)
    assert result.output == line_copies[0].with_name("four_three_lines.montage.json")
    assert json.loads(result.output.read_text())["lines"]
    assert not line_copies[0].with_name("four_three_lines.mp4").exists()


def test_recordings_without_a_line_are_refused(tmp_path: Path) -> None:
    idle = make_flight(tmp_path / "idle.mp4", [Span("idle", 20)])
    with pytest.raises(NoFlightContentError, match=r"No lines found. A line needs 10 s or more"):
        run_lines([idle], output=tmp_path / "out.mp4", config=FAST)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idle.mp4"]


def test_a_recording_without_a_line_is_left_out_with_a_note(
    line_copies: tuple[Path, Path], tmp_path: Path
) -> None:
    idle = make_flight(tmp_path / "idle.mp4", [Span("idle", 20)])
    result = run_lines(
        [idle, line_copies[0]], output=tmp_path / "out.mp4", config=FAST, cache_dir=tmp_path
    )
    assert {line.clip for line in result.lines} == {1}
    assert any("idle.mp4" in warning for warning in result.warnings)


def test_cli_lines_report_and_option_checks(
    line_copies: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = line_copies
    output = tmp_path / "best.mp4"
    cache = ["--cache-dir", str(tmp_path / "cache")]
    assert main([str(first), str(second), "--lines", "-o", str(output), *cache]) == 0
    printed = capsys.readouterr().out
    assert printed.count("Line:      ") == 2
    assert "four_three.ts" in printed and "sixteen_nine.mp4" in printed
    assert "events" in printed and "in 2 lines" in printed
    assert output.exists()
    # The default name beside the first input.
    assert main([str(first), str(second), "--lines", *cache]) == 0
    assert first.with_name("four_three_lines.mp4").exists()
    capsys.readouterr()
    # An existing output stops everything unless --force.
    assert main([str(first), str(second), "--lines", "-o", str(output), *cache]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main([str(first), str(second), "--lines", "-o", str(output), "--force", *cache]) == 0
    capsys.readouterr()
    for flag in (["--trim"], ["--length", "30"], ["--min-stars", "3"]):
        with pytest.raises(SystemExit) as exc:
            main([str(first), "--lines", *flag])
        assert exc.value.code == 2
        assert flag[0] in capsys.readouterr().err
    idle = make_flight(tmp_path / "idle.mp4", [Span("idle", 20)])
    assert main([str(idle), "--lines", "-o", str(tmp_path / "none.mp4")]) == 1
    message = "No lines found. A line needs 10 s or more of tricks, close passes or fast flying."
    assert message in capsys.readouterr().err
