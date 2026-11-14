"""What kind of recording a file is: analog goggles, HDZero or an action camera."""

from __future__ import annotations

import subprocess
from contextlib import closing
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from fpvedit import measure
from fpvedit.analysis import Analysis, AnalysisConfig, analyze, cache_path
from fpvedit.cli import main
from fpvedit.flight import Screen
from fpvedit.highlights import SelectionConfig
from fpvedit.measure import decode_args, decode_frames, hardware_decode_args
from fpvedit.media import (
    AudioStream,
    MediaInfo,
    Tools,
    VideoStream,
    find_tools,
    parse_probe,
    probe,
)
from fpvedit.montage import MontageConfig
from fpvedit.picture import Area
from fpvedit.pipeline import PipelineConfig, run_montage
from fpvedit.render import EncodeConfig
from fpvedit.source import (
    PictureEvidence,
    SourceConfig,
    SourceKind,
    classify,
    frame_evidence,
    gyro_log_beside,
    parse_choice,
)
from media_fixtures import Span, ffmpeg_available, make_clip, make_flight

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH")

# Short recordings: keep all of them, so no flying has to be found in the plain fixtures.
KEEP_ALL = AnalysisConfig(selection=SelectionConfig(short_input_s=60))


def info(
    width: int = 1280,
    height: int = 720,
    rate: float = 60,
    tags: tuple[tuple[str, str], ...] = (),
    path: Path = Path("flight.mp4"),
) -> MediaInfo:
    return MediaInfo(
        path=path,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration=10.0,
        start_time=0.0,
        video=VideoStream(
            index=0,
            codec="h264",
            width=width,
            height=height,
            sample_aspect_ratio=Fraction(1),
            frame_rate=rate,
            nominal_rate=Fraction(rate).limit_denominator(1001),
        ),
        audio=AudioStream(index=1, codec="aac", sample_rate=48000, channels=2),
        tags=tags,
    )


FILLS_FRAME = PictureEvidence(active_aspect=16 / 9, bars_share=0.0, noise=0.001, osd_found=True)
PILLARBOXED = PictureEvidence(active_aspect=4 / 3, bars_share=0.26, noise=0.01, osd_found=True)


@pytest.mark.parametrize(
    ("size", "rate", "camera"),
    [
        ((1280, 720), 50, False),  # what the goggles write
        ((1280, 720), 60, False),
        ((1280, 720), 90, False),
        ((960, 540), 90, False),
        ((640, 480), 30, False),  # an older analog DVR
        ((1920, 1080), 30, False),  # HDZero's 1080p30 may be this: size alone does not decide
        ((1920, 1080), 29.97, False),
        ((1920, 1080), 60, True),
        ((1920, 1080), 25, True),
        ((2704, 1520), 30, True),
        ((3840, 2160), 60, True),
        ((1280, 720), 120, True),  # faster than the goggles record
    ],
)
def test_frame_size_and_rate_a_goggle_never_writes(
    size: tuple[int, int], rate: float, camera: bool
) -> None:
    assert bool(frame_evidence(info(*size, rate), SourceConfig())) is camera


def test_a_full_frame_clean_picture_is_digital() -> None:
    source = classify(info(), FILLS_FRAME, gyro_log=False)
    assert source.kind is SourceKind.DIGITAL
    assert source.kind.label == "HDZero"
    assert "fills the frame" in source.reason


def test_bars_around_a_noisy_picture_are_analog() -> None:
    source = classify(info(), PILLARBOXED, gyro_log=False)
    assert source.kind is SourceKind.ANALOG
    assert source.kind.label == "Analog"
    assert "bars" in source.reason


def test_a_wide_analog_picture_with_thin_bars_is_analog() -> None:
    thin = PictureEvidence(active_aspect=1.65, bars_share=0.07, noise=0.002, osd_found=True)
    assert classify(info(), thin, gyro_log=False).kind is SourceKind.ANALOG


def test_a_4k_clip_is_an_action_camera_even_with_detail_that_looks_like_an_osd() -> None:
    source = classify(info(3840, 2160, 60), FILLS_FRAME, gyro_log=False)
    assert source.kind is SourceKind.ACTION_CAM
    assert source.kind.label == "Action cam"
    assert "3840x2160" in source.reason and "scenery" in source.reason


def test_a_maker_in_the_tags_makes_an_action_camera() -> None:
    tagged = info(tags=(("handler_name", "GoPro AVC encoder"),))
    assert classify(tagged, FILLS_FRAME, gyro_log=False).kind is SourceKind.ACTION_CAM
    other = info(tags=(("encoder", "Lavf62.3.100"), ("handler_name", "VideoHandler")))
    assert classify(other, FILLS_FRAME, gyro_log=False).kind is SourceKind.DIGITAL


def test_a_gyro_log_beside_the_recording_makes_an_action_camera(tmp_path: Path) -> None:
    recording = tmp_path / "RC_0042.mp4"
    recording.write_bytes(b"")
    assert not gyro_log_beside(recording)
    (tmp_path / "rc_0042.GCSV").write_text("not read\n")
    assert gyro_log_beside(recording)
    assert classify(info(path=recording), FILLS_FRAME).kind is SourceKind.ACTION_CAM
    (tmp_path / "RC_0043.gcsv").write_text("another recording's\n")
    assert not gyro_log_beside(tmp_path / "RC_0044.mp4")


def test_a_camera_is_overruled_when_an_osd_is_found_in_a_picture_that_looks_analog() -> None:
    """Editing by the OSD is the safe choice: no OSD is assumed only when none was found."""
    source = classify(info(1920, 1080, 60), PILLARBOXED, gyro_log=False)
    assert source.kind is SourceKind.ANALOG
    assert "OSD was found" in source.reason
    no_osd = PictureEvidence(4 / 3, 0.26, 0.01, osd_found=False)
    assert classify(info(1920, 1080, 60), no_osd, gyro_log=False).kind is SourceKind.ACTION_CAM


def test_a_clean_full_frame_picture_without_an_osd_says_so() -> None:
    clean = PictureEvidence(16 / 9, 0.0, 0.001, osd_found=False)
    source = classify(info(), clean, gyro_log=False)
    assert source.kind is SourceKind.DIGITAL
    assert "no OSD" in source.reason


def test_the_user_can_name_the_kind() -> None:
    assert parse_choice("auto") is None
    assert parse_choice("hdzero") is SourceKind.DIGITAL
    assert parse_choice("action") is SourceKind.ACTION_CAM
    assert parse_choice("analog") is SourceKind.ANALOG
    with pytest.raises(ValueError, match="unknown source kind"):
        parse_choice("dji")


def test_tags_naming_the_device_are_read_from_the_probe() -> None:
    data = {
        "format": {"format_name": "mov,mp4", "duration": "5", "tags": {"encoder": "Lavf61.1"}},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 2160,
                "r_frame_rate": "60/1",
                "tags": {"handler_name": "RunCam HEVC", "language": "und", "Make": "RunCam"},
            }
        ],
    }
    tags = dict(parse_probe(Path("x.mp4"), data).tags)
    assert tags == {"encoder": "Lavf61.1", "handler_name": "RunCam HEVC", "make": "RunCam"}


@pytest.mark.integration
@needs_ffmpeg
class TestOnRecordings:
    """Generated recordings, analysed for real."""

    @pytest.fixture
    def folder(self, tmp_path: Path) -> Path:
        (tmp_path / "clips").mkdir()
        return tmp_path / "clips"

    def analyse(self, path: Path, config: AnalysisConfig = KEEP_ALL) -> Analysis:
        return analyze(find_tools(), probe(find_tools(), path), config)

    def test_a_pillarboxed_picture_with_analog_noise_is_analog(self, folder: Path) -> None:
        path = make_clip(
            folder / "analog.mp4", pillarbox=True, grain=12, osd=True, seconds=5, fps=60
        )
        analysis = self.analyse(path)
        assert analysis.source.kind is SourceKind.ANALOG
        assert analysis.flight.has_osd, "an analog recording is still read by its OSD"
        assert (analysis.area.width, analysis.area.height) == (948, 720)

    def test_a_clean_full_frame_720p_picture_is_hdzero(self, folder: Path) -> None:
        path = make_clip(folder / "digital.mp4", osd=True, seconds=5, fps=60)
        analysis = self.analyse(path)
        assert analysis.source.kind is SourceKind.DIGITAL
        assert analysis.flight.has_osd

    def test_1080p30_without_a_tag_is_not_called_an_action_camera(self, folder: Path) -> None:
        path = make_clip(folder / "hd30.mp4", width=1920, height=1080, fps=30, seconds=5)
        source = self.analyse(path).source
        assert source.kind is SourceKind.DIGITAL
        assert "no OSD" in source.reason

    def test_4k60_with_a_maker_tag_is_an_action_camera_and_ignores_osd_like_detail(
        self, folder: Path
    ) -> None:
        path = make_clip(
            folder / "thumb.mp4",
            width=3840,
            height=2160,
            fps=60,
            seconds=3,
            osd=True,
            handler="RunCam AVC",
            audio=True,
        )
        # The same picture as an analog recording is read by its OSD ...
        as_hdzero = self.analyse(
            path, AnalysisConfig(selection=KEEP_ALL.selection, source="hdzero")
        )
        assert as_hdzero.flight.has_osd
        # ... but a camera never uses OSD events.
        analysis = self.analyse(path)
        assert analysis.source.kind is SourceKind.ACTION_CAM
        assert not analysis.flight.has_osd
        assert not [e for e in analysis.flight.events if e.kind in (Screen.ARMED, Screen.STATS)]
        assert [(a, round(b)) for a, b in analysis.flight.armed] == [(0.0, 3)]
        assert (analysis.area.width, analysis.area.height) == (3840, 2160)

    def test_a_gyro_log_beside_the_recording_makes_it_an_action_camera(self, folder: Path) -> None:
        path = make_clip(folder / "RC_0001.mp4", seconds=3)
        assert self.analyse(path).source.kind is SourceKind.DIGITAL
        (folder / "RC_0001.gcsv").write_text("gyro log, never read\n")
        source = self.analyse(path).source
        assert source.kind is SourceKind.ACTION_CAM
        assert ".gcsv" in source.reason

    def test_the_users_choice_wins(self, folder: Path) -> None:
        path = make_clip(folder / "forced.mp4", pillarbox=True, grain=12, osd=True, seconds=5)
        selection = KEEP_ALL.selection
        forced = self.analyse(path, AnalysisConfig(selection=selection, source="action"))
        assert forced.source.kind is SourceKind.ACTION_CAM
        assert forced.source.forced
        assert not forced.flight.has_osd
        digital = self.analyse(path, AnalysisConfig(selection=selection, source="hdzero"))
        assert digital.source.kind is SourceKind.DIGITAL and digital.flight.has_osd

    def test_changing_the_choice_does_not_measure_again(self, folder: Path, tmp_path: Path) -> None:
        path = make_clip(folder / "cached.mp4", pillarbox=True, grain=12, osd=True, seconds=5)
        tools, media = find_tools(), probe(find_tools(), path)
        auto = AnalysisConfig(selection=KEEP_ALL.selection)
        forced = AnalysisConfig(selection=KEEP_ALL.selection, source="action")
        assert cache_path(tmp_path, tools, media, auto) == cache_path(
            tmp_path, tools, media, forced
        )
        first = analyze(tools, media, auto, cache_dir=tmp_path)
        again = analyze(tools, media, forced, cache_dir=tmp_path)
        assert not first.from_cache and again.from_cache
        assert first.source.kind is SourceKind.ANALOG
        assert again.source.kind is SourceKind.ACTION_CAM

    def test_the_command_line_names_the_kind(
        self, folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = make_clip(folder / "cli.mp4", pillarbox=True, grain=12, osd=True, seconds=5)
        analysis_file = tmp_path / "cli.mp4"
        code = main(["--trim", "--analyze-only", str(path), "-o", str(analysis_file)])
        assert code == 0
        assert "Source:    Analog" in capsys.readouterr().out
        code = main(["--trim", "--analyze-only", "--force", "--source", "action", str(path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "Source:    Action cam" in out


FAST = PipelineConfig(encode=EncodeConfig(preset="ultrafast"))
# Short clips: shots of 3 to 6 seconds, so a few seconds of footage make a montage.
SHORT_SHOTS = PipelineConfig(
    encode=EncodeConfig(preset="ultrafast"),
    montage=MontageConfig(shot_min_s=3.0, shot_max_s=6.0, lead_s=1.0, tail_s=1.0),
)


@pytest.mark.integration
@needs_ffmpeg
def test_a_90_fps_recording_is_edited_at_90_fps(tmp_path: Path) -> None:
    clip = make_flight(
        tmp_path / "fast.mp4",
        [Span("active", 6), Span("flip", 10), Span("active", 6)],
        fps=90,
    )
    assert probe(find_tools(), clip).video.nominal_rate == Fraction(90)
    result = run_montage([clip], config=FAST)
    assert result.shots
    info = probe(find_tools(), result.output, count_frames=True)
    assert info.video.nominal_rate == Fraction(90)
    assert info.video.frame_count == sum(round(shot.duration * 90) for shot in result.shots)
    assert info.audio is not None, "the recording's sound is kept"


@pytest.mark.integration
@needs_ffmpeg
def test_90_and_25_fps_recordings_make_one_montage(tmp_path: Path) -> None:
    fast = make_flight(
        tmp_path / "fast.mp4", [Span("active", 6), Span("flip", 10), Span("active", 6)], fps=90
    )
    slow = make_flight(
        tmp_path / "slow.ts", [Span("active", 6), Span("flip", 10), Span("active", 6)], fps=25
    )
    result = run_montage([fast, slow], config=FAST)
    assert {shot.clip for shot in result.shots} == {0, 1}
    info = probe(find_tools(), result.output, count_frames=True)
    assert info.video.nominal_rate in (Fraction(90), Fraction(25))
    assert info.duration == pytest.approx(result.edited_duration, abs=0.1)


@pytest.mark.integration
@needs_ffmpeg
def test_a_4k_action_camera_clip_goes_through_plan_and_render(tmp_path: Path) -> None:
    small = make_flight(
        tmp_path / "small.mp4", [Span("active", 3), Span("flip", 4), Span("active", 3)], fps=24
    )
    clip = tmp_path / "RC_0007.mp4"
    subprocess.run(
        [
            *("ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(small)),
            *("-vf", "scale=3840:2160:flags=bicubic,format=yuv420p"),
            *("-c:v", "libx264", "-preset", "ultrafast", "-g", "12", "-c:a", "copy"),
            *("-metadata:s:v:0", "handler_name=RunCam AVC", str(clip)),
        ],
        check=True,
        capture_output=True,
    )
    result = run_montage([clip], config=SHORT_SHOTS, keep_intermediates=True)
    assert result.sources[0] is not None
    assert result.sources[0].kind is SourceKind.ACTION_CAM
    assert result.shots, "a flip in a no-OSD clip makes a shot"
    info = probe(find_tools(), result.output, count_frames=True)
    assert (info.video.width, info.video.height) == (3840, 2160)
    assert info.video.nominal_rate == Fraction(24)
    assert info.audio is not None


def test_large_recordings_are_decoded_the_fast_way() -> None:
    tools = find_tools() if ffmpeg_available() else Tools("ffmpeg", "ffprobe", 7)
    area = Area(0, 0, 3840, 2160)
    big = decode_args(tools, info(3840, 2160, 60), area, (640, 360), 60.0)
    assert big[big.index("-skip_loop_filter") :][:2] == ["-skip_loop_filter", "all"]
    usual = decode_args(tools, info(1280, 720, 60), Area(0, 0, 1280, 720), (640, 360), 60.0)
    assert "-skip_loop_filter" not in usual, "720p recordings are decoded exactly as before"
    full_hd = decode_args(tools, info(1920, 1080, 30), Area(0, 0, 1920, 1080), (640, 360), 30.0)
    assert "-skip_loop_filter" not in full_hd


def test_gpu_decoding_is_only_for_large_upright_whole_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measure, "nvidia_decoding", lambda _ffmpeg: True)
    tools = Tools("ffmpeg", "ffprobe", 7)

    def args(recording: MediaInfo, area: Area) -> list[str] | None:
        return hardware_decode_args(tools, recording, area, (640, 360), 60.0)

    big = info(3840, 2160, 60)
    whole = Area(0, 0, 3840, 2160)
    command = args(big, whole)
    assert command is not None
    assert "scale_cuda=1280:720:interp_algo=bilinear:format=nv12" in " ".join(command)
    assert args(info(1280, 720, 60), Area(0, 0, 1280, 720)) is None
    assert args(info(1920, 1080, 30), Area(0, 0, 1920, 1080)) is None
    assert args(big, Area(0, 240, 3840, 1680)) is None, "the GPU scaler cannot crop"
    vp9 = replace(big, video=replace(big.video, codec="vp9"))
    assert args(vp9, whole) is None
    turned = replace(big, video=replace(big.video, rotation=90))
    assert args(turned, whole) is None
    monkeypatch.setattr(measure, "nvidia_decoding", lambda _ffmpeg: False)
    assert args(big, whole) is None


def test_gpu_decoding_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(measure.HARDWARE_OFF, "1")
    measure.nvidia_decoding.cache_clear()
    try:
        assert not measure.nvidia_decoding("ffmpeg")
    finally:
        measure.nvidia_decoding.cache_clear()


@pytest.mark.integration
@needs_ffmpeg
def test_a_failing_gpu_decode_falls_back_to_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(tmp_path / "big.mp4", width=2560, height=1440, fps=10, seconds=2)
    tools, media = find_tools(), probe(find_tools(), clip)
    area, size = Area(0, 0, 2560, 1440), (640, 360)

    def failing(*_args: object) -> list[str]:
        return [tools.ffmpeg, "-nostdin", "-v", "error", "-i", str(tmp_path / "missing"), "-"]

    monkeypatch.setattr(measure, "hardware_decode_args", failing)
    report = measure.DecodeReport()
    with closing(decode_frames(tools, media, area, size, 10.0, report)) as frames:
        decoded = list(frames)
    assert len(decoded) == 20 and report.frames == 20
    assert report.warnings == [], "the failed attempt is not reported: the CPU did the work"


@pytest.mark.integration
@needs_ffmpeg
def test_gpu_and_cpu_decoding_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever this machine does (GPU or not), the frames match the CPU's closely."""
    clip = make_clip(tmp_path / "big.mp4", width=2560, height=1440, fps=10, seconds=2, osd=True)
    tools, media = find_tools(), probe(find_tools(), clip)
    area, size = Area(0, 0, 2560, 1440), (640, 360)

    def run() -> np.ndarray:
        with closing(decode_frames(tools, media, area, size, 10.0, measure.DecodeReport())) as it:
            return np.stack(list(it)).astype(np.int16)

    measure.nvidia_decoding.cache_clear()
    best = run()
    monkeypatch.setenv(measure.HARDWARE_OFF, "1")
    measure.nvidia_decoding.cache_clear()
    try:
        cpu = run()
    finally:
        measure.nvidia_decoding.cache_clear()
    assert best.shape == cpu.shape
    assert float(np.abs(best - cpu).mean()) < 2.0
