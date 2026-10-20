from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from fpvedit import media
from fpvedit.errors import InvalidInputError, MediaToolError, ToolNotFoundError

PROBE = {
    "format": {"format_name": "mpegts", "duration": "36.021333", "start_time": "1.400000"},
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1440,
            "height": 1080,
            "sample_aspect_ratio": "4:3",
            "avg_frame_rate": "60000/1001",
        },
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
    ],
}


def test_parse_probe_reads_streams_and_timing() -> None:
    info = media.parse_probe(Path("flight.ts"), PROBE)
    assert info.duration == pytest.approx(36.021333)
    assert info.start_time == pytest.approx(1.4)
    assert info.video.codec == "h264"
    assert info.video.frame_rate == pytest.approx(59.94, abs=0.01)
    assert info.video.sample_aspect_ratio == Fraction(4, 3)
    assert info.display_aspect == pytest.approx(16 / 9)
    assert info.audio is not None and info.audio.sample_rate == 48000


def test_parse_probe_without_audio_or_duration() -> None:
    data = {"format": {}, "streams": [PROBE["streams"][0]]}  # type: ignore[index]
    info = media.parse_probe(Path("x.ts"), data)
    assert info.audio is None
    assert info.duration is None
    assert info.start_time == 0.0


def test_parse_probe_ignores_cover_art_and_requires_video() -> None:
    cover = {"codec_type": "video", "width": 10, "height": 10, "disposition": {"attached_pic": 1}}
    with pytest.raises(InvalidInputError, match="no video stream"):
        media.parse_probe(Path("a.mp4"), {"streams": [cover], "format": {}})


def test_parse_probe_reads_rotation_and_swaps_display_aspect() -> None:
    video = dict(PROBE["streams"][0])  # type: ignore[index]
    video["sample_aspect_ratio"] = "1:1"
    video["width"], video["height"] = 1920, 1080
    video["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": -90}]
    info = media.parse_probe(Path("phone.mp4"), {"format": {}, "streams": [video]})
    assert info.video.rotation == 270
    assert info.display_aspect == pytest.approx(9 / 16)


def test_stream_indexes_with_audio_first() -> None:
    data = {
        "format": {"start_time": "1.4"},
        "streams": [
            {**PROBE["streams"][1], "index": 0},  # type: ignore[index]
            {**PROBE["streams"][0], "index": 1},  # type: ignore[index]
        ],
    }
    info = media.parse_probe(Path("dvr.ts"), data)
    assert (info.video.index, info.audio.index) == (1, 0)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("stream", "rate", "colour_range"),
    [
        ({"r_frame_rate": "60/1", "avg_frame_rate": "1526940000/25448999"}, Fraction(60), None),
        ({"r_frame_rate": "90000/1", "avg_frame_rate": "30000/1001"}, Fraction(30000, 1001), None),
        ({"r_frame_rate": "0/0", "avg_frame_rate": "1526940000/25448999"}, Fraction(60), None),
        ({"r_frame_rate": "0/0", "avg_frame_rate": "27182/907"}, Fraction(30000, 1001), None),
        ({"r_frame_rate": "0/0", "pix_fmt": "yuvj420p"}, Fraction(60000, 1001), "pc"),
        ({"r_frame_rate": "0/0", "avg_frame_rate": "0/0"}, None, None),
        ({"r_frame_rate": "25/1", "color_range": "tv"}, Fraction(25), "tv"),
    ],
)
def test_nominal_rate_and_colour_range(
    stream: dict[str, str], rate: Fraction | None, colour_range: str | None
) -> None:
    data = {"format": {}, "streams": [{**PROBE["streams"][0], **stream}]}  # type: ignore[index]
    info = media.parse_probe(Path("dvr.ts"), data)
    assert info.video.nominal_rate == rate
    assert info.video.color_range == colour_range


@pytest.mark.parametrize(
    ("value", "expected"),
    [("16:9", Fraction(16, 9)), ("30000/1001", Fraction(30000, 1001)), ("0:1", None), ("", None)],
)
def test_parse_ratio(value: str, expected: Fraction | None) -> None:
    assert media.parse_ratio(value) == expected


@pytest.mark.parametrize(
    ("text", "major"),
    [
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright", 6),
        ("ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright", 9),
        ("ffmpeg version n7.1 Copyright", 7),
        ("ffmpeg version N-118000-g1234 Copyright", None),
        ("ffmpeg version N-112000-gabc\nlibavfilter     9. 12.100 /  9. 12.100", 6),
        ("ffmpeg version 2026-10-01-git-abc\nlibavfilter    12.  1.102 / 12.  1.102", 9),
    ],
)
def test_parse_ffmpeg_major(text: str, major: int | None) -> None:
    assert media.parse_ffmpeg_major(text) == major


def test_filter_script_option_depends_on_version() -> None:
    assert media.Tools("f", "p", 6).filter_script_option == "-filter_complex_script"
    assert media.Tools("f", "p", 7).filter_script_option == "-/filter_complex"
    assert media.Tools("f", "p", None).filter_script_option == "-/filter_complex"


def test_find_tools_reports_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fpvedit.media.shutil.which", lambda name: None)
    with pytest.raises(ToolNotFoundError, match="ffmpeg and ffprobe not found"):
        media.find_tools()


def test_run_tool_includes_stderr_tail_in_error() -> None:
    import sys

    script = "import sys; sys.stderr.write('bad input\\n'); sys.exit(3)"
    with pytest.raises(MediaToolError, match="exit code 3") as exc:
        media.run_tool([sys.executable, "-c", script], action="probing")
    assert "bad input" in str(exc.value)


def test_probe_rejects_missing_file(tmp_path: Path) -> None:
    tools = media.Tools("ffmpeg", "ffprobe", 6)
    with pytest.raises(InvalidInputError, match="not found"):
        media.probe(tools, tmp_path / "missing.ts")


def test_variable_frame_rate_is_detected() -> None:
    steady = media.parse_probe(Path("a.ts"), PROBE).video
    assert not media.is_variable_rate(steady)
    phone = {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080}
    phone |= {"r_frame_rate": "30/1", "avg_frame_rate": "2700/119"}
    data = {"format": {}, "streams": [phone]}
    assert media.is_variable_rate(media.parse_probe(Path("phone.mp4"), data).video)


def _dvr_times() -> np.ndarray:
    """60 fps for 40 s, 1.13 s of signal loss, then 35 frames at the 50 fps no-signal rate."""
    flight = np.arange(2400) / 60
    tail = flight[-1] + 1.13 + np.arange(35) / 50
    return np.concatenate([flight, tail])


def test_steady_recording_with_signal_loss_is_constant_rate() -> None:
    # A real recording (20261008-134959): its average rate is 2.9 % low only because of the ending.
    stream = media.parse_probe(Path("a.ts"), PROBE).video
    times = _dvr_times()
    assert media.regular_step_share(times, Fraction(60)) > 0.98
    assert not media.is_variable_rate(stream, times)
    gaps = media.frame_gaps(times, Fraction(60))
    assert len(gaps) == 1 and gaps[0][1] == pytest.approx(1.13)


def test_jittery_frame_times_are_variable_rate() -> None:
    stream = media.parse_probe(Path("a.ts"), PROBE).video
    steps = np.random.default_rng(1).uniform(0.010, 0.030, 3000)  # a phone at ~50 fps
    assert media.is_variable_rate(stream, np.cumsum(steps))
