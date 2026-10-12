from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from fpv_editor import media
from fpv_editor.errors import InvalidInputError, MediaToolError, ToolNotFoundError

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


def test_audio_lead_from_stream_start_times() -> None:
    data = {
        "format": {"start_time": "1.4"},
        "streams": [
            {**PROBE["streams"][0], "index": 0, "start_time": "1.921"},  # type: ignore[index]
            {**PROBE["streams"][1], "index": 1, "start_time": "1.400"},  # type: ignore[index]
        ],
    }
    info = media.parse_probe(Path("dvr.ts"), data)
    assert info.audio_lead == pytest.approx(0.521)
    assert (info.video.index, info.audio.index) == (0, 1)  # type: ignore[union-attr]


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
    ],
)
def test_parse_ffmpeg_major(text: str, major: int | None) -> None:
    assert media.parse_ffmpeg_major(text) == major


def test_filter_script_option_depends_on_version() -> None:
    assert media.Tools("f", "p", 6).filter_script_option == "-filter_complex_script"
    assert media.Tools("f", "p", 7).filter_script_option == "-/filter_complex"
    assert media.Tools("f", "p", None).filter_script_option == "-/filter_complex"


def test_find_tools_reports_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fpv_editor.media.shutil.which", lambda name: None)
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
