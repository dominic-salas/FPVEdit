from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from fpvedit.errors import (
    InvalidInputError,
    MediaToolError,
    OutputExistsError,
    ToolNotFoundError,
)
from fpvedit.highlights import Interval
from fpvedit.media import Tools
from fpvedit.render import (
    NVENC,
    RENDER_ACTION,
    EncodeConfig,
    build_filter_graph,
    default_output_path,
    encoders,
    expected_frame_count,
    finalize_output,
    resolve_output_path,
    snap_to_frames,
    temporary_output,
    with_fallback,
)


def test_default_output_name_and_collision_suffix(tmp_path: Path) -> None:
    source = tmp_path / "flight 01.ts"
    assert default_output_path(source) == tmp_path / "flight 01_highlight.mp4"
    (tmp_path / "flight 01_highlight.mp4").write_bytes(b"x")
    (tmp_path / "flight 01_highlight-2.mp4").write_bytes(b"x")
    assert default_output_path(source) == tmp_path / "flight 01_highlight-3.mp4"


def test_explicit_existing_output_requires_force(tmp_path: Path) -> None:
    source = tmp_path / "a.ts"
    existing = tmp_path / "out.mp4"
    existing.write_bytes(b"keep me")
    with pytest.raises(OutputExistsError, match="--force"):
        resolve_output_path(source, existing, force=False)
    assert resolve_output_path(source, existing, force=True) == existing.resolve()
    assert existing.read_bytes() == b"keep me"


def test_output_may_never_be_the_input(tmp_path: Path) -> None:
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    with pytest.raises(InvalidInputError, match="same as an input"):
        resolve_output_path(source, source, force=True)
    link = tmp_path / "b.mp4"
    try:
        link.hardlink_to(source)
    except OSError:
        pytest.skip("hard links not supported here")
    with pytest.raises(InvalidInputError, match="same as an input"):
        resolve_output_path(source, link, force=True)


@pytest.mark.parametrize("name", ["out.mov", "out"])
def test_output_must_be_mp4(tmp_path: Path, name: str) -> None:
    with pytest.raises(InvalidInputError, match=r"\.mp4"):
        resolve_output_path(tmp_path / "a.ts", tmp_path / name, force=False)


def test_output_folder_must_exist(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="folder does not exist"):
        resolve_output_path(tmp_path / "a.ts", tmp_path / "missing" / "o.mp4", force=False)


def test_filter_graph_selects_frames_and_writes_limited_range_at_source_rate() -> None:
    graph = build_filter_graph(
        [Interval(1.0, 4.5), Interval(10.0, 12.0)], video_index=0, rate=Fraction(60)
    )
    assert graph.startswith("[0:0]setpts=PTS-STARTPTS,select='")
    # Half a frame (1/120 s) of margin before each boundary.
    assert "between(t,0.991667,4.491667)+between(t,9.991667,11.991667)" in graph
    assert "setpts=N*1/60/TB,fps=60/1" in graph
    assert "out_range=tv" in graph and "out_color_matrix=bt709" in graph
    assert graph.endswith("format=yuv420p[v]")
    assert "[a]" not in graph and "atrim" not in graph


def test_filter_graph_keeps_ntsc_rates_exact() -> None:
    graph = build_filter_graph([Interval(0.0, 3.0)], video_index=1, rate=Fraction(30000, 1001))
    assert graph.startswith("[0:1]")
    assert "setpts=N*1001/30000/TB,fps=30000/1001" in graph


def test_filter_graph_requires_intervals() -> None:
    with pytest.raises(ValueError):
        build_filter_graph([], video_index=0, rate=Fraction(60))


def test_snap_moves_boundaries_to_the_nearest_frame() -> None:
    snapped = snap_to_frames(
        [Interval(1.005, 2.0), Interval(3.0, 3.001), Interval(5.012, 5.04)], Fraction(60)
    )
    assert snapped[0] == Interval(1.0, 2.0)
    # 3.0-3.001 holds no frame once rounded; 5.012-5.04 holds the frame at 5.0167 s.
    assert snapped[1:] == (Interval(301 / 60, 302 / 60),)


def test_snap_tolerates_float_noise_on_frame_times() -> None:
    snapped = snap_to_frames([Interval(2.0000000001, 3.9999999)], Fraction(60))
    assert snapped == (Interval(2.0, 4.0),)


def test_expected_frame_count() -> None:
    rate = Fraction(60)
    intervals = snap_to_frames([Interval(0.0, 1.0), Interval(2.5, 2.75)], rate)
    assert expected_frame_count(intervals, rate) == 60 + 15


def test_expected_frame_count_allows_for_missing_frames() -> None:
    rate = Fraction(60)
    times = np.concatenate([np.arange(240) / 60, 5.0 + np.arange(120) / 60])  # 1 s gap
    intervals = snap_to_frames([Interval(1.0, 6.0)], rate)
    assert expected_frame_count(intervals, rate) == 300
    assert expected_frame_count(intervals, rate, times) == 180 + 60


def test_finalize_moves_temporary_into_place(tmp_path: Path) -> None:
    final = tmp_path / "out.mp4"
    temporary = temporary_output(final)
    assert temporary.parent == tmp_path and temporary.name.endswith(".partial.mp4")
    temporary.write_bytes(b"video")
    finalize_output(temporary, final, force=False)
    assert final.read_bytes() == b"video"
    assert not temporary.exists()


def test_finalize_never_replaces_a_file_that_appeared_meanwhile(tmp_path: Path) -> None:
    final = tmp_path / "out.mp4"
    temporary = temporary_output(final)
    final.write_bytes(b"someone else's")
    with pytest.raises(OutputExistsError):
        finalize_output(temporary, final, force=False)
    assert final.read_bytes() == b"someone else's"


def test_cpu_device_never_tries_the_gpu() -> None:
    tools = Tools("ffmpeg-that-does-not-exist", "ffprobe", 9)
    cpu = EncodeConfig(preset="ultrafast")
    assert encoders(tools, "cpu", cpu) == [cpu]
    assert encoders(tools, "auto", cpu) == [cpu]  # no NVENC here: CPU only
    with pytest.raises(ToolNotFoundError, match="cuda"):
        encoders(tools, "cuda", cpu)
    with pytest.raises(InvalidInputError, match="device"):
        encoders(tools, "gpu", cpu)


def test_fallback_moves_on_only_when_the_encode_fails() -> None:
    cpu = EncodeConfig()
    tried: list[str] = []

    def encode_fails_on_gpu(encode: EncodeConfig) -> str:
        tried.append(encode.video_codec)
        if encode == NVENC:
            raise MediaToolError(RENDER_ACTION, 1, "No NVENC capable devices found")
        return "done"

    assert with_fallback([NVENC, cpu], encode_fails_on_gpu) == ("done", cpu)
    assert tried == ["h264_nvenc", "libx264"]

    def join_fails(encode: EncodeConfig) -> str:
        raise MediaToolError("joining the montage", 1, "broken")

    with pytest.raises(MediaToolError, match="joining"):
        with_fallback([NVENC, cpu], join_fails)
