from __future__ import annotations

from pathlib import Path

import pytest

from fpv_editor.errors import InvalidInputError, OutputExistsError
from fpv_editor.highlights import Interval
from fpv_editor.render import (
    build_filter_graph,
    default_output_path,
    finalize_output,
    resolve_output_path,
    temporary_output,
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
    with pytest.raises(InvalidInputError, match="same as the input"):
        resolve_output_path(source, source, force=True)
    link = tmp_path / "b.mp4"
    try:
        link.hardlink_to(source)
    except OSError:
        pytest.skip("hard links not supported here")
    with pytest.raises(InvalidInputError, match="same as the input"):
        resolve_output_path(source, link, force=True)


@pytest.mark.parametrize("name", ["out.mov", "out"])
def test_output_must_be_mp4(tmp_path: Path, name: str) -> None:
    with pytest.raises(InvalidInputError, match=r"\.mp4"):
        resolve_output_path(tmp_path / "a.ts", tmp_path / name, force=False)


def test_output_folder_must_exist(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="folder does not exist"):
        resolve_output_path(tmp_path / "a.ts", tmp_path / "missing" / "o.mp4", force=False)


def test_filter_graph_with_audio() -> None:
    graph = build_filter_graph(
        [Interval(1.0, 4.5), Interval(10.0, 12.0)], video_index=0, audio_index=1
    )
    assert (
        "[0:0]setpts=PTS-STARTPTS,trim=start=1.000000:end=4.500000,setpts=PTS-STARTPTS,"
        "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0[v0]"
    ) in graph
    assert "[0:1]asetpts=PTS-STARTPTS,atrim=start=10.000000:end=12.000000" in graph
    assert "afade=t=out:st=1.990000" in graph
    assert graph.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")


def test_filter_graph_shifts_audio_by_its_lead() -> None:
    graph = build_filter_graph([Interval(0.2, 3.0)], video_index=1, audio_index=0, audio_lead=0.5)
    assert "[0:1]setpts=PTS-STARTPTS,trim=start=0.200000:end=3.000000" in graph
    assert "atrim=start=0.700000:end=3.500000" in graph


def test_filter_graph_clamps_audio_that_starts_after_video() -> None:
    graph = build_filter_graph([Interval(0.2, 3.0)], video_index=0, audio_index=1, audio_lead=-0.5)
    assert "atrim=start=0.000000:end=2.500000" in graph


def test_filter_graph_without_audio() -> None:
    graph = build_filter_graph([Interval(0.0, 3.0)], video_index=0)
    assert "atrim" not in graph
    assert graph.endswith("[v0]concat=n=1:v=1:a=0[v]")


def test_filter_graph_requires_intervals() -> None:
    with pytest.raises(ValueError):
        build_filter_graph([], video_index=0)


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
