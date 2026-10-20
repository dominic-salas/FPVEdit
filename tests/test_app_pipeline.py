"""What the app relies on: progress and cancel, rendering edited shots in any order, and
the window itself (offscreen, when PySide6 is installed)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fpv_editor.cli import (
    ClipJob,
    PipelineConfig,
    combine_clips,
    montage_parts,
    plan_in_worker,
    plan_montage,
    render_montage,
)
from fpv_editor.errors import CancelledError
from fpv_editor.media import find_tools, probe, run_tool_with_progress
from fpv_editor.montage import Shot
from fpv_editor.render import EncodeConfig
from media_fixtures import Span, ffmpeg_available, make_flight

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH"),
]

FAST = PipelineConfig(encode=EncodeConfig(preset="ultrafast"))


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    folder = tmp_path_factory.mktemp("app-clips")
    small = make_flight(folder / "small.ts", [Span("idle", 4), Span("flip", 10), Span("idle", 4)])
    large = make_flight(
        folder / "large.mp4",
        [Span("active", 6), Span("flip", 10), Span("active", 6)],
        width=480,
        height=360,
        fps=60,
    )
    return small, large


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_is_reported_and_cancel_stops(
    clips: tuple[Path, Path], tmp_path: Path, workers: int
) -> None:
    tools = find_tools()
    infos = [probe(tools, clip) for clip in clips]
    seen: list[tuple[str, float]] = []
    plan = plan_montage(
        tools,
        infos,
        config=FAST,
        cache_dir=tmp_path,
        progress=lambda t, s: seen.append((t, s)),
        workers=workers,
    )
    assert plan.shots
    shares = [share for _, share in seen]
    assert shares == sorted(shares) and shares[0] >= 0 and shares[-1] <= 1
    texts = {text for text, _ in seen}
    if workers == 1:
        assert texts == {f"Analysing {clip.name}" for clip in clips}
    else:
        assert all(text.startswith("Analysing 2 clips") for text in texts)

    def cancel(text: str, share: float) -> None:
        if share > 0.1:
            raise CancelledError("cancelled")

    cache = tmp_path / "cancelled"
    with pytest.raises(CancelledError):
        plan_montage(tools, infos, config=FAST, cache_dir=cache, progress=cancel, workers=workers)
    if workers == 1:
        assert not list(cache.glob("*.npz"))  # the cancelled clip was not cached


def test_parallel_analysis_gives_the_same_plan(clips: tuple[Path, Path]) -> None:
    tools = find_tools()
    infos = [probe(tools, clip) for clip in clips]
    alone = plan_montage(tools, infos, config=FAST, workers=1)
    together = plan_montage(tools, infos, config=FAST, workers=2)
    assert together.shots == alone.shots
    assert together.candidates == alone.candidates
    assert together.warnings == alone.warnings


def test_reordered_shots_render_in_the_new_order(clips: tuple[Path, Path], tmp_path: Path) -> None:
    tools = find_tools()
    infos = [probe(tools, clip) for clip in clips]
    plan = plan_montage(tools, infos, config=FAST)
    large = next(clip for clip in plan.clips if clip.info.path.name == "large.mp4")
    later, earlier = Shot(large.index, 8, 12, 1.0, ()), Shot(large.index, 1, 4, 1.0, ())
    # Going back in time within one clip needs a second part.
    assert len(montage_parts(plan.clips, [later, earlier])) == 2
    assert len(montage_parts(plan.clips, [earlier, later])) == 1
    seen: list[float] = []
    output = tmp_path / "reordered.mp4"
    length = render_montage(
        tools,
        plan.clips,
        [later, earlier, *plan.shots],
        output,
        force=False,
        encode=FAST.encode,
        progress=lambda _, share: seen.append(share),
    )
    assert output.is_file()  # validated inside render_montage
    assert length == pytest.approx(7 + sum(s.duration for s in plan.shots), abs=0.1)
    assert seen[-1] == 1.0
    assert not list(tmp_path.glob("*.partial.mp4"))


def test_cancelled_render_leaves_no_output(clips: tuple[Path, Path], tmp_path: Path) -> None:
    tools = find_tools()
    plan = plan_montage(tools, [probe(tools, clip) for clip in clips], config=FAST)
    output = tmp_path / "cancelled.mp4"

    def cancel(text: str, share: float) -> None:
        if text.startswith("Rendering") and share > 0:
            raise CancelledError("cancelled")

    with pytest.raises(CancelledError):
        render_montage(
            tools, plan.clips, plan.shots, output, force=False, encode=FAST.encode, progress=cancel
        )
    assert list(tmp_path.iterdir()) == []


def test_tool_progress_counts_frames(tmp_path: Path) -> None:
    tools = find_tools()
    frames: list[int] = []
    args = [
        *(tools.ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi"),
        *("-i", "testsrc=size=64x64:rate=30", "-frames:v", "90", "-f", "null", "-"),
    ]
    run_tool_with_progress(args, action="test", on_frames=frames.append)
    assert frames and frames[-1] == 90


def test_window_lists_and_edits_the_shots(clips: tuple[Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from fpv_editor.app.myratings import MyRatings
    from fpv_editor.app.settings import AppSettings
    from fpv_editor.app.window import MainWindow, video_files

    app = QApplication.instance() or QApplication([])
    tools = find_tools()
    # A settings file of its own, so the test never touches the user's settings.
    settings = AppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    window = MainWindow(tools, None, settings)
    window.show()
    try:
        window.add_paths([clips[0].parent])  # a folder: its recordings are added
        assert [p.name for p in window.clip_paths()] == sorted(c.name for c in clips)
        assert video_files([clips[0].with_name("missing.ts")]) == []
        plan = plan_montage(tools, [probe(tools, p) for p in window.clip_paths()], config=FAST)
        window._plan_ready(plan)
        review = window._review
        assert len(review._timeline._blocks) == len(plan.shots)
        review._picked(0, 0.0)  # a click on the first shot on the timeline
        before = window._current_shots()[0]
        review._trim(False, 0.25)
        assert window._current_shots()[0].start == pytest.approx(before.start + 0.25)
        review.remove_selected()
        assert len(review._timeline._blocks) == len(plan.shots) - 1
        window._undo()
        window._undo()
        assert window._current_shots() == plan.shots
        review._mine = MyRatings(tmp_path / "my-ratings.jsonl")  # not the user's file
        review.rate_current(4)  # rating is off by default: ignored
        assert review._mine.stars(plan.infos[0].path, window._current_shots()[0]) is None
        settings.learn_taste = True
        window._show_rating_controls()
        review.rate_current(4)  # the pilot's own stars for the open shot
        assert review._mine.stars(plan.infos[0].path, window._current_shots()[0]) == 4
        if len(plan.shots) > 1:
            review._moved(0, len(plan.shots))  # drag the first shot to the end
            assert window._current_shots()[-1] == plan.shots[0]
            window._undo()
        assert window._current_shots() == plan.shots
        app.processEvents()
    finally:
        window.close()


def test_clips_analysed_one_by_one_combine_into_the_same_plan(clips: tuple[Path, Path]) -> None:
    """What the app does: analyse each clip on its own (any order), then combine."""

    class Never:
        def is_set(self) -> bool:
            return False

    class Drop:
        def put(self, item: object) -> None:
            pass

    tools = find_tools()
    infos = [probe(tools, clip) for clip in clips]
    expected = plan_montage(tools, infos, config=FAST, workers=1)
    alone = [
        plan_in_worker(ClipJob(tools, info, 7 + i, FAST, FAST.montage, None), Drop(), Never())
        for i, info in enumerate(reversed(infos))
    ]
    results = [result.moved_to(len(infos) - 1 - i) for i, result in enumerate(alone)]
    combined = combine_clips(infos, results, FAST.montage)
    assert combined.shots == expected.shots
    assert sorted(combined.candidates, key=repr) == sorted(expected.candidates, key=repr)
    assert [c.index for c in combined.clips] == [c.index for c in expected.clips]
    assert combined.warnings == expected.warnings


def test_window_analyses_clips_in_the_background(clips: tuple[Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import time

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from fpv_editor.app.settings import AppSettings
    from fpv_editor.app.window import MainWindow

    app = QApplication.instance() or QApplication([])
    tools = find_tools()
    settings = AppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    window = MainWindow(tools, tmp_path / "cache", settings)
    window.show()

    def wait(done: object, seconds: float = 120) -> None:
        end = time.monotonic() + seconds
        while not done() and time.monotonic() < end:  # type: ignore[operator]
            app.processEvents()
            time.sleep(0.05)
        assert done(), "timed out"  # type: ignore[operator]

    try:
        window.add_paths(list(clips))
        paths = window.clip_paths()

        def analysed() -> bool:
            states = [window._analysis.state(p) for p in paths]
            return all(s is not None and s.finished for s in states)

        wait(analysed)
        window._find_shots()  # nothing left to analyse: the shots are picked at once
        wait(lambda: not window._busy(), 5)
        expected = plan_montage(tools, [probe(tools, p) for p in paths], workers=1)
        assert window._current_shots() == expected.shots
        assert window._pages.currentIndex() == window.REVIEW
        # Reordering the clips needs no new analysis either.
        window._clips.insertItem(0, window._clips.takeItem(1))
        window._clips_changed()
        window._find_shots()
        wait(lambda: not window._busy(), 5)
        again = plan_montage(tools, [probe(tools, p) for p in window.clip_paths()], workers=1)
        assert window._current_shots() == again.shots
    finally:
        window.close()
        app.processEvents()
