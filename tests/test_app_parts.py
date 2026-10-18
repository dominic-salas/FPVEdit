"""App pieces without a window: settings, time left, the cache, the icon (offscreen Qt)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from fpv_editor.app.dialogs import cache_size, clear_cache
from fpv_editor.app.settings import AppSettings
from fpv_editor.app.window import time_left, use_folders_first


def settings_in(folder: Path) -> AppSettings:
    return AppSettings(QSettings(str(folder / "settings.ini"), QSettings.Format.IniFormat))


def test_settings_are_remembered_and_bad_values_fall_back(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    assert settings.encoder == "auto"
    assert settings.output_folder is None
    settings.encoder = "cpu"
    settings.output_folder = tmp_path
    again = settings_in(tmp_path)
    assert (again.encoder, again.output_folder) == ("cpu", tmp_path)
    again.encoder = "quantum"
    assert again.encoder == "auto"
    again.output_folder = None
    assert again.output_folder is None


def test_time_left() -> None:
    assert time_left(1, 0.5) == ""  # too early to say
    assert time_left(30, 0.01) == ""
    assert time_left(30, 0.5) == "less than a minute left"
    assert time_left(120, 0.5) == "about 2 min left"


def test_clearing_the_cache_deletes_only_measurement_files(tmp_path: Path) -> None:
    (tmp_path / "clip.abc.measurements.npz").write_bytes(b"x" * 10)
    keep = tmp_path / "flight.mp4"
    keep.write_bytes(b"footage")
    assert cache_size(tmp_path) == 10
    assert clear_cache(tmp_path) == 1
    assert keep.read_bytes() == b"footage"
    assert cache_size(tmp_path) == 0 and clear_cache(tmp_path / "missing") == 0


def test_ffmpeg_folders_go_first_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "elsewhere")
    use_folders_first([tmp_path])
    assert os.environ["PATH"] == os.pathsep.join([str(tmp_path), "elsewhere"])


def test_icon_draws_at_every_size() -> None:
    from fpv_editor.app.look import SIZES, app_icon, icon_pixmap

    _ = QApplication.instance() or QApplication([])  # widgets need QApplication later
    for size in SIZES:
        image = icon_pixmap(size).toImage()
        assert image.width() == size
        assert image.pixelColor(size // 2, size // 2).alpha() == 255  # the tile is opaque
        assert image.pixelColor(0, 0).alpha() == 0  # rounded corner
    assert not app_icon().isNull()


def test_recording_time_comes_from_the_name_or_the_file(tmp_path: Path) -> None:
    from datetime import datetime

    from fpv_editor.app.clips import clock, recorded_at

    named = recorded_at(Path("D:/clips/20261010-203859.ts"))
    assert named is not None and named.from_name
    assert named.when == datetime(2026, 10, 10, 20, 38, 59)
    assert named.label() == "REC 10 OCT 2026 20:38"
    other = tmp_path / "hdz_0061.mp4"
    other.write_bytes(b"")
    found = recorded_at(other)
    assert found is not None and not found.from_name and found.label().startswith("FILE ")
    not_a_date = tmp_path / "20261399-999999.mp4"  # month 13: not a date
    not_a_date.write_bytes(b"")
    fallback = recorded_at(not_a_date)
    assert fallback is not None and not fallback.from_name
    assert recorded_at(tmp_path / "missing.mp4") is None
    assert (clock(189.9), clock(59.4), clock(3723)) == ("3:10", "0:59", "1:02:03")


def test_timeline_click_picks_and_drag_moves_a_shot() -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from fpv_editor.app.timeline import RULER, Block, Timeline

    QApplication.instance() or QApplication([])
    timeline = Timeline()
    timeline.resize(306, 90)
    timeline.set_blocks([Block(10.0, i, f"{i}", "", None) for i in range(3)])
    picked: list[tuple[int, float]] = []
    moved: list[tuple[int, int]] = []
    seeked: list[float] = []
    timeline.picked.connect(lambda i, share: picked.append((i, share)))
    timeline.moved.connect(lambda i, gap: moved.append((i, gap)))
    timeline.seeked.connect(seeked.append)
    y = RULER + 20
    QTest.mouseClick(timeline, Qt.MouseButton.LeftButton, pos=QPoint(150, y))
    assert picked and picked[0][0] == 1 and 0.4 < picked[0][1] < 0.6
    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, pos=QPoint(20, y))
    QTest.mouseMove(timeline, QPoint(120, y))
    QTest.mouseMove(timeline, QPoint(290, y))
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, pos=QPoint(290, y))
    assert moved == [(0, 3)]  # the first shot dropped after the last
    QTest.mouseClick(timeline, Qt.MouseButton.LeftButton, pos=QPoint(250, 4))  # the ruler
    assert len(seeked) == 1 and 24 < seeked[0] < 26


def test_dot_letters_share_the_x_height_and_baseline() -> None:
    from fpv_editor.app.look import BASELINE, GLYPHS, ROWS, X_TOP

    ascenders, descenders = set("bdfhiklt"), set("gjpqy")
    for letter, rows in GLYPHS.items():
        assert len(rows) == ROWS and all(len(row) == 5 for row in rows), letter
        lit = [y for y, row in enumerate(rows) if "#" in row]
        assert max(lit) == (ROWS - 1 if letter in descenders else BASELINE), letter
        if letter not in ascenders:
            assert min(lit) == X_TOP, letter


def test_montages_default_to_the_videos_folder_not_the_clips_card(tmp_path: Path) -> None:
    from fpv_editor.app.clips import default_output_folder

    card = tmp_path / "DCIM"
    card.mkdir()
    videos = tmp_path / "Videos" / "FPV Editor"
    other = tmp_path / "montages"
    other.mkdir()
    clips = [card / "a.ts", card / "b.ts"]
    assert default_output_folder(None, clips, videos) == videos
    assert default_output_folder(card, clips, videos) == videos  # never the clips' folder
    assert default_output_folder(other, clips, videos) == other  # the folder used last
    assert default_output_folder(tmp_path / "gone", clips, videos) == videos
