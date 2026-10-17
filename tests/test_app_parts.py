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
    assert settings.length_s == 90 and settings.encoder == "auto"
    assert settings.output_folder is None
    settings.length_s = 45
    settings.encoder = "cpu"
    settings.output_folder = tmp_path
    again = settings_in(tmp_path)
    assert (again.length_s, again.encoder, again.output_folder) == (45, "cpu", tmp_path)
    again.encoder = "quantum"
    again.length_s = 99999
    assert again.encoder == "auto" and again.length_s == 1800
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
