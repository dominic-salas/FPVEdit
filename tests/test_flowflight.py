"""The combined app is the same window with tabs; FPVEdit alone is unchanged."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from media_fixtures import ffmpeg_available

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="needs FFmpeg")


def make_window(tmp_path: Path, combined: bool):  # type: ignore[no-untyped-def]
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from fpvedit.app.settings import AppSettings
    from fpvedit.app.window import FLOWFLIGHT, FPVEDIT, MainWindow
    from fpvedit.media import find_tools

    _ = QApplication.instance() or QApplication([])
    settings = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    return MainWindow(find_tools(), None, settings, None, FLOWFLIGHT if combined else FPVEDIT)


def test_tabs_switch_pages_and_edit_comes_back(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=True)
    assert window.windowTitle().startswith("FlowFlight")
    window._open_pilot()
    assert window._pages.currentIndex() == window.PILOT
    assert window._steps.isHidden()
    window._open_clean()
    assert window._pages.currentIndex() == window.CLEAN
    window._open_edit()
    assert window._pages.currentIndex() == window.CLIPS
    assert not window._steps.isHidden()
    assert window._modes[window.CLIPS].property("state") == "on"


def test_fpvedit_alone_has_no_tabs(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=False)
    assert window.windowTitle().startswith("FPVEdit")
    assert window._modes == {}
    assert not window._pilot_button.isHidden()


def test_every_letter_of_both_names_can_be_drawn() -> None:
    from fpvedit.app.look import GLYPHS

    assert set("fpvedit" + "flowflight") <= set(GLYPHS)
