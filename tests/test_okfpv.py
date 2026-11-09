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
    from fpvedit.app.window import FPVEDIT, OKFPV, MainWindow
    from fpvedit.media import find_tools

    _ = QApplication.instance() or QApplication([])
    settings = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    return MainWindow(find_tools(), None, settings, None, OKFPV if combined else FPVEDIT)


def test_tabs_switch_pages_and_edit_comes_back(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=True)
    assert window.windowTitle().startswith("okFPV")
    window._open_pilot()
    assert window._pages.currentIndex() == window.PILOT
    assert window._steps.isHidden()
    window._open_clean()
    assert window._pages.currentIndex() == window.CLEAN
    window._open_edit()
    assert window._pages.currentIndex() == window.CLIPS
    assert not window._steps.isHidden()
    assert window._modes[window.CLIPS].property("state") == "on"


def test_the_modules_are_flowflight_fpvedit_and_anahd(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=True)
    assert [mode.text() for mode in window._modes.values()] == ["FLOWFLIGHT", "FPVEDIT", "ANAHD"]


def test_fpvedit_alone_has_no_tabs(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=False)
    assert window.windowTitle().startswith("FPVEdit")
    assert window._modes == {}
    assert not window._pilot_button.isHidden()


def test_every_letter_of_both_names_can_be_drawn() -> None:
    from fpvedit.app.look import GLYPHS

    assert set("fpvedit" + "okfpv") <= set(GLYPHS)


def test_leaderboard_view_and_score_card_stay_on_this_computer(tmp_path: Path) -> None:
    from dataclasses import replace

    from PySide6.QtWidgets import QApplication

    from fpvedit.app.pilotpage import PilotPage
    from fpvedit.history import FlightHistory
    from fpvedit.scoring import SCORER_VERSION, FlightScore, Status

    _ = QApplication.instance() or QApplication([])
    good = FlightScore(Status.SCORED, SCORER_VERSION, 60.0, "daylight", freestyle=40.0, flow=50.0)
    history = FlightHistory(":memory:")
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    history.record([(first, good), (second, replace(good, freestyle=70.0))])
    page = PilotPage(history, show_back=False)
    page.show()
    page.refresh()
    page._view.setCurrentIndex(1)
    assert page._list.item(0).text().startswith("1.  b.mp4")
    assert page._current is not None and page._current.name == "b.mp4"
    card = tmp_path / "card.png"
    assert page.save_card(card) and card.stat().st_size > 0
    assert page._share.isVisible()
