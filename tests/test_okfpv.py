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
    assert window._pages.currentIndex() == window.HOME
    assert window._steps.isHidden()
    window._open_pilot()
    assert window._pages.currentIndex() == window.PILOT
    assert window._steps.isHidden()
    window._open_anahd()
    assert window._pages.currentIndex() == window.ANAHD
    window._open_edit()
    assert window._pages.currentIndex() == window.CLIPS
    assert not window._steps.isHidden()
    assert window._modes[window.CLIPS].property("state") == "on"


def test_the_modules_are_flowflight_fpvedit_and_anahd(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=True)
    assert [mode.toolTip() for mode in window._modes.values()] == [
        "FlowFlight",
        "FPVEdit",
        "AnaHD",
    ]
    assert all(mode.isHidden() for mode in window._modes.values())  # Home shows the big panels
    window._open_pilot()
    assert not any(mode.isHidden() for mode in window._modes.values())
    assert window._theme_button.isHidden()
    assert window._theme_action in window._menu_button.menu().actions()


def test_home_rows_open_their_modules(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=True)
    window._home.rows["anahd"].click()
    assert window._pages.currentIndex() == window.ANAHD
    window._open_home()
    window._home.rows["fpvedit"].click()
    assert window._pages.currentIndex() == window.CLIPS
    assert window._home.rows["anahd"].status.text() == "NOT INSTALLED"


def test_home_panels_fill_the_window(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    window = make_window(tmp_path, combined=True)
    window.resize(1400, 860)
    window.show()
    QApplication.processEvents()
    panels = list(window._home.rows.values())
    assert sum(panel.width() for panel in panels) >= window.width() - 4
    assert all(panel.height() >= window.height() - 60 for panel in panels)


def test_fpvedit_alone_has_no_tabs(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=False)
    assert window.windowTitle().startswith("FPVEdit")
    assert window._modes == {}
    assert not window._pilot_button.isHidden()


def test_every_letter_of_both_names_can_be_drawn() -> None:
    from fpvedit.app.look import GLYPHS

    assert set("fpvedit" + "okfpv" + "flowflight" + "anahd") <= set(GLYPHS)


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


def test_small_dot_text_is_drawn_solid_and_can_change_size() -> None:
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    from fpvedit.app.look import DotTitle

    _ = QApplication.instance() or QApplication([])
    title = DotTitle("l", cell=4)
    title.set_colour("#ffffff")
    image = QImage(title.size(), QImage.Format.Format_RGB32)
    image.fill(QColor(0, 0, 0))
    title.render(image)
    x = 2 * 4 + 2  # the middle of the letter's stem
    column = [image.pixelColor(x, y).lightness() for y in range(4, 24)]
    assert min(column) == 255  # no gaps between the dots of a stroke
    title.set_cell(6)
    assert title.height() == 9 * 6


def _side_click(window, button) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtTest import QTest

    QTest.mouseClick(window._pages.currentWidget(), button)
    window._side_press = 0.0  # the next click is a new one


def _back(window) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import Qt

    _side_click(window, Qt.MouseButton.BackButton)


def _forward(window) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import Qt

    _side_click(window, Qt.MouseButton.ForwardButton)


def _with_shots(window, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pretend a montage was found and exported, so Review and Done can be shown."""
    monkeypatch.setattr(window, "_current_shots", lambda: [object()])
    monkeypatch.setattr(window._review, "refresh", lambda: None)
    window._last_output = Path("montage.mp4")


def test_back_walks_review_done_and_forward_redoes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    window = make_window(tmp_path, combined=False)
    _with_shots(window, monkeypatch)
    window._show_page(window.DONE)
    _back(window)
    assert window._pages.currentIndex() == window.REVIEW
    _back(window)
    assert window._pages.currentIndex() == window.CLIPS
    _back(window)  # nothing before the first page
    assert window._pages.currentIndex() == window.CLIPS
    _forward(window)
    assert window._pages.currentIndex() == window.REVIEW
    _forward(window)
    assert window._pages.currentIndex() == window.DONE
    _forward(window)
    assert window._pages.currentIndex() == window.DONE


def test_navigating_another_way_clears_forward(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    window = make_window(tmp_path, combined=False)
    _with_shots(window, monkeypatch)
    window._show_page(window.REVIEW)
    _back(window)
    window._show_page(window.RATE)
    _forward(window)
    assert window._pages.currentIndex() == window.RATE
    _back(window)
    assert window._pages.currentIndex() == window.CLIPS


def test_back_keys_work_and_one_click_acts_once(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    window = make_window(tmp_path, combined=False)
    _with_shots(window, monkeypatch)
    window._show_page(window.DONE)
    QTest.keyClick(window._pages.currentWidget(), Qt.Key.Key_Back)
    assert window._pages.currentIndex() == window.REVIEW
    QTest.keyClick(window._pages.currentWidget(), Qt.Key.Key_Forward)
    assert window._pages.currentIndex() == window.DONE
    # a key right after the button press is the same click
    QTest.mouseClick(window._pages.currentWidget(), Qt.MouseButton.BackButton)
    QTest.keyClick(window._pages.currentWidget(), Qt.Key.Key_Back)
    assert window._pages.currentIndex() == window.REVIEW


def test_rate_and_pilot_go_back_to_clips_in_fpvedit(tmp_path: Path) -> None:
    window = make_window(tmp_path, combined=False)
    window._show_page(window.RATE)
    _back(window)
    assert window._pages.currentIndex() == window.CLIPS
    if window._pilot is not None:
        window._show_page(window.PILOT)
        _back(window)
        assert window._pages.currentIndex() == window.CLIPS


def test_okfpv_modules_go_back_home_and_forward_returns(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    window = make_window(tmp_path, combined=True)
    window._open_edit()
    assert window._pages.currentIndex() == window.CLIPS
    _back(window)
    assert window._pages.currentIndex() == window.HOME
    _back(window)  # Home is the top
    assert window._pages.currentIndex() == window.HOME
    _forward(window)
    assert window._pages.currentIndex() == window.CLIPS
    _with_shots(window, monkeypatch)
    window._show_page(window.REVIEW)
    _back(window)
    assert window._pages.currentIndex() == window.CLIPS  # not the Edit tab's remembered page
    if window._pilot is not None:
        window._open_pilot()
        _back(window)
        assert window._pages.currentIndex() == window.HOME


def test_side_buttons_are_ignored_while_working(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    window = make_window(tmp_path, combined=False)
    _with_shots(window, monkeypatch)
    window._show_page(window.REVIEW)
    monkeypatch.setattr(window, "_busy", lambda: True)
    _back(window)
    assert window._pages.currentIndex() == window.REVIEW
    monkeypatch.undo()
    window._pages.setCurrentIndex(window.WORK)
    _back(window)
    assert window._pages.currentIndex() == window.WORK


def test_back_closes_a_review_side_panel_first(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    window = make_window(tmp_path, combined=False)
    _with_shots(window, monkeypatch)
    window._show_page(window.REVIEW)
    review = window._review
    monkeypatch.setattr(review, "_play_montage", lambda *_: None)
    review._choosing = "swap"
    _back(window)
    assert review._choosing is None
    assert window._pages.currentIndex() == window.REVIEW
    _back(window)
    assert window._pages.currentIndex() == window.CLIPS
