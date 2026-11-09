"""The AnaHD module stays in the window: it runs there, or offers its download there."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

from fpvedit.app.anahdpage import AnaHDPage, parse_version

_APP = QApplication.instance() or QApplication([])


class FakeSession(QObject):
    ready = Signal(QWidget)
    stopped = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.started: list[tuple[Path, list[str], str]] = []

    def start(self, program: Path, arguments: Sequence[str], theme: str) -> None:
        self.running = True
        self.started.append((program, list(arguments), theme))

    def request_close(self) -> None:
        self.running = False
        self.stopped.emit()


class FakeGetter(QObject):
    progress = Signal(float, float)
    installing = Signal()
    installed = Signal()
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.starts = 0
        self.cancels = 0

    def start(self) -> None:
        self.starts += 1

    def cancel(self) -> None:
        self.cancels += 1


def page_with(
    program: Path | None,
    version: tuple[int, ...] | None = (0, 1, 5),
    clips: list[Path] | None = None,
) -> tuple[AnaHDPage, FakeSession, FakeGetter, dict[str, Path | None]]:
    where: dict[str, Path | None] = {"program": program}
    session, getter = FakeSession(), FakeGetter()
    page = AnaHDPage(
        lambda: clips or [],
        lambda: "dark",
        lambda: where["program"],
        lambda: version,
        session,
        getter,
        True,
    )
    return page, session, getter, where


def test_installed_anahd_starts_with_the_clips_and_fills_the_page() -> None:
    program = Path("AnaHD.exe")
    page, session, _, _ = page_with(program, clips=[Path("a b.mp4"), Path("c.mp4")])
    page.refresh()
    assert session.started == [(program, ["a b.mp4", "c.mp4"], "dark")]
    window = QWidget()
    session.ready.emit(window)
    assert page._stack.currentWidget() is window
    page.refresh()
    assert len(session.started) == 1


def test_without_anahd_the_download_happens_on_the_page_then_it_opens() -> None:
    page, session, getter, where = page_with(None)
    page.refresh()
    assert page._heading.text() == "AnaHD is not installed"
    assert page.status_text() == "NOT INSTALLED"
    assert session.started == []
    page._action.click()
    assert getter.starts == 1
    getter.progress.emit(500_000_000, 2_000_000_000)
    assert "0.50 of 2.00 GB" in page._status.text()
    getter.installing.emit()
    assert page._heading.text() == "Installing AnaHD"
    where["program"] = Path("AnaHD.exe")
    getter.installed.emit()
    assert len(session.started) == 1


def test_an_older_anahd_is_offered_the_update_instead_of_opening() -> None:
    page, session, getter, _ = page_with(Path("AnaHD.exe"), (0, 1, 4))
    page.refresh()
    assert session.started == [] and page._action.text() == "UPDATE ANAHD"
    page._action.click()
    assert getter.starts == 1


def test_failed_and_cancelled_downloads_can_be_tried_again() -> None:
    page, _, getter, _ = page_with(None)
    page.refresh()
    page._action.click()
    getter.failed.emit("The download did not finish.")
    assert page._status.text() == "The download did not finish."
    page._action.click()
    assert getter.starts == 2
    page._cancel_download()
    assert getter.cancels == 1 and page._heading.text() == "AnaHD is not installed"


def test_when_anahd_stops_the_page_offers_to_open_it_again() -> None:
    program = Path("AnaHD.exe")
    page, session, _, _ = page_with(program)
    page.refresh()
    window = QWidget()
    session.ready.emit(window)
    session.request_close()
    assert page._heading.text() == "AnaHD stopped"
    assert page._stack.currentIndex() == 0
    page._action.click()
    assert len(session.started) == 2


def test_version_text() -> None:
    assert parse_version("0.1.5") == (0, 1, 5)
    assert parse_version("beta") is None
