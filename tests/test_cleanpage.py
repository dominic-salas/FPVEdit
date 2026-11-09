"""The Clean tab finds AnaHD and hands it the clips unchanged."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _app() -> object:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_finds_the_installed_program(tmp_path: Path) -> None:
    from fpvedit.app.cleanpage import find_anahd

    assert find_anahd(tmp_path) is None
    program = tmp_path / "AnaHD" / "AnaHD.exe"
    program.parent.mkdir()
    program.write_bytes(b"")
    assert find_anahd(tmp_path) == program


def test_passes_the_clips_as_separate_arguments(tmp_path: Path) -> None:
    from fpvedit.app.cleanpage import CleanPage

    program = tmp_path / "AnaHD.exe"
    clips = [Path("E:/my clips/a b.mp4"), Path("c.ts")]
    calls: list[tuple[Path, list[str]]] = []

    def launch(path: Path, arguments: Sequence[str]) -> bool:
        calls.append((path, list(arguments)))
        return True

    page = CleanPage(lambda: clips, lambda: program, launch)
    assert page._action.text() == "CLEAN 2 CLIPS IN ANAHD"
    page._act()
    assert calls == [(program, [str(clips[0]), str(clips[1])])]
    assert "opening" in page._status.text()


def test_no_clips_just_opens_it(tmp_path: Path) -> None:
    from fpvedit.app.cleanpage import CleanPage

    page = CleanPage(list, lambda: tmp_path / "AnaHD.exe", lambda _p, _a: True)
    assert page._action.text() == "OPEN ANAHD"


def test_says_so_when_it_cannot_start(tmp_path: Path) -> None:
    from fpvedit.app.cleanpage import CleanPage

    page = CleanPage(list, lambda: tmp_path / "AnaHD.exe", lambda _p, _a: False)
    page._act()
    assert "could not" in page._status.text()


def test_offers_the_download_without_it() -> None:
    from fpvedit.app.cleanpage import CleanPage

    page = CleanPage(list, lambda: None)
    assert page._action.text() == "GET ANAHD"
