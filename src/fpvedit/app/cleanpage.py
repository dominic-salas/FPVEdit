"""The Clean tab: hand the clips to AnaHD, which cleans analog goggle footage.

AnaHD needs PyTorch and an NVIDIA card, so it stays a separate program that the pilot can
install or leave out. The original recordings are only passed along, never changed here.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from fpvedit.app.widgets import button, plural

ANAHD_PAGE = "https://dvsalas.com/anahd"
Launcher = Callable[[Path, Sequence[str]], bool]


def find_anahd(local_programs: Path | None = None) -> Path | None:
    """The installed AnaHD program, if there is one."""
    if local_programs is None:
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            return None
        local_programs = Path(base) / "Programs"
    program = local_programs / "AnaHD" / "AnaHD.exe"
    return program if program.is_file() else None


def start_detached(program: Path, arguments: Sequence[str]) -> bool:
    return bool(QProcess.startDetached(str(program), list(arguments))[0])


class CleanPage(QWidget):
    def __init__(
        self,
        clips: Callable[[], list[Path]],
        find: Callable[[], Path | None] = find_anahd,
        launch: Launcher = start_detached,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._clips = clips
        self._find = find
        self._launch = launch
        self._text = QLabel()
        self._text.setProperty("role", "hero")
        self._text.setWordWrap(True)
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status = QLabel()
        self._status.setProperty("role", "hint")
        self._action = button("OPEN ANAHD", self._act, primary=True)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.addStretch(1)
        layout.addWidget(self._text)
        layout.addWidget(self._action, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._status, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        """Look for AnaHD again and show what the button will do."""
        self._status.setText("")
        if self._find() is None:
            self._text.setText(
                "Cleaning analog goggle footage needs AnaHD, a free separate download "
                "for PCs with an NVIDIA card."
            )
            self._action.setText("GET ANAHD")
            return
        self._text.setText(
            "AnaHD repairs breakup and upscales analog footage. Score and edit your original "
            "recordings; cleaning a copy never changes them."
        )
        count = len(self._clips())
        self._action.setText(f"CLEAN {plural(count, 'CLIP')} IN ANAHD" if count else "OPEN ANAHD")

    def _act(self) -> None:
        program = self._find()
        if program is None:
            QDesktopServices.openUrl(QUrl(ANAHD_PAGE))
            return
        if self._launch(program, [str(path) for path in self._clips()]):
            self._status.setText("AnaHD is opening in its own window.")
        else:
            self._status.setText("AnaHD could not be started.")
