"""The app's dialogs: settings, About, and help when FFmpeg is missing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from fpv_editor import __version__
from fpv_editor.app.settings import AppSettings
from fpv_editor.media import INSTALL_HINT

CACHE_PATTERN = "*.measurements.npz"


def cache_size(folder: Path | None) -> int:
    if folder is None or not folder.is_dir():
        return 0
    return sum(path.stat().st_size for path in folder.glob(CACHE_PATTERN))


def clear_cache(folder: Path | None) -> int:
    """Delete the app's own measurement files (never anything else); returns how many."""
    if folder is None or not folder.is_dir():
        return 0
    removed = 0
    for path in folder.glob(CACHE_PATTERN):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, cache_dir: Path | None, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._settings = settings
        self._cache_dir = cache_dir
        self._folder = settings.output_folder

        self._length = QDoubleSpinBox()
        self._length.setRange(10, 1800)
        self._length.setDecimals(0)
        self._length.setSuffix(" s")
        self._length.setValue(settings.length_s)

        self._encoder = QComboBox()
        self._encoder.addItem("NVIDIA GPU when available, else CPU", "auto")
        self._encoder.addItem("CPU only", "cpu")
        self._encoder.setCurrentIndex(max(0, self._encoder.findData(settings.encoder)))

        self._folder_label = QLabel()
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose_folder)
        beside = QPushButton("Beside the first clip")
        beside.clicked.connect(lambda: self._set_folder(None))
        folder_row = QHBoxLayout()
        folder_row.addWidget(self._folder_label, 1)
        folder_row.addWidget(choose)
        folder_row.addWidget(beside)
        self._set_folder(self._folder)

        self._cache_label = QLabel()
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear_cache)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self._cache_label, 1)
        cache_row.addWidget(clear)
        self._show_cache()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("Montage length to aim for", self._length)
        form.addRow("Video encoding", self._encoder)
        form.addRow("Save montages in", folder_row)
        form.addRow("Saved measurements", cache_row)
        form.addRow(buttons)

    def _set_folder(self, folder: Path | None) -> None:
        self._folder = folder
        self._folder_label.setText(str(folder) if folder else "the first clip's folder")

    def _choose_folder(self) -> None:
        start = str(self._folder or "")
        name = QFileDialog.getExistingDirectory(self, "Save montages in", start)
        if name:
            self._set_folder(Path(name))

    def _show_cache(self) -> None:
        megabytes = cache_size(self._cache_dir) / 1e6
        self._cache_label.setText(
            f"{megabytes:.0f} MB (makes finding shots in the same clips again take seconds)"
        )

    def _clear_cache(self) -> None:
        clear_cache(self._cache_dir)
        self._show_cache()

    def _save(self) -> None:
        self._settings.length_s = self._length.value()
        self._settings.encoder = str(self._encoder.currentData())
        self._settings.output_folder = self._folder
        self.accept()


def about_text(ffmpeg_version: str) -> str:
    return (
        f"<h3>FPV Editor {__version__}</h3>"
        "<p>Turns FPV recordings into a montage of their best shots. Everything runs on "
        "this computer; footage is never uploaded.</p>"
        "<p>MIT License. Uses Qt for Python (PySide6, LGPL-3.0), NumPy (BSD), OpenCV "
        "(Apache-2.0) and FFmpeg"
        + (f" ({ffmpeg_version})" if ffmpeg_version else "")
        + " (LGPL/GPL, run as a separate program). Licence texts are in the "
        "<code>licenses</code> folder of the installation.</p>"
    )


def ask_for_ffmpeg(parent: QWidget | None) -> Path | None:
    """Explain that FFmpeg is missing and let the user point to it; None if they cancel."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("FFmpeg not found")
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText("FPV Editor needs FFmpeg to read and write video, and could not find it.")
    box.setInformativeText(
        f"{INSTALL_HINT}\n\nIf FFmpeg is already installed somewhere else, choose the folder "
        "that contains ffmpeg and ffprobe."
    )
    choose = box.addButton("Choose folder…", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.exec()
    if box.clickedButton() is not choose:
        return None
    name = QFileDialog.getExistingDirectory(parent, "Folder with ffmpeg and ffprobe")
    return Path(name) if name else None
