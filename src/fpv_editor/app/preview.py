"""A looping preview of one shot, decoded by FFmpeg and shown in a label.

Qt's own media player needs the large PySide6-Addons package and its own decoder; FFmpeg
is already required and reads every recording the render reads.
"""

from __future__ import annotations

import subprocess
import threading
import time

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from fpv_editor.app.frames import preview_args, preview_size
from fpv_editor.media import NO_WINDOW, MediaInfo, Tools

PREVIEW_HEIGHT = 360
PREVIEW_FPS = 30


class _Playback(QThread):
    """Plays ``start``-``end`` again and again at real speed until stopped."""

    frame = Signal(QImage, float)  # picture, its time on the analysis timeline

    def __init__(
        self, tools: Tools, info: MediaInfo, start: float, end: float, first: float, parent: QObject
    ) -> None:
        super().__init__(parent)
        self._tools, self._info = tools, info
        self._start, self._end, self._first = start, end, first
        self._stop = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
        self.wait()

    def run(self) -> None:
        size = preview_size(self._info, PREVIEW_HEIGHT)
        frame_bytes = size[0] * size[1] * 3
        begin = self._first
        while not self._stop.is_set():
            args = preview_args(self._tools, self._info, begin, self._end, size, PREVIEW_FPS)
            with self._lock:
                if self._stop.is_set():
                    return
                try:
                    process = subprocess.Popen(
                        args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        creationflags=NO_WINDOW,
                    )
                except OSError:
                    return
                self._process = process
            if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
                return
            shown = 0
            clock = time.monotonic()
            with process:
                while not self._stop.is_set():
                    data = process.stdout.read(frame_bytes)
                    if len(data) < frame_bytes:
                        break
                    image = QImage(data, size[0], size[1], 3 * size[0], QImage.Format.Format_RGB888)
                    self.frame.emit(image.copy(), begin + shown / PREVIEW_FPS)
                    shown += 1
                    delay = clock + shown / PREVIEW_FPS - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                if process.poll() is None:
                    process.kill()
            if shown == 0:
                return  # nothing decodable here; do not spin
            begin = self._start


class PreviewView(QLabel):
    """Shows the shot being previewed, scaled to fit, with its time below."""

    timeChanged = Signal(float)

    def __init__(self, tools: Tools, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tools = tools
        self._playback: _Playback | None = None
        self._image: QImage | None = None
        self._generation = 0  # frames of an older playback can still be queued
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: black; color: grey;")
        self.setText("Select a shot to preview it")

    def play(self, info: MediaInfo, start: float, end: float, first: float | None = None) -> None:
        """Loop ``start``-``end`` of ``info``; the first pass begins at ``first``."""
        self.stop()
        begin = start if first is None else min(max(first, start), end)
        self._generation += 1
        playback = _Playback(self._tools, info, start, end, begin, self)
        playback.frame.connect(
            lambda image, seconds, generation=self._generation: self._show(
                generation, image, seconds
            )
        )
        self._playback = playback
        playback.start()

    def stop(self) -> None:
        if self._playback is not None:
            self._generation += 1
            self._playback.stop()
            self._playback.deleteLater()
            self._playback = None

    def clear_picture(self) -> None:
        self.stop()
        self._image = None
        self.clear()
        self.setText("Select a shot to preview it")

    def _show(self, generation: int, image: QImage, seconds: float) -> None:
        if generation != self._generation:
            return
        self._image = image
        self._paint()
        self.timeChanged.emit(seconds)

    def _paint(self) -> None:
        if self._image is not None:
            pixmap = QPixmap.fromImage(self._image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._paint()
