"""The review player: plays the montage shot after shot, or one shot on a loop, with
frames decoded by FFmpeg and shown in a label.

Qt's own media player needs the large PySide6-Addons package and its own decoder; FFmpeg
is already required and reads every recording the render reads. While a shot plays, the
FFmpeg process for the next one is already started (it seeks and waits on its pipe), so
one shot follows the next without a pause.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from fpv_editor.app.frames import preview_args, preview_size
from fpv_editor.media import NO_WINDOW, MediaInfo, Tools

PREVIEW_HEIGHT = 360
PREVIEW_FPS = 30


@dataclass(frozen=True)
class Segment:
    """``start``-``end`` (seconds on the analysis timeline) of one recording."""

    info: MediaInfo
    start: float
    end: float


class _Playback(QThread):
    """Plays segments in order from ``first`` seconds into segment ``index``, at real
    speed; with ``loop`` it starts over after the last, otherwise it ends there."""

    frame = Signal(QImage, int, float)  # picture, segment, its time in that recording
    ended = Signal()

    def __init__(
        self,
        tools: Tools,
        segments: Sequence[Segment],
        index: int,
        first: float,
        loop: bool,
        height: int,
        parent: QObject,
    ) -> None:
        super().__init__(parent)
        self._tools = tools
        self._height = height
        self._segments = tuple(segments)
        self._index, self._first, self._loop = index, first, loop
        self._stop = threading.Event()
        self._processes: list[subprocess.Popen[bytes]] = []
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for process in self._processes:
                if process.poll() is None:
                    process.kill()
        self.wait()

    def _open(self, segment: Segment, begin: float) -> subprocess.Popen[bytes] | None:
        size = preview_size(segment.info, self._height)
        args = preview_args(self._tools, segment.info, begin, segment.end, size, PREVIEW_FPS)
        with self._lock:
            if self._stop.is_set():
                return None
            try:
                process = subprocess.Popen(
                    args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW
                )
            except OSError:
                return None
            self._processes.append(process)
            return process

    def _close(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        if process.stdout is not None:
            process.stdout.close()
        process.wait()
        with self._lock:
            if process in self._processes:
                self._processes.remove(process)

    def _following(self, index: int) -> int | None:
        if index + 1 < len(self._segments):
            return index + 1
        return 0 if self._loop else None

    def run(self) -> None:
        index, begin = self._index, self._first
        process = self._open(self._segments[index], begin)
        empty_in_a_row = 0
        try:
            while process is not None and not self._stop.is_set():
                segment = self._segments[index]
                following = self._following(index)
                upcoming = (
                    self._open(self._segments[following], self._segments[following].start)
                    if following is not None
                    else None
                )
                shown = self._show(process, index, segment, begin)
                self._close(process)
                if self._stop.is_set():
                    self._close(upcoming)
                    return
                empty_in_a_row = 0 if shown else empty_in_a_row + 1
                if following is None or empty_in_a_row >= len(self._segments):
                    self._close(upcoming)
                    break  # the end, or nothing decodable anywhere: do not spin
                index, begin, process = following, self._segments[following].start, upcoming
        finally:
            with self._lock:
                left = list(self._processes)
            for leftover in left:
                self._close(leftover)
        if not self._stop.is_set():
            self.ended.emit()

    def _show(
        self, process: subprocess.Popen[bytes], index: int, segment: Segment, begin: float
    ) -> int:
        """Read ``process``'s frames at real speed; returns how many were shown."""
        width, height = preview_size(segment.info, self._height)
        frame_bytes = width * height * 3
        if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
            return 0
        shown = 0
        clock = time.monotonic()
        while not self._stop.is_set():
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            image = QImage(data, width, height, 3 * width, QImage.Format.Format_RGB888)
            self.frame.emit(image.copy(), index, begin + shown / PREVIEW_FPS)
            shown += 1
            delay = clock + shown / PREVIEW_FPS - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
        return shown


class PreviewView(QLabel):
    """Shows the playing picture, scaled to fit.

    ``position`` reports (segment, seconds in its recording) for every frame shown;
    ``ended`` fires when a sequence played to its end.
    """

    position = Signal(int, float)
    ended = Signal()
    doubleClicked = Signal()

    def __init__(self, tools: Tools, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tools = tools
        self._playback: _Playback | None = None
        self._image: QImage | None = None
        self._generation = 0  # frames of an older playback can still be queued
        self.height_px = PREVIEW_HEIGHT  # decoded picture height (more in full screen)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: black; color: grey;")

    @property
    def playing(self) -> bool:
        return self._playback is not None

    def play(
        self,
        segments: Sequence[Segment],
        index: int = 0,
        first: float | None = None,
        *,
        loop: bool = False,
    ) -> None:
        """Play ``segments`` from ``first`` seconds (in its recording) into ``index``."""
        self.stop()
        if not segments:
            return
        segment = segments[index]
        begin = segment.start if first is None else min(max(first, segment.start), segment.end)
        self._generation += 1
        playback = _Playback(self._tools, segments, index, begin, loop, self.height_px, self)
        generation = self._generation
        playback.frame.connect(
            lambda image, at, seconds: self._show(generation, image, at, seconds)
        )
        playback.ended.connect(lambda: self._ended(generation))
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

    def _show(self, generation: int, image: QImage, index: int, seconds: float) -> None:
        if generation != self._generation:
            return
        self._image = image
        self._paint()
        self.position.emit(index, seconds)

    def _ended(self, generation: int) -> None:
        if generation == self._generation:
            self.stop()
            self.ended.emit()

    def _paint(self) -> None:
        if self._image is not None:
            pixmap = QPixmap.fromImage(self._image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(pixmap)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.doubleClicked.emit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._paint()
