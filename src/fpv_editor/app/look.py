"""Things drawn in code, so no image or font file has to ship: the app icon, the
dot-matrix title and the status dot."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

SIZES = (16, 24, 32, 48, 64, 128, 256)


def icon_pixmap(size: int) -> QPixmap:
    """A rounded tile with a play mark and four rotor rings: a montage of quad footage."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    gradient = QLinearGradient(0, 0, s, s)
    gradient.setColorAt(0.0, QColor("#ff8a00"))
    gradient.setColorAt(1.0, QColor("#e5232e"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawRoundedRect(QRectF(0.04 * s, 0.04 * s, 0.92 * s, 0.92 * s), 0.2 * s, 0.2 * s)
    if size >= 32:  # rotor rings only where they can be seen
        ring = QColor(255, 255, 255, 110)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = painter.pen()
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setColor(ring)
        pen.setWidthF(max(1.0, 0.03 * s))
        painter.setPen(pen)
        for cx, cy in ((0.27, 0.27), (0.73, 0.27), (0.27, 0.73), (0.73, 0.73)):
            painter.drawEllipse(QPointF(cx * s, cy * s), 0.15 * s, 0.15 * s)
        painter.setPen(Qt.PenStyle.NoPen)
    play = QPainterPath()
    play.moveTo(0.40 * s, 0.30 * s)
    play.lineTo(0.72 * s, 0.50 * s)
    play.lineTo(0.40 * s, 0.70 * s)
    play.closeSubpath()
    painter.setBrush(QColor("white"))
    painter.drawPath(play)
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    icon = QIcon()
    for size in SIZES:
        icon.addPixmap(icon_pixmap(size))
    return icon


# A 5x7 dot-matrix face for the title, after the Doto face on dvsalas.com. Drawn as dots,
# so it needs no font file. Letters not listed are left blank.
GLYPHS: dict[str, tuple[str, ...]] = {
    "f": ("..##.", ".#...", "####.", ".#...", ".#...", ".#...", ".#..."),
    "p": (".....", "####.", "#...#", "#...#", "####.", "#....", "#...."),
    "v": (".....", ".....", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "e": (".....", ".###.", "#...#", "#####", "#....", "#...#", ".###."),
    "d": ("....#", "....#", ".####", "#...#", "#...#", "#...#", ".####"),
    "i": ("..#..", ".....", ".##..", "..#..", "..#..", "..#..", ".###."),
    "t": (".#...", ".#...", "####.", ".#...", ".#...", ".#..#", "..##."),
    "o": (".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###."),
    "r": (".....", ".....", "#.##.", "##..#", "#....", "#....", "#...."),
}


class DotTitle(QWidget):
    """Large lowercase text drawn as a grid of dots."""

    def __init__(self, text: str, cell: int = 7, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text.lower()
        self._cell = cell
        self._colour = QColor("#ededed")
        columns = sum(6 for _ in self._text) - 1
        self.setFixedSize(columns * cell, 7 * cell)

    def set_colour(self, colour: str) -> None:
        self._colour = QColor(colour)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._colour)
        radius = self._cell * 0.36
        for index, letter in enumerate(self._text):
            rows = GLYPHS.get(letter)
            if rows is None:
                continue
            for y, row in enumerate(rows):
                for x, bit in enumerate(row):
                    if bit == "#":
                        centre = QPointF((index * 6 + x + 0.5) * self._cell, (y + 0.5) * self._cell)
                        painter.drawEllipse(centre, radius, radius)
        painter.end()


class StatusDot(QWidget):
    """A small dot: still and grey when idle, pulsing in the accent colour while busy."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._idle = QColor("#6e6e6e")
        self._accent = QColor("#e5322d")
        self._busy = False
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

    def set_colours(self, idle: str, accent: str) -> None:
        self._idle, self._accent = QColor(idle), QColor(accent)
        self.update()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 1) % 30
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        colour = QColor(self._accent if self._busy else self._idle)
        if self._busy:
            colour.setAlphaF(0.35 + 0.65 * abs(15 - self._phase) / 15)
        painter.setBrush(colour)
        painter.drawEllipse(QPointF(5, 5), 3.5, 3.5)
        painter.end()
