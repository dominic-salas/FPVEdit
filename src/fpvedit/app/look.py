"""Things drawn in code, so no image or font file has to ship: the app icon, the
dot-matrix title and the status dot."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

SIZES = (16, 24, 32, 48, 64, 128, 256)


def icon_pixmap(size: int) -> QPixmap:
    """The app icon: FPV goggles with the horizon rolling over in the visor, as a flip looks
    from inside them. Small sizes keep fewer, stronger trails, which would otherwise fade."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    painter.setPen(QPen(QColor("#3d3d3d"), max(1.0, s / 64)))
    painter.setBrush(QColor("#121212"))
    edge = max(0.5, s / 128)
    painter.drawRoundedRect(QRectF(edge, edge, s - 2 * edge, s - 2 * edge), 0.2 * s, 0.2 * s)
    goggles = QPainterPath()
    goggles.addRoundedRect(QRectF(0.1 * s, 0.26 * s, 0.8 * s, 0.5 * s), 0.14 * s, 0.14 * s)
    nose = QPainterPath()
    nose.addEllipse(QPointF(0.5 * s, 0.8 * s), 0.11 * s, 0.14 * s)
    goggles = goggles.subtracted(nose)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#1f1f1f"))
    painter.drawPath(goggles)
    painter.save()
    painter.setClipPath(goggles)
    small = size < 64
    trails = (
        ((-34, 120, 0.05), (-12, 190, 0.05))
        if small
        else (
            (-58, 50, 0.03),
            (-36, 95, 0.03),
            (-14, 155, 0.03),
        )
    )
    centre = QPointF(s / 2, 0.5 * s)
    for angle, alpha, width in (*trails, (10, 255, 0.08 if small else 0.05)):
        colour = QColor("#f2a93b")
        colour.setAlpha(alpha)
        painter.setPen(QPen(colour, width * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        dx, dy = math.cos(math.radians(angle)) * s, math.sin(math.radians(angle)) * s
        painter.drawLine(centre - QPointF(dx, dy), centre + QPointF(dx, dy))
    painter.restore()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor("#ededed"), max(1.0, (0.05 if small else 0.035) * s)))
    painter.drawPath(goggles)
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    icon = QIcon()
    for size in SIZES:
        icon.addPixmap(icon_pixmap(size))
    return icon


# A 5-dot-wide dot-matrix face for the title, after the Doto face on dvsalas.com. Drawn
# as dots, so it needs no font file. Rows 0-1 hold ascenders, rows 2-6 the x-height (the
# baseline is row 6) and rows 7-8 descenders, so every lowercase letter lines up. Letters
# not listed are left blank.
X_TOP, BASELINE, ROWS = 2, 6, 9
GLYPHS: dict[str, tuple[str, ...]] = {
    "a": (".....", ".....", ".###.", "....#", ".####", "#...#", ".####", ".....", "....."),
    "d": ("....#", "....#", ".####", "#...#", "#...#", "#...#", ".####", ".....", "....."),
    "e": (".....", ".....", ".###.", "#...#", "#####", "#....", ".###.", ".....", "....."),
    "f": ("..##.", ".#...", "####.", ".#...", ".#...", ".#...", ".#...", ".....", "....."),
    "g": (".....", ".....", ".####", "#...#", "#...#", "#...#", ".####", "....#", ".###."),
    "h": ("#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#", ".....", "....."),
    "i": ("..#..", ".....", ".##..", "..#..", "..#..", "..#..", ".###.", ".....", "....."),
    "k": ("#....", "#....", "#..#.", "#.#..", "##...", "#.#..", "#..#.", ".....", "....."),
    "l": (".##..", "..#..", "..#..", "..#..", "..#..", "..#..", ".###.", ".....", "....."),
    "n": (".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#", ".....", "....."),
    "o": (".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###.", ".....", "....."),
    "p": (".....", ".....", "####.", "#...#", "#...#", "#...#", "####.", "#....", "#...."),
    "r": (".....", ".....", "#.##.", "##..#", "#....", "#....", "#....", ".....", "....."),
    "t": (".#...", ".#...", "####.", ".#...", ".#...", ".#..#", "..##.", ".....", "....."),
    "v": (".....", ".....", "#...#", "#...#", "#...#", ".#.#.", "..#..", ".....", "....."),
    "w": (".....", ".....", "#...#", "#...#", "#.#.#", "#.#.#", ".#.#.", ".....", "....."),
}


class DotTitle(QWidget):
    """Lowercase text drawn as a grid of dots, in the accent colour from ``accent_from`` on."""

    def __init__(
        self,
        text: str,
        cell: int = 7,
        parent: QWidget | None = None,
        accent_from: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._text = text.lower()
        self._cell = cell
        self._colour = QColor("#ededed")
        self._accent = self._colour
        self._accent_from = len(self._text) if accent_from is None else accent_from
        columns = sum(6 for _ in self._text) - 1
        self.setFixedSize(columns * cell, ROWS * cell)

    def set_colour(self, colour: str, accent: str | None = None) -> None:
        self._colour = QColor(colour)
        self._accent = QColor(accent or colour)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = self._cell * 0.36
        for index, letter in enumerate(self._text):
            rows = GLYPHS.get(letter)
            if rows is None:
                continue
            painter.setBrush(self._accent if index >= self._accent_from else self._colour)
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
        self._idle = QColor("#8c8c8c")
        self._accent = QColor("#f2a93b")
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
