"""The app icon, drawn in code so no image file has to ship or be loaded."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPixmap

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
