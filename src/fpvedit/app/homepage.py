"""okFPV's modules: three full-height panels on Home, and the same three as small tabs elsewhere."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import pairwise

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QEnterEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fpvedit.app.look import DotTitle
from fpvedit.app.theme import DARK, Palette

WIDE_PANEL = 620
TAB_SIZE = (52, 44)
TAB_ICON = 22


@dataclass(frozen=True)
class Module:
    name: str
    accent_from: int
    label: str
    heading: str
    text: str
    on_dark: str
    on_light: str

    def colour(self, palette: Palette) -> str:
        return self.on_dark if palette.name == DARK.name else self.on_light


FLOWFLIGHT = Module(
    "flowflight",
    4,
    "FlowFlight",
    "Score your flights",
    "Track Freestyle and Flow over time.",
    "#F5A524",
    "#A86500",
)
FPVEDIT = Module(
    "fpvedit",
    3,
    "FPVEdit",
    "Cut your best shots",
    "Finds the best moments and makes one edit.",
    "#4FD6A2",
    "#0A7D55",
)
ANAHD = Module(
    "anahd",
    3,
    "AnaHD",
    "Clean analog footage",
    "Repairs breakup and upscales to HD.",
    "#F2559B",
    "#BE1F6E",
)
MODULES = (FLOWFLIGHT, FPVEDIT, ANAHD)


def _tinted(colour: str, alpha: float) -> QColor:
    tint = QColor(colour)
    tint.setAlphaF(alpha)
    return tint


def _draw_flow(painter: QPainter, area: QRectF, colour: str, small: bool) -> None:
    side = area.width()
    points = [(0.04, 0.82), (0.24, 0.62), (0.42, 0.70), (0.64, 0.34), (0.94, 0.14)]

    def at(x: float, y: float) -> QPointF:
        return QPointF(area.x() + x * side, area.y() + y * side)

    painter.setPen(Qt.PenStyle.NoPen)
    if not small:
        painter.setBrush(_tinted(colour, 0.25))
        for index, height in enumerate((0.2, 0.34, 0.26, 0.5, 0.68)):
            left = area.x() + (0.02 + index * 0.2) * side
            bar = QRectF(left, area.bottom() - height * side, side * 0.14, height * side)
            painter.drawRoundedRect(bar, 2, 2)
    path = QPainterPath(at(*points[0]))
    for start, end in pairwise(points):
        middle = (start[0] + end[0]) / 2
        path.cubicTo(at(middle, start[1]), at(middle, end[1]), at(*end))
    pen = QPen(QColor(colour), max(2.0, side * (0.09 if small else 0.05)))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    if not small:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(colour))
        for point in points[1:]:
            painter.drawEllipse(at(*point), side * 0.055, side * 0.055)


def _draw_edit(painter: QPainter, area: QRectF, colour: str, playhead: str, small: bool) -> None:
    side = area.width()
    rows = [
        [(0.0, 0.34), (0.40, 0.22), (0.68, 0.32)],
        [(0.12, 0.26), (0.44, 0.40)],
        [(0.0, 0.18), (0.24, 0.30), (0.60, 0.40)],
    ]
    painter.setPen(Qt.PenStyle.NoPen)
    for y, row in enumerate(rows):
        for index, (x, width) in enumerate(row):
            chosen = y == 1 and index == 1
            painter.setBrush(_tinted(colour, 1.0 if chosen else 0.4 + 0.14 * index))
            clip = QRectF(
                area.x() + x * side, area.y() + (0.16 + y * 0.26) * side, width * side, 0.18 * side
            )
            painter.drawRoundedRect(clip, 2 if small else 3, 2 if small else 3)
    painter.setPen(QPen(QColor(playhead), max(1.5, side * 0.025)))
    x = area.x() + 0.62 * side
    painter.drawLine(QPointF(x, area.y() + 0.06 * side), QPointF(x, area.y() + 0.94 * side))


def _draw_analog(painter: QPainter, area: QRectF, colour: str, grain: str, small: bool) -> None:
    rows = 6 if small else 12
    cell = area.width() / rows
    noise = random.Random(7)  # noqa: S311 - the same grain every time, not security
    painter.setPen(Qt.PenStyle.NoPen)
    for column in range(rows):
        clean = column / (rows - 1)
        for row in range(rows):
            jitter = (1 - clean) * cell * 0.4
            x = area.x() + (column + 0.5) * cell + noise.uniform(-1, 1) * jitter
            y = area.y() + (row + 0.5) * cell + noise.uniform(-1, 1) * jitter
            if clean < 0.45 and noise.random() < 0.35:
                continue
            tone = colour if clean > 0.6 or noise.random() < 0.5 + clean else grain
            painter.setBrush(_tinted(tone, 0.4 + 0.6 * clean))
            radius = cell * (0.16 + 0.3 * clean)
            painter.drawEllipse(QPointF(x, y), radius, radius)


def draw_module_art(
    painter: QPainter, module: Module, area: QRectF, palette: Palette, small: bool = False
) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    colour = module.colour(palette)
    if module is FLOWFLIGHT:
        _draw_flow(painter, area, colour, small)
    elif module is FPVEDIT:
        _draw_edit(painter, area, colour, palette.text, small)
    else:
        _draw_analog(painter, area, colour, palette.muted, small)
    painter.restore()


class ModuleArt(QWidget):
    """The module's picture at the left of whatever room the panel gives it."""

    def __init__(self, module: Module) -> None:
        super().__init__()
        self._module = module
        self._palette = DARK

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        side = min(self.width(), self.height())
        top = (self.height() - side) / 2
        painter = QPainter(self)
        draw_module_art(painter, self._module, QRectF(0, top, side, side), self._palette)
        painter.end()


class ModulePanel(QPushButton):
    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module
        self._palette = DARK
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(f"{module.label}. {module.text}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.title = DotTitle(module.name, cell=4, accent_from=module.accent_from)
        self._art = ModuleArt(module)
        heading = QLabel(module.heading)
        heading.setProperty("role", "heading")
        heading.setWordWrap(True)
        what = QLabel(module.text)
        what.setProperty("role", "hero")
        what.setWordWrap(True)
        self.status = QLabel()
        self.status.setProperty("role", "status")
        self._column = QVBoxLayout(self)
        self._column.setSpacing(0)
        self._column.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignLeft)
        self._column.addSpacing(24)
        self._column.addWidget(self._art, 1)
        self._column.addSpacing(24)
        self._column.addWidget(heading)
        self._column.addSpacing(6)
        self._column.addWidget(what)
        self._column.addSpacing(18)
        self._column.addWidget(self.status)
        for child in (self.title, self._art, heading, what, self.status):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._set_room(False)

    def _set_room(self, wide: bool) -> None:
        margin = 72 if wide else 44
        self._column.setContentsMargins(margin, margin, margin, margin)
        self.title.set_cell(6 if wide else 4)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        colour = self.module.colour(palette)
        self.title.set_colour(palette.text, colour)
        self._art.set_palette(palette)
        self.status.setStyleSheet(f"color: {colour};")
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._set_room(event.size().width() >= WIDE_PANEL)
        super().resizeEvent(event)

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: object) -> None:
        self._hovered = False
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        active = self._hovered or self.hasFocus()
        colour = self.module.colour(self._palette)
        painter = QPainter(self)
        painter.fillRect(self.rect(), _tinted(colour, 0.16 if active else 0.09))
        painter.fillRect(0, 0, self.width(), 6 if active else 3, QColor(colour))
        painter.end()


class ModuleTab(QPushButton):
    """A module once it has shrunk into the top strip: its icon and colour."""

    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module
        self._palette = DARK
        self._hovered = False
        self.setFixedSize(*TAB_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(module.label)
        self.setAccessibleName(module.label)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.update()

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: object) -> None:
        self._hovered = False
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        on = self.property("state") == "on"
        colour = self.module.colour(self._palette)
        painter = QPainter(self)
        if on or self._hovered or self.hasFocus():
            painter.fillRect(self.rect(), _tinted(colour, 0.16 if on else 0.09))
        if on:
            painter.fillRect(0, 0, self.width(), 3, QColor(colour))
        area = QRectF(
            (self.width() - TAB_ICON) / 2, (self.height() - TAB_ICON) / 2, TAB_ICON, TAB_ICON
        )
        if not on:
            painter.setOpacity(0.85)
        draw_module_art(painter, self.module, area, self._palette, small=True)
        painter.end()


class SandwichButton(QPushButton):
    """Three lines that open the menu."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(40, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Menu")
        self.setAccessibleName("Menu")
        self._colour = QColor(DARK.muted)

    def set_colour(self, colour: str) -> None:
        self._colour = QColor(colour)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self._colour, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        left, right = self.width() / 2 - 9, self.width() / 2 + 9
        for offset in (-6, 0, 6):
            y = self.height() / 2 + offset
            painter.drawLine(QPointF(left, y), QPointF(right, y))
        painter.end()


class HomePage(QWidget):
    """``chosen`` carries the name of the module the pilot picked."""

    chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows: dict[str, ModulePanel] = {}
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)
        for module in MODULES:
            panel = ModulePanel(module)
            panel.clicked.connect(lambda _=False, chosen=module.name: self.chosen.emit(chosen))
            self.rows[module.name] = panel
            row.addWidget(panel, 1)

    def set_palette(self, palette: Palette) -> None:
        for panel in self.rows.values():
            panel.set_palette(palette)

    def set_status(self, name: str, status: str) -> None:
        self.rows[name].status.setText(status)
