"""The montage timeline under the player: one block per shot, as wide as the shot is
long, showing its thumbnail and the recording it comes from (colour and name).

Click a block to jump there (and pick the shot for editing), click the ruler to jump
without picking, and drag a block left or right to move the shot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from fpv_editor.app import theme

# One colour per source recording; muted so the thumbnails stay the focus.
CLIP_COLOURS = (
    "#5b8def",
    "#e0a03a",
    "#4fb39a",
    "#c76bd6",
    "#d9665b",
    "#9bb04f",
    "#5fb4d6",
    "#d68aa8",
)
RULER = 18  # px above the blocks for the time scale
BLOCKS = 58  # block height
GAP = 3  # px between blocks
DRAG_START = 5  # px the mouse moves before a press becomes a drag


# 1 to 5 stars: red, orange, yellow, light green, green.
RATING_COLOURS = ("#d9534f", "#e8873a", "#e3c341", "#9cc24f", "#3fb37f")


def rating_colour(stars: int) -> str:
    return RATING_COLOURS[min(5, max(1, stars)) - 1]


def stars_text(stars: int) -> str:
    return "★" * stars + "☆" * (5 - stars)


def clip_colour(clip: int) -> str:
    return CLIP_COLOURS[clip % len(CLIP_COLOURS)]


@dataclass(frozen=True)
class Block:
    duration: float
    clip: int  # source recording, for the colour
    label: str  # e.g. "03  low light.mp4"
    tooltip: str
    thumbnail: QPixmap | None
    colour: str | None = None  # instead of the recording's colour (e.g. the rating's)


def _tick_step(total: float, width: int) -> float:
    """A round time step that leaves at least ~70 px between ruler labels."""
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
        if total <= 0 or width * step / total >= 70:
            return float(step)
    return 1200.0


class Timeline(QWidget):
    picked = Signal(int, float)  # shot, share of it where it was clicked
    seeked = Signal(float)  # seconds into the montage (a click on the ruler)
    moved = Signal(int, int)  # shot, gap it was dropped into (0..count)

    def __init__(self) -> None:
        super().__init__()
        self._blocks: tuple[Block, ...] = ()
        self._position = 0.0
        self._selected: int | None = None
        self._palette = theme.DARK
        self._fonts = theme.Fonts("Segoe UI", "Consolas", None)
        self._press: tuple[int, float] | None = None  # block under the press, x
        self._drag_x: float | None = None
        self._hover: int | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(RULER + BLOCKS + 4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(600, RULER + BLOCKS + 4)

    def set_theme(self, palette: theme.Palette, fonts: theme.Fonts) -> None:
        self._palette, self._fonts = palette, fonts
        self.update()

    def set_blocks(self, blocks: Sequence[Block]) -> None:
        self._blocks = tuple(blocks)
        if self._selected is not None and self._selected >= len(self._blocks):
            self._selected = None
        self.update()

    def set_position(self, seconds: float) -> None:
        self._position = seconds
        self.update()

    def set_selected(self, index: int | None) -> None:
        self._selected = index
        self.update()

    # ----- geometry -----

    @property
    def total(self) -> float:
        return sum(block.duration for block in self._blocks)

    def _scale(self) -> float:
        """Pixels per second."""
        usable = self.width() - 2 - GAP * max(0, len(self._blocks) - 1)
        return usable / self.total if self.total > 0 else 0.0

    def _rects(self) -> list[QRectF]:
        scale = self._scale()
        rects: list[QRectF] = []
        x = 1.0
        for block in self._blocks:
            width = max(4.0, block.duration * scale)
            rects.append(QRectF(x, RULER, width, BLOCKS))
            x += width + GAP
        return rects

    def _x_of(self, seconds: float) -> float:
        """Where ``seconds`` into the montage is drawn (gaps included)."""
        x = 1.0
        scale = self._scale()
        left = seconds
        for block in self._blocks:
            if left <= block.duration:
                return x + left * scale
            left -= block.duration
            x += block.duration * scale + GAP
        return x - GAP

    def _seconds_at(self, x: float) -> float:
        seconds = 0.0
        for block, rect in zip(self._blocks, self._rects(), strict=True):
            if x < rect.right() + GAP / 2:
                share = min(max((x - rect.left()) / rect.width(), 0.0), 1.0)
                return seconds + share * block.duration
            seconds += block.duration
        return seconds

    def _block_at(self, x: float, y: float) -> int | None:
        if y < RULER:
            return None
        for index, rect in enumerate(self._rects()):
            if rect.left() <= x <= rect.right():
                return index
        return None

    def _gap_at(self, x: float) -> int:
        """The gap between blocks nearest to ``x``: 0 before the first, n after the last."""
        for index, rect in enumerate(self._rects()):
            if x < rect.center().x():
                return index
        return len(self._blocks)

    # ----- mouse -----

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._blocks:
            return
        x, y = event.position().x(), event.position().y()
        block = self._block_at(x, y)
        if block is None:
            self.seeked.emit(self._seconds_at(x))
        else:
            self._press = (block, x)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x, y = event.position().x(), event.position().y()
        if self._press is not None:
            if self._drag_x is not None or abs(x - self._press[1]) >= DRAG_START:
                self._drag_x = x
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.update()
            return
        hover = self._block_at(x, y)
        if hover != self._hover:
            self._hover = hover
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if hover is not None
                else Qt.CursorShape.ArrowCursor
            )
            if hover is not None:
                QToolTip.showText(
                    event.globalPosition().toPoint(), self._blocks[hover].tooltip, self
                )
            else:
                QToolTip.hideText()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press is None:
            return
        block, _ = self._press
        x = event.position().x()
        dragged = self._drag_x is not None
        self._press, self._drag_x = None, None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        if dragged:
            gap = self._gap_at(x)
            if gap not in (block, block + 1):
                self.moved.emit(block, gap)
        else:
            rect = self._rects()[block]
            self.picked.emit(block, min(max((x - rect.left()) / rect.width(), 0.0), 1.0))

    def leaveEvent(self, event: object) -> None:
        self._hover = None
        self.update()

    # ----- drawing -----

    def paintEvent(self, event: QPaintEvent) -> None:
        p = self._palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mono = theme.mono_font(self._fonts, 7.5)
        painter.setFont(mono)
        rects = self._rects()

        # Ruler.
        total = self.total
        step = _tick_step(total, self.width())
        painter.setPen(QColor(p.faint))
        tick = 0.0
        while total > 0 and tick <= total + 1e-6:
            x = self._x_of(tick)
            painter.drawLine(QPointF(x, RULER - 5), QPointF(x, RULER - 2))
            minutes, seconds = divmod(round(tick), 60)
            label = f"{minutes}:{seconds:02d}"
            if x + 30 < self.width():
                painter.drawText(QPointF(x + 3, RULER - 6), label)
            tick += step

        # Blocks.
        for index, (block, rect) in enumerate(zip(self._blocks, rects, strict=True)):
            dragging = self._press is not None and self._drag_x is not None
            faded = dragging and self._press is not None and self._press[0] == index
            self._draw_block(painter, block, rect, index, faded)

        # Drop marker while dragging.
        if self._press is not None and self._drag_x is not None:
            gap = self._gap_at(self._drag_x)
            x = rects[gap].left() - GAP / 2 if gap < len(rects) else rects[-1].right() + GAP / 2
            painter.setPen(QPen(QColor(p.accent), 2))
            painter.drawLine(QPointF(x, RULER - 2), QPointF(x, RULER + BLOCKS + 2))

        # Playhead.
        if self._blocks:
            x = self._x_of(self._position)
            painter.setPen(QPen(QColor(p.accent), 2))
            painter.drawLine(QPointF(x, RULER - 2), QPointF(x, RULER + BLOCKS + 2))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(p.accent))
            painter.drawEllipse(QPointF(x, RULER - 2), 4, 4)
        painter.end()

    def _draw_block(
        self, painter: QPainter, block: Block, rect: QRectF, index: int, faded: bool
    ) -> None:
        p = self._palette
        colour = QColor(block.colour or clip_colour(block.clip))
        painter.save()
        if faded:
            painter.setOpacity(0.35)
        clip_path_rect = rect.toRect()
        painter.setClipRect(clip_path_rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(p.surface_2))
        painter.drawRect(rect)
        if block.thumbnail is not None and not block.thumbnail.isNull():
            scaled = block.thumbnail.scaledToHeight(
                int(rect.height()), Qt.TransformationMode.SmoothTransformation
            )
            # Repeat the frame along long blocks, like a film strip.
            x = rect.left()
            while x < rect.right():
                painter.drawPixmap(int(x), int(rect.top()), scaled)
                x += max(1, scaled.width())
        shade = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        shade.setColorAt(0.0, QColor(0, 0, 0, 40))
        shade.setColorAt(1.0, QColor(0, 0, 0, 190))
        painter.fillRect(rect, shade)
        if block.colour:  # rated: the whole block takes the rating's colour
            tinted = QColor(colour)
            tinted.setAlpha(110)
            painter.fillRect(rect, tinted)
        stripe = 5 if block.colour else 3
        painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), stripe), colour)
        if rect.width() > 26:
            painter.setPen(QColor("#ffffff"))
            label_rect = QRect(
                int(rect.left()) + 6, int(rect.bottom()) - 18, int(rect.width()) - 10, 14
            )
            text = painter.fontMetrics().elidedText(
                block.label, Qt.TextElideMode.ElideRight, label_rect.width()
            )
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft, text)
        painter.setClipping(False)
        selected = index == self._selected
        hovered = index == self._hover
        if selected or hovered:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            edge = QColor(p.text if selected else p.border_strong)
            painter.setPen(QPen(edge, 2 if selected else 1))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
        painter.restore()
