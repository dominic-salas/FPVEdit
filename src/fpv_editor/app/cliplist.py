"""The clip list of step 1: one card per clip with a thumbnail, its length, when it was
recorded and how its analysis is going.

`ClipReader` probes each added clip and grabs a thumbnail in a background thread; the
cards are drawn by `ClipCard` in the app's theme.
"""

from __future__ import annotations

import queue
from pathlib import Path

from PySide6.QtCore import (
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRect,
    QSize,
    Qt,
    QThread,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from fpv_editor.app import theme
from fpv_editor.app.frames import grab_frame
from fpv_editor.errors import FPVEditError
from fpv_editor.media import Tools, probe

PATH = Qt.ItemDataRole.UserRole  # str: the clip's file
DETAILS = Qt.ItemDataRole.UserRole + 1  # str: length, rate, recording time
STATUS = Qt.ItemDataRole.UserRole + 2  # str: what the analysis says, upper case
SHARE = Qt.ItemDataRole.UserRole + 3  # float 0..1 while analysing, else None
THUMB = Qt.ItemDataRole.UserRole + 4  # QPixmap
PROBLEM = Qt.ItemDataRole.UserRole + 5  # bool: the status is a problem (shown in red)

CARD_HEIGHT = 84
THUMB_SIZE = QSize(112, 63)
COMPACT_HEIGHT = 52  # the read-only list on the progress screen
COMPACT_THUMB = QSize(64, 36)
THUMB_HEIGHT = 126  # grabbed at twice the drawn size for high-DPI screens
THUMB_AT = 0.3  # thumbnail from 30 % into the clip: past take-off, into the flight


class ClipReader(QThread):
    """Probes clips and grabs a thumbnail of each, one at a time."""

    probed = Signal(object)  # MediaInfo
    failed = Signal(object, str)  # path, why
    thumbnail = Signal(object, bytes)  # path, PNG

    def __init__(self, tools: Tools, parent: QObject) -> None:
        super().__init__(parent)
        self._tools = tools
        self._requests: queue.Queue[Path | None] = queue.Queue()

    def request(self, path: Path) -> None:
        self._requests.put(path)

    def stop(self) -> None:
        while not self._requests.empty():
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        self._requests.put(None)
        self.wait()

    def run(self) -> None:
        while (path := self._requests.get()) is not None:
            try:
                info = probe(self._tools, path)
            except FPVEditError as exc:
                self.failed.emit(path, str(exc))
                continue
            self.probed.emit(info)
            png = grab_frame(self._tools, info, THUMB_AT * (info.duration or 0.0), THUMB_HEIGHT)
            if png is not None:
                self.thumbnail.emit(path, png)


class ClipList(QListWidget):
    """The clips, in montage order: drag to reorder, drop files or folders to add."""

    dropped = Signal(list)

    def __init__(self, *, compact: bool = False) -> None:
        """``compact``: smaller cards that only show progress (no editing, no drops)."""
        super().__init__()
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.card = ClipCard(self, compact=compact)
        self.setItemDelegate(self.card)
        if compact:
            self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        else:
            self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            self.dropped.emit([Path(url.toLocalFile()) for url in event.mimeData().urls()])
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self.count() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(
                self.viewport().rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Drop recordings (.ts, .mp4) or folders here",
            )


class ClipCard(QStyledItemDelegate):
    """Draws a clip as a card: number, thumbnail, name, details, analysis status."""

    def __init__(self, parent: QObject, *, compact: bool = False) -> None:
        super().__init__(parent)
        self.compact = compact
        self.height = COMPACT_HEIGHT if compact else CARD_HEIGHT
        self.thumb = COMPACT_THUMB if compact else THUMB_SIZE
        self.palette = theme.DARK
        self.fonts = theme.Fonts("Segoe UI", "Consolas", None)

    def set_theme(self, palette: theme.Palette, fonts: theme.Fonts) -> None:
        self.palette, self.fonts = palette, fonts

    def sizeHint(
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:
        return QSize(option.rect.width(), self.height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        p = self.palette
        state = option.state
        rect: QRect = option.rect.adjusted(0, 3, -1, -4)
        selected = bool(state & QStyle.StateFlag.State_Selected)
        hovered = bool(state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = p.surface_2 if selected else p.surface if hovered else p.bg
        edge = p.border_strong if selected or hovered else p.border
        painter.setPen(QPen(QColor(edge), 1))
        painter.setBrush(QColor(fill))
        painter.drawRoundedRect(rect, 8, 8)

        # Number in playing order.
        mono = theme.mono_font(self.fonts, 8)
        painter.setFont(mono)
        painter.setPen(QColor(p.faint))
        number = QRect(rect.left() + 14, rect.top(), 28, rect.height())
        painter.drawText(number, Qt.AlignmentFlag.AlignVCenter, f"{index.row() + 1:02d}")

        # Thumbnail (or a placeholder while it loads).
        box = QRect(
            number.right() + 4,
            rect.top() + (rect.height() - self.thumb.height()) // 2,
            self.thumb.width(),
            self.thumb.height(),
        )
        pixmap = index.data(THUMB)
        painter.setPen(QPen(QColor(p.border), 1))
        painter.setBrush(QColor(p.surface_2))
        painter.drawRoundedRect(box, 4, 4)
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                box.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            source = QRect(
                (scaled.width() - box.width()) // 2,
                (scaled.height() - box.height()) // 2,
                box.width(),
                box.height(),
            )
            painter.drawPixmap(box, scaled, source)

        # Status on the right: what the analysis is doing, with a thin progress line.
        status = str(index.data(STATUS) or "")
        share = index.data(SHARE)
        status_width = 150 if self.compact else 190
        right = QRect(rect.right() - status_width - 16, rect.top(), status_width, rect.height())
        painter.setFont(mono)
        painter.setPen(QColor(p.accent if index.data(PROBLEM) else p.muted))
        painter.drawText(right, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, status)
        if isinstance(share, float):
            below = 10 if self.compact else 13
            track = QRect(right.right() - 120, right.center().y() + below, 120, 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(p.border))
            painter.drawRect(track)
            painter.setBrush(QColor(p.accent))
            painter.drawRect(QRect(track.left(), track.top(), round(120 * share), 2))

        # Name and details.
        text_left = box.right() + 18
        text = QRect(text_left, rect.top(), right.left() - text_left - 12, rect.height())
        name_font = painter.font()
        name_font.setFamily(self.fonts.sans)
        name_font.setPointSizeF(9.5 if self.compact else 10.5)
        name_font.setLetterSpacing(name_font.letterSpacingType(), 0)
        painter.setFont(name_font)
        painter.setPen(QColor(p.text))
        name = Path(str(index.data(PATH))).name
        metrics = painter.fontMetrics()
        name = metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, text.width())
        top = QRect(text.left(), text.top(), text.width(), text.height() // 2 - 2)
        painter.drawText(top, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, name)
        painter.setFont(mono)
        painter.setPen(QColor(p.faint))
        bottom = QRect(text.left(), text.center().y() + 4, text.width(), text.height() // 2)
        details = str(index.data(DETAILS) or "READING…")
        if self.compact:
            details = details.split("   ")[0]  # just the length
        painter.drawText(bottom, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, details)
        painter.restore()


def thumbnail_pixmap(png: bytes) -> QPixmap | None:
    pixmap = QPixmap()
    return pixmap if pixmap.loadFromData(png) else None
