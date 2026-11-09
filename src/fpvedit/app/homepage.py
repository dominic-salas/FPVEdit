"""The first thing okFPV shows: its three modules, each with what it does and where it stands."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from fpvedit.app.look import DotTitle

MODULES = (
    ("flowflight", 4, "Scores each flight and tracks your Freestyle and Flow."),
    ("fpvedit", 3, "Picks the best shots from your recordings and cuts one edit."),
    ("anahd", 3, "Repairs analog breakup and upscales to HD. Optional download."),
)
NAME_COLUMN = 200


class ModuleRow(QPushButton):
    def __init__(self, name: str, accent_from: int, text: str) -> None:
        super().__init__()
        self.setProperty("role", "module")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(f"{name}. {text}")
        self.setFixedHeight(76)
        self.title = DotTitle(name, cell=3, accent_from=accent_from)
        self.status = QLabel()
        self.status.setProperty("role", "status")
        what = QLabel(text)
        what.setProperty("role", "muted")
        column = QWidget()
        column.setFixedWidth(NAME_COLUMN)
        inner = QHBoxLayout(column)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignLeft)
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(16)
        row.addWidget(column)
        row.addWidget(what, 1)
        row.addWidget(self.status)
        for child in (column, self.title, what, self.status):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class HomePage(QWidget):
    """``chosen`` carries the name of the module the pilot picked."""

    chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = DotTitle("okfpv", cell=7, accent_from=2)
        intro = QLabel("Score, edit and clean your FPV flights on this PC.")
        intro.setProperty("role", "hero")
        self.rows: dict[str, ModuleRow] = {}
        content = QVBoxLayout()
        content.setSpacing(0)
        content.addWidget(self._title)
        content.addSpacing(16)
        content.addWidget(intro)
        content.addSpacing(40)
        for name, accent_from, text in MODULES:
            row = ModuleRow(name, accent_from, text)
            row.clicked.connect(lambda _=False, chosen=name: self.chosen.emit(chosen))
            self.rows[name] = row
            content.addWidget(row)
        wide = QHBoxLayout()
        wide.addLayout(content, 5)
        wide.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(72, 0, 72, 0)
        layout.addStretch(2)
        layout.addLayout(wide)
        layout.addStretch(3)

    def set_colours(self, text: str, accent: str) -> None:
        self._title.set_colour(text, accent)
        for row in self.rows.values():
            row.title.set_colour(text, accent)

    def set_status(self, name: str, status: str) -> None:
        self.rows[name].status.setText(status)
