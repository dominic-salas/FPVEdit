"""The Pilot page: Freestyle and Flow over time, and a scorecard for every flight."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fpvedit.app.widgets import button, plural
from fpvedit.history import Flight, FlightHistory, Summary, insight, summarise

CONDITIONS = (("All flights", None), ("Daylight", "daylight"), ("Low light", "low light"))
LABEL_WIDTH = 96


def _label(text: str = "", role: str = "eyebrow") -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", role)
    return widget


class TrendChart(QWidget):
    """Freestyle and Flow levels, flight by flight, on one 0-100 scale."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._summary: Summary | None = None
        self.setMinimumHeight(150)

    def show_summary(self, summary: Summary | None) -> None:
        self._summary = summary
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        summary = self._summary
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        faint = palette.color(QPalette.ColorRole.PlaceholderText)
        area = QRectF(self.rect()).adjusted(30, 22, -12, -22)
        for value in (0, 50, 100):
            y = area.bottom() - area.height() * value / 100
            painter.setPen(QPen(palette.color(QPalette.ColorRole.AlternateBase), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(faint)
            painter.drawText(QRectF(0, y - 8, 24, 16), Qt.AlignmentFlag.AlignRight, str(value))
        if summary is None:
            painter.end()
            return
        count = len(summary.freestyle)
        step = area.width() / max(1, count - 1)
        if summary.baseline < count:
            x = area.left() + step * (summary.baseline - 1)
            painter.setPen(QPen(faint, 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.drawText(
                QRectF(x + 6, area.top() - 18, 160, 16),
                Qt.AlignmentFlag.AlignLeft,
                "AFTER FIRST IMPORT",
            )
        accent = palette.color(QPalette.ColorRole.Link)
        text = palette.color(QPalette.ColorRole.Text)
        self._line(painter, summary.freestyle, area, step, accent)
        self._line(painter, summary.flow, area, step, text)
        painter.setPen(accent)
        painter.drawText(QRectF(area.left(), self.height() - 18, 90, 16), 0, "FREESTYLE")
        painter.setPen(text)
        painter.drawText(QRectF(area.left() + 92, self.height() - 18, 90, 16), 0, "FLOW")
        painter.end()

    @staticmethod
    def _line(
        painter: QPainter, values: Sequence[float], area: QRectF, step: float, colour: QColor
    ) -> None:
        points = [
            QPointF(area.left() + step * i, area.bottom() - area.height() * value / 100)
            for i, value in enumerate(values)
        ]
        painter.setPen(QPen(colour, 2))
        if len(points) == 1:
            painter.setBrush(colour)
            painter.drawEllipse(points[0], 3, 3)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            painter.drawPolyline(QPolygonF(points))


class PartBar(QWidget):
    """One named part of a score, 0-100."""

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self._value = 0.0
        self.setFixedHeight(22)

    def show_value(self, value: float) -> None:
        self._value = value
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        palette = self.palette()
        painter.setPen(palette.color(QPalette.ColorRole.Text))
        height = self.height()
        painter.drawText(
            QRectF(0, 0, LABEL_WIDTH, height), Qt.AlignmentFlag.AlignVCenter, self._name
        )
        right = self.width() - 34
        width = max(0, right - LABEL_WIDTH)
        track = QRectF(LABEL_WIDTH, height / 2 - 2, width, 4)
        painter.fillRect(track, palette.color(QPalette.ColorRole.AlternateBase))
        painter.fillRect(
            QRectF(track.left(), track.top(), width * self._value / 100, 4),
            palette.color(QPalette.ColorRole.Link),
        )
        painter.drawText(
            QRectF(right, 0, 34, height),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            f"{self._value:.0f}",
        )
        painter.end()


class PilotPage(QWidget):
    """``back`` when the user leaves."""

    back = Signal()

    def __init__(self, history: FlightHistory, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history = history
        self._flights: list[Flight] = []
        self._shown: list[Flight] = []
        self._current: Flight | None = None

        self._filter = QComboBox()
        for text, value in CONDITIONS:
            self._filter.addItem(text, value)
        self._filter.currentIndexChanged.connect(lambda _: self._apply_filter())
        top = QHBoxLayout()
        top.addWidget(button("←  CLIPS", self.back.emit))
        top.addStretch(1)
        top.addWidget(self._filter)

        stats = QHBoxLayout()
        stats.setSpacing(32)
        self._stats: dict[str, QLabel] = {}
        self._changes: dict[str, QLabel] = {}
        for key, title in (
            ("freestyle", "FREESTYLE"),
            ("flow", "FLOW"),
            ("best", "BEST FREESTYLE / FLOW"),
            ("flights", "FLIGHTS COUNTED"),
        ):
            column = QVBoxLayout()
            column.setSpacing(2)
            column.addWidget(_label(title))
            self._stats[key] = _label("-", "stat")
            column.addWidget(self._stats[key])
            self._changes[key] = _label("", "hint")
            column.addWidget(self._changes[key])
            stats.addLayout(column)
        stats.addStretch(1)

        self._chart = TrendChart()
        self._insight = _label("", "hint")
        self._insight.setWordWrap(True)
        note = _label(
            "An estimate from the motion in your recordings, compared only with your own "
            "flights. It does not name tricks.",
            "hint",
        )
        note.setWordWrap(True)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._select)

        self._card_title = _label("", "eyebrow")
        self._card_title.setWordWrap(True)
        self._card_scores = _label("", "stat")
        self._reason = _label("", "hint")
        self._reason.setWordWrap(True)
        self._bars: dict[str, PartBar] = {
            key: PartBar(name)
            for key, name in (
                ("difficulty", "Difficulty"),
                ("variety", "Variety"),
                ("linking", "Linking"),
                ("continuity", "Continuity"),
                ("smoothness", "Smoothness"),
            )
        }
        self._facts = _label("", "hint")
        self._facts.setWordWrap(True)
        self._anyway = QPushButton("COUNT IT ANYWAY")
        self._anyway.clicked.connect(self._toggle_anyway)
        self._anyway.hide()

        card = QFrame()
        card.setFixedWidth(340)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 0, 0, 0)
        card_layout.setSpacing(8)
        card_layout.addWidget(self._card_title)
        card_layout.addWidget(self._card_scores)
        card_layout.addWidget(self._reason)
        for bar in self._bars.values():
            card_layout.addWidget(bar)
        card_layout.addWidget(self._facts)
        card_layout.addWidget(self._anyway)
        card_layout.addStretch(1)

        lower = QHBoxLayout()
        lower.addWidget(self._list, 1)
        lower.addWidget(card)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)
        layout.addLayout(top)
        layout.addLayout(stats)
        layout.addWidget(self._chart, 1)
        layout.addWidget(self._insight)
        layout.addLayout(lower, 1)
        layout.addWidget(note)

    def refresh(self) -> None:
        """Read the saved flights again."""
        self._flights = self._history.flights()
        self._apply_filter()

    def _apply_filter(self) -> None:
        wanted = self._filter.currentData()
        self._shown = [f for f in self._flights if wanted is None or f.conditions == wanted]
        summary = summarise(self._shown)
        self._chart.show_summary(summary)
        self._insight.setText(
            insight(self._shown) if summary else "Find shots in some clips to score your flights."
        )
        self._show_stats(summary)
        self._list.blockSignals(True)
        self._list.clear()
        for flight in reversed(self._shown):
            self._list.addItem(QListWidgetItem(self._row_text(flight)))
        self._list.blockSignals(False)
        if self._shown:
            newest_first = list(reversed(self._shown))
            scored = (i for i, f in enumerate(newest_first) if f.status == "scored")
            self._list.setCurrentRow(next(scored, 0))
        else:
            self._clear_card()

    def _show_stats(self, summary: Summary | None) -> None:
        if summary is None:
            for label in self._stats.values():
                label.setText("-")
            for label in self._changes.values():
                label.setText("")
            return
        self._stats["freestyle"].setText(f"{summary.freestyle[-1]:.0f}")
        self._stats["flow"].setText(f"{summary.flow[-1]:.0f}")
        self._stats["best"].setText(f"{summary.best_freestyle:.0f} / {summary.best_flow:.0f}")
        self._stats["flights"].setText(str(summary.flights))
        beyond = summary.flights > summary.baseline
        for key, level, start in (
            ("freestyle", summary.freestyle[-1], summary.start_freestyle),
            ("flow", summary.flow[-1], summary.start_flow),
        ):
            self._changes[key].setText(
                f"{level - start:+.0f} since your first import" if beyond else "starting level"
            )

    @staticmethod
    def _row_text(flight: Flight) -> str:
        if flight.status != "scored":
            return f"{flight.name}   NOT SCORED: {flight.status}"
        scores = f"{flight.freestyle:.0f} / {flight.flow:.0f}"
        if flight.disqualified:
            mark = "COUNTED BY YOU" if flight.count_anyway else "DISQUALIFIED"
            return f"{flight.name}   {scores}   {mark}"
        return f"{flight.name}   {scores}"

    def _select(self, row: int) -> None:
        if not 0 <= row < len(self._shown):
            self._clear_card()
            return
        flight = list(reversed(self._shown))[row]
        self._current = flight
        self._card_title.setText(flight.name)
        scored = flight.status == "scored"
        self._card_scores.setText(f"{flight.freestyle:.0f} / {flight.flow:.0f}" if scored else "-")
        if flight.disqualified:
            self._reason.setText(f"Disqualified: {flight.disqualified}")
        elif scored:
            self._reason.setText("")
        else:
            self._reason.setText(f"Not scored: {flight.status}.")
        for key, bar in self._bars.items():
            bar.show_value(float(getattr(flight, key)))
            bar.setVisible(scored)
        minutes, seconds = divmod(round(flight.flying_s), 60)
        self._facts.setText(
            f"{plural(flight.turns, 'turn')}, {flight.crashes} crash{'' if flight.crashes == 1 else 'es'}, "
            f"{minutes}:{seconds:02d} of flying, "
            f"{flight.conditions}."
            if scored
            else f"{minutes}:{seconds:02d} of flying, {flight.conditions}."
        )
        self._anyway.setVisible(bool(flight.disqualified))
        self._anyway.setText("STOP COUNTING IT" if flight.count_anyway else "COUNT IT ANYWAY")

    def _clear_card(self) -> None:
        self._current = None
        self._card_title.setText("")
        self._card_scores.setText("")
        self._reason.setText("")
        self._facts.setText("")
        self._anyway.hide()
        for bar in self._bars.values():
            bar.hide()

    def _toggle_anyway(self) -> None:
        if self._current is not None:
            self._history.count_anyway(self._current.id, not self._current.count_anyway)
            self.refresh()
