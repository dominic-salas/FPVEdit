"""Rate shots: one shot plays on a loop, press 1-5, the next one comes. No montage.

Shots come from the clips on step 1 as soon as each one is analysed: every distinct shot
of a clip, the clips taken in turn and each clip's shots in a shuffled order, so a few
minutes of rating cover strong and weak shots of every clip. Shots already rated are
skipped. The app's own stars are not shown here, so they do not sway the rating.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from fpvedit.app.myratings import MyRatings
from fpvedit.app.preview import PreviewView, Segment
from fpvedit.app.timeline import rating_colour
from fpvedit.media import MediaInfo, Tools
from fpvedit.montage import Shot

GOAL = 150  # ratings for learning a pilot's taste from what the shots show
MIN_WATCH_S = 1.0  # a rating within this long of a shot appearing is ignored
WORDS = {1: "BORING", 2: "MEH", 3: "OK", 4: "GOOD", 5: "SICK"}


def interleave(per_clip: list[list[Shot]], seed: int = 0) -> list[Shot]:
    """Each clip's shots shuffled, then one from each clip in turn."""
    rng = random.Random(seed)  # noqa: S311 - an order to watch shots in, not security
    queues = [rng.sample(shots, len(shots)) for shots in per_clip]
    order: list[Shot] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop())
    return order


def _label(text: str = "", role: str = "eyebrow") -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", role)
    return widget


def _button(text: str, slot: Callable[[], object]) -> QPushButton:
    widget = QPushButton(text)
    widget.clicked.connect(slot)
    return widget


class RatePage(QWidget):
    """``back`` when the user leaves."""

    back = Signal()

    def __init__(self, tools: Tools, mine: MyRatings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mine = mine
        self._queue: list[tuple[MediaInfo, Shot]] = []
        self._seen: set[str] = set()  # shots queued or skipped this session
        self._current: tuple[MediaInfo, Shot] | None = None
        self._last: tuple[MediaInfo, Shot] | None = None  # the shot rated last (for undo)
        self._shown_at = 0.0  # when the shot on screen started (time.monotonic)
        self._waiting = False  # clips are still being analysed

        self._video = PreviewView(tools)
        self._progress = _label(role="hero")
        self._about = _label()
        self._about.setWordWrap(True)
        stars = QHBoxLayout()
        stars.setSpacing(10)
        for number in range(1, 6):
            star = QPushButton(f"{number}\n{'★' * number}\n{WORDS[number]}")
            star.setMinimumHeight(92)
            star.setStyleSheet(
                f"QPushButton {{ color: {rating_colour(number)}; border-color: "
                f"{rating_colour(number)}; font-size: 11pt; }}"
            )
            star.clicked.connect(functools.partial(self.rate, number))
            stars.addWidget(star, 1)
        controls = QHBoxLayout()
        controls.addWidget(_button("←  BACK", self.back.emit))
        controls.addStretch(1)
        controls.addWidget(_label("PRESS 1-5 TO RATE   S TO SKIP   BACKSPACE TO UNDO"))
        controls.addStretch(1)
        self._undo_button = _button("↶  UNDO LAST", self.undo)
        controls.addWidget(self._undo_button)
        controls.addWidget(_button("SKIP  →", self.skip))

        column = QVBoxLayout(self)
        column.setContentsMargins(40, 28, 40, 24)
        column.setSpacing(10)
        column.addWidget(_label("RATE SHOTS"))
        column.addWidget(_label("How good is this shot?", "heading"))
        column.addWidget(self._progress)
        self._note = _label()
        column.addWidget(self._note)
        column.addWidget(self._video, 1)
        column.addWidget(self._about)
        column.addLayout(stars)
        column.addLayout(controls)

        for number in range(1, 6):
            key = QShortcut(QKeySequence(str(number)), self, functools.partial(self.rate, number))
            key.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            key.setAutoRepeat(False)  # a held key gives one rating, not one per repeat
        for sequence in ("Backspace", "Ctrl+Z"):
            undo = QShortcut(QKeySequence(sequence), self, self.undo)
            undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            undo.setAutoRepeat(False)
        for sequence in ("S", "Right"):
            skip = QShortcut(QKeySequence(sequence), self, self.skip)
            skip.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        leave = QShortcut(QKeySequence("Escape"), self, self.back.emit)
        leave.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._show_progress()

    def add_shots(self, info: MediaInfo, shots: list[Shot]) -> None:
        """Queue the not yet rated shots of one analysed clip (mixed in with the rest)."""
        fresh = [
            s
            for s in shots
            if self._mine.stars(info.path, s) is None and self._key(info, s) not in self._seen
        ]
        for shot in fresh:
            self._seen.add(self._key(info, shot))
        by_clip: dict[str, list[tuple[MediaInfo, Shot]]] = {}
        for item in [*self._queue, *((info, s) for s in fresh)]:
            by_clip.setdefault(str(item[0].path), []).append(item)
        order = interleave([[s for _, s in items] for items in by_clip.values()])
        lookup = {(id(s)): i for items in by_clip.values() for i, s in items}
        self._queue = [(lookup[id(s)], s) for s in order]
        if self._current is None:
            self._next()
        self._show_progress()

    def set_waiting(self, waiting: bool) -> None:
        self._waiting = waiting
        if self._current is None:
            self._show_progress()

    def start(self) -> None:
        if self._current is None:
            self._next()
        else:
            self._play()

    def stop(self) -> None:
        self._video.stop()

    def rate(self, stars: int) -> None:
        if self._current is None:
            return
        if time.monotonic() - self._shown_at < MIN_WATCH_S:
            return  # too soon to have seen it: a key pressed twice or held
        info, shot = self._current
        try:
            self._mine.rate(info.path, shot, stars)
        except OSError as exc:
            self._about.setText(f"COULD NOT SAVE YOUR RATING: {exc}")
            return
        self._last = self._current
        self._next()
        self._toast(f"SAVED {stars}★ FOR THE LAST SHOT   BACKSPACE UNDOES IT")

    def undo(self) -> None:
        """Forget the last rating and show that shot again."""
        if self._last is None:
            return
        info, shot = self._last
        try:
            self._mine.unrate(info.path, shot)
        except OSError as exc:
            self._about.setText(f"COULD NOT UNDO: {exc}")
            return
        if self._current is not None:
            self._queue.insert(0, self._current)
        self._current, self._last = self._last, None
        self._show_progress()
        self._play()
        self._toast("RATING UNDONE: RATE THIS SHOT AGAIN")

    def _toast(self, text: str) -> None:
        self._note.setText(text)
        QTimer.singleShot(
            2500, lambda: self._note.setText("") if self._note.text() == text else None
        )

    def skip(self) -> None:
        if self._current is not None:
            self._next()

    @staticmethod
    def _key(info: MediaInfo, shot: Shot) -> str:
        return f"{info.path}|{shot.start:.2f}"

    def _next(self) -> None:
        self._current = self._queue.pop(0) if self._queue else None
        self._show_progress()
        if self._current is None:
            self._video.stop()
            self._video.clear_picture()
            return
        self._play()

    def _play(self) -> None:
        if self._current is None:
            return
        info, shot = self._current
        self._shown_at = time.monotonic()
        self._video.play([Segment(info, shot.start, shot.end)], loop=True)
        tricks = ", ".join(dict.fromkeys(m.name for m in shot.moments)) or "flying"
        self._about.setText(
            f"{info.path.name.upper()}   {shot.start // 60:.0f}:{shot.start % 60:04.1f}   "
            f"{shot.duration:.0f} S   {tricks.upper()}"
        )

    def _show_progress(self) -> None:
        total = self._mine.count
        left = len(self._queue) + (1 if self._current is not None else 0)
        if self._current is None and self._waiting:
            text = "Analysing your clips; shots appear here as each one is ready."
        elif self._current is None:
            text = "No shots left to rate. Add more clips on step 1."
        elif total >= GOAL:
            text = f"{total} rated. That's enough to tune it; keep going if you like."
        else:
            text = f"{total} of {GOAL} rated   ({left} shots waiting)"
        self._progress.setText(text)
