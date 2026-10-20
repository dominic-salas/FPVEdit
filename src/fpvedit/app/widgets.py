"""Small helpers shared by the app's pages."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QPushButton


def timecode(seconds: float) -> str:
    """'1:05.3': minutes and seconds to a tenth."""
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{rest:04.1f}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def button(text: str, slot: Callable[[], object], *, primary: bool = False) -> QPushButton:
    widget = QPushButton(text)
    if primary:
        widget.setProperty("primary", True)
    widget.clicked.connect(slot)
    return widget
