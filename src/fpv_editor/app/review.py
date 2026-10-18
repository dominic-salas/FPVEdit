"""Step 3: the montage as a player with a timeline, and editing one shot at a time.

The player plays the whole montage, shot after shot, and the timeline under it shows
which recording each shot comes from. Clicking a shot on the timeline jumps there and
opens it in the side panel (trim, swap, remove, or loop just that shot); dragging a shot
moves it. Every edit goes through `ShotList`, so it can be undone.
"""

from __future__ import annotations

import functools
import queue
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fpv_editor.app import theme
from fpv_editor.app.frames import grab_frame
from fpv_editor.app.model import ShotList, drop_position, locate, montage_time, shot_starts
from fpv_editor.app.music import MusicPreview
from fpv_editor.app.preview import PREVIEW_HEIGHT, PreviewView, Segment
from fpv_editor.app.settings import AppSettings
from fpv_editor.app.timeline import Block, Timeline, clip_colour
from fpv_editor.errors import FPVEditError
from fpv_editor.media import MUSIC_SUFFIXES, AudioFile, MediaInfo, Tools, probe_audio
from fpv_editor.montage import Shot

THUMB_HEIGHT = 144  # grabbed once, drawn at timeline, list and panel sizes
TRIM_STEPS = (-0.5, 0.5)
SIDE_WIDTH = 360
FULL_SCREEN_HEIGHT = 720  # decoded picture height in full screen
HINT = "CLICK A SHOT TO EDIT IT   DRAG TO MOVE   SPACE PLAY   F FULL SCREEN"


def clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{rest:04.1f}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def describe(shot: Shot) -> str:
    counts: dict[str, int] = {}
    for moment in shot.moments:
        counts[moment.name] = counts.get(moment.name, 0) + 1
    names = [
        f"{count} {name}{'es' if name.endswith('ss') else 's'}" if count > 1 else name
        for name, count in counts.items()
    ]
    return ", ".join(names) or "flying"


def label(text: str = "", role: str = "eyebrow") -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", role)
    return widget


def button(text: str, slot: Callable[[], object], *, primary: bool = False) -> QPushButton:
    widget = QPushButton(text)
    if primary:
        widget.setProperty("primary", True)
    widget.clicked.connect(slot)
    return widget


class ThumbnailLoader(QThread):
    """Grabs frames one at a time from a queue of requests."""

    ready = Signal(object, bytes)  # key (path, seconds), PNG

    def __init__(self, tools: Tools, parent: QObject) -> None:
        super().__init__(parent)
        self._tools = tools
        self._requests: queue.Queue[tuple[MediaInfo, float] | None] = queue.Queue()

    def request(self, info: MediaInfo, seconds: float) -> None:
        self._requests.put((info, seconds))

    def stop(self) -> None:
        while not self._requests.empty():  # drop what is still waiting
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        self._requests.put(None)
        self.wait()

    def run(self) -> None:
        while (request := self._requests.get()) is not None:
            info, seconds = request
            png = grab_frame(self._tools, info, seconds, THUMB_HEIGHT)
            if png is not None:
                self.ready.emit((info.path, seconds), png)


class FullScreen(QWidget):
    """The player and timeline on their own, full screen; ``closing`` when it goes."""

    closing = Signal()

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("FPV Editor")
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(32, 24, 32, 20)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closing.emit()
        super().closeEvent(event)


def describe_music(tracks: Sequence[AudioFile], montage: float) -> str:
    """How the music fits the montage, in one line."""
    if not tracks:
        return "NO MUSIC. THE MONTAGE IS SILENT."
    music = sum(track.duration for track in tracks)
    if music > montage + 0.05:
        return f"{clock(music)} OF MUSIC, CUT AND FADED OUT AT {clock(montage)}"
    return f"MUSIC ENDS AT {clock(music)}; THE LAST {clock(montage - music)} ARE SILENT"


class ReviewPage(QWidget):
    """The review step. ``changed`` fires after every edit of the shots."""

    changed = Signal()

    def __init__(
        self, tools: Tools, settings: AppSettings | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._tools = tools
        self._settings = settings or AppSettings()
        self._music: list[AudioFile] = []
        self._music_preview = MusicPreview(tools)
        self._full_screen: FullScreen | None = None
        self._infos: tuple[MediaInfo, ...] = ()
        self._shots: ShotList | None = None
        self._summary = ""
        self._selected: int | None = None
        self._position = 0.0  # seconds into the montage
        self._mode = "montage"  # or "shot" (one shot looping) or "preview" (a candidate)
        self._alts: list[Shot] = []
        self._choosing: str | None = None  # "swap" or "add"
        self._thumbs: dict[tuple[Path, float], QPixmap] = {}
        self._requested: set[tuple[Path, float]] = set()
        self._palette = theme.DARK
        self._loader = ThumbnailLoader(tools, self)
        self._loader.ready.connect(self._thumbnail_ready)
        self._loader.start()
        self._build(tools)

    # ----- layout -----

    def _build(self, tools: Tools) -> None:
        self._video = PreviewView(tools)
        self._video.position.connect(self._playing_at)
        self._video.ended.connect(self._ended)
        self._video.doubleClicked.connect(self.toggle_full_screen)
        self._play = button("▶  PLAY", self.toggle_play)
        self._play.setFixedWidth(96)
        self._time = label()
        self._now = QLabel()
        self._now.setProperty("role", "caption")
        self._now.setTextFormat(Qt.TextFormat.RichText)
        transport = QHBoxLayout()
        transport.setSpacing(12)
        transport.addWidget(self._play)
        transport.addWidget(self._time)
        transport.addStretch(1)
        transport.addWidget(self._now)
        self._full_button = button("⛶  FULL SCREEN", self.toggle_full_screen)
        self._full_button.setProperty("flat", True)
        transport.addWidget(self._full_button)
        self._timeline = Timeline()
        self._timeline.picked.connect(self._picked)
        self._timeline.seeked.connect(self._seek)
        self._timeline.moved.connect(self._moved)
        self._legend = QLabel()
        self._legend.setProperty("role", "caption")
        self._legend.setTextFormat(Qt.TextFormat.RichText)
        self._legend.setWordWrap(True)
        legend_row = QHBoxLayout()
        legend_row.addWidget(self._legend, 1)
        legend_row.addWidget(label(HINT))
        stage = QWidget()
        self._stage = stage
        stage_column = QVBoxLayout(stage)
        stage_column.setContentsMargins(0, 0, 0, 0)
        stage_column.setSpacing(10)
        stage_column.addWidget(self._video, 1)
        stage_column.addLayout(transport)
        stage_column.addWidget(self._timeline)
        stage_column.addLayout(legend_row)

        self._side = QStackedWidget()
        self._side.addWidget(self._build_overview())
        self._side.addWidget(self._build_editor())
        self._side.addWidget(self._build_chooser())
        self._side.setFixedWidth(SIDE_WIDTH)

        body = QHBoxLayout()
        self._body = body
        body.setSpacing(28)
        body.addWidget(stage, 1)
        body.addWidget(self._side)

        self._heading_summary = label(role="muted")
        self._heading_summary.setWordWrap(True)
        column = QVBoxLayout(self)
        column.setContentsMargins(40, 28, 40, 16)
        column.setSpacing(6)
        column.addWidget(label("STEP 3 OF 4"))
        column.addWidget(label("Your montage", "heading"))
        column.addWidget(self._heading_summary)
        column.addSpacing(14)
        column.addLayout(body, 1)

        self._add_shortcuts(self)
        delete = QShortcut(QKeySequence.StandardKey.Delete, self, self.remove_selected)
        delete.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def _add_shortcuts(self, widget: QWidget) -> None:
        """Space, F, Esc and the arrows, on the page and on the full-screen window."""
        keys: list[tuple[QKeySequence, Callable[[], object]]] = [
            (QKeySequence(Qt.Key.Key_Space), self.toggle_play),
            (QKeySequence(Qt.Key.Key_F), self.toggle_full_screen),
            (QKeySequence(Qt.Key.Key_Escape), self.leave_full_screen),
            (QKeySequence(Qt.Key.Key_Left), functools.partial(self._step_shot, -1)),
            (QKeySequence(Qt.Key.Key_Right), functools.partial(self._step_shot, 1)),
        ]
        for sequence, slot in keys:
            shortcut = QShortcut(sequence, widget, slot)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def _build_overview(self) -> QWidget:
        self._overview_total = label(role="hero")
        self._message = label()
        self._message.setWordWrap(True)
        self._music_list = QListWidget()
        self._music_list.currentRowChanged.connect(lambda _: self._refresh_music_buttons())
        self._music_fit = label()
        self._music_fit.setWordWrap(True)
        self._music_up = button("↑", lambda: self._move_music(-1))
        self._music_up.setToolTip("Play this track earlier")
        self._music_down = button("↓", lambda: self._move_music(1))
        self._music_down.setToolTip("Play this track later")
        self._music_remove = button("REMOVE", self._remove_music)
        music_buttons = QHBoxLayout()
        music_buttons.setSpacing(6)
        music_buttons.addWidget(button("+  ADD MUSIC", self._add_music))
        music_buttons.addStretch(1)
        music_buttons.addWidget(self._music_up)
        music_buttons.addWidget(self._music_down)
        music_buttons.addWidget(self._music_remove)
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(label("THE MONTAGE"))
        column.addWidget(self._overview_total)
        column.addWidget(button("+  ADD A SHOT", lambda: self._open_chooser("add")))
        column.addSpacing(18)
        column.addWidget(label("MUSIC   TRACKS PLAY IN THIS ORDER"))
        column.addWidget(self._music_list)
        column.addWidget(self._music_fit)
        column.addLayout(music_buttons)
        column.addSpacing(18)
        column.addWidget(self._message)
        column.addStretch(1)
        return panel

    def _build_editor(self) -> QWidget:
        self._shot_number = label()
        self._shot_picture = QLabel()
        self._shot_picture.setFixedSize(SIDE_WIDTH, round(SIDE_WIDTH * 9 / 16))
        self._shot_picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._shot_picture.setStyleSheet("background: black; border-radius: 6px;")
        self._shot_title = label(role="hero")
        self._shot_source = QLabel()
        self._shot_source.setProperty("role", "caption")
        self._shot_source.setTextFormat(Qt.TextFormat.RichText)
        self._shot_source.setWordWrap(True)
        trims = QHBoxLayout()
        trims.setSpacing(6)
        for end in (False, True):
            trims.addWidget(label("END" if end else "START"))
            for step in TRIM_STEPS:
                trim = button(f"{step:+g}S", functools.partial(self._trim, end, step))
                trim.setToolTip(
                    f"{'Lengthen' if (step > 0) == end else 'Shorten'} the shot at its "
                    f"{'end' if end else 'start'} by {abs(step):g} s"
                )
                trims.addWidget(trim)
            if not end:
                trims.addSpacing(10)
        trims.addStretch(1)
        self._loop_button = button("↻  LOOP THIS SHOT", self._toggle_loop)
        self._loop_button.setToolTip("Play only this shot, again and again")
        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self._loop_button)
        swap = button("SWAP", lambda: self._open_chooser("swap"))
        swap.setToolTip("Pick another shot from the same recording instead")
        actions.addWidget(swap)
        actions.addWidget(button("REMOVE", self.remove_selected))
        actions.addStretch(1)
        steps = QHBoxLayout()
        steps.setSpacing(6)
        self._previous = button("←  PREVIOUS", lambda: self._step_shot(-1))
        self._next = button("NEXT  →", lambda: self._step_shot(1))
        steps.addWidget(self._previous)
        steps.addWidget(self._next)
        steps.addStretch(1)
        steps.addWidget(button("DONE", self._deselect))
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        column.addWidget(self._shot_number)
        column.addWidget(self._shot_picture)
        column.addWidget(self._shot_title)
        column.addWidget(self._shot_source)
        column.addSpacing(6)
        column.addWidget(label("TRIM"))
        column.addLayout(trims)
        column.addSpacing(6)
        column.addLayout(actions)
        column.addStretch(1)
        column.addLayout(steps)
        return panel

    def _build_chooser(self) -> QWidget:
        self._chooser_title = label()
        self._chooser_title.setWordWrap(True)
        self._alternatives = QListWidget()
        self._alternatives.setIconSize(QPixmap(96, 54).size())
        self._alternatives.setWordWrap(True)
        self._alternatives.currentRowChanged.connect(self._alternative_selected)
        self._use = button("USE THIS SHOT", self._use_alternative, primary=True)
        buttons = QHBoxLayout()
        buttons.addWidget(button("CANCEL", self._close_chooser))
        buttons.addStretch(1)
        buttons.addWidget(self._use)
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self._chooser_title)
        column.addWidget(self._alternatives, 1)
        column.addLayout(buttons)
        return panel

    # ----- the window's side -----

    def set_montage(
        self, infos: Sequence[MediaInfo], shots: ShotList | None, summary: str = ""
    ) -> None:
        """Show a new montage (or none); playback starts over. The music stays."""
        self._pause_all()
        self._infos, self._shots, self._summary = tuple(infos), shots, summary
        self._selected, self._position, self._mode = None, 0.0, "montage"
        self._choosing = None
        self._side.setCurrentIndex(0)
        self._video.clear_picture()
        self.refresh()

    def activate(self) -> None:
        """The page is shown: play the montage from where it was."""
        if self._current() and self._choosing is None:
            self._play_montage(self._position if self._position < self._total() - 0.1 else 0.0)

    def deactivate(self) -> None:
        self.leave_full_screen()
        self._pause_all()
        self._show_play_state()

    def shutdown(self) -> None:
        self.leave_full_screen()
        self._pause_all()
        self._loader.stop()

    @property
    def music(self) -> tuple[AudioFile, ...]:
        return tuple(self._music)

    # ----- full screen -----

    def toggle_full_screen(self) -> None:
        if self._full_screen is None:
            self.enter_full_screen()
        else:
            self.leave_full_screen()

    def enter_full_screen(self) -> None:
        if self._full_screen is not None or not self._current():
            return
        window = FullScreen()
        self._add_shortcuts(window)
        self._body.removeWidget(self._stage)
        window.layout_.addWidget(self._stage)
        window.closing.connect(self._full_screen_closed)
        self._full_screen = window
        self._full_button.setText("EXIT FULL SCREEN")
        self._video.height_px = FULL_SCREEN_HEIGHT
        window.showFullScreen()
        self._restart_playback()

    def leave_full_screen(self) -> None:
        if self._full_screen is not None:
            self._full_screen.close()  # puts the player back through `closing`

    def _full_screen_closed(self) -> None:
        if self._full_screen is None:
            return
        self._full_screen.layout_.removeWidget(self._stage)
        self._body.insertWidget(0, self._stage, 1)
        self._full_screen.deleteLater()
        self._full_screen = None
        self._full_button.setText("⛶  FULL SCREEN")
        self._video.height_px = PREVIEW_HEIGHT
        self._restart_playback()

    def _restart_playback(self) -> None:
        """Play on at the new picture size, if something was playing."""
        if not self._video.playing:
            return
        if self._mode == "montage":
            self._play_montage(self._position)
        elif self._mode == "shot":
            self._loop_selected()

    # ----- music -----

    def _add_music(self) -> None:
        start = str(self._settings.music_folder or "")
        pattern = " ".join(f"*{suffix}" for suffix in MUSIC_SUFFIXES)
        names, _ = QFileDialog.getOpenFileNames(
            self, "Add music", start, f"Music ({pattern});;All files (*)"
        )
        if not names:
            return
        self._settings.music_folder = Path(names[0]).parent
        problems = []
        for name in names:
            try:
                self._music.append(probe_audio(self._tools, Path(name)))
            except FPVEditError as exc:
                problems.append(str(exc))
        self._say("; ".join(problems) if problems else "")
        self._music_changed(len(self._music) - 1)

    def _remove_music(self) -> None:
        row = self._music_list.currentRow()
        if 0 <= row < len(self._music):
            del self._music[row]
            self._music_changed(min(row, len(self._music) - 1))

    def _move_music(self, step: int) -> None:
        row = self._music_list.currentRow()
        to = row + step
        if 0 <= row < len(self._music) and 0 <= to < len(self._music):
            self._music.insert(to, self._music.pop(row))
            self._music_changed(to)

    def _music_changed(self, row: int) -> None:
        self._refresh_music(row)
        self.changed.emit()
        if self._mode == "montage" and self._video.playing:
            self._music_preview.play(self._music, self._position, self._total())

    def _refresh_music(self, row: int | None = None) -> None:
        keep = self._music_list.currentRow() if row is None else row
        self._music_list.blockSignals(True)
        self._music_list.clear()
        for number, track in enumerate(self._music, start=1):
            self._music_list.addItem(f"{number:02d}   {track.path.name}   {clock(track.duration)}")
        self._music_list.blockSignals(False)
        self._music_list.setVisible(bool(self._music))
        self._music_list.setFixedHeight(min(len(self._music), 5) * 34 + 6)
        if self._music:
            self._music_list.setCurrentRow(min(max(keep, 0), len(self._music) - 1))
        fit = describe_music(self._music, self._total())
        if self._music and self._music_preview.ffplay is None:
            fit += " (NOT HEARD IN THIS PREVIEW: FFPLAY WAS NOT FOUND)"
        self._music_fit.setText(fit)
        self._refresh_music_buttons()

    def _refresh_music_buttons(self) -> None:
        row = self._music_list.currentRow()
        has = 0 <= row < len(self._music)
        self._music_remove.setEnabled(has)
        self._music_up.setEnabled(has and row > 0)
        self._music_down.setEnabled(has and row < len(self._music) - 1)

    def _pause_all(self) -> None:
        self._music_preview.stop()
        self._video.stop()

    def _play_video(
        self,
        segments: Sequence[Segment],
        index: int = 0,
        first: float | None = None,
        *,
        loop: bool = False,
    ) -> None:
        """Play pictures only (a shot on a loop, a candidate): the music stays quiet."""
        self._music_preview.stop()
        self._video.play(segments, index, first, loop=loop)

    def set_theme(self, palette: theme.Palette, fonts: theme.Fonts) -> None:
        self._palette = palette
        self._timeline.set_theme(palette, fonts)
        self.refresh()

    def undo(self) -> None:
        if self._shots is not None and self._shots.can_undo:
            self._shots.undo()
            self._selected = None
            self._edited(min(self._position, self._total()))

    def remove_selected(self) -> None:
        index = self._selected
        if index is None or self._shots is None or self._choosing is not None:
            return
        start = shot_starts(self._current())[index]
        self._shots.remove(index)
        self._selected = None
        self._say("Shot removed. Ctrl+Z brings it back.")
        self._edited(start)

    # ----- state -----

    def _current(self) -> tuple[Shot, ...]:
        return self._shots.shots if self._shots is not None else ()

    def _total(self) -> float:
        return sum(shot.duration for shot in self._current())

    def _info(self, shot: Shot) -> MediaInfo:
        return self._infos[shot.clip]

    def _name(self, shot: Shot) -> str:
        return self._info(shot).path.name

    def _segments(self) -> list[Segment]:
        return [Segment(self._info(s), s.start, s.end) for s in self._current()]

    def refresh(self) -> None:
        """Bring the timeline, the panels and the texts up to date."""
        shots = self._current()
        total = self._total()
        blocks = [
            Block(
                duration=shot.duration,
                clip=shot.clip,
                label=f"{number:02d}  {self._name(shot)}",
                tooltip=(
                    f"Shot {number}: {describe(shot)}\n{self._name(shot)}, "
                    f"{clock(shot.start)} to {clock(shot.end)} ({shot.duration:.1f} s)"
                ),
                thumbnail=self._thumbnail(shot),
            )
            for number, shot in enumerate(shots, start=1)
        ]
        self._timeline.set_blocks(blocks)
        self._timeline.set_selected(self._selected if self._mode != "preview" else None)
        self._timeline.set_position(self._position)
        used = sorted({shot.clip for shot in shots})
        self._legend.setText(
            "   ".join(
                f"<span style='color:{clip_colour(clip)}'>■</span> {self._infos[clip].path.name}"
                for clip in used
            )
        )
        self._overview_total.setText(f"{plural(len(shots), 'shot')}, {clock(total)}")
        self._refresh_music()
        self._heading_summary.setText(
            f"{plural(len(shots), 'shot')}, {clock(total)} long. {self._summary}"
        )
        self._show_time()
        self._show_editor()

    def _show_editor(self) -> None:
        shots = self._current()
        index = self._selected
        if self._choosing is not None:
            return
        if index is None or index >= len(shots):
            self._side.setCurrentIndex(0)
            return
        shot = shots[index]
        self._side.setCurrentIndex(1)
        self._shot_number.setText(f"SHOT {index + 1:02d} OF {len(shots):02d}")
        pixmap = self._thumbnail(shot)
        if pixmap is not None:
            self._shot_picture.setPixmap(
                pixmap.scaled(
                    self._shot_picture.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._shot_picture.clear()
        self._shot_title.setText(describe(shot).capitalize())
        self._shot_source.setText(
            f"<span style='color:{clip_colour(shot.clip)}'>■</span> "
            f"{self._name(shot).upper()}<br>{clock(shot.start)} TO {clock(shot.end)}   "
            f"{shot.duration:.1f} S"
        )
        self._previous.setEnabled(index > 0)
        self._next.setEnabled(index < len(shots) - 1)
        looping = self._mode == "shot" and self._video.playing
        self._loop_button.setText("▶  PLAY THE MONTAGE" if looping else "↻  LOOP THIS SHOT")

    def _say(self, text: str) -> None:
        self._message.setText(text.upper())

    # ----- thumbnails -----

    def _thumb_key(self, shot: Shot) -> tuple[Path, float]:
        seconds = shot.moments[0].time if shot.moments else (shot.start + shot.end) / 2
        return (self._info(shot).path, round(seconds, 2))

    def _thumbnail(self, shot: Shot) -> QPixmap | None:
        key = self._thumb_key(shot)
        if key in self._thumbs:
            return self._thumbs[key]
        if key not in self._requested:
            self._requested.add(key)
            self._loader.request(self._info(shot), key[1])
        return None

    def _thumbnail_ready(self, key: object, png: bytes) -> None:
        pixmap = QPixmap()
        if pixmap.loadFromData(png) and isinstance(key, tuple):
            self._thumbs[key] = pixmap
            self.refresh()
            for i, shot in enumerate(self._alts):
                if i < self._alternatives.count() and self._thumb_key(shot) == key:
                    self._alternatives.item(i).setIcon(QIcon(pixmap))

    # ----- playback -----

    def toggle_play(self) -> None:
        if not self._current() or self._choosing is not None:
            return
        if self._video.playing:
            self._pause_all()
            self._show_play_state()
        elif self._mode == "shot" and self._selected is not None:
            self._loop_selected()
        else:
            self._play_montage(self._position if self._position < self._total() - 0.1 else 0.0)

    def _play_montage(self, seconds: float) -> None:
        shots = self._current()
        if not shots:
            return
        self._mode = "montage"
        index, clip_seconds = locate(shots, seconds)
        self._position = montage_time(shots, index, clip_seconds)
        self._play_video(self._segments(), index, clip_seconds)
        self._music_preview.play(self._music, self._position, self._total())
        self._show_play_state()
        self.refresh()

    def _loop_selected(self, first: float | None = None) -> None:
        if self._selected is None:
            return
        shot = self._current()[self._selected]
        self._mode = "shot"
        self._play_video([Segment(self._info(shot), shot.start, shot.end)], 0, first, loop=True)
        self._show_play_state()
        self._show_editor()

    def _toggle_loop(self) -> None:
        if self._mode == "shot" and self._video.playing:
            index = self._selected
            start = shot_starts(self._current())[index] if index is not None else 0.0
            self._play_montage(start)
        else:
            self._loop_selected()

    def _show_play_state(self) -> None:
        playing = self._video.playing
        self._play.setText("❚❚  PAUSE" if playing else "▶  PLAY")
        if self._selected is not None:
            self._show_editor()

    def _playing_at(self, segment: int, seconds: float) -> None:
        shots = self._current()
        if self._mode == "montage" and 0 <= segment < len(shots):
            self._position = montage_time(shots, segment, seconds)
            self._timeline.set_position(self._position)
            self._show_now(segment)
        elif self._mode == "shot" and self._selected is not None:
            self._position = montage_time(shots, self._selected, seconds)
            self._timeline.set_position(self._position)
            self._show_now(self._selected, looping=True)
        self._show_time()

    def _show_now(self, index: int, *, looping: bool = False) -> None:
        shots = self._current()
        shot = shots[index]
        state = "LOOPING" if looping else "NOW"
        self._now.setText(
            f"{state}  {index + 1:02d}/{len(shots):02d}   "
            f"<span style='color:{clip_colour(shot.clip)}'>■</span> "
            f"{self._name(shot).upper()}   {describe(shot).upper()}"
        )

    def _show_time(self) -> None:
        self._time.setText(f"{clock(self._position)} / {clock(self._total())}")

    def _ended(self) -> None:
        self._music_preview.stop()
        self._position = self._total()
        self._timeline.set_position(self._position)
        self._show_play_state()
        self._show_time()

    def _seek(self, seconds: float) -> None:
        if self._choosing is None and self._current():
            self._play_montage(seconds)

    def _picked(self, index: int, share: float) -> None:
        """A click on a shot: open it for editing and play from where it was clicked."""
        if self._choosing is not None:
            self._close_chooser()
        shots = self._current()
        if not 0 <= index < len(shots):
            return
        self._selected = index
        self._play_montage(shot_starts(shots)[index] + share * shots[index].duration)

    def _deselect(self) -> None:
        self._selected = None
        if self._mode == "shot":
            self._mode = "montage"
            if self._video.playing:
                self._play_montage(self._position)
        self.refresh()

    def _step_shot(self, step: int) -> None:
        shots = self._current()
        if not shots or self._choosing is not None:
            return
        current = self._selected if self._selected is not None else locate(shots, self._position)[0]
        target = current + step
        target = min(max(0, target), len(shots) - 1)
        self._selected = target
        if self._mode == "shot":
            self._loop_selected()
            self.refresh()
        else:
            self._play_montage(shot_starts(shots)[target])

    # ----- edits -----

    def _edited(self, play_from: float) -> None:
        """After an edit: tell the window, redraw, and play from ``play_from``."""
        self.changed.emit()
        if not self._current():
            self._pause_all()
            self._video.clear_picture()
            self._position = 0.0
            self.refresh()
            return
        if self._mode == "shot" and self._selected is not None:
            self.refresh()
            self._loop_selected()
        else:
            self._play_montage(min(play_from, max(0.0, self._total() - 0.05)))

    def _moved(self, index: int, gap: int) -> None:
        if self._shots is None or self._choosing is not None:
            return
        to = drop_position(index, gap)
        self._shots.move(index, to)
        self._selected = to
        self._mode = "montage"
        self._say(f"Shot moved to number {to + 1}. Ctrl+Z undoes it.")
        self._edited(shot_starts(self._current())[to])

    def _trim(self, end: bool, step: float) -> None:
        index = self._selected
        if index is None or self._shots is None:
            return
        if end:
            self._shots.trim(index, end=step)
        else:
            self._shots.trim(index, start=step)
        shot = self._current()[index]
        start = shot_starts(self._current())[index]
        # An end change plays the last seconds; a start change plays from the start.
        if self._mode == "shot":
            self.changed.emit()
            self.refresh()
            self._loop_selected(max(shot.start, shot.end - 3) if end else None)
        else:
            self._edited(start + max(0.0, shot.duration - 3) if end else start)

    # ----- choosing another shot -----

    def _open_chooser(self, mode: str) -> None:
        if self._shots is None:
            return
        index = self._selected
        if mode == "swap":
            if index is None:
                return
            self._alts = self._shots.alternatives(index)
            title = f"SWAP SHOT {index + 1} FOR ONE OF THESE (SAME RECORDING)"
        else:
            self._alts = self._shots.alternatives(limit=40)
            title = "ADD ONE OF THESE SHOTS (ALL RECORDINGS)"
        self._choosing = mode
        self._chooser_title.setText(title if self._alts else "NO OTHER SHOTS TO OFFER")
        self._alternatives.blockSignals(True)
        self._alternatives.clear()
        for shot in self._alts:
            pixmap = self._thumbnail(shot)
            icon = QIcon(pixmap) if pixmap is not None else QIcon()
            text = (
                f"{describe(shot)}\n{self._name(shot)}   {clock(shot.start)}  ·  "
                f"{shot.duration:.1f} s"
            )
            self._alternatives.addItem(QListWidgetItem(icon, text))
        self._alternatives.blockSignals(False)
        self._use.setText("USE INSTEAD" if mode == "swap" else "ADD TO MONTAGE")
        self._use.setEnabled(bool(self._alts))
        self._side.setCurrentIndex(2)
        if self._alts:
            self._alternatives.setCurrentRow(0)
        else:
            self._pause_all()
            self._show_play_state()

    def _close_chooser(self) -> None:
        if self._choosing is None:
            return
        self._choosing = None
        self._mode = "montage"
        self._play_montage(self._position)

    def _alternative_selected(self, row: int) -> None:
        if 0 <= row < len(self._alts):
            shot = self._alts[row]
            self._mode = "preview"
            self._play_video([Segment(self._info(shot), shot.start, shot.end)], loop=True)
            self._now.setText(
                f"PREVIEW, NOT IN THE MONTAGE   "
                f"<span style='color:{clip_colour(shot.clip)}'>■</span> "
                f"{self._name(shot).upper()}   {describe(shot).upper()}"
            )
            self._show_play_state()

    def _use_alternative(self) -> None:
        row = self._alternatives.currentRow()
        if self._shots is None or not 0 <= row < len(self._alts) or self._choosing is None:
            return
        chosen = self._alts[row]
        try:
            if self._choosing == "swap" and self._selected is not None:
                self._shots.replace(self._selected, chosen)
                position = self._selected
                self._say(f"Shot {position + 1} swapped. Ctrl+Z undoes it.")
            else:
                position = self._shots.add(chosen)
                self._say(f"Shot added as number {position + 1}. Ctrl+Z undoes it.")
        except ValueError as exc:
            self._say(f"Not possible: {exc}.")
            return
        self._choosing = None
        self._mode = "montage"
        self._selected = position
        self._edited(shot_starts(self._current())[position])
