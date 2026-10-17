"""The app window (Qt). Slow work runs in background threads: analysis, thumbnails, render.

Nothing leaves this computer: clips are read in place, measurements are cached in the
user's local app-data folder, and the montage is written where the user chooses.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QStandardPaths, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QPainter,
    QPaintEvent,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fpv_editor import __version__
from fpv_editor.app import theme
from fpv_editor.app.analyser import BackgroundAnalysis
from fpv_editor.app.cliplist import (
    DETAILS,
    PATH,
    PROBLEM,
    SHARE,
    STATUS,
    THUMB,
    ClipList,
    ClipReader,
    thumbnail_pixmap,
)
from fpv_editor.app.clips import clock as clock_short
from fpv_editor.app.clips import describe_clip
from fpv_editor.app.dialogs import SettingsDialog, about_text, ask_for_ffmpeg
from fpv_editor.app.frames import grab_frame
from fpv_editor.app.logs import start_logging
from fpv_editor.app.look import DotTitle, StatusDot, app_icon
from fpv_editor.app.model import ShotList
from fpv_editor.app.preview import PreviewView
from fpv_editor.app.settings import AppSettings
from fpv_editor.cli import (
    MONTAGE_SUFFIX,
    MontagePlan,
    PipelineConfig,
    Progress,
    combine_clips,
    render_montage,
)
from fpv_editor.errors import CancelledError, FPVEditError
from fpv_editor.media import MediaInfo, Tools, find_tools, run_tool
from fpv_editor.montage import MomentKind, Shot
from fpv_editor.render import (
    NVENC,
    EncodeConfig,
    default_output_path,
    encoders,
    resolve_output_path,
    with_fallback,
)

VIDEO_SUFFIXES = (".ts", ".mp4")
OUTPUT_MARKERS = (MONTAGE_SUFFIX, "_highlight")  # our own renders, skipped in dropped folders
THUMB_HEIGHT = 72
TRIM_STEPS = (-0.5, 0.5)
log = logging.getLogger("fpv_editor.app")


def video_files(paths: Sequence[Path]) -> list[Path]:
    """Recordings among dropped files and folders (a folder's top level only, by name)."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if p.is_file() and _is_recording(p)))
        elif path.is_file() and _is_recording(path):
            found.append(path)
    return found


def _is_recording(path: Path) -> bool:
    stem = path.stem.lower()
    ours = any(marker in stem for marker in OUTPUT_MARKERS)
    return path.suffix.lower() in VIDEO_SUFFIXES and not ours


def clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{rest:04.1f}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def time_left(elapsed: float, share: float) -> str:
    """'about 2 min left' from the time so far and the share done ('' while too early)."""
    if share < 0.03 or elapsed < 3:
        return ""
    seconds = elapsed * (1 - share) / share
    if seconds < 60:
        return "less than a minute left"
    return f"about {round(seconds / 60)} min left"


def describe(shot: Shot) -> str:
    counts: dict[MomentKind, int] = {}
    for moment in shot.moments:
        counts[moment.kind] = counts.get(moment.kind, 0) + 1
    names = [
        f"{count} {kind.value}{'es' if kind is MomentKind.PROXIMITY else 's'}"
        if count > 1
        else kind.value
        for kind, count in counts.items()
    ]
    return ", ".join(names) or "flying"


class Task(QThread):
    """Runs ``work(progress)`` in the background; ``cancel()`` stops it at the next report."""

    progressed = Signal(str, float)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, work: Callable[[Progress], object], parent: QObject) -> None:
        super().__init__(parent)
        self._work = work
        self._cancel = threading.Event()
        self.started_at = time.monotonic()

    def cancel(self) -> None:
        self._cancel.set()

    def _report(self, text: str, share: float) -> None:
        if self._cancel.is_set():
            raise CancelledError("cancelled")
        self.progressed.emit(text, share)

    def run(self) -> None:
        try:
            result = self._work(self._report)
        except CancelledError:
            log.info("cancelled")
            self.failed.emit("Cancelled.")
        except FPVEditError as exc:
            log.warning("failed: %s", exc)
            self.failed.emit(str(exc))
        except Exception:  # a defect: show it in full rather than lose it in a thread
            log.exception("unexpected error")
            self.failed.emit(f"Unexpected error:\n{traceback.format_exc()}")
        else:
            self.succeeded.emit(result)


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


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "heading")
    return label


def eyebrow(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "eyebrow")
    return label


def button(text: str, slot: Callable[[], object], *, primary: bool = False) -> QPushButton:
    widget = QPushButton(text)
    if primary:
        widget.setProperty("primary", True)
    widget.clicked.connect(slot)
    return widget


def restyle(widget: QWidget, name: str, value: object) -> None:
    """Change a style-sheet property and repaint the widget with it."""
    if widget.property(name) != value:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class DotGrid(QWidget):
    """A plain area with the site's faint dot grid behind its children."""

    def __init__(self) -> None:
        super().__init__()
        self._dot = QColor(theme.DARK.dot)

    def set_palette(self, palette: theme.Palette) -> None:
        self._dot = QColor(palette.dot)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._dot)
        step = 16
        for y in range(step // 2, self.height(), step):
            for x in range(step // 2, self.width(), step):
                painter.drawEllipse(QPointF(x, y), 1.0, 1.0)
        painter.end()


class Steps(QWidget):
    """The four steps in the header; the current one is lit, reachable ones are clickable."""

    chosen = Signal(int)
    NAMES = ("ADD CLIPS", "FIND SHOTS", "REVIEW", "EXPORT")

    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._buttons: list[QPushButton] = []
        for index, name in enumerate(self.NAMES):
            if index:
                line = eyebrow("——")
                row.addWidget(line)
            step = QPushButton(f"{index + 1}  {name}")
            step.setProperty("role", "step")
            step.clicked.connect(lambda _=False, i=index: self.chosen.emit(i))
            row.addWidget(step)
            self._buttons.append(step)

    def show_state(self, current: int, reachable: set[int]) -> None:
        for index, step in enumerate(self._buttons):
            state = "on" if index == current else ("done" if index < current else "off")
            restyle(step, "state", state)
            step.setEnabled(index in reachable and index != current)


class MainWindow(QMainWindow):
    """Four screens, one step each: add clips, find shots, review, export."""

    CLIPS, WORK, REVIEW, DONE = range(4)

    def __init__(
        self,
        tools: Tools,
        cache_dir: Path | None,
        settings: AppSettings | None = None,
        log_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"FPV Editor {__version__}")
        self.setWindowIcon(app_icon())
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self._tools = tools
        self._cache_dir = cache_dir
        self._settings = settings or AppSettings()
        self._log_file = log_file
        self._notes: tuple[str, ...] = ()
        self._plan: MontagePlan | None = None
        self._shots: ShotList | None = None
        self._task: Task | None = None
        self._task_page = self.CLIPS  # where to return if the work fails or is cancelled
        self._output: Path | None = None  # chosen by the user; None = automatic name
        self._force = False  # the user confirmed replacing ``_output``
        self._thumbs: dict[tuple[Path, float], QPixmap] = {}
        self._requested: set[tuple[Path, float]] = set()
        self._loop: tuple[float, float] | None = None  # preview range, seconds
        self._alts: list[Shot] = []  # the shots offered by the chooser, in list order
        self._choosing: str | None = None  # "swap" or "add" while the chooser is open
        self._last_output: Path | None = None
        self._last_summary = ""
        self._fonts = theme.load_fonts()
        self._infos: dict[Path, MediaInfo] = {}  # probed clips, by resolved path
        self._unreadable: dict[Path, str] = {}  # clips that could not be probed, and why
        self._clip_notes: dict[Path, str] = {}  # what each clip gave the last plan
        self._finding: tuple[float, float] | None = None  # (started, share then) while waiting

        self._analysis = BackgroundAnalysis(tools, cache_dir, self)
        self._analysis.changed.connect(self._analysis_changed)
        self._reader = ClipReader(tools, self)
        self._reader.probed.connect(self._clip_probed)
        self._reader.failed.connect(self._clip_unreadable)
        self._reader.thumbnail.connect(self._clip_thumbnail)
        self._reader.start()

        self._loader = ThumbnailLoader(tools, self)
        self._loader.ready.connect(self._thumbnail_ready)
        self._loader.start()

        self._build()
        self._build_menus()
        self._apply_theme(self._settings.theme)
        geometry = self._settings.geometry
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._refresh()

    # ----- layout -----

    def _build(self) -> None:
        logo = QLabel("fpv<span style='color:#e5322d'>.</span>editor")
        logo.setProperty("role", "logo")
        self._steps = Steps()
        self._steps.chosen.connect(self._go_to_step)
        self._theme_button = QPushButton()
        self._theme_button.setProperty("flat", True)
        self._theme_button.clicked.connect(self._toggle_theme)
        self._menu_button = QPushButton("MENU")
        self._menu_button.setProperty("flat", True)
        header = QFrame()
        header.setProperty("role", "header")
        header.setFixedHeight(56)
        row = QHBoxLayout(header)
        row.setContentsMargins(24, 0, 16, 0)
        row.addWidget(logo)
        row.addStretch(1)
        row.addWidget(self._steps)
        row.addStretch(1)
        row.addWidget(self._theme_button)
        row.addWidget(self._menu_button)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_clips_page())
        self._pages.addWidget(self._build_work_page())
        self._pages.addWidget(self._build_review_page())
        self._pages.addWidget(self._build_done_page())

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self._pages, 1)
        self.setCentralWidget(central)

    def _build_clips_page(self) -> QWidget:
        # Empty: one big drop target that says what to do.
        self._drop_zone = DotGrid()
        self._title = DotTitle("fpv editor", cell=10)
        intro = QLabel(
            "Drop your FPV recordings (.ts or .mp4) or a whole folder anywhere on this "
            "window. FPV Editor finds the best flips, spins and close passes and cuts them "
            "into one montage. You check it before anything is saved."
        )
        intro.setProperty("role", "hero")
        intro.setWordWrap(True)
        intro.setMaximumWidth(560)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(button("ADD CLIPS", self._choose_clips, primary=True))
        buttons.addWidget(button("ADD A FOLDER", self._choose_folder))
        buttons.addStretch(1)
        zone = QVBoxLayout(self._drop_zone)
        zone.setContentsMargins(72, 48, 72, 48)
        zone.addStretch(2)
        zone.addWidget(eyebrow("● STEP 1 OF 4   RUNS ON THIS PC. NOTHING IS UPLOADED"))
        zone.addSpacing(18)
        zone.addWidget(self._title)
        zone.addSpacing(24)
        zone.addWidget(intro)
        zone.addSpacing(24)
        zone.addLayout(buttons)
        zone.addStretch(3)

        # With clips: the list in playing order, and one next step.
        self._clips = ClipList()
        self._clips.dropped.connect(self.add_paths)
        self._clips.model().rowsMoved.connect(lambda *_: self._clips_changed())
        self._clips_summary = eyebrow()
        tools = QHBoxLayout()
        tools.setSpacing(8)
        tools.addWidget(button("ADD MORE", self._choose_clips))
        tools.addWidget(button("ADD A FOLDER", self._choose_folder))
        self._remove_clips_button = button("REMOVE SELECTED", self._remove_clips)
        tools.addWidget(self._remove_clips_button)
        tools.addStretch(1)
        tools.addWidget(self._clips_summary)
        self._length = QDoubleSpinBox()
        self._length.setRange(10, 1800)
        self._length.setDecimals(0)
        self._length.setSuffix(" S")
        self._length.setValue(self._settings.length_s)
        self._length.valueChanged.connect(self._length_changed)
        self._find = button("FIND SHOTS  →", self._find_shots, primary=True)
        list_page = DotGrid()
        self._list_page = list_page
        column = QVBoxLayout(list_page)
        column.setContentsMargins(72, 40, 72, 0)
        column.setSpacing(10)
        column.addWidget(eyebrow("STEP 1 OF 4"))
        column.addWidget(heading("Your clips"))
        hint = QLabel(
            "Shots from these clips play in this order. Drag to reorder; drop more "
            "anywhere on the window. Each clip is analysed as soon as it is added."
        )
        hint.setProperty("role", "muted")
        column.addWidget(hint)
        column.addSpacing(8)
        column.addWidget(self._clips, 1)
        column.addLayout(tools)
        column.addSpacing(16)

        self._clip_stack = QStackedWidget()
        self._clip_stack.addWidget(self._drop_zone)
        self._clip_stack.addWidget(list_page)
        self._clips_bar = self._action_bar([eyebrow("MONTAGE LENGTH"), self._length], [self._find])
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._clip_stack, 1)
        layout.addWidget(self._clips_bar)
        return page

    def _build_work_page(self) -> QWidget:
        self._work_step = eyebrow()
        self._work_title = heading("")
        self._percent = QLabel()
        self._percent.setProperty("role", "big-number")
        self._work_status = QLabel()
        self._work_status.setProperty("role", "muted")
        self._work_left = eyebrow()
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.setFixedWidth(520)
        self._dot = StatusDot()
        self._cancel = button("CANCEL", self._cancel_task)
        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        status_row.addWidget(self._dot)
        status_row.addWidget(self._work_status, 1)
        page = DotGrid()
        self._work_page = page
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._work_step)
        column.addWidget(self._work_title)
        column.addSpacing(8)
        column.addWidget(self._percent)
        column.addWidget(self._progress)
        column.addSpacing(12)
        column.addLayout(status_row)
        column.addWidget(self._work_left)
        column.addSpacing(20)
        cancel_row = QHBoxLayout()
        cancel_row.addWidget(self._cancel)
        cancel_row.addStretch(1)
        column.addLayout(cancel_row)
        # While finding shots: every clip with its own progress, on the right.
        self._work_clips = ClipList(compact=True)
        self._work_clips.setFixedWidth(520)
        self._work_clips_box = QWidget()
        clips_column = QVBoxLayout(self._work_clips_box)
        clips_column.setContentsMargins(0, 0, 0, 0)
        clips_column.addWidget(eyebrow("CLIPS"))
        clips_column.addWidget(self._work_clips)
        row = QHBoxLayout(page)
        row.setContentsMargins(72, 48, 72, 48)
        row.setSpacing(48)
        row.addWidget(left, 1, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._work_clips_box, 0, Qt.AlignmentFlag.AlignVCenter)
        return page

    def _build_review_page(self) -> QWidget:
        # The preview, with what you can do to the shot on screen right under it.
        self._video = PreviewView(self._tools)
        self._video.timeChanged.connect(self._show_time)
        self._caption_left = eyebrow()
        self._now = eyebrow()
        caption = QHBoxLayout()
        caption.addWidget(self._caption_left)
        caption.addStretch(1)
        caption.addWidget(self._now)

        self._previous = button("←", lambda: self._step_shot(-1))
        self._previous.setToolTip("Previous shot")
        self._next = button("→", lambda: self._step_shot(1))
        self._next.setToolTip("Next shot")
        self._trim_buttons: list[QPushButton] = []
        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self._previous)
        controls.addWidget(self._next)
        controls.addSpacing(16)
        for end in (False, True):
            controls.addWidget(eyebrow("END" if end else "START"))
            for step in TRIM_STEPS:
                trim = button(f"{step:+g}S", functools.partial(self._trim, end, step))
                trim.setToolTip(
                    f"{'Lengthen' if (step > 0) == end else 'Shorten'} the shot at its "
                    f"{'end' if end else 'start'} by {abs(step):g} s"
                )
                controls.addWidget(trim)
                self._trim_buttons.append(trim)
            controls.addSpacing(12)
        controls.addStretch(1)
        self._swap_button = button("SWAP SHOT", lambda: self._open_chooser("swap"))
        self._swap_button.setToolTip("Pick another shot from the same clip instead")
        self._remove_shot = button("REMOVE", self._remove_current)
        controls.addWidget(self._swap_button)
        controls.addWidget(self._remove_shot)

        stage = QWidget()
        stage_column = QVBoxLayout(stage)
        stage_column.setContentsMargins(0, 0, 0, 0)
        stage_column.setSpacing(10)
        stage_column.addWidget(self._video, 1)
        stage_column.addLayout(caption)
        stage_column.addLayout(controls)

        # Right: the shots in playing order, or (while choosing) other shots to use.
        self._shot_list = QListWidget()
        self._shot_list.setIconSize(QPixmap(96, THUMB_HEIGHT).size())
        self._shot_list.setWordWrap(True)
        self._shot_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._shot_list.currentRowChanged.connect(self._shot_selected)
        self._shot_list.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self._reorder))
        self._total = eyebrow()
        shots_side = QWidget()
        side = QVBoxLayout(shots_side)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)
        side.addWidget(eyebrow("SHOTS IN PLAYING ORDER   DRAG TO REORDER"))
        side.addWidget(self._shot_list, 1)
        side.addWidget(self._total)
        side.addWidget(button("+  ADD A SHOT", lambda: self._open_chooser("add")))

        self._chooser_title = eyebrow()
        self._alternatives = QListWidget()
        self._alternatives.setIconSize(QPixmap(96, THUMB_HEIGHT).size())
        self._alternatives.setWordWrap(True)
        self._alternatives.currentRowChanged.connect(self._alternative_selected)
        self._use = button("USE THIS SHOT", self._use_alternative, primary=True)
        chooser_buttons = QHBoxLayout()
        chooser_buttons.addWidget(button("CANCEL", self._close_chooser))
        chooser_buttons.addStretch(1)
        chooser_buttons.addWidget(self._use)
        chooser = QWidget()
        chooser_column = QVBoxLayout(chooser)
        chooser_column.setContentsMargins(0, 0, 0, 0)
        chooser_column.setSpacing(8)
        chooser_column.addWidget(self._chooser_title)
        chooser_column.addWidget(self._alternatives, 1)
        chooser_column.addLayout(chooser_buttons)

        self._side = QStackedWidget()
        self._side.addWidget(shots_side)
        self._side.addWidget(chooser)
        self._side.setFixedWidth(400)

        body = QHBoxLayout()
        body.setSpacing(24)
        body.addWidget(stage, 1)
        body.addWidget(self._side)

        self._review_summary = QLabel()
        self._review_summary.setProperty("role", "muted")
        self._review_message = eyebrow()
        top = QWidget()
        column = QVBoxLayout(top)
        column.setContentsMargins(40, 28, 40, 16)
        column.setSpacing(6)
        column.addWidget(eyebrow("STEP 3 OF 4"))
        column.addWidget(heading("Review your montage"))
        column.addWidget(self._review_summary)
        column.addSpacing(14)
        column.addLayout(body, 1)
        column.addWidget(self._review_message)

        self._notes_button = button("", self._show_notes)
        self._notes_button.setProperty("flat", True)
        self._output_label = eyebrow()
        self._export = button("EXPORT MONTAGE  →", self._start_render, primary=True)
        back = button("←  CLIPS", lambda: self._show_page(self.CLIPS))
        bar = self._action_bar(
            [back, self._notes_button],
            [self._output_label, button("CHANGE", self._choose_output), self._export],
        )
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top, 1)
        layout.addWidget(bar)
        return page

    def _build_done_page(self) -> QWidget:
        self._done_title = DotTitle("done", cell=12)
        self._done_text = QLabel()
        self._done_text.setProperty("role", "hero")
        self._done_text.setWordWrap(True)
        self._done_path = eyebrow()
        self._done_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(button("OPEN VIDEO", self._open_output, primary=True))
        buttons.addWidget(button("SHOW IN FOLDER", self._show_output))
        buttons.addWidget(button("←  BACK TO REVIEW", lambda: self._show_page(self.REVIEW)))
        buttons.addWidget(button("NEW MONTAGE", self._new_montage))
        buttons.addStretch(1)
        page = DotGrid()
        self._done_page = page
        column = QVBoxLayout(page)
        column.setContentsMargins(72, 48, 72, 48)
        column.addStretch(2)
        column.addWidget(eyebrow("STEP 4 OF 4"))
        column.addSpacing(16)
        column.addWidget(self._done_title)
        column.addSpacing(24)
        column.addWidget(self._done_text)
        column.addWidget(self._done_path)
        column.addSpacing(24)
        column.addLayout(buttons)
        column.addStretch(3)
        return page

    def _action_bar(self, left: Sequence[QWidget], right: Sequence[QWidget]) -> QFrame:
        """The bottom bar of a step: going back on the left, the next step on the right."""
        bar = QFrame()
        bar.setProperty("role", "footer")
        bar.setFixedHeight(68)
        row = QHBoxLayout(bar)
        row.setContentsMargins(40, 0, 40, 0)
        row.setSpacing(10)
        for widget in left:
            row.addWidget(widget)
        row.addStretch(1)
        for widget in right:
            row.addWidget(widget)
        return bar

    def _build_menus(self) -> None:
        def action(text: str, slot: Callable[[], object], key: object = None) -> QAction:
            item = QAction(text, self)
            if key is not None:
                item.setShortcut(key)  # type: ignore[call-overload]
            item.triggered.connect(slot)
            self.addAction(item)  # shortcuts work with the menu closed
            return item

        menu = QMenu(self)
        self._menu_button.setMenu(menu)
        menu.addAction(action("Add clips…", self._choose_clips, QKeySequence.StandardKey.Open))
        menu.addAction(action("Add a folder of clips…", self._choose_folder))
        menu.addAction(action("Remove all clips", self._clear_clips))
        menu.addSeparator()
        menu.addAction(action("Undo", self._undo, QKeySequence.StandardKey.Undo))
        delete = action("Remove shot", self._remove_current, QKeySequence.StandardKey.Delete)
        self.removeAction(delete)  # only while the shot list has focus
        delete.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self._shot_list.addAction(delete)
        menu.addSeparator()
        menu.addAction(action("Settings…", self._open_settings, "Ctrl+,"))
        menu.addAction(action("Show the log file", self._show_log))
        menu.addAction(action("About FPV Editor", self._about))
        menu.addSeparator()
        menu.addAction(action("Quit", self.close, QKeySequence.StandardKey.Quit))

    # ----- steps -----

    def _show_page(self, page: int) -> None:
        if page != self.REVIEW:
            self._close_chooser()
            self._video.stop()
        self._pages.setCurrentIndex(page)
        if page == self.REVIEW:
            row = self._shot_list.currentRow()
            if row >= 0:
                self._shot_selected(row)
        self._refresh_steps()

    def _go_to_step(self, step: int) -> None:
        """A click on a step in the header."""
        if self._busy():
            return
        if step == 0:
            self._show_page(self.CLIPS)
        elif step == 1:
            self._find_shots()
        elif step == 2 and self._current_shots():
            self._show_page(self.REVIEW)
        elif step == 3 and self._last_output is not None:
            self._show_page(self.DONE)

    def _refresh_steps(self) -> None:
        page = self._pages.currentIndex()
        if page == self.WORK:
            current = 3 if self._task_page == self.REVIEW else 1
        else:
            current = {self.CLIPS: 0, self.REVIEW: 2, self.DONE: 3}[page]
        reachable = {0}
        if self._clips.count():
            reachable.add(1)
        if self._current_shots():
            reachable.add(2)
        if self._last_output is not None:
            reachable.add(3)
        self._steps.show_state(current, set() if self._busy() else reachable)

    # ----- clips -----

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and not self._busy():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self.add_paths([Path(url.toLocalFile()) for url in event.mimeData().urls()])

    def clip_paths(self) -> list[Path]:
        return [Path(self._clips.item(i).data(PATH)) for i in range(self._clips.count())]

    def _clip_items(self, path: Path) -> list[QListWidgetItem]:
        key = path.resolve()
        items = (self._clips.item(i) for i in range(self._clips.count()))
        return [item for item in items if Path(item.data(PATH)).resolve() == key]

    def _clip_probed(self, info: object) -> None:
        if not isinstance(info, MediaInfo):
            raise TypeError(f"expected MediaInfo, got {type(info).__name__}")
        self._infos[info.path] = info
        for item in self._clip_items(info.path):
            item.setData(DETAILS, describe_clip(info))
        self._analysis.add(info)  # start measuring now; Find shots then waits less
        self._refresh_clips()

    def _clip_unreadable(self, path: object, why: str) -> None:
        if isinstance(path, Path):
            self._unreadable[path.resolve()] = why
            for item in self._clip_items(path):
                item.setData(DETAILS, "NOT A READABLE RECORDING")
                item.setToolTip(f"{path}\n{why}")
        self._analysis_changed()

    def _clip_thumbnail(self, path: object, png: bytes) -> None:
        pixmap = thumbnail_pixmap(png)
        if isinstance(path, Path) and pixmap is not None:
            for item in self._clip_items(path):
                item.setData(THUMB, pixmap)

    def _clip_status(self, path: Path) -> tuple[str, float | None, bool]:
        """What a clip's card says on the right: (text, progress share or None, problem)."""
        key = path.resolve()
        if key in self._unreadable:
            return "CAN'T READ", None, True
        if key not in self._infos:
            return "READING", None, False
        state = self._analysis.state(key)
        if state is None:
            return "NOT ANALYSED YET", None, False
        if state.error is not None:
            return "ANALYSIS FAILED", None, True
        if state.result is not None:
            if key in self._clip_notes:
                return self._clip_notes[key], None, False
            clip = state.result.clip
            if clip is None:
                return "NOTHING USABLE", None, True
            return f"READY   {clock_short(clip.analysed_duration)} FLYING", None, False
        if not state.started:
            return "QUEUED", None, False
        return f"ANALYSING {state.share:.0%}", state.share, False

    def _refresh_clips(self) -> None:
        """Update every clip card and the summary under the list."""
        ready = 0
        footage = 0.0
        for i in range(self._clips.count()):
            item = self._clips.item(i)
            path = Path(item.data(PATH))
            text, share, problem = self._clip_status(path)
            item.setData(STATUS, text)
            item.setData(SHARE, share)
            item.setData(PROBLEM, problem)
            state = self._analysis.state(path)
            ready += state is not None and state.finished
            info = self._infos.get(path.resolve())
            footage += (info.duration or 0.0) if info is not None else 0.0
        self._mirror_clips()
        count = self._clips.count()
        parts = [plural(count, "clip"), f"{clock_short(footage)} of footage"]
        if count:
            parts.append("all analysed" if ready == count else f"{ready} of {count} analysed")
        self._clips_summary.setText("   ".join(parts).upper())

    def add_paths(self, paths: list[Path]) -> None:
        if self._busy():
            return
        known = {p.resolve() for p in self.clip_paths()}
        added = 0
        for path in video_files(paths):
            if path.resolve() in known:
                continue
            known.add(path.resolve())
            item = QListWidgetItem(path.name)
            item.setData(PATH, str(path))
            item.setToolTip(str(path))
            self._clips.addItem(item)
            self._reader.request(path)
            added += 1
        if added:
            self._clips_changed()
            self._show_page(self.CLIPS)

    def _choose_clips(self) -> None:
        start = str(self._settings.clip_folder or "")
        names, _ = QFileDialog.getOpenFileNames(
            self, "Add clips", start, "Recordings (*.ts *.mp4 *.TS *.MP4)"
        )
        if names:
            self._settings.clip_folder = Path(names[0]).parent
            self.add_paths([Path(n) for n in names])

    def _choose_folder(self) -> None:
        start = str(self._settings.clip_folder or "")
        name = QFileDialog.getExistingDirectory(self, "Add the clips in a folder", start)
        if name:
            self._settings.clip_folder = Path(name)
            self.add_paths([Path(name)])

    def _clear_clips(self) -> None:
        if self._busy():
            return
        self._clips.clear()
        self._clips_changed()
        self._show_page(self.CLIPS)

    def _remove_clips(self) -> None:
        if self._busy():
            return
        for item in self._clips.selectedItems():
            self._clips.takeItem(self._clips.row(item))
        self._clips_changed()

    def _new_montage(self) -> None:
        self._last_output = None
        self._clear_clips()

    def _clips_changed(self) -> None:
        if self._plan is not None:
            self._plan = None
            self._shots = None
            self._set_notes(())
            self._clip_notes.clear()
        self._refresh()

    def _length_changed(self, seconds: float) -> None:
        self._settings.length_s = seconds

    def _mirror_clips(self) -> None:
        """The progress screen's list shows the same clips, cards and progress."""
        target = self._work_clips
        if target.count() != self._clips.count():
            target.clear()
            for _ in range(self._clips.count()):
                target.addItem(QListWidgetItem())
        for i in range(self._clips.count()):
            source, mirror = self._clips.item(i), target.item(i)
            for role in (PATH, DETAILS, STATUS, SHARE, PROBLEM, THUMB):
                mirror.setData(role, source.data(role))
        rows = min(max(1, target.count()), 9)  # taller lists scroll
        target.setFixedHeight(rows * target.card.height + 4)

    def _annotate_clips(self, plan: MontagePlan) -> None:
        """Show what each clip gave: its usable length and shots, or why it gave none."""
        used = {clip.index: clip for clip in plan.clips}
        shots = {index: sum(1 for s in plan.shots if s.clip == index) for index in used}
        self._clip_notes.clear()
        for index, info in enumerate(plan.infos):
            notes = [n for n in plan.warnings if n.startswith(info.path.name)]
            if index in used:
                usable = clock_short(used[index].analysed_duration)
                note = f"{usable} FLYING   {plural(shots[index], 'shot').upper()}"
            else:
                note = "NOTHING USABLE"
            self._clip_notes[info.path.resolve()] = note
            for item in self._clip_items(info.path):
                item.setToolTip("\n".join([str(info.path), *notes]))
        self._refresh_clips()

    def _set_notes(self, notes: Sequence[str]) -> None:
        self._notes = tuple(notes)
        self._notes_button.setText(f"NOTES ({len(self._notes)})")
        self._notes_button.setVisible(bool(self._notes))

    def _show_notes(self) -> None:
        QMessageBox.information(self, "Notes", "\n\n".join(self._notes))

    # ----- finding shots -----

    def _find_shots(self) -> None:
        """Wait for the clips' background analysis to finish, then pick the shots."""
        if not self._clips.count() or self._busy():
            return
        self._analysis.retry_failed()
        for path in self.clip_paths():
            info = self._infos.get(path.resolve())
            if info is not None:
                self._analysis.add(info)  # again after a cancel; no-op when done or running
        log.info("finding shots in %d clips", self._clips.count())
        self._task_page = self.CLIPS
        self._finding = (time.monotonic(), self._analysed_share()[0])
        self._show_work("STEP 2 OF 4", "Finding the best shots")
        self._analysis_changed()

    def _analysed_share(self) -> tuple[float, str]:
        """Share of the clips' footage analysed so far (by length), and what is going on."""
        total = done = 0.0
        running = waiting = 0
        for path in self.clip_paths():
            key = path.resolve()
            info = self._infos.get(key)
            weight = (info.duration or 60.0) if info is not None else 60.0
            total += weight
            state = self._analysis.state(key)
            if key in self._unreadable or (state is not None and state.finished):
                done += weight
            elif state is not None:
                done += weight * state.share
                running += state.started
                waiting += not state.started
            else:
                waiting += 1
        if running:
            text = f"Analysing {plural(running, 'clip')} at a time"
            if waiting:
                text += f", {waiting} waiting"
        elif waiting:
            text = "Reading the clips"
        else:
            text = "Picking the shots"
        return (done / total if total else 1.0), text

    def _analysis_changed(self) -> None:
        self._refresh_clips()
        if self._finding is None:
            return
        share, text = self._analysed_share()
        started, share_then = self._finding
        progress = (share - share_then) / max(1e-9, 1 - share_then)
        self._show_progress(text, share, time_left(time.monotonic() - started, progress))
        if share >= 1.0:
            self._finish_finding()

    def _finish_finding(self) -> None:
        infos: list[MediaInfo] = []
        results = []
        warnings: dict[int, list[str]] = {}
        for path in self.clip_paths():
            key = path.resolve()
            if key in self._unreadable:
                warnings[-1] = [*warnings.get(-1, []), f"{path.name}: {self._unreadable[key]}"]
                continue
            state = self._analysis.state(key)
            if state is None or not state.finished:
                return  # not ready after all (e.g. added again after a cancel)
            index = len(infos)
            infos.append(state.info)
            if state.result is not None:
                results.append(state.result.moved_to(index))
            else:
                warnings[index] = [f"{path.name}: could not be analysed ({state.error})"]
        config = dataclasses.replace(PipelineConfig().montage, length_s=self._length.value())
        plan = combine_clips(infos, results, config, warnings)
        self._end_finding()
        self._plan_ready(plan)

    def _end_finding(self) -> None:
        self._finding = None
        self._dot.set_busy(False)
        self._refresh_buttons()
        self._refresh_steps()

    def _plan_ready(self, result: object) -> None:
        if not isinstance(result, MontagePlan):
            raise TypeError(f"expected a MontagePlan, got {type(result).__name__}")
        self._plan = result
        durations = {clip.index: clip.analysed_duration for clip in result.clips}
        self._shots = ShotList(result.shots, result.candidates, durations)
        self._output, self._force = None, False
        self._annotate_clips(result)
        self._set_notes(result.warnings)
        self._last_summary = (
            f"Picked from {result.footage_duration / 60:.0f} min of usable flying in "
            f"{plural(len(result.clips), 'clip')}."
        )
        self._refresh()
        if result.shots:
            self._shot_list.setCurrentRow(0)
            self._show_page(self.REVIEW)
        else:
            self._show_page(self.CLIPS)
            QMessageBox.information(
                self,
                "No shots found",
                "No flip, spin or close pass was found in these clips.\n\n"
                + "\n".join(result.warnings),
            )

    # ----- shots -----

    def _info(self, shot: Shot) -> MediaInfo:
        if self._plan is None:
            raise RuntimeError("no montage plan")
        return self._plan.infos[shot.clip]

    def _path(self, shot: Shot) -> Path:
        return self._info(shot).path

    def _thumb_key(self, shot: Shot) -> tuple[Path, float]:
        seconds = shot.moments[0].time if shot.moments else (shot.start + shot.end) / 2
        return (self._path(shot), round(seconds, 2))

    def _icon(self, shot: Shot) -> QIcon:
        key = self._thumb_key(shot)
        if key in self._thumbs:
            return QIcon(self._thumbs[key])
        if key not in self._requested:
            self._requested.add(key)
            self._loader.request(self._info(shot), key[1])
        return QIcon()

    def _thumbnail_ready(self, key: object, png: bytes) -> None:
        pixmap = QPixmap()
        if pixmap.loadFromData(png) and isinstance(key, tuple):
            self._thumbs[key] = pixmap
            self._refresh_icons()

    def _shot_text(self, number: int | None, shot: Shot) -> str:
        prefix = f"{number:02d}   " if number is not None else ""
        return (
            f"{prefix}{describe(shot)}\n{self._path(shot).name}   "
            f"{clock(shot.start)}  ·  {shot.duration:.1f} s"
        )

    def _refresh(self, *, preview: bool = True) -> None:
        """Bring every screen up to date; with ``preview`` the shot on screen restarts."""
        row = self._shot_list.currentRow()
        self._shot_list.blockSignals(True)
        self._shot_list.clear()
        for number, shot in enumerate(self._current_shots(), start=1):
            item = QListWidgetItem(self._icon(shot), self._shot_text(number, shot))
            item.setData(Qt.ItemDataRole.UserRole, number - 1)
            self._shot_list.addItem(item)
        self._shot_list.blockSignals(False)
        count = self._shot_list.count()
        if count:
            self._shot_list.blockSignals(not preview or not self._on_review())
            self._shot_list.setCurrentRow(min(max(row, 0), count - 1))
            self._shot_list.blockSignals(False)
        else:
            self._stop_preview()
            if self._on_review():
                self._show_page(self.CLIPS)
        clips = self._clips.count()
        self._clip_stack.setCurrentIndex(1 if clips else 0)
        self._clips_bar.setVisible(bool(clips))
        self._refresh_clips()
        self._refresh_total()
        self._refresh_output()
        self._refresh_buttons()
        self._refresh_steps()

    def _on_review(self) -> bool:
        return self._pages.currentIndex() == self.REVIEW

    def _refresh_total(self) -> None:
        shots = self._current_shots()
        total = sum(shot.duration for shot in shots)
        self._total.setText(f"{plural(len(shots), 'SHOT').upper()}   {clock(total)} TOTAL")
        self._review_summary.setText(
            f"{plural(len(shots), 'shot')}, {clock(total)} long. {self._last_summary} Click a "
            "shot to watch it; swap, trim or remove the ones you don't like."
        )

    def _refresh_icons(self) -> None:
        for widget, shots in (
            (self._shot_list, self._current_shots()),
            (self._alternatives, self._alts),
        ):
            for i, shot in enumerate(shots):
                if i < widget.count():
                    widget.item(i).setIcon(self._icon(shot))

    def _current_shots(self) -> tuple[Shot, ...]:
        return self._shots.shots if self._shots is not None else ()

    def _refresh_buttons(self) -> None:
        busy = self._busy()
        index = self._selected_index()
        # While choosing another shot, only the chooser's buttons act.
        selected = index is not None and self._choosing is None
        for trim in self._trim_buttons:
            trim.setEnabled(selected and not busy)
        self._remove_shot.setEnabled(selected and not busy)
        self._swap_button.setEnabled(selected and not busy)
        count = len(self._current_shots())
        self._previous.setEnabled(selected and index is not None and index > 0)
        self._next.setEnabled(selected and index is not None and index < count - 1)
        self._find.setEnabled(self._clips.count() > 0 and not busy)
        self._export.setEnabled(bool(count) and not busy)
        self._use.setEnabled(self._alternatives.currentRow() >= 0 and not busy)
        drag = (
            QAbstractItemView.DragDropMode.NoDragDrop
            if busy
            else QAbstractItemView.DragDropMode.InternalMove
        )
        self._clips.setDragDropMode(drag)
        self._shot_list.setDragDropMode(drag)

    def _selected_index(self) -> int | None:
        row = self._shot_list.currentRow()
        return row if row >= 0 and self._shots is not None else None

    def _shot_selected(self, row: int) -> None:
        if self._shots is not None and 0 <= row < len(self._shots.shots) and self._on_review():
            self._close_chooser()
            self._preview(self._shots.shots[row])
        self._refresh_buttons()

    def _step_shot(self, step: int) -> None:
        index = self._selected_index()
        if index is not None:
            self._shot_list.setCurrentRow(min(max(0, index + step), len(self._current_shots()) - 1))

    def _reorder(self) -> None:
        if self._shots is None or self._busy():
            return
        order = [
            int(self._shot_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._shot_list.count())
        ]
        moved_to = self._shot_list.currentRow()
        self._shots.reorder(order)
        self._refresh()
        self._shot_list.setCurrentRow(moved_to)

    def _trim(self, end: bool, step: float) -> None:
        index = self._selected_index()
        if index is None or self._shots is None or self._busy():
            return
        if end:
            self._shots.trim(index, end=step)
        else:
            self._shots.trim(index, start=step)
        self._refresh(preview=False)
        shot = self._shots.shots[index]
        if self._on_review():
            # An end change plays the last seconds; a start change replays from the start.
            self._preview(shot, from_seconds=max(shot.start, shot.end - 3) if end else None)

    def _remove_current(self) -> None:
        index = self._selected_index()
        if index is not None and self._shots is not None and not self._busy():
            self._shots.remove(index)
            self._message("Shot removed. Ctrl+Z brings it back.")
            self._refresh()

    def _undo(self) -> None:
        if self._shots is not None and self._shots.can_undo and not self._busy():
            self._shots.undo()
            self._refresh()

    def _message(self, text: str) -> None:
        self._review_message.setText(text.upper())

    # ----- choosing another shot -----

    def _open_chooser(self, mode: str) -> None:
        if self._shots is None or self._busy():
            return
        index = self._selected_index()
        if mode == "swap":
            if index is None:
                return
            self._alts = self._shots.alternatives(index)
            title = f"SWAP SHOT {index + 1} FOR ONE OF THESE (SAME CLIP)"
        else:
            self._alts = self._shots.alternatives(limit=40)
            title = "ADD ONE OF THESE SHOTS (ALL CLIPS)"
        self._choosing = mode
        self._refresh_buttons()
        self._chooser_title.setText(title if self._alts else "NO OTHER SHOTS TO OFFER")
        self._alternatives.blockSignals(True)
        self._alternatives.clear()
        for shot in self._alts:
            self._alternatives.addItem(
                QListWidgetItem(self._icon(shot), self._shot_text(None, shot))
            )
        self._alternatives.blockSignals(False)
        self._use.setText("USE INSTEAD" if mode == "swap" else "ADD TO MONTAGE")
        self._side.setCurrentIndex(1)
        if self._alts:
            self._alternatives.setCurrentRow(0)
        self._refresh_buttons()

    def _close_chooser(self) -> None:
        if self._choosing is None:
            return
        self._choosing = None
        self._side.setCurrentIndex(0)
        self._refresh_buttons()
        index = self._selected_index()
        if index is not None and self._on_review():
            self._preview(self._current_shots()[index])

    def _alternative_selected(self, row: int) -> None:
        if 0 <= row < len(self._alts):
            self._preview(self._alts[row])
        self._refresh_buttons()

    def _use_alternative(self) -> None:
        row = self._alternatives.currentRow()
        if self._shots is None or not 0 <= row < len(self._alts) or self._choosing is None:
            return
        chosen = self._alts[row]
        try:
            if self._choosing == "swap":
                index = self._selected_index()
                if index is None:
                    return
                self._shots.replace(index, chosen)
                position = index
                self._message(f"Shot {index + 1} swapped. Ctrl+Z undoes it.")
            else:
                position = self._shots.add(chosen)
                self._message(f"Shot added as number {position + 1}. Ctrl+Z undoes it.")
        except ValueError as exc:
            self._message(f"Not possible: {exc}.")
            return
        self._choosing = None
        self._side.setCurrentIndex(0)
        self._refresh()
        self._shot_list.setCurrentRow(position)

    # ----- preview -----

    def _preview(self, shot: Shot, from_seconds: float | None = None) -> None:
        self._loop = (shot.start, shot.end)
        shots = self._current_shots()
        if shot in shots and self._choosing is None:
            number = shots.index(shot) + 1
            self._caption_left.setText(
                f"FIG. {number:02d}   SHOT {number} OF {len(shots)}   {describe(shot).upper()}"
            )
        else:
            self._caption_left.setText(f"PREVIEW   NOT IN THE MONTAGE   {describe(shot).upper()}")
        self._video.play(self._info(shot), shot.start, shot.end, from_seconds)

    def _show_time(self, seconds: float) -> None:
        if self._loop is not None:
            start, end = self._loop
            self._now.setText(f"{clock(seconds)}   ({clock(start)} to {clock(end)})")

    def _stop_preview(self) -> None:
        self._loop = None
        self._caption_left.setText("")
        self._video.clear_picture()
        self._now.setText("")

    # ----- export -----

    def _default_output(self) -> Path | None:
        if self._plan is None or not self._plan.infos:
            return None
        first = self._plan.infos[0].path
        folder = self._settings.output_folder
        try:
            if folder is not None and folder.is_dir():
                return default_output_path(folder / first.name, MONTAGE_SUFFIX)
            return resolve_output_path(
                [info.path for info in self._plan.infos], None, False, suffix=MONTAGE_SUFFIX
            )
        except FPVEditError:
            return None

    def _refresh_output(self) -> None:
        output = self._output or self._default_output()
        self._output_label.setText(f"SAVES AS  {output.name}" if output else "")
        self._output_label.setToolTip(str(output) if output else "")

    def _choose_output(self) -> None:
        start = str(self._output or self._default_output() or "")
        name, _ = QFileDialog.getSaveFileName(self, "Save montage as", start, "MP4 video (*.mp4)")
        if name:
            path = Path(name)
            if path.suffix.lower() != ".mp4":
                path = path.with_suffix(".mp4")
            # The dialog asks before returning an existing file, but not about the name
            # changed here.
            if path.exists() and path != Path(name):
                answer = QMessageBox.question(
                    self, "Replace file?", f"{path} already exists. Replace it?"
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self._output, self._force = path, path.exists()
            self._refresh_output()

    def _start_render(self) -> None:
        if self._plan is None or self._shots is None or self._busy():
            return
        plan, shots = self._plan, self._shots.shots
        tools, force, device = self._tools, self._force, self._settings.encoder
        try:
            final = resolve_output_path(
                [info.path for info in plan.infos],
                self._output or self._default_output(),
                force,
                suffix=MONTAGE_SUFFIX,
            )
        except FPVEditError as exc:
            QMessageBox.warning(self, "Cannot export", str(exc))
            return

        def work(progress: Progress) -> object:
            started = time.monotonic()
            progress("Checking for NVIDIA GPU encoding", 0.0)
            choices = encoders(tools, device, EncodeConfig())
            log.info("rendering %d shots to %s", len(shots), final)
            length, used = with_fallback(
                choices,
                lambda encode: render_montage(
                    tools, plan.clips, shots, final, force=force, encode=encode, progress=progress
                ),
            )
            where = "the GPU (NVENC)" if used == NVENC else "the CPU"
            if used != choices[0]:
                where += " (GPU encoding failed, so the CPU was used)"
            return final, length, time.monotonic() - started, where

        self._run(work, self._rendered, self.REVIEW, "STEP 4 OF 4", "Exporting your montage")

    def _rendered(self, result: object) -> None:
        if not isinstance(result, tuple):
            raise TypeError(f"expected the render result, got {type(result).__name__}")
        final, length, elapsed, where = result
        self._last_output = final
        self._output, self._force = None, False
        self._done_text.setText(
            f"Your montage is ready: {clock(length)} long, "
            f"{plural(len(self._current_shots()), 'shot')}. "
            f"Made on {where} in {elapsed:.0f} s and checked frame by frame."
        )
        self._done_path.setText(str(final))
        self._refresh()
        self._show_page(self.DONE)

    def _open_output(self) -> None:
        if self._last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output)))

    def _show_output(self) -> None:
        if self._last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output.parent)))

    def _apply_theme(self, name: str) -> None:
        palette = theme.THEMES.get(name, theme.DARK)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            theme.apply(app, palette, self._fonts)
        for grid in (self._drop_zone, self._list_page, self._work_page, self._done_page):
            grid.set_palette(palette)
        for clips in (self._clips, self._work_clips):
            clips.card.set_theme(palette, self._fonts)
            clips.viewport().update()
        self._title.set_colour(palette.text)
        self._done_title.set_colour(palette.text)
        self._dot.set_colours(palette.faint, palette.accent)
        self._theme_button.setText("LIGHT" if palette is theme.DARK else "DARK")
        self._settings.theme = palette.name

    def _toggle_theme(self) -> None:
        self._apply_theme("light" if self._settings.theme == "dark" else "dark")

    # ----- background work -----

    def _busy(self) -> bool:
        return self._task is not None or self._finding is not None

    def _run(
        self,
        work: Callable[[Progress], object],
        done: Callable[[object], None],
        back_to: int,
        step: str,
        title: str,
    ) -> None:
        task = Task(work, self)
        task.progressed.connect(self._progressed)
        task.succeeded.connect(done)
        task.failed.connect(self._failed)
        task.finished.connect(self._task_finished)
        self._task = task
        self._task_page = back_to
        self._show_work(step, title)
        task.start()

    def _show_work(self, step: str, title: str) -> None:
        finding = self._finding is not None
        self._work_clips_box.setVisible(finding)
        self._work_step.setText(step)
        self._work_title.setText(title)
        self._show_progress("Starting…", 0.0, "")
        self._dot.set_busy(True)
        self._cancel.setEnabled(True)
        self._video.stop()
        self._pages.setCurrentIndex(self.WORK)
        self._refresh_buttons()
        self._refresh_steps()

    def _show_progress(self, text: str, share: float, left: str) -> None:
        self._progress.setValue(round(1000 * share))
        self._percent.setText(f"{share:.0%}")
        self._work_status.setText(text)
        self._work_left.setText(left.upper())

    def _progressed(self, text: str, share: float) -> None:
        left = time_left(time.monotonic() - self._task.started_at, share) if self._task else ""
        self._show_progress(text, share, left)

    def _failed(self, message: str) -> None:
        self._show_page(self._task_page)
        if message != "Cancelled.":
            QMessageBox.warning(self, "Something went wrong", message)

    def _task_finished(self) -> None:
        if self._task is not None:
            self._task.deleteLater()
        self._task = None
        self._dot.set_busy(False)
        self._refresh_buttons()
        self._refresh_steps()

    def _cancel_task(self) -> None:
        if self._finding is not None:
            self._analysis.cancel()
            self._end_finding()
            self._show_page(self._task_page)
            return
        if self._task is not None:
            self._task.cancel()
            self._cancel.setEnabled(False)
            self._work_status.setText("Stopping…")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._task is not None:
            answer = QMessageBox.question(
                self, "Stop and quit?", "FPV Editor is still working. Stop and quit?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._task.cancel()
            QApplication.processEvents()
            self._task.wait()
        self._settings.geometry = self.saveGeometry()
        self._analysis.shutdown()
        self._reader.stop()
        self._loader.stop()
        self._video.stop()
        super().closeEvent(event)

    # ----- menu -----

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self._cache_dir, self)
        if dialog.exec():
            self._length.setValue(self._settings.length_s)
            self._refresh_output()

    def _show_log(self) -> None:
        if self._log_file is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._log_file.parent)))
        else:
            QMessageBox.information(self, "Log file", "The log file could not be created.")

    def _about(self) -> None:
        try:
            version = run_tool([self._tools.ffmpeg, "-version"], action="ffmpeg -version")
            first = version.stdout.splitlines()[0] if version.stdout else ""
        except FPVEditError:
            first = ""
        QMessageBox.about(self, "About FPV Editor", about_text(first))


def data_folder() -> Path | None:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base) if base else None


def cache_folder() -> Path | None:
    base = data_folder()
    return base / "measurements" if base else None


def ffmpeg_folders(settings: AppSettings) -> list[Path]:
    """The FFmpeg folder the user chose, looked in before PATH (a copy shipped with the
    standalone app is found by `find_tools` first anyway)."""
    folder = settings.ffmpeg_folder
    return [folder] if folder is not None and folder.is_dir() else []


def use_folders_first(folders: Sequence[Path]) -> None:
    """Put ``folders`` at the front of this process's PATH (for FFmpeg)."""
    if folders:
        rest = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join([*(str(f) for f in folders), rest])


def locate_tools(settings: AppSettings) -> Tools | None:
    """FFmpeg, asking the user where it is until it is found or they give up."""
    use_folders_first(ffmpeg_folders(settings))
    while True:
        try:
            return find_tools()
        except FPVEditError as exc:
            log.warning("FFmpeg not found: %s", exc)
            folder = ask_for_ffmpeg(None)
            if folder is None:
                return None
            settings.ffmpeg_folder = folder
            use_folders_first([folder])


def run(argv: Sequence[str]) -> int:
    app = QApplication(list(argv))
    app.setApplicationName("fpv-editor")
    app.setApplicationDisplayName("FPV Editor")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    base = data_folder()
    log_file = start_logging(base / "logs") if base else None
    log.info("FPV Editor %s starting", __version__)
    settings = AppSettings()
    theme.apply(app, theme.THEMES.get(settings.theme, theme.DARK), theme.load_fonts())
    tools = locate_tools(settings)
    if tools is None:
        return 1
    window = MainWindow(tools, cache_folder(), settings, log_file)
    window.add_paths([Path(arg) for arg in argv[1:]])
    window.show()
    return app.exec()
