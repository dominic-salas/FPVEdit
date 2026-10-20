"""The app window (Qt). Slow work runs in background threads: analysis, thumbnails, render.

Nothing leaves this computer: clips are read in place, measurements are cached in the
user's local app-data folder, and the montage is written where the user chooses.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QStandardPaths, Qt, QThread, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QPainter,
    QPaintEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
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

from fpvedit import __version__, scenery
from fpvedit.app import theme
from fpvedit.app.analyser import BackgroundAnalysis
from fpvedit.app.cliplist import (
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
from fpvedit.app.clips import clock as clock_short
from fpvedit.app.clips import default_output_folder, describe_clip, is_removable
from fpvedit.app.dialogs import ExportDialog, SettingsDialog, about_text, ask_for_ffmpeg
from fpvedit.app.logs import start_logging
from fpvedit.app.look import DotTitle, StatusDot, app_icon
from fpvedit.app.model import ShotList
from fpvedit.app.ratepage import RatePage
from fpvedit.app.review import ReviewPage, describe_music
from fpvedit.app.settings import AppSettings
from fpvedit.app.widgets import button, plural, timecode
from fpvedit.errors import CancelledError, FPVEditError
from fpvedit.media import MediaInfo, Tools, find_tools, run_tool
from fpvedit.montage import Shot, pick_shots
from fpvedit.pipeline import (
    MONTAGE_SUFFIX,
    MontagePlan,
    PipelineConfig,
    Progress,
    combine_clips,
    keep_rated,
    render_montage,
)
from fpvedit.render import (
    NVENC,
    EncodeConfig,
    default_output_path,
    encoders,
    resolve_output_path,
    with_fallback,
)

VIDEO_SUFFIXES = (".ts", ".mp4")
OUTPUT_MARKERS = (MONTAGE_SUFFIX, "_highlight")  # our own renders, skipped in dropped folders
log = logging.getLogger("fpvedit.app")


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


def time_left(elapsed: float, share: float) -> str:
    """'about 2 min left' from the time so far and the share done ('' while too early)."""
    if share < 0.03 or elapsed < 3:
        return ""
    seconds = elapsed * (1 - share) / share
    if seconds < 60:
        return "less than a minute left"
    return f"about {round(seconds / 60)} min left"


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


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "heading")
    return label


def eyebrow(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "eyebrow")
    return label


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

    CLIPS, WORK, REVIEW, DONE, RATE = range(5)

    def __init__(
        self,
        tools: Tools,
        cache_dir: Path | None,
        settings: AppSettings | None = None,
        log_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"FPVEdit {__version__}")
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
        self._open_after = False  # open the folder once the export is done
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

        self._build()
        self._build_menus()
        self._apply_theme(self._settings.theme)
        geometry = self._settings.geometry
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._refresh()

    # ----- layout -----

    def _build(self) -> None:
        logo = QLabel("FPV<span style='color:#e5322d'>Edit</span>")
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
        self._rate = RatePage(self._tools, self._review.ratings)
        self._rate.back.connect(lambda: self._show_page(self.CLIPS))
        self._pages.addWidget(self._rate)
        self._rated_clips: set[Path] = set()  # clips whose shots went to the rating screen

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
        self._title = DotTitle("fpvedit", cell=10)
        intro = QLabel(
            "Drop your FPV recordings (.ts or .mp4) or a whole folder anywhere on this "
            "window. FPVEdit finds the best flips, spins and close passes and cuts them "
            "into one montage. You check it before anything is saved."
        )
        intro.setProperty("role", "hero")
        intro.setWordWrap(True)
        intro.setMaximumWidth(560)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(button("ADD CLIPS", self._choose_clips, primary=True))
        buttons.addWidget(button("ADD A FOLDER", self._choose_folder))
        buttons.addSpacing(24)
        self._quick_rate = button("★  JUST RATE SHOTS", self._rate_folder)
        self._quick_rate.setToolTip("Pick a folder of recordings and rate their shots, no montage")
        buttons.addWidget(self._quick_rate)
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
        self._find = button("FIND SHOTS  →", self._find_shots, primary=True)
        self._rate_button = button("RATE SHOTS", self._open_rating)
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
        self._clips_bar = self._action_bar(
            [
                eyebrow(
                    "EVERY FLIP, SPIN AND CLOSE PASS FROM EVERY CLIP. REMOVE ANY YOU DON'T WANT."
                )
            ],
            [self._rate_button, self._find],
        )
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
        base = data_folder()
        mine = base / "my-ratings.jsonl" if base is not None else None
        self._review = ReviewPage(self._tools, self._settings, ratings_file=mine)
        self._review.changed.connect(self._review_changed)

        self._notes_button = button("", self._show_notes)
        self._notes_button.setProperty("flat", True)
        self._output_label = eyebrow()
        self._export = button("EXPORT MONTAGE  →", self._start_render, primary=True)
        back = button("←  CLIPS", lambda: self._show_page(self.CLIPS))
        bar = self._action_bar(
            [back, self._notes_button],
            [self._output_label, self._export],
        )
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._review, 1)
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
        menu.addSeparator()
        menu.addAction(action("Settings…", self._open_settings, "Ctrl+,"))
        menu.addAction(action("Show the log file", self._show_log))
        self._rate_action = action("Rate shots…", self._open_rating, "Ctrl+R")
        menu.addAction(self._rate_action)
        menu.addAction(action("Show my shot ratings", self._show_my_ratings))
        menu.addAction(action("About FPVEdit", self._about))
        menu.addSeparator()
        menu.addAction(action("Quit", self.close, QKeySequence.StandardKey.Quit))

    # ----- steps -----

    def _show_page(self, page: int) -> None:
        if page == self.CLIPS:
            self._refresh_clips()  # catch up on progress made while another page was shown
        if page != self.REVIEW:
            self._review.deactivate()
        if page != self.RATE:
            self._rate.stop()
        self._pages.setCurrentIndex(page)
        if page == self.REVIEW:
            self._review.activate()
        elif page == self.RATE:
            self._rate.start()
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
            current = {self.CLIPS: 0, self.REVIEW: 2, self.DONE: 3, self.RATE: 0}[page]
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

    def _rate_folder(self) -> None:
        """Straight to rating: the clips of a folder (the last one used is offered first)."""
        start = str(self._settings.clip_folder or "")
        name = QFileDialog.getExistingDirectory(self, "Rate the shots of the clips in", start)
        if not name:
            return
        self._settings.clip_folder = Path(name)
        self.add_paths([Path(name)])
        self._open_rating()

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
            self._review.set_montage((), None)
        self._refresh()

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

    def _open_rating(self) -> None:
        """The rating screen: shots of the clips on step 1, one at a time, no montage."""
        if self._busy():
            return
        if not self._clips.count():
            QMessageBox.information(
                self, "Rate shots", "Add some clips first; their shots will come up to rate."
            )
            return
        for path in self.clip_paths():
            info = self._infos.get(path.resolve())
            if info is not None:
                self._analysis.add(info)
        self._show_page(self.RATE)
        self._feed_rating()

    def _feed_rating(self) -> None:
        """Give the rating screen the shots of every clip analysed since last time."""
        waiting = False
        for path in self.clip_paths():
            key = path.resolve()
            state = self._analysis.state(key)
            if state is None or not state.finished:
                waiting = waiting or key not in self._unreadable
                continue
            if key in self._rated_clips or state.result is None:
                continue
            self._rated_clips.add(key)
            result = state.result.moved_to(0)
            shots = pick_shots(result.candidates, 1, PipelineConfig().montage)
            self._rate.add_shots(state.info, list(shots))
        self._rate.set_waiting(waiting)

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
        if self._pages.currentIndex() == self.RATE:
            self._feed_rating()  # the clip cards are not on screen: keep the player smooth
            return
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
        config = PipelineConfig().montage  # every distinct shot; no target length
        plan = combine_clips(infos, results, config, warnings)
        self._end_finding()
        learn = self._settings.learn_taste
        min_stars = self._settings.min_stars if learn else 1
        ratings = self._review.ratings if learn else None
        if not plan.shots or not scenery.available():
            self._plan_ready(keep_rated(self._tools, plan, min_stars=min_stars, ratings=ratings))
            return
        tools, cache = self._tools, self._cache_dir

        def work(progress: Progress) -> object:
            progress("Rating the shots (with the image model)", 0.0)
            return keep_rated(
                tools,
                plan,
                min_stars=min_stars,
                cache_dir=cache,
                ratings=ratings,
                progress=progress,
            )

        self._run(work, self._plan_ready, self.CLIPS, "STEP 2 OF 4", "Finding the best shots")

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
        if result.left_out:
            self._last_summary += (
                f" Left out {plural(len(result.left_out), 'shot')} rated below "
                f"{self._settings.min_stars}★; Add a shot brings them back."
            )
        self._review.set_montage(result.infos, self._shots, self._last_summary)
        self._refresh()
        if result.shots:
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

    def _refresh(self) -> None:
        """Bring every screen up to date."""
        if not self._current_shots() and self._on_review():
            self._show_page(self.CLIPS)
        clips = self._clips.count()
        self._clip_stack.setCurrentIndex(1 if clips else 0)
        self._clips_bar.setVisible(bool(clips))
        self._refresh_clips()
        self._refresh_output()
        self._refresh_buttons()
        self._refresh_steps()

    def _on_review(self) -> bool:
        return self._pages.currentIndex() == self.REVIEW

    def _current_shots(self) -> tuple[Shot, ...]:
        return self._shots.shots if self._shots is not None else ()

    def _refresh_buttons(self) -> None:
        busy = self._busy()
        self._find.setEnabled(self._clips.count() > 0 and not busy)
        self._export.setEnabled(bool(self._current_shots()) and not busy)
        drag = (
            QAbstractItemView.DragDropMode.NoDragDrop
            if busy
            else QAbstractItemView.DragDropMode.InternalMove
        )
        self._clips.setDragDropMode(drag)

    def _review_changed(self) -> None:
        """An edit on the review page."""
        self._refresh()
        self._refresh_buttons()
        self._refresh_steps()

    def _undo(self) -> None:
        if self._on_review() and not self._busy():
            self._review.undo()

    # ----- export -----

    def _default_output(self) -> Path | None:
        if self._plan is None or not self._plan.infos:
            return None
        first = self._plan.infos[0].path
        sources = [info.path for info in self._plan.infos]
        folder = default_output_folder(self._settings.output_folder, sources, videos_folder())
        try:
            return default_output_path(folder / first.name, MONTAGE_SUFFIX)
        except FPVEditError:
            return None

    def _refresh_output(self) -> None:
        output = self._output or self._default_output()
        text = f"SAVES TO  {output}" if output else ""
        metrics = self._output_label.fontMetrics()
        self._output_label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, 560))
        tip = f"{output}\nYou can change it when you click Export." if output else ""
        self._output_label.setToolTip(tip)

    def _start_render(self) -> None:
        """Ask where and how (the export window), then render."""
        if self._plan is None or self._shots is None or self._busy():
            return
        default = self._output or self._default_output()
        if default is None:
            QMessageBox.warning(self, "Cannot export", "No place to save the montage was found.")
            return
        music = self._review.music
        total = sum(shot.duration for shot in self._shots.shots)
        fit = describe_music(music, total).capitalize()
        default.parent.mkdir(parents=True, exist_ok=True)  # the Videos\FPVEdit folder
        dialog = ExportDialog(
            default, self._settings, fit, self, remember=lambda f: not is_removable(f)
        )
        if not dialog.exec() or dialog.choice is None:
            return
        choice = dialog.choice
        self._output, self._force = choice.path, choice.replace
        self._open_after = choice.open_folder
        self._refresh_output()
        plan, shots = self._plan, self._shots.shots
        tools, force, device, vertical = self._tools, self._force, choice.encoder, choice.vertical
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
                    tools,
                    plan.clips,
                    shots,
                    final,
                    force=force,
                    encode=encode,
                    progress=progress,
                    music=music,
                    vertical=vertical,
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
            f"Your montage is ready: {timecode(length)} long, "
            f"{plural(len(self._current_shots()), 'shot')}. "
            f"Made on {where} in {elapsed:.0f} s and checked frame by frame."
        )
        self._done_path.setText(f"SAVED TO  {final}")
        self._refresh()
        self._show_page(self.DONE)
        if self._open_after:
            self._show_output()

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
        self._review.set_theme(palette, self._fonts)
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
        self._review.deactivate()
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
                self, "Stop and quit?", "FPVEdit is still working. Stop and quit?"
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
        self._review.shutdown()
        super().closeEvent(event)

    # ----- menu -----

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self._cache_dir, self, self._review.ratings)
        dialog.exec()  # ratings may have been cleared even if it was cancelled
        self._refresh_output()
        self._review.refresh()  # the timeline colours may have changed

    def _show_my_ratings(self) -> None:
        base = data_folder()
        if base is not None:
            base.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(base)))

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
        QMessageBox.about(self, "About FPVEdit", about_text(first))


def videos_folder() -> Path:
    """``FPVEdit`` in the user's Videos folder (made when it is first needed)."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MoviesLocation)
    return (Path(base) if base else Path.home() / "Videos") / "FPVEdit"


def data_folder(name: str = "FPVEdit") -> Path | None:
    r"""Where ratings, saved measurements and logs live: %LOCALAPPDATA%\FPVEdit on Windows,
    ~/.local/share/FPVEdit on Linux."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
    return Path(base) / name if base else None


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


APP_NAME = "FPVEdit"


def run(argv: Sequence[str]) -> int:
    app = QApplication(list(argv))
    # Settings need an organisation name to be stored at all (without one, Qt keeps them
    # only until the app closes): HKEY_CURRENT_USER\Software\FPVEdit on Windows.
    app.setOrganizationName(APP_NAME)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    base = data_folder()
    log_file = start_logging(base / "logs") if base else None
    log.info("FPVEdit %s starting", __version__)
    settings = AppSettings()
    theme.apply(app, theme.THEMES.get(settings.theme, theme.DARK), theme.load_fonts())
    tools = locate_tools(settings)
    if tools is None:
        return 1
    window = MainWindow(tools, cache_folder(), settings, log_file)
    window.add_paths([Path(arg) for arg in argv[1:]])
    window.show()
    return app.exec()
