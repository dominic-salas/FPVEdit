"""The app window (Qt). Slow work runs in background threads: analysis, thumbnails, render.

Nothing leaves this computer: clips are read in place, measurements are cached in the
user's local app-data folder, and the montage is written where the user chooses.
"""

from __future__ import annotations

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
    QDragMoveEvent,
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
    QComboBox,
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
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fpv_editor import __version__
from fpv_editor.app import theme
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
    Progress,
    plan_montage,
    render_montage,
)
from fpv_editor.errors import CancelledError, FPVEditError
from fpv_editor.media import MediaInfo, Tools, find_tools, probe, run_tool
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
TRIM_STEPS = (-1.0, -0.25, 0.25, 1.0)
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


class ClipList(QListWidget):
    """The clips, in montage order: drag to reorder, drop files or folders to add."""

    dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
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


class Panel(QFrame):
    """A bordered panel with a numbered label, like the sections of dvsalas.com."""

    def __init__(self, title: str, note: str = "") -> None:
        super().__init__()
        self.setProperty("role", "panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        top = QHBoxLayout()
        label = QLabel(title)
        label.setProperty("role", "eyebrow")
        top.addWidget(label)
        top.addStretch(1)
        if note:
            hint = QLabel(note)
            hint.setProperty("role", "eyebrow")
            top.addWidget(hint)
        layout.addLayout(top)
        self.body = layout


def panel(title: str, note: str = "") -> Panel:
    return Panel(title, note)


class Hero(QWidget):
    """What the preview shows before a shot is chosen: the title over a dot grid."""

    def __init__(self) -> None:
        super().__init__()
        self._dot = QColor(theme.DARK.dot)
        self._title = DotTitle("fpv editor", cell=9)
        badge = QLabel("● RUNS ON THIS PC. NOTHING IS UPLOADED")
        badge.setProperty("role", "eyebrow")
        self._text = QLabel(
            "Drop FPV recordings or a folder anywhere on this window, then find shots. "
            "The best flips, spins and close passes are picked for you: watch them, swap "
            "or trim them, and render one montage."
        )
        self._text.setProperty("role", "hero")
        self._text.setWordWrap(True)
        self._text.setMaximumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.addStretch(1)
        layout.addWidget(badge)
        layout.addSpacing(14)
        layout.addWidget(self._title)
        layout.addSpacing(20)
        layout.addWidget(self._text)
        layout.addStretch(2)

    def set_palette(self, palette: theme.Palette) -> None:
        self._dot = QColor(palette.dot)
        self._title.set_colour(palette.text)
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


class MainWindow(QMainWindow):
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
        self.resize(1280, 800)
        self.setAcceptDrops(True)
        self._tools = tools
        self._cache_dir = cache_dir
        self._settings = settings or AppSettings()
        self._log_file = log_file
        self._notes: tuple[str, ...] = ()
        self._plan: MontagePlan | None = None
        self._shots: ShotList | None = None
        self._task: Task | None = None
        self._output: Path | None = None  # chosen by the user; None = automatic name
        self._force = False  # the user confirmed replacing ``_output``
        self._thumbs: dict[tuple[Path, float], QPixmap] = {}
        self._requested: set[tuple[Path, float]] = set()
        self._loop: tuple[float, float] | None = None  # preview range, seconds
        self._alts: list[Shot] = []  # the alternatives listed, in list order
        self._last_output: Path | None = None

        self._loader = ThumbnailLoader(tools, self)
        self._loader.ready.connect(self._thumbnail_ready)
        self._loader.start()

        self._build()
        self._build_menus()
        self._set_notes(())
        self._apply_theme(self._settings.theme)
        geometry = self._settings.geometry
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._refresh()

    # ----- layout -----

    def _build(self) -> None:
        # ---- header: logo, the three steps, encoder and theme
        logo = QLabel("fpv<span style='color:#e5322d'>.</span>editor")
        logo.setProperty("role", "logo")
        self._steps = [QLabel(text) for text in ("01 CLIPS", "02 SHOTS", "03 RENDER")]
        for step_label in self._steps:
            step_label.setProperty("role", "step")
        self._theme_button = QPushButton()
        self._theme_button.setProperty("flat", True)
        self._theme_button.clicked.connect(self._toggle_theme)
        header = QFrame()
        header.setProperty("role", "header")
        header.setFixedHeight(52)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(20, 0, 16, 0)
        header_row.setSpacing(18)
        header_row.addWidget(logo)
        header_row.addSpacing(18)
        for step_label in self._steps:
            header_row.addWidget(step_label)
        header_row.addStretch(1)
        self._encoder_label = QLabel()
        self._encoder_label.setProperty("role", "eyebrow")
        header_row.addWidget(self._encoder_label)
        header_row.addWidget(self._theme_button)
        self._menu_button = QPushButton("MENU")
        self._menu_button.setProperty("flat", True)
        header_row.addWidget(self._menu_button)

        # ---- 01 clips
        self._clips = ClipList()
        self._clips.dropped.connect(self.add_paths)
        self._clips.model().rowsMoved.connect(lambda *_: self._clips_changed())
        add = QPushButton("ADD CLIPS")
        add.clicked.connect(self._choose_clips)
        remove = QPushButton("REMOVE")
        remove.clicked.connect(self._remove_clips)
        self._length = QDoubleSpinBox()
        self._length.setRange(10, 1800)
        self._length.setDecimals(0)
        self._length.setSuffix(" S")
        self._length.setValue(self._settings.length_s)
        self._length.valueChanged.connect(self._length_changed)
        length_label = QLabel("LENGTH")
        length_label.setProperty("role", "eyebrow")
        self._find = QPushButton("FIND SHOTS")
        self._find.setProperty("primary", True)
        self._find.clicked.connect(self._find_shots)
        clip_buttons = QHBoxLayout()
        clip_buttons.setSpacing(6)
        for widget in (add, remove):
            clip_buttons.addWidget(widget)
        clip_buttons.addStretch(1)
        for control in (length_label, self._length, self._find):
            clip_buttons.addWidget(control)
        clips_panel = panel("01 / CLIPS", "DRAG TO REORDER")
        clips_panel.body.addWidget(self._clips)
        clips_panel.body.addLayout(clip_buttons)

        # ---- 02 shots
        self._shot_list = QListWidget()
        self._shot_list.setIconSize(QPixmap(112, THUMB_HEIGHT).size())
        self._shot_list.setWordWrap(True)
        self._shot_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._shot_list.currentRowChanged.connect(self._shot_selected)
        self._shot_list.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self._reorder))
        self._total = QLabel()
        self._total.setProperty("role", "caption")
        shots_panel = panel("02 / SHOTS", "IN PLAYING ORDER")
        shots_panel.body.addWidget(self._shot_list, 1)
        shots_panel.body.addWidget(self._total)

        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(clips_panel)
        left.addWidget(shots_panel)
        left.setSizes([230, 600])
        left.setChildrenCollapsible(False)

        # ---- 03 preview: the dot-matrix title until a shot is chosen
        self._video = PreviewView(self._tools)
        self._video.timeChanged.connect(self._show_time)
        self._hero = Hero()
        self._stage = QStackedWidget()
        self._stage.addWidget(self._hero)
        self._stage.addWidget(self._video)
        self._caption_left = QLabel()
        self._caption_left.setProperty("role", "eyebrow")
        self._now = QLabel()
        self._now.setProperty("role", "eyebrow")
        caption = QHBoxLayout()
        caption.addWidget(self._caption_left)
        caption.addStretch(1)
        caption.addWidget(self._now)

        trim = QHBoxLayout()
        trim.setSpacing(4)
        self._trim_buttons: list[QPushButton] = []
        for end in (False, True):
            label = QLabel("END" if end else "START")
            label.setProperty("role", "eyebrow")
            trim.addWidget(label)
            for step in TRIM_STEPS:
                button = QPushButton(f"{step:+g}S")
                button.setToolTip(f"Move the shot's {'end' if end else 'start'} by {step:+g} s")
                button.clicked.connect(lambda _=False, e=end, s=step: self._trim(e, s))
                trim.addWidget(button)
                self._trim_buttons.append(button)
            trim.addSpacing(14)
        trim.addStretch(1)
        self._remove_shot = QPushButton("REMOVE SHOT")
        self._remove_shot.clicked.connect(self._remove_current)
        trim.addWidget(self._remove_shot)

        preview_panel = panel("03 / PREVIEW", "LOOPS THE SELECTED SHOT")
        preview_panel.body.addWidget(self._stage, 1)
        preview_panel.body.addLayout(caption)
        preview_panel.body.addLayout(trim)

        # ---- 04 other shots
        self._mode = QComboBox()
        self._mode.addItems(["SWAP: OTHER SHOTS FROM THIS CLIP", "ADD: SHOTS FROM ALL CLIPS"])
        self._mode.currentIndexChanged.connect(lambda _: self._refresh_alternatives())
        self._alternatives = QListWidget()
        self._alternatives.setIconSize(QPixmap(112, THUMB_HEIGHT).size())
        self._alternatives.setWordWrap(True)
        self._alternatives.currentRowChanged.connect(self._alternative_selected)
        self._swap = QPushButton("USE INSTEAD")
        self._swap.clicked.connect(self._use_alternative)
        self._add = QPushButton("ADD TO MONTAGE")
        self._add.clicked.connect(self._add_alternative)
        alt_buttons = QHBoxLayout()
        alt_buttons.setSpacing(6)
        alt_buttons.addWidget(self._mode, 1)
        alt_buttons.addWidget(self._swap)
        alt_buttons.addWidget(self._add)
        alt_panel = panel("04 / OTHER SHOTS", "CLICK TO PREVIEW")
        alt_panel.body.addLayout(alt_buttons)
        alt_panel.body.addWidget(self._alternatives, 1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(preview_panel)
        right.addWidget(alt_panel)
        right.setSizes([560, 260])
        right.setChildrenCollapsible(False)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        main.addWidget(right)
        main.setSizes([460, 820])
        main.setChildrenCollapsible(False)

        # ---- footer: status, output, render
        self._dot = StatusDot()
        self._status = QLabel()
        self._status.setProperty("role", "status")
        self._status.setWordWrap(True)
        self._notes_button = QPushButton()
        self._notes_button.setProperty("flat", True)
        self._notes_button.clicked.connect(self._show_notes)
        self._output_label = QLabel()
        self._output_label.setProperty("role", "eyebrow")
        self._save_as = QPushButton("SAVE AS")
        self._save_as.clicked.connect(self._choose_output)
        self._open = QPushButton("SHOW IN FOLDER")
        self._open.clicked.connect(self._show_output)
        self._cancel = QPushButton("CANCEL")
        self._cancel.clicked.connect(self._cancel_task)
        self._render = QPushButton("RENDER MONTAGE")
        self._render.setProperty("primary", True)
        self._render.clicked.connect(self._start_render)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)

        footer = QFrame()
        footer.setProperty("role", "footer")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(20, 0, 16, 12)
        footer_layout.setSpacing(10)
        footer_layout.addWidget(self._progress)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._dot)
        row.addWidget(self._status, 1)
        row.addWidget(self._notes_button)
        row.addSpacing(12)
        row.addWidget(self._output_label)
        for widget in (self._save_as, self._open, self._cancel, self._render):
            row.addWidget(widget)
        footer_layout.addLayout(row)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 12)
        body_layout.addWidget(main)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(body, 1)
        layout.addWidget(footer)
        self.setCentralWidget(central)

    def _build_menus(self) -> None:
        def action(text: str, slot: Callable[[], object], key: object = None) -> QAction:
            item = QAction(text, self)
            if key is not None:
                item.setShortcut(key)  # type: ignore[call-overload]
            item.triggered.connect(slot)
            self.addAction(item)  # shortcuts work with the menu closed
            return item

        # One MENU button in the header instead of a menu bar, like the site's header.
        menus = QMenu(self)
        self._menu_button.setMenu(menus)
        file = menus.addMenu("&File")
        file.addAction(action("Add clips…", self._choose_clips, QKeySequence.StandardKey.Open))
        file.addAction(action("Add a folder of clips…", self._choose_folder))
        file.addAction(action("Remove selected clips", self._remove_clips))
        file.addAction(action("Remove all clips", self._clear_clips))
        file.addSeparator()
        file.addAction(action("Find shots", self._find_shots, "F5"))
        file.addAction(action("Render montage", self._start_render, "Ctrl+R"))
        file.addSeparator()
        file.addAction(action("Settings…", self._open_settings, "Ctrl+,"))
        file.addSeparator()
        file.addAction(action("Quit", self.close, QKeySequence.StandardKey.Quit))
        edit = menus.addMenu("&Edit")
        edit.addAction(action("Undo", self._undo, QKeySequence.StandardKey.Undo))
        delete = action("Remove shot", self._remove_current, QKeySequence.StandardKey.Delete)
        self.removeAction(delete)  # only while the shot list has focus
        delete.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        edit.addAction(delete)
        self._shot_list.addAction(delete)
        self._shot_list.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        help_menu = menus.addMenu("&Help")
        help_menu.addAction(action("Show the log file", self._show_log))
        help_menu.addAction(action("About FPV Editor", self._about))

    # ----- clips -----

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self.add_paths([Path(url.toLocalFile()) for url in event.mimeData().urls()])

    def clip_paths(self) -> list[Path]:
        return [
            Path(self._clips.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._clips.count())
        ]

    def add_paths(self, paths: list[Path]) -> None:
        if self._busy():
            return
        known = {p.resolve() for p in self.clip_paths()}
        for path in video_files(paths):
            if path.resolve() in known:
                continue
            known.add(path.resolve())
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self._clips.addItem(item)
        self._clips_changed()

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

    def _remove_clips(self) -> None:
        if self._busy():
            return
        for item in self._clips.selectedItems():
            self._clips.takeItem(self._clips.row(item))
        self._clips_changed()

    def _clips_changed(self) -> None:
        if self._plan is not None:
            self._plan = None
            self._shots = None
            self._set_notes(())
            for i in range(self._clips.count()):
                item = self._clips.item(i)
                item.setText(Path(item.data(Qt.ItemDataRole.UserRole)).name)
            self._status.setText("The clips changed: press Find shots again (it is quick now).")
        self._refresh()

    def _length_changed(self, seconds: float) -> None:
        self._settings.length_s = seconds
        self._refresh_total()

    def _annotate_clips(self, plan: MontagePlan) -> None:
        """Show what each clip gave: its usable length and shots, or why it gave none."""
        used = {clip.index: clip for clip in plan.clips}
        shots = {index: sum(1 for s in plan.shots if s.clip == index) for index in used}
        for row, info in enumerate(plan.infos):
            item = self._clips.item(row)
            if item is None:
                continue
            notes = [n for n in plan.warnings if n.startswith(info.path.name)]
            if row in used:
                usable = clock(used[row].analysed_duration)
                text = f"{info.path.name}  ({usable}, {plural(shots[row], 'shot')})"
            else:
                text = f"{info.path.name}  (nothing usable)"
            item.setText(text)
            item.setToolTip("\n".join([str(info.path), *notes]))

    def _set_notes(self, notes: Sequence[str]) -> None:
        self._notes = tuple(notes)
        self._notes_button.setText(f"Notes ({len(self._notes)})")
        self._notes_button.setVisible(bool(self._notes))

    def _show_notes(self) -> None:
        QMessageBox.information(self, "Notes", "\n\n".join(self._notes))

    # ----- analysis -----

    def _find_shots(self) -> None:
        paths = self.clip_paths()
        if not paths or self._busy():
            return
        tools, cache, length = self._tools, self._cache_dir, self._length.value()

        def work(progress: Progress) -> object:
            progress("Reading the clips", 0.0)
            infos = [probe(tools, path) for path in paths]
            log.info("finding shots in %d clips, length %.0f s", len(infos), length)
            return plan_montage(tools, infos, length=length, cache_dir=cache, progress=progress)

        self._run(work, self._plan_ready, "Analysing")

    def _plan_ready(self, result: object) -> None:
        if not isinstance(result, MontagePlan):
            raise TypeError(f"expected a MontagePlan, got {type(result).__name__}")
        self._plan = result
        durations = {clip.index: clip.analysed_duration for clip in result.clips}
        self._shots = ShotList(result.shots, result.candidates, durations)
        self._output, self._force = None, False
        self._annotate_clips(result)
        self._set_notes(result.warnings)
        if result.shots:
            self._status.setText(
                f"{len(result.shots)} shots picked from {result.footage_duration / 60:.1f} min "
                "of usable footage. Click a shot to watch it; swap, trim or drag shots, then "
                "render."
            )
        else:
            self._status.setText(
                "No flip, spin or close pass found in these clips. See Notes for why."
            )
        self._refresh()
        if result.shots:
            self._shot_list.setCurrentRow(0)

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
        prefix = f"{number}. " if number is not None else ""
        return (
            f"{prefix}{self._path(shot).name}\n{clock(shot.start)}-{clock(shot.end)} "
            f"({shot.duration:.1f} s)  {describe(shot)}"
        )

    def _refresh(self, *, preview: bool = True) -> None:
        """Rebuild the lists; with ``preview`` the selected shot starts playing again."""
        row = self._shot_list.currentRow()
        self._shot_list.blockSignals(True)
        self._shot_list.clear()
        if self._shots is not None:
            for number, shot in enumerate(self._shots.shots, start=1):
                item = QListWidgetItem(self._icon(shot), self._shot_text(number, shot))
                item.setData(Qt.ItemDataRole.UserRole, number - 1)
                self._shot_list.addItem(item)
        self._shot_list.blockSignals(False)
        count = self._shot_list.count()
        if count:
            self._shot_list.blockSignals(not preview)
            self._shot_list.setCurrentRow(min(max(row, 0), count - 1))
            self._shot_list.blockSignals(False)
        else:
            self._stop_preview()
        self._refresh_total()
        self._refresh_alternatives()
        self._refresh_output()
        self._refresh_buttons()

    def _refresh_total(self) -> None:
        if self._shots is not None:
            self._total.setText(
                f"{len(self._shots.shots)} shots, {self._shots.duration:.1f} s "
                f"(target {self._length.value():.0f} s)"
            )
        elif self._clips.count():
            self._total.setText("Press Find shots (F5).")
        else:
            self._total.setText("Add clips, then press Find shots.")

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
        selected = self._shot_list.currentRow() >= 0
        for button in self._trim_buttons:
            button.setEnabled(selected and not busy)
        self._remove_shot.setEnabled(selected and not busy)
        self._find.setEnabled(self._clips.count() > 0 and not busy)
        primary = self._plan is None
        if self._find.property("primary") != primary:
            self._find.setProperty("primary", primary)
            self._find.style().unpolish(self._find)
            self._find.style().polish(self._find)
        self._render.setEnabled(bool(self._current_shots()) and not busy)
        self._save_as.setEnabled(self._plan is not None and not busy)
        self._cancel.setEnabled(busy)
        self._cancel.setVisible(busy)
        self._open.setVisible(self._last_output is not None and self._last_output.exists())
        self._dot.set_busy(busy)
        step = 0 if self._plan is None else 1
        if self._last_output is not None or (busy and self._plan is not None and step == 1):
            step = 2 if busy or self._last_output is not None else step
        for index, label in enumerate(self._steps):
            label.setProperty("role", "step-on" if index == step else "step")
            label.style().unpolish(label)
            label.style().polish(label)
        encoder = "GPU WHEN AVAILABLE" if self._settings.encoder == "auto" else "CPU ONLY"
        self._encoder_label.setText(f"ENCODER: {encoder}")
        swap_mode = self._mode.currentIndex() == 0
        alt = self._alternatives.currentRow() >= 0
        self._swap.setEnabled(alt and selected and swap_mode and not busy)
        self._add.setEnabled(alt and not swap_mode and not busy)
        self._open.setEnabled(self._last_output is not None and self._last_output.exists())
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
        if self._shots is not None and 0 <= row < len(self._shots.shots):
            self._preview(self._shots.shots[row])
        if self._mode.currentIndex() == 0:
            self._refresh_alternatives()
        self._refresh_buttons()

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
        # Show the changed end: the start loops anyway; an end change plays its last seconds.
        self._preview(shot, from_seconds=max(shot.start, shot.end - 3) if end else None)

    def _remove_current(self) -> None:
        index = self._selected_index()
        if index is not None and self._shots is not None and not self._busy():
            self._shots.remove(index)
            self._refresh()

    def _undo(self) -> None:
        if self._shots is not None and self._shots.can_undo and not self._busy():
            self._shots.undo()
            self._refresh()

    # ----- alternatives -----

    def _refresh_alternatives(self) -> None:
        self._alternatives.blockSignals(True)
        self._alternatives.clear()
        self._alts = []
        if self._shots is not None:
            swap_mode = self._mode.currentIndex() == 0
            index = self._selected_index()
            if swap_mode and index is not None:
                self._alts = self._shots.alternatives(index)
            elif not swap_mode:
                self._alts = self._shots.alternatives(limit=40)
            for shot in self._alts:
                text = self._shot_text(None, shot) + f"  score {shot.score:.1f}"
                self._alternatives.addItem(QListWidgetItem(self._icon(shot), text))
        self._alternatives.blockSignals(False)
        self._refresh_buttons()

    def _alternative_selected(self, row: int) -> None:
        if 0 <= row < len(self._alts):
            self._preview(self._alts[row])
        self._refresh_buttons()

    def _use_alternative(self) -> None:
        index = self._selected_index()
        row = self._alternatives.currentRow()
        if index is None or self._shots is None or not 0 <= row < len(self._alts):
            return
        try:
            self._shots.replace(index, self._alts[row])
        except ValueError as exc:
            self._status.setText(f"Not swapped: {exc}.")
            return
        self._refresh()

    def _add_alternative(self) -> None:
        row = self._alternatives.currentRow()
        if self._shots is None or not 0 <= row < len(self._alts):
            return
        try:
            position = self._shots.add(self._alts[row])
        except ValueError as exc:
            self._status.setText(f"Not added: {exc}.")
            return
        self._refresh()
        self._shot_list.setCurrentRow(position)

    # ----- preview -----

    def _preview(self, shot: Shot, from_seconds: float | None = None) -> None:
        self._loop = (shot.start, shot.end)
        self._stage.setCurrentWidget(self._video)
        shots = self._current_shots()
        if shot in shots:
            number = shots.index(shot) + 1
            self._caption_left.setText(f"FIG. {number:02d}   SHOT {number} OF {len(shots)}")
        else:
            self._caption_left.setText("FIG. --   ANOTHER SHOT, NOT IN THE MONTAGE")
        self._video.play(self._info(shot), shot.start, shot.end, from_seconds)

    def _show_time(self, seconds: float) -> None:
        if self._loop is not None:
            start, end = self._loop
            self._now.setText(f"{clock(seconds)}  (shot {clock(start)}-{clock(end)})")

    def _stop_preview(self) -> None:
        self._loop = None
        self._stage.setCurrentWidget(self._hero)
        self._caption_left.setText("")
        self._video.clear_picture()
        self._now.setText("")

    # ----- render -----

    def _default_output(self) -> Path | None:
        if self._plan is None:
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
        self._output_label.setText(f"SAVES TO  {output}" if output else "")

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
            QMessageBox.warning(self, "Cannot render", str(exc))
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
            where = "GPU (NVENC)" if used == NVENC else "CPU (libx264)"
            if used != choices[0]:
                where += " (GPU encoding failed, so the CPU was used)"
            return final, length, time.monotonic() - started, where

        self._run(work, self._rendered, "Rendering")

    def _rendered(self, result: object) -> None:
        if not isinstance(result, tuple):
            raise TypeError(f"expected the render result, got {type(result).__name__}")
        final, length, elapsed, where = result
        self._last_output = final
        self._output, self._force = None, False
        self._status.setText(
            f"Done: {final} ({clock(length)}, rendered on the {where} in {elapsed:.0f} s, checked)."
        )
        self._refresh()

    def _apply_theme(self, name: str) -> None:
        palette = theme.THEMES.get(name, theme.DARK)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            theme.apply(app, palette, theme.load_fonts())
        self._hero.set_palette(palette)
        self._dot.set_colours(palette.faint, palette.accent)
        self._theme_button.setText("LIGHT" if palette is theme.DARK else "DARK")
        self._settings.theme = palette.name

    def _toggle_theme(self) -> None:
        self._apply_theme("light" if self._settings.theme == "dark" else "dark")

    def _show_output(self) -> None:
        if self._last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output.parent)))

    # ----- background work -----

    def _busy(self) -> bool:
        return self._task is not None

    def _run(
        self, work: Callable[[Progress], object], done: Callable[[object], None], what: str
    ) -> None:
        task = Task(work, self)
        task.progressed.connect(self._progressed)
        task.succeeded.connect(done)
        task.failed.connect(self._failed)
        task.finished.connect(self._task_finished)
        self._task = task
        self._progress.setValue(0)
        self._status.setText(f"{what}…")
        self._refresh_buttons()
        task.start()

    def _progressed(self, text: str, share: float) -> None:
        self._progress.setValue(round(1000 * share))
        left = time_left(time.monotonic() - self._task.started_at, share) if self._task else ""
        self._status.setText(f"{text}  ({left})" if left else text)

    def _failed(self, message: str) -> None:
        self._status.setText(message.splitlines()[0] if message else "Failed.")
        if message != "Cancelled.":
            QMessageBox.warning(self, "Something went wrong", message)

    def _task_finished(self) -> None:
        if self._task is not None:
            self._task.deleteLater()
        self._task = None
        self._progress.setValue(0)
        self._refresh_buttons()

    def _cancel_task(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._status.setText("Stopping…")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._task is not None:
            answer = QMessageBox.question(
                self, "Stop and quit?", "FPV Editor is still working. Stop and quit?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._task.cancel()
            self._status.setText("Stopping…")
            QApplication.processEvents()
            self._task.wait()
        self._settings.geometry = self.saveGeometry()
        self._loader.stop()
        self._video.stop()
        super().closeEvent(event)

    # ----- menus -----

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
