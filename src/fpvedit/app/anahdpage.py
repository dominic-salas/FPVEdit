"""The AnaHD module: AnaHD running inside this window, or the prompt to download it.

AnaHD needs PyTorch and an NVIDIA card, so it stays an optional program of its own. Its
window is placed in this page as a child, which keeps its GPU work and memory in its own
process. The original recordings are only passed along, never changed here.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    Qt,
    QTemporaryDir,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QDesktopServices, QWindow
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import QLabel, QProgressBar, QStackedLayout, QVBoxLayout, QWidget

from fpvedit.app.widgets import button

ANAHD_PAGE = "https://dvsalas.com/anahd"
RELEASE_API = "https://api.github.com/repos/dominic-salas/AnaHD/releases/latest"
UNINSTALL_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    r"\{3C9E4A71-8B2D-4F6A-9E15-7D0B2A6C4E83}_is1"
)
MIN_VERSION = (0, 1, 5)  # the first AnaHD that can run inside another window
FIND_WINDOW_MS = 60_000
Version = tuple[int, ...]


def find_anahd(local_programs: Path | None = None) -> Path | None:
    """The installed AnaHD program, if there is one."""
    if local_programs is None:
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            return None
        local_programs = Path(base) / "Programs"
    program = local_programs / "AnaHD" / "AnaHD.exe"
    return program if program.is_file() else None


def parse_version(text: str) -> Version | None:
    try:
        return tuple(int(part) for part in text.strip().split("."))
    except ValueError:
        return None


def installed_version() -> Version | None:
    """The installed AnaHD's version as its installer recorded it."""
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
            return parse_version(str(winreg.QueryValueEx(key, "DisplayVersion")[0]))
    except OSError:
        return None


def _windows_of(pid: int) -> list[int]:
    """The visible top-level windows of a process that are big enough to be its main one."""
    if sys.platform != "win32":
        return []
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[int] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(handle: int, _extra: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        rect = wintypes.RECT()
        user32.GetWindowRect(handle, ctypes.byref(rect))
        if owner.value == pid and user32.IsWindowVisible(handle) and rect.right - rect.left > 400:
            found.append(int(handle))
        return True

    user32.EnumWindows(enum_proc(visit), 0)
    return found


def _ask_to_close(handle: int) -> None:
    if sys.platform == "win32":
        ctypes.windll.user32.PostMessageW(handle, 0x0010, 0, 0)  # WM_CLOSE


class Session(Protocol):
    """A running AnaHD."""

    ready: Any
    stopped: Any

    @property
    def running(self) -> bool: ...
    def start(self, program: Path, arguments: Sequence[str], theme: str) -> None: ...
    def request_close(self) -> None: ...


class Getter(Protocol):
    """Downloads and installs AnaHD."""

    progress: Any
    installing: Any
    installed: Any
    failed: Any

    def start(self) -> None: ...
    def cancel(self) -> None: ...


class EmbeddedAnaHD(QObject):
    """AnaHD's process, with its window as a child widget once it has opened one."""

    ready = Signal(QWidget)
    stopped = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._host = parent
        self._process = QProcess(self)
        self._process.finished.connect(self._ended)
        self._process.errorOccurred.connect(self._error)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._look_for_window)
        self._waited = 0
        self._handle = 0

    @property
    def running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def start(self, program: Path, arguments: Sequence[str], theme: str) -> None:
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("ANAHD_EMBEDDED", "1")
        environment.insert("ANAHD_THEME", theme)
        self._process.setProcessEnvironment(environment)
        self._waited = 0
        self._handle = 0
        self._process.start(str(program), list(arguments))
        self._timer.start()

    def request_close(self) -> None:
        """AnaHD asks first if it is cleaning, so this may leave it running."""
        if self._handle:
            _ask_to_close(self._handle)
        else:
            self._process.kill()

    def _look_for_window(self) -> None:
        self._waited += self._timer.interval()
        windows = _windows_of(int(self._process.processId()))
        if windows:
            self._timer.stop()
            self._handle = windows[0]
            foreign = QWindow.fromWinId(self._handle)
            self.ready.emit(QWidget.createWindowContainer(foreign, self._host))
        elif self._waited >= FIND_WINDOW_MS:
            self._timer.stop()
            self._process.kill()

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._ended()

    def _ended(self, *_status: object) -> None:
        self._timer.stop()
        self._handle = 0
        self.stopped.emit()


class ReleaseGetter(QObject):
    """Fetches the latest AnaHD installer, checks its SHA-256 and runs it for this user."""

    progress = Signal(float, float)
    installing = Signal()
    installed = Signal()
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._network = QNetworkAccessManager(self)
        self._reply: QNetworkReply | None = None
        self._folder: QTemporaryDir | None = None
        self._digest = hashlib.sha256()
        self._expected = ""
        self._setup_url = ""
        self._setup_name = ""
        self._file: Any = None

    def start(self) -> None:
        self._expected = ""
        self._get(RELEASE_API, self._release_read)

    def cancel(self) -> None:
        reply, self._reply = self._reply, None
        if reply is not None:
            reply.abort()
        self._close_file()

    def _get(self, url: str, done: Callable[[QNetworkReply], None]) -> QNetworkReply:
        request = QNetworkRequest(QUrl(url))
        request.setRawHeader(b"User-Agent", b"okFPV")
        reply = self._network.get(request)
        self._reply = reply
        reply.finished.connect(lambda: self._finished(reply, done))
        return reply

    def _finished(self, reply: QNetworkReply, done: Callable[[QNetworkReply], None]) -> None:
        reply.deleteLater()
        if reply is not self._reply:
            return  # cancelled
        if reply.error() != QNetworkReply.NetworkError.NoError:
            self._close_file()
            self.failed.emit("The download did not finish. Check your connection and try again.")
            return
        done(reply)

    def _release_read(self, reply: QNetworkReply) -> None:
        release = json.loads(bytes(reply.readAll().data()))
        assets = {asset["name"]: asset["browser_download_url"] for asset in release["assets"]}
        setups = [name for name in assets if name.endswith("-setup.exe")]
        if not setups or "SHA256SUMS.txt" not in assets:
            self.failed.emit("This AnaHD release has no installer to download.")
            return
        self._setup_name = setups[0]
        self._setup_url = assets[self._setup_name]
        self._get(assets["SHA256SUMS.txt"], self._sums_read)

    def _sums_read(self, reply: QNetworkReply) -> None:
        for line in bytes(reply.readAll().data()).decode("utf-8", "replace").splitlines():
            parts = line.replace("*", " ").split()
            if len(parts) == 2 and parts[1].rsplit("/", 1)[-1] == self._setup_name:
                self._expected = parts[0].lower()
        if not self._expected:
            self.failed.emit("The AnaHD installer could not be checked, so it was not downloaded.")
            return
        self._folder = QTemporaryDir()
        self._file = (Path(self._folder.path()) / self._setup_name).open("wb")
        self._digest = hashlib.sha256()
        setup = self._get(self._setup_url, self._setup_read)
        setup.readyRead.connect(lambda: self._write(setup))
        setup.downloadProgress.connect(
            lambda done, total: self.progress.emit(float(done), float(total))
        )

    def _write(self, reply: QNetworkReply) -> None:
        data = bytes(reply.readAll().data())
        self._digest.update(data)
        self._file.write(data)

    def _setup_read(self, reply: QNetworkReply) -> None:
        self._write(reply)
        path = Path(self._file.name)
        self._close_file()
        if self._digest.hexdigest() != self._expected:
            self.failed.emit("The download was damaged, so it was not installed. Try again.")
            return
        self._install(path)

    def _install(self, setup: Path) -> None:
        self.installing.emit()
        process = QProcess(self)
        process.finished.connect(lambda code, _status: self._install_done(int(code)))
        process.errorOccurred.connect(lambda _error: self._install_done(1))
        process.start(str(setup), ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"])

    def _install_done(self, code: int) -> None:
        if code == 0:
            self.installed.emit()
        else:
            self.failed.emit("AnaHD could not be installed.")

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class AnaHDPage(QWidget):
    def __init__(
        self,
        clips: Callable[[], list[Path]],
        theme: Callable[[], str],
        find: Callable[[], Path | None] | None = None,
        version: Callable[[], Version | None] | None = None,
        session: Session | None = None,
        getter: Getter | None = None,
        can_install: bool = sys.platform == "win32",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._clips = clips
        self._theme = theme
        self._find = find or find_anahd
        self._version = version or installed_version
        self._session: Session = session or EmbeddedAnaHD(self)
        self._getter: Getter = getter or ReleaseGetter(self)
        self._can_install = can_install
        self._busy_getting = False
        self._embedded: QWidget | None = None

        self._session.ready.connect(self._show_embedded)
        self._session.stopped.connect(self._ended)
        self._getter.progress.connect(self._progress)
        self._getter.installing.connect(self._installing)
        self._getter.installed.connect(self._installed)
        self._getter.failed.connect(self._failed)

        self._heading = QLabel()
        self._heading.setProperty("role", "heading")
        self._text = QLabel()
        self._text.setProperty("role", "hero")
        self._text.setWordWrap(True)
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(3)
        self._bar.hide()
        self._status = QLabel()
        self._status.setProperty("role", "status")
        self._action = button("DOWNLOAD ANAHD", self._act, primary=True)
        self._cancel = button("CANCEL", self._cancel_download)
        self._cancel.hide()
        column = QVBoxLayout()
        column.setSpacing(16)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._heading)
        column.addWidget(self._text)
        column.addWidget(self._bar)
        column.addWidget(self._status)
        column.addSpacing(8)
        column.addWidget(self._action, alignment=Qt.AlignmentFlag.AlignLeft)
        column.addWidget(self._cancel, alignment=Qt.AlignmentFlag.AlignLeft)
        narrow = QWidget()
        narrow.setLayout(column)
        narrow.setMaximumWidth(640)
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(72, 0, 72, 0)
        outer.addStretch(2)
        outer.addWidget(narrow)
        outer.addStretch(3)
        self._stack = QStackedLayout(self)
        self._stack.addWidget(panel)

    @property
    def running(self) -> bool:
        return self._session.running

    def request_close(self) -> None:
        self._session.request_close()

    def on_stopped(self, slot: Callable[[], object]) -> None:
        self._session.stopped.connect(slot)

    def status_text(self) -> str:
        """Where AnaHD stands, for the Home page."""
        if self._session.running:
            return "OPEN"
        if self._find() is None:
            return "NOT INSTALLED"
        return "UPDATE AVAILABLE" if self._outdated() else "INSTALLED"

    def refresh(self) -> None:
        """Show AnaHD, starting it if needed, or say what it takes to get it."""
        if self._session.running or self._busy_getting:
            return
        program = self._find()
        if program is None:
            self._offer(
                "AnaHD is not installed",
                "AnaHD repairs analog breakup and upscales your goggle footage to HD. It is a "
                "free optional download of about 2 GB and needs an NVIDIA graphics card.",
                "DOWNLOAD ANAHD",
            )
        elif self._outdated():
            self._offer(
                "AnaHD needs an update",
                "This version of AnaHD cannot open inside okFPV yet. Your settings are kept.",
                "UPDATE ANAHD",
            )
        else:
            self._start(program)

    def _outdated(self) -> bool:
        version = self._version()
        return version is not None and version < MIN_VERSION

    def _offer(self, heading: str, text: str, action: str) -> None:
        self._stack.setCurrentIndex(0)
        self._heading.setText(heading)
        self._text.setText(text)
        self._bar.hide()
        self._cancel.hide()
        self._status.setText("")
        self._action.setText(action if self._can_install else "OPEN ANAHD WEBSITE")
        self._action.show()

    def _waiting(self, heading: str) -> None:
        self._stack.setCurrentIndex(0)
        self._heading.setText(heading)
        self._text.setText("")
        self._action.hide()

    def _start(self, program: Path) -> None:
        self._waiting("Opening AnaHD")
        self._bar.hide()
        self._cancel.hide()
        self._status.setText("")
        self._session.start(program, [str(path) for path in self._clips()], self._theme())

    def _show_embedded(self, widget: QWidget) -> None:
        self._embedded = widget
        self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)

    def _ended(self) -> None:
        widget, self._embedded = self._embedded, None
        if widget is not None:
            self._stack.removeWidget(widget)
            widget.deleteLater()
        self._offer(
            "AnaHD stopped", "AnaHD closed. Your recordings were not changed.", "OPEN ANAHD"
        )
        self._action.setText("OPEN ANAHD")

    def _act(self) -> None:
        program = self._find()
        if program is not None and not self._outdated():
            self._start(program)
        elif self._can_install:
            self._busy_getting = True
            self._waiting("Downloading AnaHD")
            self._bar.setRange(0, 0)
            self._bar.show()
            self._cancel.show()
            self._status.setText("Connecting")
            self._getter.start()
        else:
            QDesktopServices.openUrl(QUrl(ANAHD_PAGE))

    def _progress(self, done: float, total: float) -> None:
        if total > 0:
            self._bar.setRange(0, 1000)
            self._bar.setValue(int(1000 * done / total))
            self._status.setText(f"{done / 1e9:.2f} of {total / 1e9:.2f} GB")

    def _installing(self) -> None:
        self._heading.setText("Installing AnaHD")
        self._bar.setRange(0, 0)
        self._cancel.hide()
        self._status.setText("This can take a few minutes")

    def _installed(self) -> None:
        self._busy_getting = False
        self.refresh()

    def _failed(self, why: str) -> None:
        self._busy_getting = False
        self.refresh()
        self._status.setText(why)

    def _cancel_download(self) -> None:
        self._getter.cancel()
        self._busy_getting = False
        self.refresh()
