"""Clips are analysed in the background from the moment they are added.

Measuring every frame is most of the wait, and it depends only on the clip, not on the
montage length or the clip order. So each clip goes to a pool of worker processes as
soon as it is known, and Find shots only waits for what is not finished yet. Results are
kept per file, so reordering clips or changing the length needs no new analysis; the
measurement cache makes the next session's analysis of the same clips quick too.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from fpv_editor.cli import ClipJob, ClipResult, PipelineConfig, default_workers, plan_in_worker
from fpv_editor.errors import CancelledError, FPVEditError
from fpv_editor.media import MediaInfo, Tools

log = logging.getLogger("fpv_editor.app")
POLL_MS = 200


@dataclass
class ClipState:
    info: MediaInfo
    job: int
    share: float = 0.0  # of the clip measured so far
    started: bool = False
    result: ClipResult | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.result is not None or self.error is not None


class BackgroundAnalysis(QObject):
    """Analyses each added clip once, in worker processes; ``changed`` on any progress."""

    changed = Signal()

    def __init__(
        self,
        tools: Tools,
        cache_dir: Path | None,
        parent: QObject | None = None,
        workers: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._tools = tools
        self._cache_dir = cache_dir
        self._config = PipelineConfig()
        self._workers = workers if workers is not None else default_workers()
        self._states: dict[Path, ClipState] = {}
        self._futures: dict[Path, Future[ClipResult]] = {}
        self._ids: dict[int, Path] = {}  # job index -> clip
        self._next_id = 0
        self._manager: Any = None
        self._shares: Any = None
        self._stop: Any = None
        self._pool: ProcessPoolExecutor | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    def state(self, path: Path) -> ClipState | None:
        return self._states.get(path.resolve())

    def add(self, info: MediaInfo) -> None:
        """Start analysing ``info`` unless it is analysed, under way or waiting already."""
        path = info.path.resolve()
        state = self._states.get(path)
        if state is not None and (state.result is not None or path in self._futures):
            return
        pool = self._start_pool()
        job_id = self._next_id
        self._next_id += 1
        self._ids[job_id] = path
        self._states[path] = ClipState(info, job_id)
        job = ClipJob(
            self._tools, info, job_id, self._config, self._config.montage, self._cache_dir
        )
        self._futures[path] = pool.submit(plan_in_worker, job, self._shares, self._stop)
        self._timer.start()
        self.changed.emit()

    def retry_failed(self) -> None:
        """Clips whose analysis failed get one more try (a file that was still copying)."""
        for state in list(self._states.values()):
            if state.error is not None:
                self.add(state.info)

    def cancel(self) -> None:
        """Stop every unfinished analysis; finished results are kept."""
        if self._stop is not None:
            self._stop.set()
            self._stop = self._manager.Event()  # for the next jobs
        for path, future in list(self._futures.items()):
            future.cancel()
            state = self._states.get(path)
            if state is not None and not state.finished:
                del self._states[path]
        self._futures.clear()
        self.changed.emit()

    def shutdown(self) -> None:
        self._timer.stop()
        if self._stop is not None:
            self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    def _start_pool(self) -> ProcessPoolExecutor:
        if self._pool is not None:
            return self._pool
        context = multiprocessing.get_context("spawn")  # the same on Windows and Linux
        self._manager = context.Manager()
        self._shares = self._manager.Queue()
        self._stop = self._manager.Event()
        self._pool = ProcessPoolExecutor(self._workers, mp_context=context)
        return self._pool

    def _poll(self) -> None:
        changed = False
        while True:
            try:
                job_id, share = self._shares.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            state = self._states.get(self._ids.get(job_id, Path()))
            if state is not None and state.job == job_id and not state.finished:
                state.share, state.started, changed = share, True, True
        for path, future in list(self._futures.items()):
            if not future.done():
                continue
            del self._futures[path]
            changed = True
            state = self._states.get(path)
            if state is None:
                continue
            if future.cancelled():
                del self._states[path]
                continue
            try:
                state.result = future.result()
                state.share = 1.0
            except CancelledError:
                del self._states[path]
            except FPVEditError as exc:
                log.warning("analysing %s failed: %s", path.name, exc)
                state.error = str(exc)
            except Exception as exc:  # a defect in a worker: keep the app running, log it
                log.exception("analysing %s failed", path.name)
                state.error = f"unexpected error: {exc}"
        if not self._futures:
            self._timer.stop()
        if changed:
            self.changed.emit()
