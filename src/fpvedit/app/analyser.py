"""Clips are analysed in the background from the moment they are added.

Measuring every frame is most of the wait, and it depends only on the clip, not on the
montage length or the clip order. So each clip goes to a pool of worker processes as
soon as it is known, and Find shots only waits for what is not finished yet. Results are
kept per file, so reordering clips or changing the length needs no new analysis; the
measurement cache makes the next session's analysis of the same clips quick too.
"""

from __future__ import annotations

import dataclasses
import heapq
import logging
import multiprocessing
import os
import queue
import sys
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from fpvedit.errors import CancelledError, FPVEditError
from fpvedit.media import MediaInfo, Tools
from fpvedit.pipeline import (
    ClipJob,
    ClipResult,
    PipelineConfig,
    default_workers,
    is_cached,
    plan_in_worker,
)
from fpvedit.source import AUTO, Source

log = logging.getLogger("fpvedit.app")
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

    @property
    def source(self) -> Source | None:
        """The kind of recording, once analysed (None when nothing usable was found)."""
        clip = self.result.clip if self.result is not None else None
        return clip.source if clip is not None else None


def background_priority() -> None:
    """Run this worker process (and the FFmpeg it starts) below normal priority, so the
    window and the shot playing in it stay smooth while clips are analysed."""
    if sys.platform == "win32":
        import ctypes

        below_normal = 0x4000  # BELOW_NORMAL_PRIORITY_CLASS; child processes inherit it
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)
    else:
        os.nice(10)


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
        self._choices: dict[Path, str] = {}  # clip -> kind the user named (absent: auto)
        self._found: dict[Path, Source] = {}  # clip -> what the last automatic analysis said
        self._futures: dict[Path, Future[ClipResult]] = {}
        # Clips waiting for a worker: already measured ones (cached, seconds each) go first,
        # then the shortest, so with many clips the first results come quickly.
        self._cached: deque[Path] = deque()
        self._uncached: list[tuple[float, int, Path]] = []  # heap of (length, order, clip)
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

    def choice(self, path: Path) -> str:
        """The kind the user named for ``path``: ``auto`` or a `SourceKind` value."""
        return self._choices.get(path.resolve(), AUTO)

    def found(self, path: Path) -> Source | None:
        """What analysing ``path`` by itself found, also while a kind is forced."""
        return self._found.get(path.resolve())

    def set_source(self, path: Path, choice: str) -> bool:
        """Analyse only ``path`` again with its kind forced (``auto``: found out again).

        Its measurements are cached and the choice is not part of the cache key, so this
        only repeats the editing. False when the clip is not analysed yet.
        """
        key = path.resolve()
        state = self._states.get(key)
        if state is None or not state.finished:
            return False
        if choice == self.choice(key):
            return True
        if choice == AUTO:
            self._choices.pop(key, None)
        else:
            self._choices[key] = choice
        del self._states[key]
        self.add(state.info)
        return True

    def _config_for(self, path: Path) -> PipelineConfig:
        """The settings for one clip: the shared ones with its own kind of recording."""
        choice = self._choices.get(path, AUTO)
        return self._config if choice == AUTO else dataclasses.replace(self._config, source=choice)

    def add(self, info: MediaInfo) -> None:
        """Start analysing ``info`` unless it is analysed, under way or waiting already."""
        path = info.path.resolve()
        state = self._states.get(path)
        waiting = path in self._cached or any(item[2] == path for item in self._uncached)
        if state is not None and (state.result is not None or path in self._futures or waiting):
            return
        job_id = self._next_id
        self._next_id += 1
        self._ids[job_id] = path
        self._states[path] = ClipState(info, job_id)
        cached = self._cache_dir is not None and is_cached(
            self._tools, info, self._cache_dir, self._config_for(path)
        )
        if cached:
            self._cached.append(path)
        else:
            heapq.heappush(self._uncached, (info.duration or 0.0, job_id, path))
        self._submit_more()
        self.changed.emit()

    def _submit_more(self) -> None:
        """Keep the workers busy with a short line, cached clips first."""
        while len(self._futures) < 2 * self._workers and (self._cached or self._uncached):
            path = self._cached.popleft() if self._cached else heapq.heappop(self._uncached)[2]
            state = self._states.get(path)
            if state is None:
                continue
            pool = self._start_pool()
            config = self._config_for(path)
            job = ClipJob(
                self._tools,
                state.info,
                state.job,
                config,
                config.montage,
                self._cache_dir,
            )
            self._futures[path] = pool.submit(plan_in_worker, job, self._shares, self._stop)
            self._timer.start()

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
        for path in [*self._cached, *(item[2] for item in self._uncached)]:
            self._states.pop(path, None)
        self._cached.clear()
        self._uncached.clear()
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
        self._pool = ProcessPoolExecutor(
            self._workers, mp_context=context, initializer=background_priority
        )
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
                if state.source is not None and path not in self._choices:
                    self._found[path] = state.source
            except CancelledError:
                del self._states[path]
            except FPVEditError as exc:
                log.warning("analysing %s failed: %s", path.name, exc)
                state.error = str(exc)
            except Exception as exc:  # a defect in a worker: keep the app running, log it
                log.exception("analysing %s failed", path.name)
                state.error = f"unexpected error: {exc}"
        self._submit_more()
        if not self._futures:
            self._timer.stop()
        if changed:
            self.changed.emit()
