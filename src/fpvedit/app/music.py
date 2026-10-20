"""Hearing the background music while the montage plays in the review step.

The montage's preview is silent video (the recordings' audio is never used), so the
music plays beside it with ffplay, which comes with FFmpeg: each track in turn, starting
at the point of the montage the player is at. Without ffplay the preview stays silent;
the export always has the music.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

from fpvedit.media import NO_WINDOW, AudioFile, Tools


def find_ffplay(tools: Tools) -> str | None:
    """ffplay beside the ffmpeg in use, else on PATH."""
    beside = Path(tools.ffmpeg).with_name("ffplay" + Path(tools.ffmpeg).suffix)
    if beside.is_file():
        return str(beside)
    return shutil.which("ffplay")


def track_at(tracks: Sequence[AudioFile], seconds: float) -> tuple[int, float] | None:
    """The track playing ``seconds`` into the soundtrack and the time in it; None after
    the last track ends."""
    start = 0.0
    for index, track in enumerate(tracks):
        if seconds < start + track.duration:
            return index, max(0.0, seconds - start)
        start += track.duration
    return None


class MusicPreview:
    """Plays the tracks from a point of the montage until stopped or the montage ends."""

    def __init__(self, tools: Tools) -> None:
        self.ffplay = find_ffplay(tools)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def play(self, tracks: Sequence[AudioFile], seconds: float, until: float) -> None:
        self.stop()
        if self.ffplay is None or not tracks or seconds >= until:
            return
        self._stop = threading.Event()
        stop = self._stop
        self._thread = threading.Thread(
            target=self._run, args=(self.ffplay, tuple(tracks), seconds, until, stop), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(
        self,
        ffplay: str,
        tracks: tuple[AudioFile, ...],
        seconds: float,
        until: float,
        stop: threading.Event,
    ) -> None:
        found = track_at(tracks, seconds)
        if found is None:
            return
        index, offset = found
        at = seconds
        while index < len(tracks) and at < until and not stop.is_set():
            length = min(tracks[index].duration - offset, until - at)
            args = [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet"]
            args += ["-ss", f"{offset:.3f}", "-t", f"{length:.3f}", str(tracks[index].path)]
            with self._lock:
                if stop.is_set():
                    return
                try:
                    self._process = subprocess.Popen(
                        args,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=NO_WINDOW,
                    )
                except OSError:
                    return
                process = self._process
            process.wait()
            at += length
            index, offset = index + 1, 0.0
