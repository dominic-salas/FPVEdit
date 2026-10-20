"""Error types shown to the user as one-line messages by the CLI."""

from __future__ import annotations


class FPVEditError(Exception):
    """Base class for expected, user-facing failures."""


class ToolNotFoundError(FPVEditError):
    """FFmpeg or ffprobe is not installed or not on PATH."""


class MediaToolError(FPVEditError):
    """An ffmpeg/ffprobe invocation failed."""

    def __init__(self, action: str, returncode: int, stderr: str) -> None:
        tail = "\n".join(stderr.strip().splitlines()[-8:])
        detail = f":\n{tail}" if tail else ""
        super().__init__(f"{action} failed (exit code {returncode}){detail}")
        self.action = action
        self.returncode = returncode
        self.stderr = stderr


class EncodeError(MediaToolError):
    """Encoding the video failed; another encoder (e.g. the CPU instead of NVENC) may work."""


class InvalidInputError(FPVEditError):
    """The input is missing, not a file, or has no usable video stream."""


class NoFlightContentError(FPVEditError):
    """Analysis found nothing but idle or blank footage."""


class OutputExistsError(FPVEditError):
    """The requested output already exists and --force was not given."""


class OutputLocationError(FPVEditError):
    """The output cannot be written where it was asked for (missing or read-only folder)."""


class ValidationError(FPVEditError):
    """The rendered file failed post-render validation."""


class CancelledError(FPVEditError):
    """The user stopped the work; raised from a progress callback."""
