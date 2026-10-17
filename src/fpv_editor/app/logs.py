"""The app's log file: what it did and every error, for troubleshooting (no footage, no
uploads). It stays on this computer, in the user's local app-data folder."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from types import TracebackType

LOG_NAME = "fpv-editor.log"
log = logging.getLogger("fpv_editor.app")


def start_logging(folder: Path) -> Path | None:
    """Log to ``folder``/fpv-editor.log (rotated at 1 MB, 3 kept), including uncaught
    errors in any thread. Returns the log file, or None if it cannot be written."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            folder / LOG_NAME, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("fpv_editor")
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    def uncaught(
        kind: type[BaseException], error: BaseException, trace: TracebackType | None
    ) -> None:
        log.critical("uncaught error", exc_info=(kind, error, trace))
        sys.__excepthook__(kind, error, trace)

    def uncaught_in_thread(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is not None:
            error = (args.exc_type, args.exc_value, args.exc_traceback)
            log.critical("uncaught error in thread %s", args.thread, exc_info=error)

    sys.excepthook = uncaught
    threading.excepthook = uncaught_in_thread
    return folder / LOG_NAME
