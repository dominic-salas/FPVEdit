"""Desktop app: drop clips in, check and change the shots the montage engine picks, render.

Runs entirely on this computer. Needs the ``app`` extra (PySide6): ``uv sync --extra app``.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import sys


def main(combined: bool = False) -> int:
    # In the standalone build, analysis worker processes start this same program;
    # freeze_support() runs the worker there instead of a second window.
    multiprocessing.freeze_support()
    if importlib.util.find_spec("PySide6") is None:
        message = (
            "The desktop app needs PySide6. Install it with "
            "`uv sync --extra app` or `pip install fpvedit[app]`."
        )
        print(message, file=sys.stderr)
        return 1
    from fpvedit.app.window import FLOWFLIGHT, FPVEDIT, run

    return run(sys.argv, FLOWFLIGHT if combined else FPVEDIT)


def flowflight() -> int:
    """The combined app: the same window with the Score, Edit and Clean tabs."""
    return main(combined=True)
