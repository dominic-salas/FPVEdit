"""Desktop app: drop clips in, check and change the shots the montage engine picks, render.

Runs entirely on this computer. Needs the ``app`` extra (PySide6): ``uv sync --extra app``.
"""

from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    if importlib.util.find_spec("PySide6") is None:
        message = (
            "The desktop app needs PySide6. Install it with "
            "`uv sync --extra app` or `pip install fpv-editor[app]`."
        )
        print(message, file=sys.stderr)
        return 1
    from fpv_editor.app.window import run

    return run(sys.argv)
