"""PostToolUse hook: lint and format-check a Python file right after Claude edits it.

Only the edited file is checked, so the hook stays fast. Exit code 2 reports the
problems back to Claude; it never modifies files.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    path = payload.get("tool_input", {}).get("file_path", "")
    if not path.endswith(".py"):
        return 0
    problems = []
    for args in (["check", "--quiet", path], ["format", "--check", "--quiet", path]):
        result = subprocess.run(
            [sys.executable, "-m", "ruff", *args], capture_output=True, text=True
        )
        if result.returncode != 0:
            problems.append(result.stdout + result.stderr)
    if problems:
        sys.stderr.write("".join(problems) or f"ruff reported problems in {path}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
