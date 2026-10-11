"""PreToolUse hook: before Claude runs `git commit`, refuse if prohibited files are staged.

Reads the Claude Code hook payload from stdin. Exit code 2 blocks the tool call and
shows stderr to Claude; any other command passes through untouched.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

GIT_COMMIT = re.compile(r"(^|[;&|(\s])git(\s+-[^\s]+(\s+[^\s-][^\s]*)?)*\s+commit(?![\w-])")


def is_git_commit(command: str) -> bool:
    return bool(GIT_COMMIT.search(command))


def main() -> int:
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")
    if not is_git_commit(command):
        return 0
    project_dir = Path(payload.get("cwd") or ".")
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_staged.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=project_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
