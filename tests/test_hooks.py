from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_git_commit.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guard_git_commit", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load()


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m 'msg'",
        "git -C repo commit -am x",
        "git -c user.name=x commit",
        "uv run pytest && git commit -m done",
        "cd repo; git commit",
    ],
)
def test_commit_commands_are_detected(command: str) -> None:
    assert guard.is_git_commit(command)


@pytest.mark.parametrize(
    "command",
    ["git status", "git log --grep commit", "echo git commit-tree", "git diff --cached"],
)
def test_other_commands_pass_through(command: str) -> None:
    assert not guard.is_git_commit(command)
