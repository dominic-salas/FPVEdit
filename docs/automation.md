# Automation and guardrails

Everything here is deterministic and cheap. No hook or CI job runs model inference.

| Check | Where | When | What it does |
|---|---|---|---|
| `scripts/check_staged.py` | git pre-commit hook (opt-in), Claude hook, CI | before a commit / every CI run | Refuses video, audio, model weights, credentials and files > 5 MB |
| `.claude/hooks/guard_git_commit.py` | Claude Code `PreToolUse` on Bash | when Claude runs `git commit` | Runs the check above; blocks the commit if it fails |
| `.claude/hooks/ruff_on_edit.py` | Claude Code `PostToolUse` on Edit/Write | after Claude edits a `.py` file | `ruff check` + `ruff format --check` on that file only; never rewrites it |
| `permissions.ask` | `.claude/settings.proposed.json` | commit, push, reset, clean, release | Claude must ask before running them |
| `.github/workflows/ci.yml` | GitHub Actions | push to `main`, pull requests | Ubuntu + Windows, Python 3.12/3.14: guard, ruff, mypy, pytest (no model tests) |

## Enabling

The Claude Code hooks are **proposed, not active**. To enable them after review:

```
copy .claude\settings.proposed.json .claude\settings.json      # Windows
cp .claude/settings.proposed.json .claude/settings.json        # Linux/macOS
```

The git hook is opt-in per clone:

```
git config core.hooksPath .githooks
```

On Linux/macOS the hook must be executable (`chmod +x .githooks/pre-commit`); record that in Git
once with `git add --chmod=+x .githooks/pre-commit` so every clone gets it. Git for Windows does not
need the executable bit.

Both need `uv sync` to have been run (the hooks use the project environment).

## Test tiers

- Default `uv run pytest`: unit tests + FFmpeg integration tests on generated fixtures. No GPU, no downloads.
- `uv run pytest -m model`: real model inference; downloads weights on first run. Run manually or via a manual CI job (added in milestone 3).
