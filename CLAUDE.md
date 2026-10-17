# FPVEdit — project rules for Claude

One-command, offline CLI (`fpv-editor a.ts b.ts ...`) that turns FPV freestyle recordings into one
H.264 montage of their best shots (`--trim`: one recording edited as a whole flight) (video only: the DVR's goggle audio is irrelevant). Plan and status: `PLAN.md`. Python ≥ 3.11, FFmpeg ≥ 6 on PATH.

## Commands
- Install dev env: `uv sync`
- Fast checks (run after every change): `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`
- Model tests (download weights, opt-in): `uv run pytest -m model`
- Prohibited-file check: `uv run python scripts/check_staged.py` (staged) or `--all`

## Architecture
`src/fpv_editor/`: `cli` (orchestration) → `media` (ffprobe/ffmpeg) → `analysis` (runs the next
steps, optional measurement cache): `picture` (active area, OSD mask) → `measure` + `motion`
(per-frame signals) → `flight` (OSD flight state) → `highlights` (clean flight, `--trim` edit)
→ `montage` (moments, shots, picking across clips) → `render` (shots → MP4 parts → join →
validate). Analysis never renders; rendering reads only the shot list. `app/` is the PySide6
desktop window (`uv sync --extra app`, `uv run fpv-editor-app`): it calls `cli.plan_montage` and
`cli.render_montage`; its editing rules live in `app/model.py` (no Qt).

## Conventions
- Type hints everywhere (mypy strict), `pathlib`, frozen dataclasses for config/results, no global mutable state.
- Subprocess calls use argument lists; never `shell=True`; never build commands from strings.
- Explicit error types; no broad `except` that hides defects.
- Tests must not need a GPU or model downloads unless marked `model`. Fixtures are generated at test time with FFmpeg; never commit media.
- New dependencies need a stated reason in `PLAN.md`.

## Safety (non-negotiable)
- Never modify, move or delete source footage. Never overwrite an existing output without `--force`.
- Never commit footage, renders, model weights or credentials (`.gitignore` + `scripts/check_staged.py`).
- No commit, push, release, file deletion or repo-setting change without Dom's explicit approval.
- Don't claim the model recognizes specific tricks; image-text similarity is not a probability.

## Definition of done for a change
Fast checks pass; diff reviewed; `PLAN.md` updated with the real result (passed / failed / skipped / not run / blocked).
