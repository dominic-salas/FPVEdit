# PLAN

Living plan for FPVEdit. Full rationale and sources: `docs/plan-proposal.md` (approved by Dom 2026-10-11).

## Approach (approved)
- Classic CV does most of the work: image rotation rate, flow magnitude and coherence, novelty, idle/black/static detection, OSD overlay masking.
- One optional ML model, SigLIP 2 base (`google/siglip2-base-patch16-224`, Apache-2.0): semantic novelty/dedupe and a relative scenic vs low-information signal. Kept only if it measurably beats the CV-only baseline on annotated flights. No claim of trick or gate recognition.
- Scores robust-normalized per flight, fused with documented weights, smoothed; hysteresis → intervals with lead-in/aftermath, maneuver-safe boundaries, dedupe, soft 35–65 % retention.
- One FFmpeg trim/concat re-encode (libx264; NVENC only if a probe encode works), temp file → validate → atomic move. Source never written.

## Dependencies
| Dependency | Reason | Added in |
|---|---|---|
| FFmpeg/ffprobe ≥ 6 (system) | decode `.ts` directly, render, validate | M2 |
| numpy | feature arrays, scoring | M2 |
| opencv-python-headless | optical flow, feature tracking, blur | M2 |
| torch + transformers (`ml` extra) | SigLIP 2 inference | M3 |
| pytest, ruff, mypy (dev) | tests, lint/format, types | M1 |

## Milestones
| # | Milestone | Status |
|---|---|---|
| M1 | Repository and safeguards | done pending Dom review (Windows run and CI not yet run) |
| M2 | CV-only end-to-end baseline | not started |
| M3 | SigLIP 2 integration + measured comparison | not started (blocked here: huggingface.co, download.pytorch.org denied by network policy) |
| M4 | Editing quality + evaluation on annotated flights | not started (needs Dom's annotated flights) |
| M5 | Reliability, Windows and NVIDIA validation | not started |
| M6 | Release readiness | not started |

### Acceptance criteria
- **M1**: ruff, format check, mypy and pytest pass; guard refuses `.ts`/`.mp4`/weights/credentials/>5 MB; hooks and CI reviewed before enabling; LF line endings pinned.
- **M2**: `fpv-editor synthetic.ts` writes a validated, playable MP4; on a synthetic fixture with known active/idle spans, ≥ 90 % of active and ≤ 20 % of idle seconds retained; source hash unchanged; no-audio, 2-second, truncated and existing-output cases tested.
- **M3**: opt-in model tests run real inference; CPU throughput measured; keep/demote decision from measured eval numbers.
- **M4**: time precision/recall, event recall (≥ 50 % covered), retained %, redundant segments reported vs CV-only and motion-only baselines on Dom's flights.
- **M5**: VFR, PTS discontinuities, truncated input, FFmpeg failures, NVENC and CPU fallback each tested or documented with an exact manual command.
- **M6**: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, model provenance, ADRs, validation report; CI green; three reviews reconciled.

## Decisions
- 2026-10-11: build in existing repo `dominic-salas/FPVEdit`, MIT license kept.
- 2026-10-11: Claude works in Dom's local folder; Dom commits. Claude GitHub app not yet installed on FPVEdit (needed later for CI/PRs from Claude).
- Open: enable proposed hooks (`docs/automation.md`)? Allow huggingface.co + download.pytorch.org for the cloud environment (M3)?

## Evaluation data
Dom has an edited MP4 of example clips; if the raw flights it came from are available, clips will be matched back to the raws to produce annotations. Otherwise Dom annotates 2–3 flights: one line per moment (`2:13–2:21 power loop, must keep`) plus definitely-boring stretches.

## Validation log
(newest first; exact commands and outcomes)

### 2026-10-11 — M1 checks (Linux cloud container, Python 3.13.16, uv 0.11.32)
| Command | Result |
|---|---|
| `uv lock` / `uv sync` | passed (14 packages: mypy, pytest, ruff + deps) |
| `uv run ruff check .` | passed |
| `uv run ruff format --check .` | passed (after formatting one file) |
| `uv run mypy` (strict, 8 files) | passed |
| `uv run pytest` | passed, 26 tests |
| Scratch repo, `core.hooksPath .githooks`, staged random `flight.ts`, `git commit` | refused, exit 1, commit not created |
| `guard_git_commit.py` with `git commit` payload, `.ts` staged / nothing staged | exit 2 (blocked) / exit 0 |
| `ruff_on_edit.py` on a badly formatted file / on `cli.py` | exit 2 with findings / exit 0 |
| `git status --ignored` with `.mp4`, `.safetensors`, `.env`, `*_highlight.mp4` | all ignored |
| CI workflow on GitHub | not run (Claude GitHub app not installed on FPVEdit) |
| Windows run of the checks | not run (needs Dom's PC) |
| Found and fixed | pre-commit hook was skipped when not executable on Linux; documented `git add --chmod=+x` |
