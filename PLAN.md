# PLAN

Living plan for FPVEdit. Original rationale and model research: `docs/plan-proposal.md` (approved by Dom 2026-10-11).
Revised 2026-10-11 after testing on Dom's first real flights (see "Real-flight review" below).

## Approach (revised 2026-10-11, approved by Dom 2026-10-12)
- **Flight state first.** Split the recording into armed flight, landed/disarmed, OSD menus and stats screens, and no-signal static. Read this from the burned-in Betaflight OSD when present (the "armed" text and stats screen); otherwise fall back to stillness. Downtime, menus and static never reach the edit, and the edit never starts or ends on them.
- **Measure what pilots do, at full frame rate.** One FFmpeg decode of every frame (60 fps) at ~240 px of the active 4:3 picture, with the OSD and props masked automatically. From it:
  - camera rotation (pitch/yaw/roll rate and accumulated rotation) for flips, rolls and spins;
  - flow expansion (things rushing at the camera) for weaving and gap passes, ranked within the flight;
  - the existing cheap features, subsampled to 8 fps.
- **Edit in sessions, like Dom does.** Strong moments are seeds. Each gets a lead-in and aftermath, and seeds closer than ~8 s join into one session that is never cut inside. There is no length target (Dom, 2026-10-12: edit length doesn't matter, keeping each flight's best moments does), so every session that clears a within-flight quality bar is kept. Segments are at least ~5 s, in chronological order, with hard cuts.
- **Failures are excluded where they can be seen.** Crash-disarms and repeated re-arming (from the OSD) remove the failure and a short "cut short" remainder before it. Video alone cannot see every failure (a ground tap with no disarm looks like a flip). An optional `--exclude m:ss-m:ss` list covers these without making flags required. When a good trick runs into a crash, the cut goes in dead space before the failure if the trick still looks complete; otherwise the trick goes too.
- **ML has to earn its place.** SigLIP 2 (Apache-2.0) is tested after the CV pipeline is solid. It has two candidate jobs: scenery, and learning Dom's own keep/cut preferences from his finished edits. It is kept only if it beats the CV pipeline on held-out recordings.
- **Picture enhancement, on by default** (Dom, 2026-10-12). It runs after the edit is chosen, on the kept frames only (plus 3 neighbouring frames at each cut), so it costs a fraction of the flight. Recipe from the 2026-10-12 blind test: motion-compensated fusion of ±3 frames with an OSD guard, horizontal destriping, cross-colour (rainbow) removal, a tone curve and vibrance, a 1440x1080 upscale with luma-guided chroma, and clamped sharpening. Classical CV only: AI upscalers (Real-ESRGAN, BSRGAN, SwinIR, SCUNet) looked painted and lost the blind test. `--no-enhance` gives a fast plain render. CPU cost is ~2 CPU-s per frame; a CUDA version is planned so default runs stay practical.
- **One FFmpeg render pass** that behaves the same on FFmpeg 6, 7 and 9: a select-based chain, source frame rate kept, standard limited-range BT.709 H.264, no audio. Temp file, then validate (fps, frame count, duration, range), then atomic move. Source never written.

## Dependencies
| Dependency | Reason | Added in |
|---|---|---|
| FFmpeg/ffprobe ≥ 6 (system) | decode `.ts`/`.mp4` directly, render, validate | M2 |
| numpy | feature arrays, scoring | M2 |
| opencv-python-headless | optical flow (Farneback, DIS), robust fitting | M2 |
| torch + transformers (`ml` extra) | SigLIP 2 inference, only if it passes its test | M3 |
| pytest, ruff, mypy (dev) | tests, lint/format, types | M1 |

## Milestones
| # | Milestone | Status |
|---|---|---|
| M1 | Repository and safeguards | done (Linux + Windows checks passed; CI not yet run on GitHub) |
| M2 | CV-only end-to-end baseline | done; real-flight result: close to chance on 3 of 4 recordings (see review) |
| M2.5 | Fixes and evaluation harness | in progress (started 2026-10-12) |
| M4 | Editing quality on real flights (moved before M3) | not started |
| ME | Picture enhancement, default on | prototype done (blind-tested 2026-10-12) |
| M3 | ML (SigLIP 2) as a measured addition | after M4 |
| M5 | Reliability, Windows and NVIDIA validation | not started |
| M6 | Release readiness | not started |

### Acceptance criteria
- **M2.5**:
  - Render keeps the source fps on FFmpeg 6.1/7/9, writes limited range, has no audio track, and stays under ~500 MB RAM on a 5-minute 720p60 flight.
  - Validation checks fps, frame count and the colour-range tag.
  - `.gitignore` and `check_staged.py` block `res/`, blackbox logs, .srt/.osd and analysis outputs, with tests.
  - Analysis is cropped to the active picture with an automatic OSD mask.
  - `scripts/match_edit.py` turns a finished edit + raws into a reference-edit annotation.
  - `scripts/evaluate.py` reports a same-length random-edit baseline.
  - An analyze-only mode with a feature cache lets tuning run without rendering.
- **M4**, measured leave-one-recording-out on every evaluation recording (≥ 6 recordings, at least 2 not used for any tuning):
  - Better than ≥ 90 % of same-length random edits on include time (labelled flights) and on IoU with Dom's edit (reference edits).
  - 0 s kept of labelled failures that end in a crash-disarm.
  - No cut inside a continuous session.
  - No downtime, menu, stats or static at the start or end.
  - Event recall ≥ 0.9 on labelled include moments (no length target).
  - Dom's blind rating of 3 new edits ≥ his rating of the M2 edits.
- **ME**:
  - Enhancement is applied to the cut edit only, by default; `--no-enhance` renders exactly as M2.5 does.
  - No OSD damage, no visible flicker or ghosting at cuts or in fast flips (checked on night and day flights).
  - Dom judges the enhanced edit better than the plain one.
  - Runtime per 3-minute flight measured on Dom's CPU and on the RTX 4070 Ti, reported in the validation log; the CUDA path is used automatically when it works and falls back to CPU otherwise.
- **M3**: SigLIP 2 kept only if it improves M4's held-out metrics by a pre-registered margin. Otherwise it is dropped and documented.
- **M5**: VFR, PTS discontinuities, truncated input, FFmpeg failures, 25/30 fps and interlaced DVRs, NVENC and CPU fallback each tested or documented with an exact manual command.
- **M6**: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, model provenance, ADRs, validation report; CI green; three reviews reconciled.

## Real-flight review (2026-10-11)
Data: Dom's flights 182156 and 180033 (hand-labelled, `eval/annotations/`), and clip 1 + clip 2 with Dom's own DaVinci edit, matched frame by frame (corr 0.998) into reference-edit annotations. All analysis used 640x360 60 fps proxies. Footage: analog, 4:3 pillarboxed in 720p60 HEVC full range, Betaflight OSD burned in. Audio was not analysed (Dom: irrelevant).

What was measured:
- **M2 baseline:** its score is near chance (per-sample AUC 0.49-0.62). It beats same-length random edits only on 182156, and ranks failed tricks above good ones on 180033 (failure time kept 0.93 vs 0.62 random). On the clips it keeps 92-98 % (Dom kept 43/32 %).
- **Rotation at 60 fps:**
  - It finds all 19 individual flips in 182156 and separates labelled tricks from other flying (AUC 0.82 / 0.76 vs 0.61 / 0.45 for M2 motion).
  - Fast flips are lost at 25-30 fps: tracking fails, so 1-9 of 19 are found.
  - It fires on failed tricks too.
  - It does not predict which stretches Dom keeps in his own edit (AUC 0.58 / 0.38): he cut several real flips.
- **Flow expansion** separates weaving from other flying within 180033 (AUC 0.86); it does not transfer between flights with a fixed threshold.
- **Scenic** (level horizon, steady, far): finds the one landscape shot (AUC 0.92) but was never kept under leave-one-out tuning. One example is not enough.
- **OSD flight state:** 16/17 arms and 16/17 disarms found; crash-disarms vs landings separated 17/17. One pilot, one layout.
- **Combined prototype** (rotation + expansion + flight state + sessions + OSD failure rules, ~41 % of flight time):
  - 182156: beats chance (99th percentile).
  - 180033 and clip1: above the random mean but not significant.
  - clip2: at or below chance.
  - 180033 still keeps the two failures that have no disarm.
  - Sessions do give Dom-like structure: 2-6 segments of 6-41 s.
- **Render audit:**
  - FFmpeg 6.1 peaks at ~5.9 GB RAM on a 360p proxy.
  - FFmpeg 7.0.2 silently writes 25 fps.
  - Output colour range differs by FFmpeg version.
  - The first segment's audio is ~66 ms early when audio starts late.
  - The pillarbox halves measured flow.

Conclusions:
1. The structure work (flight state, sessions, failure rules, render fixes) is clearly needed and cheap.
2. Rotation is a sound "don't cut this trick" signal, but not a measure of what Dom finds worth keeping.
3. The missing piece is Dom's taste, and four recordings cannot teach or validate it. The most valuable data is more of Dom's own finished edits with their raw files: matching makes them labels at no annotation cost.
Details and per-claim evidence: `docs/review-2026-10-11.md`.

## Known gaps (tracked)
- Failures without a disarm (ground taps) are invisible to video-only detection. Covered by `--exclude` for now; blackbox data could close it later.
- Digital systems (DJI/Walksnail/HDZero), PAL OSD, no OSD, 25/30 fps DVRs and stabilised video are untested; each needs a fallback.
- Lens field of view is unknown, so rotation degrees are approximate; thresholds are relative to the flight.
- MPEG-TS PTS discontinuities are not handled yet (M5).
- NVENC and the `--device` flag arrive with M3/M5.

## Decisions
- 2026-10-11: build in existing repo `dominic-salas/FPVEdit`, MIT license kept.
- 2026-10-11: Claude works in Dom's local folder; Dom commits. Claude GitHub app not yet installed on FPVEdit (needed later for CI/PRs from Claude).
- 2026-10-11: M3 (ML) waited for real flights; after the review it moves after M4 (proposed, needs Dom's OK).
- 2026-10-11: audio is not used for analysis (Dom).
- 2026-10-12: the output has no audio. The recorded track is goggle audio, "completely irrelevant for the edit" (Dom); he adds music himself. This replaces the spec's H.264/AAC output.
- 2026-10-11: blackbox logs are not a product input for now. The format is open; the common decoders are GPL-3.0 (blackbox-tools, orangebox), so the MIT core may only call them as external programs (the Rust `blackbox-log` crate is MIT/Apache-2.0). Their main use is ground truth for checking rotation and failure detection.
- 2026-10-12: revised approach and milestone order approved; M2.5 started (Dom).
- 2026-10-12: no edit-length target; keep each flight's best moments. Crashes are cut, using dead space before them where possible (Dom).
- 2026-10-12: the uncertain 180033 3:29-3:37 weave is left out of scoring (Dom: "if you aren't confident leave it out").
- 2026-10-12: picture enhancement (mcfuse recipe) is on by default and runs after cutting (Dom). Footage is analog, recorded by HDZero BoxPro+ goggles, always 60 fps.
- Open: enable proposed hooks (`docs/automation.md`)? Allow huggingface.co + download.pytorch.org for the cloud environment (M3)?

## Evaluation data
- Hand labels: `eval/annotations/20261009-182156.json`, `20261009-180033.json`, `20261010-203859.json` (Dom's timestamps, incomplete; 180033 corrections applied 2026-10-12 after Dom's answers; 203859 is a raw `.ts` as the tool is meant to accept).
- Reference edits: `eval/annotations/clip1.json`, `clip2.json` (from Dom's `finished edit.mp4`).
- Footage stays on Dom's PC (`D:\FPVEdit\res\testdata1`, gitignored). Only annotations are committed.
- Wanted: more finished edits plus their raws (any number), flights from other conditions (daytime cinematic, gates, a digital system), and optionally 2-3 flights with blackbox logs for ground truth.
- Protocol: leave-one-recording-out tuning with at most a few tuned parameters, a same-length random-edit baseline on every metric, a sealed set of new recordings for milestone sign-off, and Dom's blind ratings as the final arbiter.

## Validation log
(newest first; exact commands and outcomes)

### 2026-10-11 — Real-flight review (cloud container, 640x360 60 fps proxies of Dom's originals)
| Command | Result |
|---|---|
| `fpv-editor <proxy> --keep-intermediates` on 182156 / 180033 | 63 % / 62 % kept; 11/12 and 6/10 labelled events; all 3 labelled failures kept (93 % of their time) |
| same on clip1 / clip2 | 92 % / 98 % kept (flat-score mode); IoU with Dom's edit 0.47 / 0.33, equal to random |
| `python scripts/evaluate.py ANNOT ANALYSIS` | metrics above; reference-edit mode added |
| Frame matching of `finished edit.mp4` to clip 1/2 (64x48 fingerprints, 60 fps) | 9 segments, median corr 0.998, hard cuts, no speed change |
| Investigation (7 independent agents) + 2 adversarial verifiers | see "Real-flight review" and `docs/review-2026-10-11.md` |
| Runtime, 182156 proxy, FFmpeg 6.1 | analysis 24 s, render 98 s, validation 9 s |
| Windows / original 720p HEVC files | not run in this review |

### 2026-10-11 — M2 checks on Dom's PC (Windows 11, Python 3.14.5, FFmpeg 9.0.2)
| Command | Result |
|---|---|
| `uv sync --locked`, `ruff check`, `ruff format --check`, `mypy` | passed |
| `uv run pytest -v` | passed, 100 tests in 15.6 s |
| `check_staged.py --all` | passed |
| 60 s 1920×1080 synthetic flight (idle 10 / active 15 / idle 10 / active 15 / idle 10) | 9.3 s total; 54 % kept; intervals [9.5, 25.6] and [34.5, 50.6]; output h264 1920×1080 + aac, 32.3 s |
| NVENC | `h264_nvenc` available and a 2 s 1080p test encode succeeded (pipeline still uses libx264) |

### 2026-10-11 — M2 checks (Linux cloud container, Python 3.13.16, FFmpeg 6.1.1)
| Command | Result |
|---|---|
| `uv run ruff check .` / `ruff format --check .` / `mypy` (strict, 23 files) | passed |
| `uv run pytest` | passed, 100 tests (incl. 11 FFmpeg end-to-end on generated clips), ~60 s |
| `fpv-editor flight.ts --keep-intermediates` on a 36 s synthetic flight (idle 8 / active 8 / idle 8 / active 6 / idle 6) | 45 % kept, 2 segments; 100 % of active and 10 % of idle seconds kept; source SHA-256 unchanged; output h264 + aac, decodes cleanly |
| Sync regression check: same test with the audio-lead fix disabled | fails (tone onset 1.0 s vs expected 0.5 s), so the test catches the bug |
| Independent video + code-quality reviews | 1 high (A/V timeline drift when audio starts before video, common in DVR .ts), several medium/low; all fixed or listed above |
| Analysis cost | Farneback at 320 px was 20 ms/frame here; moved to 160 px (≈4× cheaper) |
| Windows run, real footage, GPU | not run yet |

### 2026-10-11 — M1 checks on Dom's PC (Windows, D:\FPVEdit)
Hardware/tools: Ryzen 9 5900X (24 threads), 31.9 GB RAM, RTX 4070 Ti (driver 610.74, CUDA 13.3), Python 3.14.5, uv 0.13.0, FFmpeg 9.0.2 (gyan.dev full build), git 2.54.0.windows.1.
| Command | Result |
|---|---|
| `uv sync --locked` | passed |
| `uv run ruff check .` / `ruff format --check .` | passed / passed (16 files) |
| `uv run mypy` | passed (8 files) |
| `uv run pytest -v` | passed, 26 tests |
| `uv run python scripts/check_staged.py --all` | passed (only LICENSE tracked so far, so this was a weak check) |
| Follow-up | Added Python 3.14 to the CI matrix because Dom's machine runs 3.14 |

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
