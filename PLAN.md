# PLAN

Living plan for FPVEdit. Original rationale and model research: `docs/plan-proposal.md` (approved by Dom 2026-10-11).
Revised 2026-10-11 after testing on Dom's first real flights (see "Real-flight review" below).

## Approach (revised 2026-10-11, approved by Dom 2026-10-12)
- **The product is a montage maker** (Dom, 2026-10-13). Dom gives it a selection of clips and gets back one montage of their best moments, not a trimmed copy of each flight. Each clip is cleaned with the flight-state rules below (no downtime, static, time on the ground, crashes or re-arm churn), then 8-15 s shots are built around the moments worth showing (flips, spins, close passes, fast flying), with cuts placed in calm moments so no trick is cut short. The best shots across all clips are picked up to a target length (90 s by default, `--length`), at most half from any one clip and never overlapping, and played in the order the clips were given, joined by hard cuts. The single-flight edit stays available as `--trim`.
- **Flight state first.** Split the recording into armed flight, landed/disarmed, OSD menus and stats screens, and no-signal static. Read this from the burned-in Betaflight OSD when present (the "armed" text and stats screen); otherwise fall back to stillness. Downtime, menus and static never reach the edit, and the edit never starts or ends on them.
- **Measure what pilots do, at full frame rate.** One FFmpeg decode of every frame (60 fps) at ~240 px of the active 4:3 picture, with the OSD and props masked automatically. From it:
  - camera rotation (pitch/yaw/roll rate and accumulated rotation) for flips, rolls and spins;
  - flow expansion (things rushing at the camera) for weaving and gap passes, ranked within the flight;
  - the existing cheap features, subsampled to 8 fps.
- **Edit by removing, like Dom does** (changed in M4, 2026-10-13). Dom keeps long uncut stretches and drops what went wrong, so the edit starts from all armed flight and removes, in order: time not flying (boot, menus, stats screens, disarmed), static of 1 s or more, sitting on the ground (3 s of stillness, or 1 s when the picture does not change at all; the moment before lift-off is kept), crashes with the failed move before them, repeated disarms, dull stretches of 15 s or more with no flip, spin, close pass or landing stunt (1 s margins), leftovers under 3 s, and short pieces where the camera hardly turns (a quad on a desk or in a hand). Tricks only protect footage from the dull rule. There is no length target (Dom, 2026-10-12). Segments stay in order, joined by hard cuts. The seed-and-session prototype was dropped: it kept ~41 % and cut real flips Dom wanted.
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
| torch + transformers + Pillow (`ml` extra) | SigLIP 2 inference: tells indoor shots from outdoor ones (CUDA wheels from download.pytorch.org cu128) | M3 |
| PyInstaller (`build` group) | the standalone Windows app (GPL-2.0 with a bootloader exception: the built app may use any licence) | R5 |
| PySide6-Essentials (`app` extra; dev) | desktop app window, drag and drop, threads (Qt for Python, LGPL-3.0, used unmodified) | UI |
| pytest, ruff, mypy (dev) | tests, lint/format, types | M1 |

## Milestones
| # | Milestone | Status |
|---|---|---|
| M1 | Repository and safeguards | done (Linux + Windows checks passed; CI not yet run on GitHub) |
| M2 | CV-only end-to-end baseline | done; real-flight result: close to chance on 3 of 4 recordings (see review) |
| M2.5 | Fixes and evaluation harness | done 2026-10-12 (Linux + Windows checks passed; render peak 520 MB on Dom's 24-thread PC, see log) |
| M4 | Editing quality on real flights (moved before M3) | done 2026-10-13 (cloud and Windows checks passed). Dom's verdict on the blind pairs: per-flight trims "don't really look like a montage of anything", which led to the montage pivot |
| MM | Montage engine: several clips in, one montage out (CLI) | done 2026-10-13 |
| MR | Dom rates montages of his own clips; tune until he is happy | done 2026-10-14 (indoor shots left to the image model and the UI) |
| UI | Local desktop app: drag clips in, preview and swap shots, render on the GPU | done 2026-10-16 (Dom uses it; speed, no-OSD crashes and mid-flip cuts fixed after his feedback) |
| R1 | Lock in: commit, CI green on GitHub, remove the unused M2 pipeline | committed and pushed 2026-10-17; CI result not checked |
| R2 | Reliability: odd and damaged input, error log, first-run FFmpeg check, safe cancel | app part done 2026-10-17; odd-input checks open |
| R3 | Picture enhancement built into the editor (Dom's separate enhancement project), GPU with CPU fallback | waiting on the enhancement project's location |
| R4 | App polish: settings, remembered folders, icon, version, time left | done 2026-10-17 (offscreen checks) |
| R5 | Windows installer with FFmpeg bundled (runs without uv or the project folder); Linux as a plain install | standalone build and installer workflow done 2026-10-17; waiting on Dom to install it |
| R6 | Public 1.0: docs, licences (Qt, FFmpeg), validation report, tagged GitHub release | not started |
| M3 | ML (SigLIP 2) as a measured addition | scenery (indoor vs outdoor) added 2026-10-18 at Dom's request; measured on his labels, not yet on held-out recordings |

Road to 1.0 (Dom, 2026-10-17): 1.0 is a public release. Claude may commit; pushing, releases and repo settings still need Dom's OK. Nothing may cost Dom money or credits: only free, local tools (no paid APIs, Hugging Face inference or cloud GPUs). The old M5 (reliability) and M6 (release readiness) are now R2 and R6; ME is R3.

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
- **MM**, measured on every labelled recording and reference edit (one montage per recording):
  - Montage time inside labelled include moments (or Dom's own edit) beats ≥ 80 % of same-length random shots from the cleaned footage on average, and no recording falls below the random mean by much.
  - 0 s of labelled failures that end in a crash-disarm; failures without a disarm reported, not hidden.
  - Every shot is 8-15 s, inside cleaned footage, and starts and ends in a calm moment.
  - Junk clips (bench, no signal, menus, sitting still) contribute no shots; a selection of only junk is refused with a clear message.
  - Clips of different sizes or frame rates are fitted to one output size and rate; the render passes validation on FFmpeg 6.1/7/9 and on Dom's PC.
- **MR**: Dom rates montages from 4-5 of his clips and prefers them to what he would cut himself in the same time, or tells us what to change.
- **UI**: runs locally on Windows with no internet; drag and drop of clips and folders; shows the chosen shots with thumbnails, lets Dom swap, remove, reorder and trim them, then renders with progress; footage never leaves the PC.
- **ME**:
  - Enhancement is applied to the cut edit only, by default; `--no-enhance` renders exactly as M2.5 does.
  - No OSD damage, no visible flicker or ghosting at cuts or in fast flips (checked on night and day flights).
  - Dom judges the enhanced edit better than the plain one.
  - Runtime per 3-minute flight measured on Dom's CPU and on the RTX 4070 Ti, reported in the validation log; the CUDA path is used automatically when it works and falls back to CPU otherwise.
- **M3**: SigLIP 2 kept only if it improves M4's held-out metrics by a pre-registered margin. Otherwise it is dropped and documented.
- **R1**: everything since `c2cc201` committed in reviewable commits; `sampling.py`, `features.py`, `scoring.py` and their tests removed; CI green on GitHub for Linux and Windows.
- **R2** (was M5): VFR, PTS discontinuities, truncated input, FFmpeg failures, 25/30/50/90/100 fps, 1080p and 4:3/16:9 digital footage without OSD, a full disk and very long clips each tested or documented with an exact manual command; the app writes an error log, checks for FFmpeg on first run with install guidance, and cancels safely in every step.
- **R3**: see ME below; the enhancement code comes from Dom's enhancement project and runs inside the editor's render, with a switch in the app and `--no-enhance` in the CLI.
- **R4**: the app remembers its settings and folders, shows its version and the time left, and has an icon.
- **R5**: a signed-off Windows installer that runs on a clean Windows 11 machine without Python or FFmpeg installed; uninstall leaves nothing behind but the measurement cache, which it offers to delete.
- **R6** (was M6): README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, third-party licences (Qt LGPL-3.0, FFmpeg LGPL build), ADRs, validation report; CI green; independent reviews reconciled; tagged release with the installer attached.

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
- Failures without a disarm (ground taps) are invisible to video-only detection. Covered by `--exclude` for now; blackbox data could close it later. Seen in M4: 180033 1:01-1:05 (ground tap) and 203859 0:16-0:32 (repeated failed gate attempts before a crash) are kept.
- A crash cuts back at most 8 s before the disarm, to the calmest moment. A long failed move before that (180033 1:08-1:12, Dom: drop the weave before it too) is partly kept.
- Dom's own clip edits (clip1/clip2) keep 32-43 %; the editor keeps 80-84 % of those flights. No measured signal predicts which flying he cut there (IoU at chance), so this is taste, not a rule. Goose (Dom kept 86 %) matches well (IoU 0.86).
- OSD menus read as stats screens. Harmless so far (it is downtime either way), but a missed arm between two menu screens can mark a few seconds as armed; the 3 s minimum removes them.
- Without any arm or disarm on screen (no OSD, or none ever shown), only pieces of 6 s or more with a median rotation of 18 deg/s or more are kept. Untested on real no-OSD flights (DJI/Walksnail DVRs record without OSD). Without an OSD, crashes are found from the picture (a sustained tumble into a dead stop); checked only by hiding the OSD signal of analog recordings (2026-10-16).
- Digital systems (DJI/Walksnail/HDZero), PAL OSD, no OSD, 25/30 fps DVRs and stabilised video are untested; each needs a fallback.
- Lens field of view is unknown, so rotation degrees are approximate; thresholds are relative to the flight.
- MPEG-TS PTS discontinuities are not handled yet (M5). FFmpeg only repairs jumps of 10 s or more; a smaller backward jump (goggles resetting their clock mid-file) would make analysis and render disagree. Not seen in Dom's files.
- Variable-frame-rate input (phones, screen captures) is refused with a clear message and a conversion command (M5). The check reads every frame's timestamp: footage counts as constant-rate when 90% of its frame steps are one frame long, so a steady DVR recording with signal loss is accepted. Gaps (missing frames) are reported and the edit plays straight across them; validation counts the frames that really exist.
- `scripts/match_edit.py` keeps all fingerprints in memory (~7 kB per frame at 60 fps): fine for a few clips, ~2-3 GB for an hour of raw footage.
- Encoding uses NVENC when a test encode works (`--device auto`, the default in the CLI and the app), with a libx264 fallback if the GPU encode fails; `--device cpu` or `cuda` forces one. Untested on machines without an NVIDIA GPU other than through the unit tests.
- Analysis runs one process per clip (a quarter of the logical CPUs at most), so a montage takes about as long as its longest clip alone (~24 s per minute of footage on Dom's PC). One long clip is not split: its rotation fit uses one random sequence per clip, and splitting would change the measurements. Splitting with a fixed chunk length (deterministic, but measurements change once) or a faster rotation fit are the next steps if single long clips need to be faster.
- Rendering a part that starts 10 s or more into its clip seeks with `-copyts -noaccurate_seek` and shifts the timeline by the first decoded frame's timestamp (ffprobe). Synthetic MPEG-TS with a nonzero start, MP4 and a recording with a gap give frame-identical output to decoding from the start; on real footage only clips that start at 0 were compared.
- The app's preview and thumbnails seek with input `-ss`; in a `.ts` whose timestamps jump (see PTS discontinuities above) they would show the wrong place. The render is unaffected.
- Closing the app during a step that reports no progress (probing, joining, the final check) waits for that step to finish.

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
- 2026-10-13: the tool becomes a montage maker from a selection of clips (Dom). Shots are long, 8-15 s (Dom's choice). The UI is a local desktop app with drag and drop that uses his GPU, not a website (Dom: "it doesn't need to be web based since it uses the gpu").
- 2026-10-14: the app uses PySide6 (Qt), approved by Dom. Its preview decodes with FFmpeg (raw frames into the window) instead of Qt Multimedia, which needs the much larger PySide6-Addons and a second decoder. The app renders with NVENC when a test encode works, otherwise libx264.
- Open: enable proposed hooks (`docs/automation.md`)? Allow huggingface.co + download.pytorch.org for the cloud environment (M3)?

## Evaluation data
- Hand labels: `eval/annotations/20261009-182156.json`, `20261009-180033.json`, `20261010-203859.json` (Dom's timestamps, incomplete; 180033 corrections applied 2026-10-12 after Dom's answers; 203859 is a raw `.ts` as the tool is meant to accept).
- Reference edits: `eval/annotations/clip1.json`, `clip2.json` (from Dom's `finished edit.mp4`).
- Reference edit `eval/annotations/hdz_0000.reference.json` (from Dom's `goose.mp4`, a 24 fps export: one take trimmed to 28.3-242.4 s with two small gaps, 86% kept). Raw and edit are in `D:\clips`.
- `D:\clips` (Dom's dump, 2026-10-12): 133 clips, 5.4 h. Test material for junk detection (bench/no-signal, menus, sitting still, stats screen), not training data. The raws of `Timeline 1`, `barrhaven 7 batteries` and `bathrobe fpv` are not in it.
- Footage stays on Dom's PC (`D:\FPVEdit\res\testdata1`, gitignored). Only annotations are committed.
- Wanted: more finished edits plus their raws (any number), flights from other conditions (daytime cinematic, gates, a digital system), and optionally 2-3 flights with blackbox logs for ground truth.
- Protocol: leave-one-recording-out tuning with at most a few tuned parameters, a same-length random-edit baseline on every metric, a sealed set of new recordings for milestone sign-off, and Dom's blind ratings as the final arbiter.

## Validation log
(newest first; exact commands and outcomes)

### 2026-10-18 — FLAC music on the SD card, coloured control hints (Dom's PC)
- Dom (2026-10-18): exporting with two FLAC tracks (with cover art) failed with "adding the music failed ... Error submitting a packet to the muxer: Invalid argument"; the log shows the montage was being saved to `F:\DCIM\100HDZRO`, the DVR's FAT32 SD card. The same two tracks muxed fine here into the PC's drive (synthetic 10, 300 and 600 s videos, and real CLI montages of `clip 1.mp4` and `20261010-203859.ts`), so the card is the likely cause; not reproduced on a FAT32 volume (none here, and Dom's card was not written to). Fix: the music is muxed in the local temporary folder and the finished file copied next to the output. New test: FLAC with an attached cover picture, twice. Waiting on Dom to retry on the card.
- Dom: the hints under the timeline should use colour to show which control does what. They are now key chips (CLICK, DRAG, SPACE, F, CTRL+Z) in colours kept apart from the recording colours, the Play and Full screen buttons take their chip's colour, and the recording legend moved to the side panel. Checked in offscreen screenshots, dark and light.
- Checks: ruff, format, mypy clean; pytest 217 passed.

### 2026-10-18 — Trick names from the turn's axis (Dom's PC)
- Dom (2026-10-18): the app only knows flips, spins and close passes; it should know more real tricks.
- Measured on every fast turn of the 16 testdata1 recordings: the camera's net pitch, yaw and roll over the second around it. Dom's labelled backflip is +232 pitch (so positive pitch is a back flip), a front flip -226, his "big side flip" a roll (-194 roll), his "trippy spin" yaw with roll the opposite way (-153 yaw, +91 roll), and rolls show yaw the same way as roll (the camera's up-tilt). A full 360 measures 200-280 degrees.
- Built (`tricks.py`): back flip, front flip, roll, yaw spin, double back/front flip, double roll by the dominant axis and those signs; mixed axes or a motion fit below 0.45 confidence (blur) stay a plain "flip"; power loop = 250 degrees of pitch over 2.5 s with no second turning 180, at least 1.5 s from any flip. The review step, the shot panel and the montage JSON show the names. A split-S namer (half roll, then half pitch) was tried and named five of Dom's labelled flips in 20261009-182156 as split-S, so it was dropped.
- Result on the 16 recordings (fast turns): 29 back flips, 19 yaw spins, 18 plain flips, 15 rolls, 4 front flips, 1 power loop. Every labelled flip, side flip and spin got a name of the right family; the power loop has no label to check, and rolls versus flips are only checked on the one side flip Dom named (he labels both as "flip").
- Not possible from this signal: tricks defined by where the quad goes rather than how it turns (dives, gaps, orbits around an object, matty flips, juicy flicks) would need the quad's attitude and path, which a DVR recording does not carry.
- Checks: ruff, format, mypy (59 files) clean; pytest 216 passed (new `tests/test_tricks.py`).

### 2026-10-18 — Image model: weak indoor shots left out (Dom's PC, RTX 4070 Ti shared with AnaHD)
- Dom (2026-10-18): compare the worst trick with the best, so lame indoor tricks drop out and weaving outdoors stays; he approved downloading the model and PyTorch.
- Motion cannot do it: over the 128 distinct shots of the 16 testdata1 recordings, the indoor clip's shots and Dom's labelled weaves have the same median image speed, rotation, brightness and contrast; only the median |divergence| differs (1.30 vs 0.94) and the ranges overlap (0.56-1.85 vs 0.16-1.72). The shot score even ranks indoor above the weaves (1.42 vs 1.17).
- SigLIP 2 base (`google/siglip2-base-patch16-224`, revision `75de2d5`, Apache-2.0, 375 M parameters, ~1.5 GB in the Hugging Face cache): indoorness = best similarity to five indoor descriptions minus best to five outdoor ones, median of the shot's frames. AUC indoor-clip shots vs Dom's labelled keepers 0.97 (vs all outdoor shots 0.96); one-word prompts "indoors"/"outdoors" gave 0.95. The 14 most indoor-looking shots were 12 indoor ones and 1 keeper.
- Rule (`montage.leave_out_weak_indoor`): a shot with indoorness > 0.03 stays only if its score per sqrt(second) reaches the 75th percentile of the session's outdoor shots; a session without outdoor shots keeps all. On the 16 recordings: 11 of the 14 indoor shots left out, none of Dom's labelled keepers; 117 shots / 20.7 min kept. Margin and quantile were chosen on these same recordings (only one indoor recording exists), so they are not validated on held-out footage.
- Runs after the shots are picked, in its own process so its GPU memory (0.8 GB peak, fp16, batches of 16) is released when done; scores per frame are cached in `scenery.json` in the measurement folder. First run on the 16 recordings 36.2 s (grabbing 3 frames per shot is most of it), cached 0.3 s. App: Find shots runs it and says how many indoor shots were left out; Add a shot brings them back. CLI: on by default when the `ml` extra is installed, `--no-scenery` to skip; without the extra, a note says it was skipped. The installer does not include the model or PyTorch.
- App check (offscreen): indoors cruising + 3 outdoor clips gave 24 shots and left 8 indoor shots out.
- Checks: ruff, format, mypy (57 files) clean; pytest 213 passed (the model is never loaded in normal tests); `pytest -m model tests/test_scenery.py` 1 passed (a drawn room scores more indoor than a drawn forest).

### 2026-10-18 — Every worthy shot instead of a target length (Dom's PC)
- Dom (2026-10-18): a montage length setting misrepresents what he wants; the montage should take all the clips and all the tricks and keep the ones worth keeping.
- Measured on the 16 raw testdata1 recordings (68 min; the two edits left out): every distinct shot (non-overlapping, 4 s apart within a clip) gives 128 shots, 22.6 min. On Dom's hand-labelled recordings (20261009-180033, 20261009-182156, 20261010-203859, clip 1, clip 2) those shots cover 28 of his 35 labelled tricks, and 29 of the 37 shots in those recordings overlap something he labelled (the labels are incomplete). Keeping only shots scoring at least a share of the 90th-percentile score dropped labelled tricks fast: share 0.3 kept 81 shots and 23/35 tricks, 0.5 kept 40 shots and 13/35. Low-scoring shots are often weaves of close passes that Dom marked as keepers, so no score threshold was added.
- Change: `MontageConfig.length_s` is None by default, so `pick_shots` keeps every distinct shot; `--length` still makes a montage of about that length, best shots first. The app's montage length box and setting are gone; step 1 says it keeps every flip, spin and close pass, and the review step is where shots are removed.
- Missed labelled tricks (7): two weaves at the start of 180033, flips at 1:31 and 4:02 of 182156, a spin and a flip at 1:34 and 1:43 of 203859, one kept moment of clip 2; these are moments the detector does not find or that sit inside the 4 s gap next to another shot.
- Checks: ruff, format, mypy clean; pytest 209 passed.

### 2026-10-18 — Full screen, background music, export window (Dom's PC)
- Dom (2026-10-18): the player must go full screen; the panel beside the video used too much space for help text; where the export goes was unclear; add background music (several tracks in order, the recordings' audio is ignored); a YouTube upload like DaVinci Resolve's would be good too.
- Full screen: a button, F or a double-click on the video moves the player and timeline to a full-screen window (Esc or F leaves); the preview decodes at 720p there instead of 360p. The help text is now one line of hints under the timeline; the panel holds the montage length, Add a shot and the music.
- Music: `--music FILE` (repeatable) in the CLI and a music list in the review step (add, reorder, remove). Tracks play one after another from the start of the montage; when they are longer they are cut and faded out over 2 s at the end, when shorter the rest is silent (`render.soundtrack_filter`). The joined video is copied, not encoded again, and the music is encoded as AAC 192 kb/s 48 kHz stereo; validation then requires the AAC track. The preview plays the music with ffplay (it ships with FFmpeg; the Windows build now copies ffplay.exe when the FFmpeg build has it), without ffplay the preview is silent and says so. Changing `media.py` renewed the measurement cache once.
- Export: Export opens a window with the folder, the file name, the encoding, how the music fits and "open the folder when it is done" (remembered); the bar shows the full path it saves to.
- Dot-matrix title: the e was a row taller than the other lowercase letters; the face now has ascender, x-height and descender rows and a test checks every letter's x-height and baseline.
- YouTube upload: not built. It needs a Google Cloud project with the YouTube Data API and an OAuth client that Dom creates (free), and it sends the montage to YouTube, so it waits for Dom's go.
- Checked: real export of `clip 1.mp4` with two music tracks through the CLI (`--length 20`): NVENC, 2 shots, h264 18.88 s and aac 18.88 s, 56 s; offscreen screenshots of the review step with music, full screen, the export window and the done page. Fast checks: ruff, format, mypy (54 files) clean; pytest 208 passed. Music in the preview and the full-screen window were not checked by hand (offscreen has no sound and no real screen).

### 2026-10-18 — Review step as a montage player (Dom's PC, offscreen)
- Dom (2026-10-18): step 3 should be a player for the whole montage, YouTube-like, with a timeline that shows which recording each shot comes from; editing a shot should be something you open per shot; reordering by drag and drop.
- Built (`app/review.py`, `app/timeline.py`, `app/preview.py`): the player plays the montage shot after shot (the next shot's FFmpeg is opened while the current one plays, so there is no pause between shots), with play/pause (Space) and the time. Under it, a timeline with one block per shot, as wide as the shot is long, its thumbnail, number and recording name, coloured by recording, a ruler and a playhead; a legend names the colours. Click a block to jump there and open the shot in the side panel (trim, swap, remove, loop just this shot, previous/next, done); click the ruler to jump; drag a block to move the shot. Every edit can be undone (Ctrl+Z). The time mapping between the montage and the recordings is in `app/model.py` (no Qt) and tested.
- Checked: offscreen screenshots of the review step playing and with a shot open (3 testdata1 clips, 9 shots / 1:31.7); app, model and window tests 26 passed, including a simulated click, drag and ruler click on the timeline. Not checked by hand on the real screen yet.

### 2026-10-17 — Clip cards, background analysis, faster motion (Dom's PC, offscreen)
- Dom's feedback on the steps, acted on: clips are now cards (`app/cliplist.py`) with a thumbnail from 30 % into the clip, the length, size and rate, and the recording time (from a Walksnail-style name `YYYYMMDD-HHMMSS`, else the file's modified time labelled `FILE`; DVR files carry no creation time, checked on Walksnail and HDZero files); step 1's list sits on the dot grid like the other steps.
- Clips are analysed in worker processes from the moment they are added (`app/analyser.py`), each card shows its progress, and Find shots only waits for what is left, with every clip's progress on the progress screen. Results are kept per file, so a new montage length or clip order picks again at once (0.01-0.02 s). The engine gained `combine_clips` and `ClipResult.moved_to` for this; `plan_montage` uses the same combining, so the CLI is unchanged.
- Motion measuring: the rotation fit scores all RANSAC candidates in one batch with the same random draws, and `np.cross`/`einsum` calls were replaced by cheaper ones. Measurements are bit-identical (all arrays of 3600 frames of `clip 1.mp4` compared); measuring those frames took 16.5 s before and 11.7 s after (-29 %). Decoding alone is ~15 % of the analysis (CPU decode of the 3:10 clip: 18.7 s for the whole clip), so the analysis is CPU-bound in Python, not in FFmpeg. The code change renews the measurement cache once.
- Timings while Dom's AnaHD SeedVR2 job was running on the same PC (so slower than normal): 3 uncached clips (3:04, 3:10, 3:20) added and Find shots clicked 8 s later: ready 71 s after adding. 1 cached + 2 uncached (4:03, 4:24): 96 s. A second Find shots on analysed clips: 0.01 s. No before/after comparison of whole runs under equal load was made.
- Fonts: still Segoe UI and Consolas/Cascadia Mono; bundling Geist, Geist Mono and Doto (OFL) waits on Dom's OK to download them.
- Not done: splitting a long clip into parallel pieces (one 9-minute clip is still analysed by one worker).
- Checks: ruff, ruff format, mypy (51 files) clean; pytest 204 passed (new: recording-time parsing, per-clip analysis combined in any order equals `plan_montage`, the window analysing in the background and picking at once after a reorder).

### 2026-10-17 — UI redesign status (handed to a new thread)
- Restyled after dvsalas.com (`app/theme.py`: dark/light palettes, mono uppercase labels, red accent; `app/look.py`: dot-matrix title, status dot) and rebuilt as four steps (`app/window.py`: add clips, find shots, review, export/done). Committed and pushed (`f73c104`, `be7103b`); checked by offscreen screenshots and the app tests, not yet by hand.
- Dom's feedback on the steps (2026-10-17), still open:
  - the theme is not consistent: step 1's text uses default fonts. Bundle Geist, Geist Mono and Doto (OFL, google/fonts `ofl/geist`, `ofl/geistmono`, `ofl/doto`, ~1 MB) in `app/fonts/` with their OFL.txt; Dom asked for the site's look, the download was not done yet;
  - clips should show a thumbnail, their length and when they were recorded (timestamp);
  - the loading screen is good, but finding shots takes long: start analysing in the background as soon as clips are added, and consider splitting long clips;
  - the review page is good; Dom will judge it after using it.

### 2026-10-17 — Installer on GitHub Actions
- `.github/workflows/release.yml` (tag `v*` or run by hand) builds the standalone app on a Windows runner with BtbN's GPL shared FFmpeg 9.0, smoke-tests it (`fpv-editor.exe --version`, bundled `ffmpeg -version`) and packs it with Inno Setup (`packaging/installer.iss`: per-user install by default, Start menu and optional desktop shortcut, offers to remove `%LOCALAPPDATA%\fpv-editor` on uninstall). A tag also drafts a GitHub release; publishing stays manual.
- Manual run 39072509889: success in 3.6 min; installer artifact 112 MB (first run failed: Git Bash rewrote `/DAppVersion` as a path; fixed with `MSYS_NO_PATHCONV`).
- CI on the pushed commits: green, except one Windows run where a cancelled render's part file was still locked when the temporary folder was cleaned up, so the cancel surfaced as a PermissionError. Fixed by ignoring cleanup errors of the temporary folder.
- Not run: installing on a clean Windows machine (R5 acceptance), an uninstall.

### 2026-10-17 — R1 lock-in, R2/R4 app polish, R5 standalone build (Dom's PC)
- R1: work since `c2cc201` committed in reviewable commits and pushed (Dom's OK); the unused M2 pipeline (`sampling`, `features`, `scoring` and their tests) removed; CI on Linux now installs Qt's system libraries and runs three jobs (Linux 3.11 and 3.14, Windows 3.14) to stay within the free Actions minutes of the private repository. CI result on GitHub: not checked yet.
- R4 app polish: File/Edit/Help menus with shortcuts; settings dialog (montage length, GPU or CPU encoding, output folder, measurement cache size with Clear, which only deletes `*.measurements.npz`); remembered settings, window size and last clip folder; clips show usable length and shot count after Find shots; warnings in a Notes button; time left during work; app icon drawn in code; version in the title; About box with licences. Checked by offscreen screenshots (empty window and the 5 testdata1 clips, 9 shots / 92.1 s).
- R2 (app part): log file in `%LOCALAPPDATA%\fpv-editor\logs` with uncaught errors from any thread; when FFmpeg is missing the app explains how to install it and lets the user choose its folder; closing while busy asks first. Odd-input checks (timestamp jumps, other rates and sizes, a full disk) are still open.
- R5 standalone build: `scripts/build_windows.py` (PyInstaller 6.22, new `build` dependency group) makes `dist\FPV Editor` with `FPV Editor.exe`, `fpv-editor.exe`, a bundled `ffmpeg` folder and `licenses` (FPV Editor MIT, Qt LGPL-3.0, NumPy, OpenCV, FFmpeg GPL or LGPL by its configuration, third-party notice). OpenCV's unused FFmpeg DLL (30 MB) is dropped. Fixes needed for the frozen build: `multiprocessing.freeze_support()` in both entry points; `find_tools` looks in the bundled `ffmpeg` folder first; the measurement cache key hashes compiled code when the sources are not shipped.
- Frozen CLI with FFmpeg removed from PATH, `low light` + `urban building dive`, fresh cache: 2 clips, 8:27 usable, 9 shots / 1:22, NVENC, validated, 244 s (266 s from source: same work). The built app itself was not clicked through by hand.
- Size: 682 MB with Dom's static gyan.dev FFmpeg (228 MB per program); the app alone is ~240 MB. The release workflow uses a shared FFmpeg build instead.
- Fast checks: ruff, ruff format (56 files), mypy (47 files) clean; app, media and pipeline tests 63 passed (the full suite last passed at 196 before these changes plus 5 new app tests).

### 2026-10-16 — Footage without OSD, flips cut short, stricter picking (Dom's PC)
- Dom (2026-10-16): his digital footage is recorded without the HDZero/Betaflight OSD, so the app must handle no OSD; picking could be "a bit more strict based on clips uploaded"; some shots are cut off mid-flip.
- None of the 18 testdata1 recordings lacks an OSD, so no-OSD footage was simulated: the same measurements of the five labelled recordings with the OSD signal removed (flight state from the picture alone). Digital footage itself (HDZero/DJI picture, frame size, rate, field of view) is still untested.
- Without the OSD, crashes were not found: 203859 kept 13 s of labelled failures in its shots (0.35 of shot time in Dom's moments, 0.94 with the OSD). Picture-only crash rule: the quad coming to rest (time on the ground, or a dead stop of 0.5 s or more: image speed under 0.08 widths/s and turning under 10 deg/s) right after a sustained tumble (90th percentile of the 0.25 s median rotation rate of 300 deg/s or more in the 1.3 s before) is treated like a crash-disarm, with the same cut back to the failed move. Speed alone is not enough without a disarm (a fast flight into a deliberate stop is not a crash). Result without OSD (one 40 s montage per recording; share of shot time in Dom's moments / labelled failure seconds in shots): 180033 0.47 / 4.0 (OSD 0.37 / 4.0), 182156 0.58 / 0 (0.58 / 0), 203859 0.91 / 1.2 (0.93 / 0.5), clip1 0.62 (0.72), clip2 0.54 (0.34). 180033 still keeps 40 s of labelled failures in its usable footage without the OSD (9.5 s with it), as failures without a crash stay invisible (known gap).
- Mid-flip cuts: the shot edges in the default 5-clip montage were all calm (max rotation within 0.3 s of a cut 280 deg/s, at the urban start). Two causes were fixed: frames whose motion could not be measured (blur in a fast flip) counted as perfectly calm, and nothing stopped a cut next to a fast turn or inside a turn the moment detector had not counted. Now no shot edge lies inside a turn (0.3 s median rate 90 deg/s or more) or within 0.4 s of a fast one (0.1 s median 150 deg/s or more); the start moves back before the turn (or after it), the end on past it (or before it), keeping 8-15 s. New test: candidate edges in or next to a flip 0 (13 with the rule disabled). Labelled recordings with the OSD, mean share of shot time in Dom's moments 0.53 → 0.59 (clip1 0.42 → 0.72, others within 0.02); failure seconds 5.5 → 4.5.
- Stricter picking, tested and not adopted: dropping picked shots whose score per square root of length is below a share of the top three's (one 90 s montage of the five labelled recordings) did not help: share in Dom's moments 0.60 at ratios up to 0.6, 0.51 at 0.7 (5 shots, 70 s). The score does not separate Dom's choices at the margin, so a score bar only shortens the montage. Waiting on Dom to say what "stricter" should mean.
- Fast checks: ruff, ruff format (55 files), mypy (46 files) clean; pytest 212 passed in 356 s (new: a tumble into a dead stop is a crash without OSD, a flip into a hover is not; no candidate shot edge in or next to a flip).

### 2026-10-15 — App shortcut (Dom's PC)
- Dom asked for an exe to open the app easily. uv already installs `.venv\Scripts\fpv-editor-app.exe` (a GUI launcher, no console window); a desktop shortcut `FPV Editor` now points to it (working folder `D:\FPVEdit`). Checked: the exe starts and stays up (offscreen, 8 s), and the parallel analysis works under `pythonw` (2 cached clips, 1.6 s). A standalone build that runs without the project folder (PyInstaller or similar) is not made; it would be ~300 MB and still need FFmpeg on PATH.

### 2026-10-14 — Speed (Dom's PC: Ryzen 9 5900X, RTX 4070 Ti, FFmpeg 9.0.2)
- Profile (one 4-min clip, 204314): measuring took 95 s, of which ~90 s is the per-frame Python work in one thread (the robust rotation fit alone 44 s, DIS flow 6 s). Decoding in FFmpeg took 38 s on the CPU (27 s with `-hwaccel cuda`) and runs alongside, so GPU decoding would not help until the Python side is faster.
- Changes: clips are analysed in parallel processes (`plan_montage(workers=...)`, default CPUs/4; progress and cancel work across processes); render parts seek to their first shot instead of decoding from the clip start, and shots of one clip more than 20 s apart get parts of their own; the CLI gains `--device auto|cpu|cuda` (NVENC by default when it works, libx264 fallback) and prints the encoder.
- 5 testdata1 clips (26:14 of usable footage), fresh cache: analysis 525 s → 246 s (bounded by the longest clip, indoors cruising), same 8 shots and 362 candidates; cached re-analysis 1.7 s.
- Same clips, cached, whole CLI run including render and validation: 170 s before → 51 s with `--device cpu`, 36 s with NVENC. Before the part split the seek alone gave 93 s / 77 s (the indoor clip's 4 shots span 55-511 s).
- Seek check on real footage: an 8 s shot at 150 s of `20261010-203859.ts` and at 497 s of `indoors cruising.mp4` rendered with and without seeking: 480 frames each, framemd5 identical.
- The app's Find shots through the process pool (offscreen, cached): 1.9 s, 8 shots.
- Fast checks: ruff, ruff format (55 files), mypy (46 files) clean; pytest 209 passed in 198 s (new: seek vs no-seek frame identity on TS/MP4/gap clips, parallel analysis gives the same plan, progress and cancel with 1 and 2 workers, device choice and encoder fallback).

### 2026-10-14 — UI desktop app (Dom's PC: Windows, Python 3.14.5, FFmpeg 9.0.2, RTX 4070 Ti)
- Built: `fpv-editor-app` (PySide6 6.12). Drop clips or folders, reorder them, Find shots (background analysis with progress and cancel, measurements cached in `%LOCALAPPDATA%\fpv-editor\measurements`), shot list with thumbnails, a looping FFmpeg preview, trim start/end, remove, drag to reorder, swap for another candidate from the same clip, add any candidate, undo, then Render (NVENC when a test encode works, else libx264; progress, cancel, validation, never overwrites without asking). The engine was split into `plan_montage` and `render_montage`; shots may now be in any order (a part per run of forward shots from one clip). The CLI's behaviour is unchanged.
- Fast checks: ruff, ruff format (54 files), mypy strict (45 files) clean; pytest 202 passed in 179 s (new: shot-list rules, progress and cancel during analysis and render with nothing cached or written, a reordered render that validates, frame-count progress, an offscreen window smoke test). `check_staged.py --all` clean.
- Real run on 5 testdata1 clips (204314, nighttime, low light, urban building dive, indoors cruising), app code path without the window: analysis 525 s (uncached), the same 8 shots / 88.4 s as the MM run, 362 candidates. Edited like Dom would (first indoor shot swapped for an alternative, second removed, last shot moved first, one trim): 7 shots, 83.0 s, rendered with NVENC in 198 s and validated: h264 1280x720 60 fps yuv420p tv, 4982 frames. Most of the render time is decoding from the start of each clip (see Known gaps). A cached re-analysis took 2.5 s.
- Independent reviews (code quality, FFmpeg): fixed a save-as path that could replace a file the dialog did not ask about, edits and drags while a task runs, Add enabled for swap candidates, cancel being ignored after the final check, the NVENC fallback catching non-encode failures, console windows flashing on every ffmpeg run, slow close with queued thumbnails, stale preview frames, double preview restarts on trim, trimmed shots losing their moments, and an import error reported as "PySide6 missing". Also fixed in the engine: the expected frame count when a clip with frame gaps is converted to another rate, and the render's read limit when audio starts before video. Window offscreen screenshot checked by eye.
- Not run: the window by hand on Dom's desktop (drag and drop, real clicks) and a CPU-only fallback render through the window. Waiting on Dom's try-out.

### 2026-10-14 — MR rating and close-pass change (cloud container; Dom's PC)
- Dom's rating of the 5-clip montage: he likes it overall and the cut points are good. Two shots from indoors cruising (139.7-147.7 and 370.3-382.0), picked almost only for close passes, are not impressive: indoors everything is close, so doorways look like near misses.
- Change: a close pass must also be `pass_prominence` (0.75) above the median approach over the 4 s around it. Labelled recordings: mean beats-random 75 % to 82 %, failure time kept unchanged (0.5 s). New unit test: a long stretch of steady closeness gives no passes, a short sudden one does. Dom's PC: ruff, format, mypy clean; pytest 188 passed.
- It does not remove the two indoor shots (re-render picked the same 8 shots). Their passes stand out from their surroundings (prominence 0.95-3.75) and the indoor clip's motion statistics (approach, speed, turning, passes per minute) match the outdoor 204314 closely. Capping the score of passes per shot lowered the label score to 64-72 %. So motion alone cannot tell indoor doorways from outdoor gaps; recognising indoor footage waits for the image model, and the UI lets Dom drop or swap a shot by hand. Dom chose to move on to the UI (2026-10-14).

### 2026-10-13 — MM montage engine (cloud container; Dom's PC)
- Dom's PC (Windows, FFmpeg 9.0.2): ruff, format, mypy pass; pytest 187 passed. Montage of 5 testdata1 clips (204314, nighttime, low light, urban building dive, indoors cruising; 26:14 of usable footage, no cache): exit 0, 706 s, 8 shots / 88.4 s, shots from all 5 clips (4 from indoors cruising, at the half-per-clip cap), 1280x720 60 fps, 5304 frames, no warnings. Waiting on Dom's rating.
- Fast checks: ruff, ruff format, mypy strict (37 files) pass; pytest 186 passed on FFmpeg 6.1.1. Render, pipeline and montage tests (44) also pass on FFmpeg 9.0.2. FFmpeg 7.0.2: not run; the static 7.0.2 build crashes (segfault) on any decode in this container, including a 3-second test file, so the result says nothing about the code.
- New tests: moments (a flip is found, calm cruising gives none), usable footage (no shots before arming, on the ground, in a crash or on the stats screen), shots (8-15 s, inside usable footage, the flip shown whole with a lead-in), a flip that crashes gives no shot, a desk recording gives none, picking (target length, best first, no overlap, a 4 s gap within a clip, at most half per clip, clip order), and end to end: two clips of different size and rate (320x180 at 30 fps .ts and 480x360 at 60 fps .mp4) become one validated montage at 480x360 with shots from both, inputs unchanged; idle clips are left out with a warning, only idle clips are refused with no output, a clip given twice is used once, `--analyze-only` writes only the plan, `--trim` with two inputs and a negative `--length` are refused.
- Shot picking on the labelled recordings (360p proxies, one 40 s montage per recording; share of montage time inside Dom's labelled moments or his own edit, vs 300 random sets of same-length shots from the same usable footage):

| Recording | Shots | In Dom's moments | Random | Beats random | Failure time kept |
|---|---|---|---|---|---|
| 180033 | 4 (42 s) | 0.57 | 0.44 | 76 % | 0.0 s (random 1.1 s) |
| 182156 | 3 (38 s) | 0.59 | 0.41 | 86 % | 0.0 s |
| 203859 | 5 (44 s) | 0.94 | 0.54 | 99 % | 0.4 s (random 3.2 s) |
| clip1 (Dom's edit) | 4 (43 s) | 0.46 | 0.49 | 44 % | - |
| clip2 (Dom's edit) | 4 (42 s) | 0.61 | 0.40 | 89 % | - |
| hdz_0000 (goose) | 4 (40 s) | 0.99 | 0.93 | 54 % | - |

  Mean 75 % (criterion: 80 %). Weights were chosen from a small grid on these same six recordings (balanced for the fewest failures over the best mean), so this is not a held-out result; Dom's ratings decide. The 0.4 s is the edge of a failed gate attempt in 203859 that has no disarm (known gap). All 11 junk clips (bench, no signal, desk, car, crash-only) give no shots.
- Five of Dom's best flights (145717, 201057, 204314, hdz_0053, nighttime; proxies) give an 89 s draft of 10 shots: 5 from 145717 (48 s, near the half-per-clip cap), 2 from 201057, 1 from 204314, 2 from nighttime, none from hdz_0053.

### 2026-10-13 — M4 checks (cloud container on proxies; Dom's PC on the originals)
- What changed:
  - New `flight.py` reads flight state from the OSD: ARMED message, arming logo, stats screen (disarm), boot screen, static and the grey no-signal screen. A recording that opens on a boot screen starts disarmed; a stats screen while disarmed means an arm was missed.
  - `highlights.py` is rewritten as the removal-based policy described under Approach.
  - `analysis.py` now caches the per-frame measurements (`*.measurements.npz`) instead of the M2 samples.
  - `sampling.py`, `features.py` and `scoring.py` are no longer used by the pipeline. They are kept until Dom OKs deleting them.
  - `scripts/evaluate.py`: the reference-edit printout had the editor's and the human edit's numbers under each other's names (metrics were right).
- OSD reading fixes:
  - Glyphs are now bright pixels with a much darker pixel within 2 px. The old rule required near-black outlines and missed the OSD on 4 night/indoor clips; the OSD is now found on all 28 recordings that have one.
  - When the stats screen stays up for a long stretch (3 of 30 recordings), its text no longer counts as part of the normal layout.
- First run on Dom's PC (Windows, originals, earlier rules): ruff, ruff format (32 files), mypy (34 files) clean; pytest 168 passed in 152 s. Full-resolution scores were close to the proxies except 182156: include time 0.85, because calm hovers (a landscape shot at 0:26-0:41, a moment inside the 4:02 flip) were cut as "on the ground". Fix: one still second only counts as ground when the picture does not change at all (ground ~0.001 frame change, hover 0.004 and up); otherwise 3 s. Dull stretches were raised from 10 s to 15 s, which keeps every labelled good moment; a short piece must also show the camera turning (median 18 deg/s), which keeps the desk and no-OSD bench clips out.
- Six evaluation recordings with the final rules (proxies), each compared with 1000 same-length random edits (share beaten in brackets):
  | Recording | Kept | Result |
  |---|---|---|
  | 180033 | 79 % | event recall 1.00 (96 %), include time 1.00 (99 %), failure time kept 0.21 (98 %), no session cut |
  | 182156 | 66 % | event recall 1.00 (99 %), include time 1.00 (100 %), no session cut |
  | 203859 | 79 % | event recall 1.00 (94 %), include time 1.00 (99 %), failure time kept 0.71 (77 %), no session cut |
  | clip1 | 85 % | IoU with Dom's edit 0.49 (53 %) |
  | clip2 | 84 % | IoU 0.38 (58 %) |
  | hdz_0000 (goose) | 88 % | IoU 0.87 (71 %) |
- Failures that end in a crash-disarm keep at most 1 s: 180033 1:14-1:19 keeps 1.0 s (the re-arm), 2:51-3:00 0 s, 3:58-4:19 0.8 s; 203859 2:03-2:07 0.5 s. The failures that are kept have no disarm (see Known gaps).
- M2 on the same proxies (2026-10-12): 180033 kept 73 % of failure time, 203859 kept 95 % of the flight, clip IoU 0.47/0.32.
- 24 clips from `D:\clips` that were never used for tuning:
  - All 11 junk clips give nothing (the CLI stops with "no flying worth keeping"): 5 bench/grey no-signal (two at 50 fps), 3 desk clips, 1 in a car, 2 without OSD, 1 short crash clip.
  - The 13 real flights keep 17-87 %. Contact sheets of 9 were checked by eye: every edit starts at take-off and ends before the stats screen, and menus, reboots and ground time are cut. Indoor flights out of a window are kept.
- The parameters were set by inspecting all six evaluation recordings, so this is not a leave-one-out result. The held-out evidence is the 24 untuned clips and Dom's blind rating.
- Cloud, final rules: ruff, ruff format and mypy clean; pytest 169 passed.

### 2026-10-12 — Frame timing fix (Linux cloud container, FFmpeg 6.1.1 / 7.0.2 / 9.0.2; Dom's PC)
- Rate check over all 133 clips in `D:\clips` (on Dom's PC): the old average-rate check refused 1 clip, `20261008-134959` (average 58.27 fps); `hdz_0020` passed at 1.96%. Both are steady 60 fps that freeze ~1.13 s at the end (signal loss) and finish with ~35 frames at 50 fps. Clip length, not timing, decided the result.
- A synthetic 60 fps TS with 1.13 s of frames missing failed the old render validation ("292 frames, expected about 360"). Now it renders and validates with exactly 292 frames.
- Linux: `ruff check`, `ruff format --check`, `mypy` clean; `pytest`: 161 passed on FFmpeg 6.1.1; media/render/pipeline tests (66) pass on FFmpeg 9.0.2. A 5:00 HEVC MPEG-TS (looped 203859) still renders and validates: 18015/18015 frames, no false gap warning.
- First Windows run: both real gap clips were accepted but failed validation (527 frames rendered, 495 expected; 1281 vs 1245). Cause: `setpts` renumbers frames but keeps their durations, so in MP4 the frame before the gap still lasts ~1.1 s and the constant-rate encoder repeated it ~60 times, then dropped later frames. The synthetic TS test missed it because enough frames followed the gap to even out the count. Fix: an `fps` filter after `setpts` resets every duration. The regression test now uses an MP4 with a gap and a 50 fps tail and checks that no frame repeats (it fails on the old graph: 248 frames vs 202). Same result on FFmpeg 6.1.1, 7.0.2 (MP4) and 9.0.2.
- Windows after the fix (FFmpeg 9.0.2): ruff, ruff format (38 files), mypy (29 files) clean; pytest 161 passed in 24 s. `20261008-134959.mp4`: gap warning (1.13 s at 39.3 s), 0:41 → 0:08, 495/495 frames. `hdz_0020.mp4`: gap warning (1.14 s at 58.5 s), 1:00 → 0:21, 1245/1245 frames. `20261010-203859.ts`: no gap warning, 3:20 → 3:09, 11345 frames as before. framemd5: no frame repeats in any output. `check_staged.py --all` clean; nothing committed.
- `scripts/match_edit.py`: title/black frames (match < 0.5) and stray cross-fade frames far from kept footage are left out instead of being marked `ignore` at random raw times.

### 2026-10-12 — M2.5 checks on Dom's PC (Windows, Python 3.14.5, FFmpeg 9.0.2 gyan.dev full build)
- `uv sync` OK; ruff, ruff format (38 files), mypy (29 files) clean; pytest 155 passed in 27 s.
- Full-resolution real flights, `--keep-intermediates --cache-dir work\m25\cache` (memory sampled every second from Get-Process PeakWorkingSet64):
  | Flight | Wall time | Render time / peak RAM | Edit | Output |
  |---|---|---|---|---|
  | 20261009-182156.mp4 | 117 s | ~76 s / 522 MB | 4:43 → 3:45 (79 %), 4 segments, flat-score warning | h264 1280x720 yuv420p tv bt709 60/1, 13472 frames |
  | 20261009-180033.mp4 | 105 s | ~62 s / 520 MB | 4:19 → 2:33 (59 %), 5 segments | same format, 9173 frames |
  | 20261010-203859.ts | 92 s | ~56 s / 522 MB | 3:20 → 3:09 (95 %), 1 segment, flat-score warning | same format, 11345 frames |
- Frame counts equal edited duration × 60 in every output; video only. Analysis decode peaked at 111 MB, validation at 91 MB.
- Active area [166, 0, 948, 720] on all three; OSD mask 10-11 % of the picture.
- Cached `--analyze-only` re-run of 203859: 0.2 s, no FFmpeg started, same result.
- `check_staged.py --all` clean; nothing in `res\` touched; nothing committed.
- RAM: the ~500 MB target is met in spirit but not to the letter (520 MB). The peak comes from x264 and HEVC decoder threads scaling with the 24-thread CPU (213 MB on 4 cores); the old render peaked near 6 GB. Accepted; revisit only if it matters.
- The plain render keeps the full 1280x720 frame with its pillarbox bars (like Dom's own edit). Cropping to the 4:3 picture comes with enhancement (ME), which outputs 1440x1080.

### 2026-10-12 — M2.5 checks (Linux cloud container, Python 3.13, FFmpeg 6.1.1 / 7.0.2 / 9.0.2)
- Fast checks: ruff, ruff format, mypy (29 files) clean; pytest 155 passed.
- Render matrix (`render(...)` + `validate_output(...)`, preset ultrafast):
  - Full-range HEVC MPEG-TS from Dom's 203859 flight looped to 5:00 (`-stream_loop`, stream copy), 30 cuts, 2:15 kept: 60/1 fps, 8107/8107 frames, yuv420p limited range, no audio, VALID on FFmpeg 6.1 (115 s, peak 213 MB) and 9.0 (107 s, peak 133 MB).
  - Full-range H.264 MP4 with late audio, 4 cuts: VALID with identical frame counts on 6.1, 7.0.2 and 9.0 (peak 147 MB). The 7.0.2 test binary (static johnvansickle build) segfaults on any MPEG-TS or HEVC input here, so 7.x was only tested on MP4.
- Independent reviews (video, code quality): no high-severity defects. Fixed: cache key now includes the measuring code and FFmpeg version; damaged cache files are re-measured; unwritable cache only warns; match_edit never replaces hand labels and marks weak matches as unknown; cut boundaries snap to the nearest frame (robust to 1 ms timebases); analysis samples aligned to i/fps; averaged frame rates snap to standard rates; git builds pick the right filter-script option; variable-frame-rate input refused clearly; more prohibited paths (blackbox CSV/event, nested work/frames/output folders).
- Picture: active area 472x360 in the 640x360 proxies and 948x720 at full resolution (true picture 480/960 wide; ~0.6 % trimmed per side); OSD mask 6-13 % of the picture, visually only OSD elements.
- Edit matching reproduces Dom's 9 segments of `finished edit.mp4` exactly (37 s on proxies).
- The M2 selector on the 640x360 proxies (now cropped and OSD-masked), with 1000 same-length random edits as baseline:
  - 182156: 79 % kept, event recall 1.00 (beats 96 % of random edits).
  - 180033: 65 % kept, event recall 0.78 (beats 89 %), but 73 % of failure time kept (random keeps 60 %).
  - 203859: kept 95 % (flat score), so it does not beat chance.
  - clip1/clip2: IoU with Dom's edit 0.47/0.32, at chance.
  - Small analysis changes move these numbers a lot, which is expected from the M2 selector. Choosing moments well is M4's job.

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
