# FPV Highlight Editor — Plan for approval

Status: **approved by Dom on 2026-10-11.** Snapshot of the planning-stage proposal; `PLAN.md` is the living version.
Date: 2026-10-11. After approval this becomes `PLAN.md` in the new repository.

## 1. Current state (inspected, not assumed)

| Item | Finding | How checked |
|---|---|---|
| Repository | **Exists:** `github.com/dominic-salas/FPVEdit`, cloned on Dom's Windows PC (inside a OneDrive folder). Branch `main`, one commit (`3b828cf Initial commit`), contents: `LICENSE` (MIT, © 2026 Dominic Salas) only. Not reachable from this cloud session yet (repo listing returns nothing). | read-only `git log`, `git status`, `.git/config` via the shared folder |
| Working tree | `git status` shows `LICENSE` modified, but the diff is line endings only (CRLF on disk, LF in the commit; `git diff --ignore-cr-at-eol` is empty). Not a real change; left untouched. | `git diff --stat` vs `--ignore-cr-at-eol` |
| Existing CLAUDE.md / code / tests | None to preserve. | recursive listing of the folder |
| Footage | None in the repo folder. | recursive listing |
| Python | 3.13.16, `uv` 0.11.32 available | `python3 --version`, `uv --version` |
| FFmpeg | 6.1.1 (Ubuntu build) with `libx264`, `h264_nvenc`, `aac` encoders, `cuda` hwaccel compiled in | `ffmpeg -encoders`, `ffmpeg -hwaccels` |
| GPU | None in this container (`nvidia-smi` not found). 4 CPU cores, 15 GB RAM. | `nvidia-smi`, `nproc`, `free -g` |
| torch / OpenCV | Not installed (correct; nothing installed yet) | `import torch`, `import cv2` fail |
| Network | PyPI reachable. **huggingface.co and download.pytorch.org are blocked** (proxy 403). | `curl` to each host |

Consequences: this is an almost-empty existing repo (MIT license only), so new files are added around it. Model inference cannot be verified in this container until the network allows `huggingface.co` (+ `download.pytorch.org` for CPU torch wheels), or you run those steps. GPU and Windows checks are steps you run, with exact commands I will provide. Your PC is Windows, so Windows is a first-class target (paths, FFmpeg filter-script files, CI on `windows-latest`). The folder sits inside OneDrive: footage, outputs and model caches must stay outside it (and out of Git), and `.gitattributes` will pin line endings to stop the CRLF noise above. GPU details of your PC aren't visible from the folder share; `nvidia-smi` output from you settles that.

**Defaults I'm assuming**: build in your existing `FPVEdit` repo; synthetic/redistributable fixtures until you share flights; Windows + NVIDIA validation run by you.

## 2. Architecture

Analysis and rendering are separate. Analysis produces a time-indexed feature table → score → edit decision list (EDL, JSON). Rendering consumes only the EDL.

```
cli.py        args, orchestration, summary print, exit codes
media.py      ffprobe/ffmpeg wrappers (argument lists only), probe dataclass, discontinuity detection
sampling.py   ffmpeg → rawvideo pipe of downscaled frames at fixed fps (no intermediate file)
features.py   CV features per time step (flow, rotation, novelty, blur, idle, overlay mask)
models.py     SemanticScorer protocol + SigLIP2 implementation (optional extra); device selection
scoring.py    robust normalization, weighted fusion, smoothing
highlights.py hysteresis → intervals, padding, merging, maneuver-safe boundaries, dedupe, soft retention
render.py     EDL → single ffmpeg trim/concat re-encode → temp file → validate → atomic move
eval/         annotation format + metrics script (not part of the runtime path)
```

Design rules: frozen dataclasses for config and results (no global state), `pathlib`, `subprocess.run([...])` never `shell=True`, typed errors (`ProbeError`, `RenderError`, `ValidationError`) mapped to clear CLI messages and non-zero exit codes. The model layer is a small `Protocol` (`embed(frames) -> ndarray`), so replacing SigLIP 2 later touches one file. A debug JSON of per-second features/scores/EDL is written with `--keep-intermediates`. Caching: not in M2; added only if measured re-run cost justifies it, keyed on input size+mtime+hash-of-first-MB, model id+revision, package version, config hash.

### Dependencies (minimal, each justified)

| Dep | Why | Notes |
|---|---|---|
| FFmpeg/ffprobe ≥ 6.0 (system) | decode `.ts` directly, render, validate | apt / `winget install Gyan.FFmpeg`; not pip-bundled |
| numpy | feature arrays, scoring | — |
| opencv-python-headless (Apache-2.0) | optical flow (Farneback), feature tracking + `estimateAffinePartial2D` for rotation, Laplacian blur | headless avoids GUI libs |
| `[ml]` extra: torch (CPU or CUDA wheel), transformers (Apache-2.0) | SigLIP 2 inference | optional so the core and tests install without torch |
| dev: pytest, ruff, mypy | tests, lint/format, types | — |

Packaging: `pyproject.toml` + `uv.lock`, console script `fpv-editor`, Python ≥ 3.11. Rejected: moviepy/PyAV (overlap with FFmpeg CLI, more breakage), decord (stale), a DB/web/plugin system (no requirement).

## 3. ML model candidates (sources at bottom)

No public model I found is trained to recognize FPV maneuvers or gate passes. I will not claim trick recognition.

| Candidate | What it actually does | Weights license | Fit for FPV | Verdict |
|---|---|---|---|---|
| **SigLIP 2 base/16-224** (`google/siglip2-base-patch16-224`) | image–text embeddings, zero-shot by prompt similarity | Apache-2.0 (model card) | per-frame semantics: scenery vs ground/grass/black/static, scene identity for novelty & dedupe. Unknown robustness to fisheye/blur/analog noise → must be measured | **Recommended, single model** |
| X-CLIP base/32 (`microsoft/xclip-base-patch32`) | video–text (8-frame clips), Kinetics-400 trained | MIT | adds temporal semantics, but Kinetics has no FPV classes; 8-frame clips at low fps cover little motion | Evaluate only if SigLIP 2 + CV leaves a measured gap |
| VideoMAE (Kinetics fine-tuned) | action classification, 400 Kinetics labels | **CC-BY-NC-4.0** | labels irrelevant to FPV; non-commercial weights | Reject as default |
| V-JEPA 2 ViT-L | self-supervised video features (326M params, F32) | MIT | strong motion features but heavy for CPU; no labels; would need a trained head we have no data for | Reject for now; revisit if annotated data grows |
| OpenAI CLIP B/32 | image–text | code MIT; HF card has no license tag | superseded by SigLIP 2 for the same role | Not chosen |

What SigLIP 2 contributes that CV can't: (a) **semantic novelty** — cosine distance of a frame's embedding to the recent past and to already-selected material, the main tool against repetitive laps over the same spot; (b) a **relative scenic/low-information signal** from contrasting prompt sets (e.g. "aerial view of a landscape" vs "close-up of grass", "black screen", "video static"), used only as within-flight z-scores, never as a probability. Cost: ~1 frame/s; CPU throughput to be measured in M3 (I have no measured number yet).

Fallback: if the model doesn't measurably beat the CV baseline on annotated flights, it stays an optional extra and the docs say so.

## 4. Scoring and interval selection

Analysis timeline: frames decoded by FFmpeg with `fps=` filter (handles VFR), times derived from output frame index relative to the probed start time; probe detects PTS discontinuities in `.ts` and warns/normalizes before any temporal decision.

Per time step (CV at 4 fps on ~320 px gray frames; SigLIP 2 at 1 fps, interpolated):

1. **Rotation rate** — similarity transform between consecutive frames from tracked features → roll/yaw-like image rotation (deg/s). The main proxy for flips/rolls/orientation changes.
2. **Flow structure** — median flow magnitude *and* coherence (how well one global motion explains the flow). Coherent fast flow ≈ controlled dynamic flight; incoherent shaky flow is down-weighted. Raw magnitude is capped so it can't dominate.
3. **Close-proximity cue** — flow magnitude in the periphery vs centre (expansion near obstacles/gaps). Labelled a hypothesis for gate/gap passes; only kept if eval supports it.
4. **Visual novelty** — histogram + SigLIP 2 embedding distance to recent window.
5. **Idle / low-information** — near-zero flow + near-zero change (on the ground, waiting), black frames, analog static (high-frequency energy with no structure), heavy blur.
6. **OSD overlay mask** — pixels that are static across the flight (OSD text, recording timer) excluded from flow/novelty.

Fusion: each feature robust-normalized within the flight (median/MAD → clipped z), weighted sum with weights in one documented config dataclass; idle acts as a gate (strong penalty), not a weight. Then Gaussian smoothing (~2 s) so scores don't oscillate.

Selection:
- **Hysteresis**: enter a segment above a high threshold, leave only below a lower one.
- **Context**: pad 1.5 s lead-in / 1.5 s aftermath; merge gaps < 3 s (a short cut is jarring); minimum segment 3 s.
- **Maneuver-safe cuts**: move each boundary up to ±1 s to the nearest local minimum of rotation/flow, never inside a high-rotation run.
- **Dedupe**: drop a segment if its embedding signature closely matches a higher-scoring kept one (only while retention stays above the floor).
- **Soft retention 35–65%**: thresholds are percentile-based with bounded adjustment; if the natural result is outside the band, nudge within limits and print a warning instead of forcing it.
- **Low confidence**: if the score has little dynamic range (flat, uninformative flight), only clear idle/black/static is removed.
- Output: EDL of `(start, end, reason, score)`; always ≥ 1 segment unless the input is entirely idle (then a clear error, no empty file).

## 5. Rendering and validation

- One FFmpeg pass: `trim`/`atrim` per segment → `concat` filter → `libx264 -crf 18 -preset medium -pix_fmt yuv420p` + `aac 192k`, `-movflags +faststart`. Re-encoding is frame-accurate; stream-copy at arbitrary cut points is not, so copy is not used for cuts. Filter graph passed via a script file (Windows command-line length limits). 10 ms audio fades at cut points only to prevent clicks (not a creative effect).
- Aspect ratio/SAR and resolution preserved; no scaling.
- `h264_nvenc` used only if `--device` allows it **and** a 1-second probe encode succeeds; otherwise libx264.
- No audio stream → video-only output (warned). Audio present but broken → retry without audio, warned.
- Output: `<stem>_highlight.mp4` beside input; if taken, `_highlight-2.mp4` etc. `--output` to an existing file requires `--force`. Render to `.<name>.partial.mp4` in the same dir, validate (ffprobe streams + duration within tolerance of EDL sum + full decode `ffmpeg -v error -i out -f null -`), then `os.replace`. Partial removed on failure. Source opened read-only and never written.
- Summary: output path, original/edited duration, retained %, elapsed time, device/encoder/model mode, warnings.

## 6. Milestones and acceptance criteria

**M1 — Safeguards.** Skeleton added to the existing repo (LICENSE kept), `.gitattributes`, concise `CLAUDE.md`, `PLAN.md`, `.gitignore` (media, weights, outputs), pyproject + lockfile, ruff/mypy/pytest configured, proposed hooks and CI (reviewed before enabling), 3 read-only reviewer agent definitions.
✔ `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` all pass on an empty-ish suite; staging-guard test proves a `.ts`/`.safetensors`/`.mp4` file is refused.

**M2 — CV-only end-to-end baseline.** probe → sample → simple features (flow, idle, novelty) → hysteresis → EDL → render → validate → summary.
✔ `fpv-editor synthetic.ts` produces a playable MP4 that passes validation; synthetic fixture with known "active" and "idle" spans retains ≥ 90% of active seconds and ≤ 20% of idle seconds; source file hash unchanged; unit tests for scoring/intervals/naming/existing-output; no-audio, 2-second, truncated, and existing-output cases tested.

**M3 — ML integration (SigLIP 2).** `[ml]` extra, device auto/cpu/cuda with fallback, pinned model revision, provenance doc. Experiment: on annotated flights, compare CV-only vs CV+SigLIP 2.
✔ Model tests (marked, opt-in) run real inference; measured CPU throughput reported; keep-or-demote decision made on measured eval numbers. **Blocked here** until `huggingface.co` is reachable or you run the commands.

**M4 — Editing quality.** Rotation, coherence, overlay mask, maneuver-safe boundaries, dedupe, soft retention; evaluation script.
✔ On your annotated flights: report time-precision/recall, event recall (event hit if ≥ 50% covered), retained %, redundant-segment count, vs. baseline and a motion-only baseline. No target numbers promised before data exists.

**M5 — Reliability.** VFR, PTS discontinuities in `.ts`, truncated files, FFmpeg failure paths, NVENC fallback, CPU fallback, Windows paths.
✔ Each listed case has a test or a documented manual check with exact command; CUDA run reported only if you run it.

**M6 — Release readiness.** README (Windows/Linux, CPU/GPU, first-run download, privacy, troubleshooting, limitations), LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, MODEL_CARD/provenance, ADRs, validation report; three independent reviews reconciled.
✔ CI green (needs GitHub); every claim in README traceable to a test or measured result.

## 7. Testing, eval, automation

- **Fixtures** generated at test time with FFmpeg `lavfi` (e.g. `testsrc2` + rotate/zoom for "maneuvers", static colour for "idle", sine for audio), muxed to `.ts`; variants: no audio, 2 s, truncated bytes, VFR, PTS jump. Nothing large committed.
- **Markers**: default `pytest` = fast, no GPU, no downloads; `-m integration` = FFmpeg end-to-end; `-m model` = real inference (manual/CI dispatch only). No mocked model output in tests that claim to prove inference.
- **Annotation format**: one JSON per flight: `[{start, end, label: maneuver|gate|cinematic|other, importance: 1–3}]`, plus a "must-cut" list (idle). Target set: 5–10 of your flights covering a clear trick, a gate pass, a slow cinematic view, fast-but-boring footage, and ambiguous footage. Held-out human review set for regressions.
- **Hooks (proposed, reviewed before enabling)**: PostToolUse on `*.py` edits → `ruff check` + `ruff format --check` on that file only; PreToolUse on `git add/commit` → guard script refusing media/weights/`.env`/files > 5 MB. No inference or full test runs in hooks.
- **CI**: GitHub Actions, Ubuntu + Windows, ruff + mypy + fast pytest + FFmpeg integration tests; separate manual workflow for model tests.

## 8. Main risks and how each gets validated

| Risk | Validation / mitigation |
|---|---|
| SigLIP 2 semantics may be weak on fisheye, motion blur, analog noise, OSD overlays | M3 eval vs CV baseline; low weight; demote if no gain |
| Real DVR `.ts` (Walksnail/HDZero/analog DVR) timestamp quirks differ from synthetic | Need 2–3 real files; probe discontinuities; fixture reproducing each quirk found |
| Image rotation ≠ actual drone attitude (horizon-free scenes, fisheye) | Check on annotated maneuvers; treat as a signal, not ground truth |
| CPU runtime on long flights | Measure on a real 10-min flight; tune sampling fps |
| Automated metrics miss aesthetics | Your review of a fixed set of outputs per scoring change |
| Model download blocked in this container | Network allowlist change, or you run M3 commands |
| Work lost when the cloud container is reclaimed | Work on a branch pushed to `FPVEdit` (needs GitHub connected) or directly in your folder |
| OneDrive syncing large media/caches or fighting Git | Keep footage/outputs/caches outside the OneDrive folder; `.gitignore` + staging guard |

## 9. Decisions for you

1. Approve this plan (or tell me what to change).
2. Where I work: **(a) recommended:** connect GitHub to Claude so I can push a branch to `dominic-salas/FPVEdit` and open PRs you review and pull on your PC; or (b) I edit your local folder directly and you commit. Either way nothing is committed or pushed without your OK.
3. Code license: keep your existing **MIT** license (compatible with SigLIP 2's Apache-2.0 weights, which are downloaded, not redistributed).
4. Network: allow `huggingface.co` and `download.pytorch.org` for this project's cloud environment, needed for M3 here.
5. Footage: 2–3 representative flights (and which goggles/DVR recorded them) whenever you can share them; they unblock M4/M5 evaluation.

## Sources

- SigLIP 2 model card (Apache-2.0): https://huggingface.co/google/siglip2-base-patch16-224 ; paper: https://arxiv.org/abs/2502.14786
- X-CLIP model card (MIT): https://huggingface.co/microsoft/xclip-base-patch32 ; paper: https://arxiv.org/abs/2208.02816
- VideoMAE Kinetics card (CC-BY-NC-4.0): https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics ; paper: https://arxiv.org/abs/2203.12602
- V-JEPA 2 card (MIT, ~326M params): https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 ; paper: https://arxiv.org/abs/2506.09985
- OpenAI CLIP card: https://huggingface.co/openai/clip-vit-base-patch32
- PyPI current versions (2026-10-11): torch 2.14.1, transformers 5.19.0 (Apache-2.0), opencv-python-headless 5.0.0.93 (Apache-2.0), numpy 2.5.4
- FFmpeg filters (trim, concat, fps): https://ffmpeg.org/ffmpeg-filters.html ; OpenCV optical flow: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- PyTorch install selector (CPU vs CUDA wheels): https://pytorch.org/get-started/locally/

License data was read from Hugging Face model metadata on 2026-10-11; weight licenses will be re-checked against the full model cards and pinned revisions in M3.
