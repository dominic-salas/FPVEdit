# FPVEdit

Offline command-line tool that turns raw FPV freestyle recordings (`.ts` or `.mp4`) into one
montage of their best shots.

```
fpv-editor flight1.ts flight2.ts flight3.mp4
```

**Status: early development (montage engine).** Each recording is first cleaned using the
burned-in Betaflight OSD (arming, disarming, the stats screen, the boot logo) and the camera's
own motion, measured on every frame: downtime, static, time on the ground, crashes together
with the move that caused them and repeated re-arming are never used. The editor then finds
flips, spins and close passes, builds 8-15 s shots around them with the cuts in calm moments,
and picks the best shots across all recordings up to the target length, in the order the
recordings were given. Recordings with no flying in them (bench tests, a quad on a desk) give
no shots. A desktop app with drag and drop, picture enhancement and the ML component are
planned. See [PLAN.md](PLAN.md) for milestones, known gaps and validation results.

## Quick start

Requires FFmpeg 6+ on PATH and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run fpv-editor path/to/flight1.ts path/to/flight2.ts -o montage.mp4
```

Options:
- `-o/--output out.mp4`: by default the montage is written next to the first recording as
  `<name>_montage.mp4` (or `_montage-2.mp4`, … if that exists).
- `--length SECONDS`: montage length to aim for (default 90). Shorter if the recordings do not
  have enough good shots.
- `--trim`: instead of a montage, edit one recording as a whole flight (cut downtime, crashes
  and dull stretches, keep the rest), written as `<name>_highlight.mp4`.
- `--force`: replace an existing output.
- `--keep-intermediates`: also write `<output name>.montage.json` (the clips and the chosen
  shots with their moments) or, with `--trim`, `<output name>.analysis.json` (flight state,
  motion signals and the reason for every cut).
- `--analyze-only`: write only that file, no video.
- `--cache-dir DIR`: reuse per-frame measurements between runs.

Clips of different sizes or frame rates are fitted to the largest size, at the frame rate of
the clip that fills most of the montage. The output is H.264 video with no audio track: DVR
audio is goggle noise, so add your own music afterwards. Source files are never modified.

## Evaluation

`eval/annotations/` holds hand-labelled moments and reference edits for test flights (the
footage itself stays on the owner's PC). `scripts/evaluate.py ANNOTATION ANALYSIS --random 1000`
scores an edit and compares it with random edits of the same length, and
`scripts/match_edit.py EDIT RAW...` turns a finished edit and its raw recordings into a
reference-edit annotation.

Footage is processed locally and never uploaded. Licensed under the [MIT License](LICENSE).
