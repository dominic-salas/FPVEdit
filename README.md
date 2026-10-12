# FPVEdit

Offline command-line tool that turns a raw FPV freestyle recording (`.ts` or `.mp4`) into one
highlight MP4, keeping the flight's flow while cutting idle and repetitive footage.

```
fpv-editor /path/to/flight.ts
```

**Status: early development (milestone 2.5).** A first computer-vision-only editor works end to
end: it finds the picture inside black bars, ignores the burned-in OSD, cuts idle/blank stretches
(on the ground, no signal) and trims the least active parts. On real flights its choices are
still close to chance; choosing moments the way a pilot would is the next milestone. Picture
enhancement and the ML component are planned. See [PLAN.md](PLAN.md) for milestones and
validation results.

## Quick start

Requires FFmpeg 6+ on PATH and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run fpv-editor path/to/flight.ts
```

Options: `-o/--output out.mp4`, `--force` (replace an existing output), `--keep-intermediates`
(also write `<output name>.analysis.json` with per-sample scores and the chosen intervals),
`--analyze-only` (write only `<input name>.analysis.json`, no video), `--cache-dir DIR` (reuse per-frame
measurements between runs). By default the output is written next to the input as
`<name>_highlight.mp4` (or `_highlight-2.mp4`, … if that exists). The output is H.264 video at
the recording's own frame rate with no audio track: DVR audio is goggle noise, so add your own
music afterwards. The source file is never modified.

## Evaluation

`eval/annotations/` holds hand-labelled moments and reference edits for test flights (the
footage itself stays on the owner's PC). `scripts/evaluate.py ANNOTATION ANALYSIS --random 1000`
scores an edit and compares it with random edits of the same length, and
`scripts/match_edit.py EDIT RAW...` turns a finished edit and its raw recordings into a
reference-edit annotation.

Footage is processed locally and never uploaded. Licensed under the [MIT License](LICENSE).
