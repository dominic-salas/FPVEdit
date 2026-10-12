# FPVEdit

Offline command-line tool that turns a raw FPV freestyle recording (`.ts` or `.mp4`) into one
highlight MP4, keeping the flight's flow while cutting idle and repetitive footage.

```
fpv-editor /path/to/flight.ts
```

**Status: early development (milestone 2 of 6).** A first computer-vision-only editor works on
synthetic test clips: it cuts idle/blank stretches (on the ground, no signal) and, when more
than ~65 % would remain, trims the least active parts. It has **not yet been tuned or evaluated
on real FPV flights**, and the planned ML component is not integrated yet. See
[PLAN.md](PLAN.md) for milestones and validation results.

## Quick start

Requires FFmpeg 6+ on PATH and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run fpv-editor path/to/flight.ts
```

Options: `-o/--output out.mp4`, `--force` (replace an existing output), `--keep-intermediates`
(also write `<output>.analysis.json` with per-sample scores and the chosen intervals). By default
the output is written next to the input as `<name>_highlight.mp4` (or `_highlight-2.mp4`, … if
that exists). The source file is never modified.

Footage is processed locally and never uploaded. Licensed under the [MIT License](LICENSE).
