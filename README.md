# FPVEdit

Offline command-line tool that turns raw FPV freestyle recordings (`.ts` or `.mp4`) into one
montage of their best shots.

```
fpvedit flight1.ts flight2.ts flight3.mp4
```

**Status: early development (montage engine).** Each recording is first cleaned using the
burned-in Betaflight OSD (arming, disarming, the stats screen, the boot logo) and the camera's
own motion, measured on every frame: downtime, static, time on the ground, crashes together
with the move that caused them and repeated re-arming are never used. The editor then finds
flips, spins and close passes, builds 8-15 s shots around them with the cuts in calm moments,
and keeps every distinct shot from every recording (each built around a trick or close
pass, never two of the same moment), in the order the recordings were given. Recordings with no flying in them (bench tests, a quad on a desk) give
no shots. A desktop app lets you check and change the shots before rendering (see below).
Picture enhancement and the ML component are planned. See [PLAN.md](PLAN.md) for milestones, known gaps and validation results.

## Quick start

Requires FFmpeg 6+ on PATH and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run fpvedit path/to/flight1.ts path/to/flight2.ts -o montage.mp4
```

Options:
- `-o/--output out.mp4`: by default the montage is written next to the first recording as
  `<name>_montage.mp4` (or `_montage-2.mp4`, … if that exists).
- `--length SECONDS`: make the montage about this long, best shots first. By default every
  distinct shot is kept, however long that makes the montage.
- `--trim`: instead of a montage, edit one recording as a whole flight (cut downtime, crashes
  and dull stretches, keep the rest), written as `<name>_highlight.mp4`.
- `--force`: replace an existing output.
- `--keep-intermediates`: also write `<output name>.montage.json` (the clips and the chosen
  shots with their moments) or, with `--trim`, `<output name>.analysis.json` (flight state,
  motion signals and the reason for every cut).
- `--analyze-only`: write only that file, no video.
- `--cache-dir DIR`: reuse per-frame measurements between runs.
- `--device auto|cpu|cuda`: encode on the NVIDIA GPU (NVENC) when FFmpeg can, otherwise the
  CPU (`auto`, the default); force the CPU (`cpu`) or the GPU (`cuda`).

Clips of different sizes or frame rates are fitted to the largest size, at the frame rate of
the clip that fills most of the montage. The output is H.264 video with no audio track: DVR
audio is goggle noise, so add your own music afterwards. Source files are never modified.

## Image model (optional)

```
uv sync --extra app --extra ml
```

adds SigLIP 2 (Apache-2.0, about 1.5 GB downloaded from Hugging Face on first use) and
PyTorch with CUDA. After the shots are picked, it compares what they show: indoor-looking
shots that are weaker than the best outdoor shots of the same session are left out (the app
can add them back). It runs on an NVIDIA GPU when there is one (under 1 GB of video memory,
freed when it is done), else on the CPU; `--no-scenery` skips it. Its similarity scores are
only compared between shots; they are not probabilities, and it does not recognise tricks.

## Desktop app

```
uv sync --extra app
uv run fpvedit-app
```

After `uv sync`, the app is also a program of its own: `.venv\Scripts\fpvedit-app.exe`.
Pin it or make a desktop shortcut to it (right-click > Send to > Desktop) to open the app
with a double-click; dropping clips onto the shortcut opens the app with them. It keeps
working after updates, as long as the project folder and its `.venv` stay where they are.

Drop recordings or folders onto the window (a folder adds the `.ts`/`.mp4` files at its top
level), drag them into order and press **Find shots**. The app shows the shots the engine
picked, with thumbnails and what each one caught (flip, spin, close pass). Click a shot to
loop it. You can trim its start or end, remove it (Delete), drag it to a new place, swap it
for another shot the engine found in the same clip, or add any shot from any clip. Ctrl+Z
undoes. **Render montage** writes the video, using the NVIDIA GPU (NVENC) when FFmpeg can,
otherwise the CPU, then checks the result.

Everything runs on your computer. Measurements are cached in your local app-data folder
(`%LOCALAPPDATA%\fpvedit\measurements` on Windows), so a second analysis of the same
clips takes seconds. You can delete that folder at any time. The app needs PySide6 (Qt for
Python, LGPL-3.0), installed by the `app` extra.

## Evaluation

`eval/annotations/` holds hand-labelled moments and reference edits for test flights (the
footage itself stays on the owner's PC). `scripts/evaluate.py ANNOTATION ANALYSIS --random 1000`
scores an edit and compares it with random edits of the same length, and
`scripts/match_edit.py EDIT RAW...` turns a finished edit and its raw recordings into a
reference-edit annotation.

Footage is processed locally and never uploaded. Licensed under the [MIT License](LICENSE).
