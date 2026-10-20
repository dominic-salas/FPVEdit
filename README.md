# FPVEdit

FPVEdit turns raw FPV freestyle recordings (`.ts` or `.mp4` from your goggles' DVR) into one
montage of their best shots: flips, rolls, spins and close passes, with the crashes, menus,
static and dull cruising left out. It runs entirely on your computer; nothing is uploaded.

## Download (Windows)

1. Download `FPVEdit-<version>-setup.exe` from the
   [latest release](https://github.com/dominic-salas/FPVEdit/releases/latest).
2. Run it. It installs for your user only (no administrator rights needed) and includes
   everything it needs, FFmpeg too. Windows may warn that the installer is from an unknown
   publisher (it is not code-signed): choose **More info > Run anyway**.
3. Start **FPVEdit** from the Start menu or the desktop shortcut.

Requirements: Windows 10 or 11 (64-bit). An NVIDIA graphics card is used for faster video
encoding when there is one; without it everything works on the CPU.

To uninstall: Windows Settings > Apps > Installed apps > FPVEdit > Uninstall, or
**Uninstall FPVEdit** in the Start menu. It asks whether to also remove your ratings, saved
measurements, settings and log (No unless you choose Yes). Your videos and montages are
never touched.

## Making a montage

1. **Add clips**: drop recordings or folders onto the window (a folder adds the `.ts` and
   `.mp4` files in it). Each clip is analysed in the background as soon as it is added.
2. **Find shots**: FPVEdit picks every distinct moment worth showing and builds an
   8-15 second shot around it, with the cuts in calm moments, never in the middle of a trick.
3. **Your montage**: the shots play one after another. Click a shot on the timeline to loop
   it; trim its start or end, remove it, drag it to a new place, swap it for another shot
   from the same clip, or add any shot from any clip. Drag the playhead to scan through.
   Add background music if you like. Ctrl+Z undoes.
4. **Export**: an H.264 MP4, as recorded (landscape) or **vertical 9:16 for phones** (the
   middle of each shot, filling the screen). Your recordings are never changed.

**Rate shots** (1-5 stars, on the montage page or the Rate shots screen) to teach FPVEdit
what you like. With Settings > My ratings turned on, the next montage leaves out shots it
predicts you would rate low; you can always add them back. Ratings never change the montage
you are editing.

### How it decides

- **What gets cut**: everything outside armed flight (boot screens, menus, the stats screen
  after disarming, static), time on the ground, crashes and repeated re-arming. It reads the
  Betaflight OSD burned into the DVR picture; without an OSD it uses the camera's motion.
- **What makes a shot**: flips and rolls (the camera turning 180 degrees or more within a
  second), spins and orbits, and sudden close passes, measured from the picture's motion.
  A short flight between two crashes can be a shot of its own; the moment just before a
  crash is kept, the impact is not.
- **What it does not do**: it does not recognise named tricks or gates beyond these motion
  patterns, and its star ratings are estimates, not judgements. You always have the final
  say in the montage page.

## Privacy

Everything runs on your computer. Nothing is uploaded and there is no account or telemetry.
Your ratings, saved measurements and log are kept in `%LOCALAPPDATA%\FPVEdit`
(Linux: `~/.local/share/FPVEdit`) and settings in the registry under
`HKEY_CURRENT_USER\Software\FPVEdit`. Settings > Saved measurements > Clear and
Settings > Clear my ratings empty them.

## Troubleshooting

- **A clip is left out**: recordings with no flying in them (a bench test, a quad on a desk)
  give no shots, and recordings with a variable frame rate (phone or screen captures) are
  refused; convert them first with `ffmpeg -i in.mp4 -fps_mode cfr -r 60 out.mp4`.
- **Export is slow**: without an NVIDIA GPU (or with Settings > Video encoding set to CPU)
  the video is encoded on the CPU, which takes longer.
- **Something went wrong**: Menu > Show the log file, and attach the log to a
  [bug report](https://github.com/dominic-salas/FPVEdit/issues/new/choose).

Known limitations are listed in [PLAN.md](PLAN.md#known-gaps-tracked).

## Command line

The installer includes the command-line tool `fpvedit.exe` (in the install folder):

```
fpvedit flight1.ts flight2.ts flight3.mp4
```

writes `flight1_montage.mp4` next to the first recording. Options:

- `-o/--output out.mp4`: where to write the montage.
- `--length SECONDS`: make the montage about this long, best shots first (by default every
  distinct shot is kept).
- `--vertical`: a 9:16 video for phones.
- `--music FILE` (repeatable): background music, cut and faded at the end.
- `--trim`: edit one recording as a whole flight instead (cut downtime, crashes and dull
  stretches, keep the rest), written as `<name>_highlight.mp4`.
- `--force`: replace an existing output (never done otherwise).
- `--device auto|cpu|cuda`: NVIDIA GPU encoding when available (default), or force CPU/GPU.
- `--keep-intermediates`, `--analyze-only`, `--cache-dir DIR`: write the chosen shots or
  analysis as JSON, only that, and reuse measurements between runs.

Clips of different sizes or frame rates are fitted to the largest size (letterboxed, never
stretched) at the frame rate of the clip that fills most of the montage. The output has no
audio from the recordings (DVR audio is goggle noise).

## Running from source (Windows or Linux)

Requires Python 3.12+, FFmpeg 6+ on PATH and [uv](https://docs.astral.sh/uv/).

```
uv sync --extra app          # the desktop app and the command line
uv run fpvedit-app           # the app
uv run fpvedit a.ts b.ts     # the command line
```

Optional image model: `uv sync --extra app --extra ml` adds SigLIP 2 (Apache-2.0, about
1.5 GB of weights downloaded from Hugging Face on first use) and PyTorch with CUDA. It
compares what each shot shows (indoors or outdoors, people, gaps, trees, sky...) so the
learned ratings can follow what you like to see. It runs on an NVIDIA GPU when there is one,
else on the CPU; `--no-scenery` skips it. Its similarity scores are compared between shots
only: they are not probabilities and do not recognise tricks. The Windows installer does not
include it (PyTorch with CUDA is several GB).

Development: see [CONTRIBUTING.md](CONTRIBUTING.md). The installer is built by
`.github/workflows/release.yml` (standalone app with PyInstaller, FFmpeg, Inno Setup).
`eval/annotations/` holds hand-labelled test flights and `scripts/evaluate.py` scores an
edit against them.

## Licence

FPVEdit is MIT licensed ([LICENSE](LICENSE)). The Windows build includes FFmpeg (GPL build,
run as a separate program; source at https://ffmpeg.org), Qt for Python (LGPL-3.0), Python,
NumPy and OpenCV; their licences are in the `licenses` folder of the installation. The
optional image model's weights are Google's SigLIP 2 (Apache-2.0), downloaded on first use
and never shipped.
