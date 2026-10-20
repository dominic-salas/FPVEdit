# Known limitations

FPVEdit was developed and tested on analog footage recorded by HDZero BoxPro+ goggles
(Betaflight OSD burned into the picture, 60 fps, `.ts` and `.mp4`). Other setups should work
but are less tested.

## Footage

- **Digital systems** (DJI, Walksnail, HDZero digital), PAL OSD, 25/30 fps DVRs and
  stabilised video are untested.
- **No OSD**: without any arm or disarm on screen, only pieces of 6 s or more with clear
  rotation are kept, and crashes are found from the picture alone (a tumble into a dead stop).
  This was checked only by hiding the OSD of analog recordings.
- **Variable frame rate** (phone or screen captures) is refused with a conversion command.
  A steady recording with signal loss is accepted; the edit plays straight across the gap.
- **MPEG-TS timestamp jumps**: FFmpeg repairs jumps of 10 s or more; a smaller backward jump
  (goggles resetting their clock mid-file) is not handled and could make the preview and the
  render disagree.
- **Lens field of view** is unknown, so rotation in degrees is approximate; thresholds are
  relative to each flight.

## Editing decisions

- **Failures without a disarm** (a ground tap, repeated failed attempts at a gap) cannot be
  seen from video alone and may be kept; remove such shots in the app before exporting.
- **Crashes**: the last 1.5 s before a crash is cut; a long failed move before that may be
  partly kept.
- **Taste**: the motion signals find flips, rolls, spins and close passes, not named tricks or
  gates, and they cannot tell which of two similar shots you prefer. Ratings (Settings >
  My ratings) help with that over time. The picture-based part of the learned ratings needs
  the optional image model, which is not in the Windows installer.
- **OSD menus** are read as stats screens. This is harmless (both are downtime).

## Performance

- Clips are analysed in parallel (one process per clip, up to a quarter of the logical CPUs),
  so a montage takes about as long as its longest clip alone; one long clip is not split.
- Encoding uses the NVIDIA GPU (NVENC) when a test encode works, with a CPU fallback.
  Machines without an NVIDIA GPU are covered by tests but little used.
- `scripts/match_edit.py` keeps every frame fingerprint in memory (about 2-3 GB for an hour
  of footage).

## App

- The preview and thumbnails seek by time; in a `.ts` whose timestamps jump they can show the
  wrong place. The render is unaffected.
- Closing the app during a step that reports no progress (probing, joining, the final check)
  waits for that step to finish.

## Distribution

- The Windows installer is not code-signed: Windows SmartScreen warns on first run.
- Only Windows has an installer. Linux runs from source (`uv sync --extra app`); macOS is
  untested.
