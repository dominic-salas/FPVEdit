# Changelog

## Unreleased

- Review page: the Music and Timeline colours labels sit on their own line, apart from
  their hints.

## 1.0.0 (2026-10-20)

First public release: a Windows installer with everything included (FFmpeg too).

- **Montage from many recordings**: add `.ts` or `.mp4` DVR recordings and FPVEdit picks
  every distinct flip, roll, spin and close pass, builds 8-15 s shots around them with the
  cuts in calm moments, and plays them in order.
- **Flight cleaning from the OSD**: boot screens, menus, the stats screen, static, time on
  the ground, crashes and repeated re-arming are left out; the moment before a crash stays,
  the impact does not; a short flight between two crashes can be a shot of its own.
- **Desktop app**: clips are analysed as soon as they are added; review the montage, trim,
  remove, reorder, swap or add shots, scan with the playhead, add music, undo.
- **Export**: H.264 MP4 as recorded or vertical 9:16 for phones; NVIDIA GPU encoding when
  available, CPU otherwise; every output is checked frame by frame before it is kept; an
  existing file is never replaced unless you choose to.
- **Ratings**: rate shots 1-5 on the montage page or the Rate shots screen; with
  Settings > My ratings on, later montages leave out shots predicted to be rated low.
- **Fast again**: measurements are saved, so a clip analysed before loads in seconds.
- **Private**: runs offline; nothing is uploaded.
- **Command line**: `fpvedit a.ts b.ts` (also `--trim`, `--vertical`, `--music`, `--length`).
- **Uninstaller**: removes the program and asks before removing your ratings and settings.

Known limitations: [docs/limitations.md](docs/limitations.md). The optional image
model (people, gaps, scenery in the learned ratings) needs a source install with the `ml`
extra; it is not in the installer.
