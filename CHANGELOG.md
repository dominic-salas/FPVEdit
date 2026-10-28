# Changelog

## Unreleased

- **Find a missed shot**: on the montage page, play any whole recording, and at the moment
  you want press M (or Mark a shot here): a shot is built around it, cut in calm moments
  and out of any turn, and added to the montage.
- **FPVEdit learns from your edits**: when you export, what you did is remembered on this
  computer (shots you removed, kept, added or marked) and teaches the taste, more weakly
  than a star rating. The taste is always learned now; Settings > My ratings only decides
  whether shots predicted low are left out.
- Shots offered to add or swap come in the order of your predicted stars.

## 1.1.0 (2026-10-22)

- A new look, the same as dvsalas.com: amber accent, softer dark and warm light
  backgrounds, thin borders, rounded solid buttons and square tool buttons.
- The fonts Space Grotesk and Geist Mono are included (SIL Open Font License), so the app
  looks the same on every computer.
- New screenshots in the README.

## 1.0.1 (2026-10-22)

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
