# Changelog

## Unreleased

- **A new text font.** The app's text is set in Instrument Sans (SIL Open Font License,
  bundled) instead of the previous face.
- **Blurred flips are found.** Where the picture smears so badly that motion tracking gives
  up, a turn of 110 degrees in a second now counts as a flip.
- **A clearer review page.** One preview instead of two, a shorter shot title, timeline
  labels that are never cut to dots, and ratings shown from grey to amber instead of red to
  green. The Rate page shows no step as current, and buttons share one shape.
- **Fewer flips left out.** Without a target length, the shots are chosen to show as many
  flips as possible; before, a flip beside a stronger shot could be dropped. On 12 test
  recordings every turn of 180 degrees or more now has a shot (6 of 49 did not).
- **The header logo is the dot-matrix title**, and the app has its own taskbar icon when run
  from source.
- **Vertical exports are 1080x1920**, the size Reels, Shorts and TikTok expect, instead of
  the recording's own height (404x720 from 720p). The Export window says when a vertical
  montage is longer than 3 minutes.
- **The montage's sound is levelled** to about -14 LUFS with peaks under -1 dBTP, for music
  and for the recordings' own sound, by one overall volume change. Very quiet sound is
  raised by at most 12 dB. Audio is now AAC at 384 kb/s.
- **H.264 follows YouTube's upload recommendation**: High profile, 2 B-frames, a closed
  group of pictures of half the frame rate.
- **Tricks are measured, not guessed.** The motion alone under-counts fast flips in the
  blur (a full flip read 130-380 degrees). With the image tools installed
  (`uv sync --extra ml`), FPVEdit now reads which way is down in the frames just before and
  after each fast turn (GeoCalib) and works out how far the turn really went. A flip that
  came back round is named as one (or a double), on the quad's own axes with the camera's
  up-tilt taken out, and is no longer rated down as "not a full turn". One that did not is
  named by how far it got ("half roll", "3/4 back flip"). Turns gravity cannot see (a flat
  yaw spin, a frame too dark or broken up) keep the old name. About 110 MB of weights
  download on first use; readings are cached.
- **Right name?** The shot panel asks whether the trick's name is right (Yes, No, or pick
  the right one under Other). Your name is shown from then on, and the answers are kept on
  this computer (`my-names.jsonl`) to measure how often the names are right.
- **The throttle is read from the OSD.** FPVEdit finds the throttle reading burned into the
  DVR picture and reads it every 1/12 s, learning the OSD's digits from the flight timer
  (no font needed). Shots get notes like "throttle punch" (cut during a flip, then punched
  out), "throttle off" (a dive) and "high throttle", which the learned taste uses. The
  rating itself is unchanged. Recordings whose throttle is not on screen work as before.
- **Fewer tricks without a shot**: a trick in the first 2 s after take-off now gets a shot
  that starts at the take-off, and a shot whose end would cut through a flip now ends just
  after it instead of being dropped. On 11 test recordings, 249 of 304 found moments get a
  shot instead of 243.
- "No shots found" is said above the clip list instead of in a dialog you have to close.
- **The montage keeps the recordings' sound**, cut exactly with the picture (in sync to
  within a couple of milliseconds across every cut). Music, when added, still replaces it.
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
