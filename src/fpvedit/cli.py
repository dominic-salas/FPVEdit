"""Command line: `fpvedit a.ts b.ts ...` makes a montage, `--trim` edits one recording."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from collections.abc import Sequence
from pathlib import Path

from fpvedit import __version__
from fpvedit.errors import (
    FPVEditError,
)
from fpvedit.pipeline import (
    DEFAULT_MIN_STARS,
    MontageResult,
    Result,
    run_montage,
    run_trim,
)
from fpvedit.render import (
    DEVICES,
    NVENC,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpvedit",
        description=(
            "Turn raw FPV freestyle recordings into one montage of their best shots, or "
            "(with --trim) edit one recording down to its flying."
        ),
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        metavar="input",
        help="recordings (.ts or .mp4), in the order their shots should appear",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "output .mp4 (default: <first input>_montage.mp4, or <input>_highlight.mp4 "
            "with --trim, beside the first input)"
        ),
    )
    parser.add_argument(
        "--length",
        type=float,
        metavar="SECONDS",
        help="make the montage about this long, best shots first (default: every "
        "distinct shot with a flip, spin or close pass, however long that is)",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="edit one recording as a whole flight (cut downtime, crashes, dull stretches) "
        "instead of making a montage",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="make a 9:16 video for phones: as tall as the clips, filled edge to edge with the "
        "middle of the picture (the sides are cut off)",
    )
    parser.add_argument(
        "--music",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="background music for the montage (repeat for more tracks, played in order; "
        "cut and faded at the end, silence after if they are shorter)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        choices=range(1, 6),
        default=DEFAULT_MIN_STARS,
        metavar="1-5",
        help=f"keep shots rated at least this many stars (default {DEFAULT_MIN_STARS}; 1 keeps "
        "every shot found)",
    )
    parser.add_argument(
        "--no-scenery",
        action="store_true",
        help="rate shots without the image model's indoor/outdoor nudge",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace the output file if it already exists"
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="video encoding: the NVIDIA GPU (NVENC) when it works, else the CPU (auto, the "
        "default); the CPU only (cpu); or the GPU only (cuda)",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="also write <output name>.analysis.json (--trim) or <output name>.montage.json "
        "(the clips and shots chosen)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="write only the analysis or montage file and no video",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="reuse per-frame measurements stored in this folder (created if missing)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


def main(argv: Sequence[str] | None = None) -> int:
    multiprocessing.freeze_support()  # analysis workers in a standalone build
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trim and len(args.inputs) != 1:
        parser.error("--trim edits exactly one recording")
    if args.trim and args.vertical:
        parser.error("--vertical is for montages")
    if args.trim and args.music:
        parser.error("--music is for montages")
    if args.trim and args.length is not None:
        parser.error("--length is for montages; --trim keeps all good flying")
    try:
        if not args.trim:
            return _report_montage(
                run_montage(
                    args.inputs,
                    output=args.output,
                    force=args.force,
                    length=args.length,
                    keep_intermediates=args.keep_intermediates,
                    analyze_only=args.analyze_only,
                    cache_dir=args.cache_dir,
                    device=args.device,
                    music=args.music,
                    scenery=not args.no_scenery,
                    min_stars=args.min_stars,
                    vertical=args.vertical,
                )
            )
        return _report_trim(
            run_trim(
                args.inputs[0],
                output=args.output,
                force=args.force,
                keep_intermediates=args.keep_intermediates,
                analyze_only=args.analyze_only,
                cache_dir=args.cache_dir,
                device=args.device,
            )
        )
    except (FPVEditError, OSError) as exc:  # OSError: disk full, permission denied, ...
        print(f"fpvedit: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("fpvedit: interrupted; no output written", file=sys.stderr)
        return 130


def _report_trim(result: Result) -> int:
    retained = result.edited_duration / result.original_duration if result.original_duration else 0
    print(f"Output:    {result.output}")
    print(f"Original:  {format_duration(result.original_duration)}")
    print(f"Edited:    {format_duration(result.edited_duration)} ({retained:.0%} retained)")
    print(f"Segments:  {len(result.selection.intervals)}")
    if result.encoder is not None:
        print(f"Encoder:   {_encoder_name(result.encoder)}")
    print(f"Time:      {result.elapsed:.1f} s")
    for warning in result.warnings:
        print(f"Warning:   {warning}", file=sys.stderr)
    return 0


def _encoder_name(codec: str) -> str:
    return "NVIDIA GPU (h264_nvenc)" if codec == NVENC.video_codec else f"CPU ({codec})"


def _report_montage(result: MontageResult) -> int:
    used = len({shot.clip for shot in result.shots})
    print(f"Output:    {result.output}")
    print(f"Clips:     {len(result.clips)} given, shots from {used}")
    print(f"Footage:   {format_duration(result.footage_duration)} of usable recordings")
    print(f"Montage:   {format_duration(result.edited_duration)} in {len(result.shots)} shots")
    if result.encoder is not None:
        print(f"Encoder:   {_encoder_name(result.encoder)}")
    print(f"Time:      {result.elapsed:.1f} s")
    for warning in result.warnings:
        print(f"Warning:   {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
