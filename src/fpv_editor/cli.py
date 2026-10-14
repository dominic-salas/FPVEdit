"""Command-line entry point: probe → analyse → pick → render → validate → move into place.

By default the clips given become one montage of their best shots (`montage.py`). With
``--trim`` a single recording is edited as a whole flight instead (`highlights.py`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpv_editor import __version__
from fpv_editor.analysis import Analysis, AnalysisConfig, analysis_document, analyze
from fpv_editor.errors import (
    FPVEditError,
    InvalidInputError,
    NoFlightContentError,
    OutputExistsError,
)
from fpv_editor.flight import FlightConfig
from fpv_editor.highlights import Interval, Selection, SelectionConfig
from fpv_editor.measure import MeasureConfig
from fpv_editor.media import (
    MediaInfo,
    Tools,
    find_tools,
    frame_gaps,
    frame_times,
    is_variable_rate,
    probe,
)
from fpv_editor.montage import MontageConfig, Shot, candidate_shots, pick_shots
from fpv_editor.picture import PictureConfig, decoded_size
from fpv_editor.render import (
    EncodeConfig,
    Fit,
    expected_frame_count,
    finalize_output,
    join_parts,
    output_rate,
    render,
    resolve_output_path,
    snap_to_frames,
    temporary_output,
    validate_output,
)


@dataclass(frozen=True)
class PipelineConfig:
    picture: PictureConfig = field(default_factory=PictureConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    montage: MontageConfig = field(default_factory=MontageConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)

    @property
    def analysis(self) -> AnalysisConfig:
        return AnalysisConfig(
            picture=self.picture,
            measure=self.measure,
            flight=self.flight,
            selection=self.selection,
        )


@dataclass(frozen=True)
class Result:
    output: Path  # the video, or the analysis file with analyze_only
    analysis_file: Path | None
    from_cache: bool  # measurements were read from --cache-dir instead of decoded
    original_duration: float
    edited_duration: float
    elapsed: float
    selection: Selection
    warnings: tuple[str, ...]


MONTAGE_SUFFIX = "_montage"


@dataclass(frozen=True)
class Clip:
    """One input of a montage that has usable footage.

    Only what rendering needs is kept: the per-frame measurements of each clip are
    dropped once its candidate shots are found, so memory does not grow with the number
    of clips.
    """

    index: int
    info: MediaInfo
    times: NDArray[np.float64] | None
    analysed_duration: float


@dataclass(frozen=True)
class MontageResult:
    output: Path  # the video, or the plan file with analyze_only
    plan_file: Path | None
    clips: tuple[Path, ...]
    shots: tuple[Shot, ...]
    footage_duration: float  # all clips together
    edited_duration: float
    elapsed: float
    warnings: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpv-editor",
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
        help=f"montage length to aim for (default {MontageConfig().length_s:g})",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="edit one recording as a whole flight (cut downtime, crashes, dull stretches) "
        "instead of making a montage",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace the output file if it already exists"
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


def analysis_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}.analysis.json")


def plan_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}.montage.json")


def run(
    source: Path,
    *,
    output: Path | None = None,
    force: bool = False,
    keep_intermediates: bool = False,
    analyze_only: bool = False,
    cache_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> Result:
    config = config or PipelineConfig()
    started = time.monotonic()
    tools = find_tools()
    info = probe(tools, source)
    times = frame_times(tools, info)
    if is_variable_rate(info.video, times):
        raise InvalidInputError(
            f"{info.path.name} has a variable frame rate (nominal {info.video.nominal_rate} "
            "fps, but its frames do not arrive at that rate), which is not supported yet. "
            "DVR recordings are constant-rate; for phone or screen captures, convert first, "
            "e.g. ffmpeg -i in.mp4 -fps_mode cfr -r 60 out.mp4"
        )
    gap_notes = _gap_warnings(info, times)
    # With analyze_only no video is written, so an existing video is no obstacle, and
    # the analysis gets a fixed name that does not depend on which videos exist.
    final = resolve_output_path(info.path, output, force or analyze_only)
    if analyze_only:
        analysis_file: Path | None = analysis_path(final if output else info.path)
    else:
        analysis_file = analysis_path(final) if keep_intermediates else None
    if analyze_only and analysis_file is not None and analysis_file.exists() and not force:
        raise OutputExistsError(f"{analysis_file} already exists; use --force to replace it")

    analysis = analyze(tools, info, config.analysis, cache_dir=cache_dir)
    warnings = [*gap_notes, *analysis.warnings]
    rate = output_rate(info)
    segments = snap_to_frames(analysis.selection.intervals, rate)
    edited_duration = expected_frame_count(segments, rate, times) / float(rate)

    if not analyze_only:
        _render_and_validate(tools, info, segments, final, force=force, config=config, times=times)
    if analysis_file is not None:
        if analysis_file.exists() and not force:
            warnings.append(f"{analysis_file.name} already exists; not replacing it (use --force)")
            analysis_file = None
        else:
            _write_analysis(analysis_file, analysis, tools, config)

    return Result(
        output=analysis_file if analyze_only and analysis_file is not None else final,
        analysis_file=analysis_file,
        from_cache=analysis.from_cache,
        original_duration=info.duration or analysis.analysed_duration,
        edited_duration=edited_duration,
        elapsed=time.monotonic() - started,
        selection=analysis.selection,
        warnings=tuple(warnings),
    )


def run_montage(
    sources: Sequence[Path],
    *,
    output: Path | None = None,
    force: bool = False,
    length: float | None = None,
    keep_intermediates: bool = False,
    analyze_only: bool = False,
    cache_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> MontageResult:
    config = config or PipelineConfig()
    montage_config = config.montage
    if length is not None:
        if length <= 0:
            raise InvalidInputError("--length must be a positive number of seconds")
        montage_config = dataclasses.replace(montage_config, length_s=length)
    started = time.monotonic()
    if not sources:
        raise InvalidInputError("no input recordings given")
    tools = find_tools()
    infos = [probe(tools, source) for source in sources]  # fail early on a bad path
    final = resolve_output_path(
        [info.path for info in infos], output, force or analyze_only, suffix=MONTAGE_SUFFIX
    )
    plan_file = plan_path(final) if analyze_only or keep_intermediates else None
    if analyze_only and plan_file is not None and plan_file.exists() and not force:
        raise OutputExistsError(f"{plan_file} already exists; use --force to replace it")

    warnings: list[str] = []
    clips: list[Clip] = []
    candidates: list[Shot] = []
    seen: set[Path] = set()
    for index, info in enumerate(infos):
        name = info.path.name
        if info.path.resolve() in seen:
            warnings.append(f"{name} was given more than once; using it once")
            continue
        seen.add(info.path.resolve())
        times = frame_times(tools, info)
        if is_variable_rate(info.video, times):
            warnings.append(
                f"{name} has a variable frame rate and was left out; convert it first, e.g. "
                "ffmpeg -i in.mp4 -fps_mode cfr -r 60 out.mp4"
            )
            continue
        try:
            analysis = analyze(tools, info, config.analysis, cache_dir=cache_dir)
        except NoFlightContentError as exc:
            warnings.append(f"{name}: nothing usable ({exc})")
            continue
        warnings.extend(
            f"{name}: {note}" for note in (*_gap_warnings(info, times), *analysis.warnings)
        )
        found = candidate_shots(
            index, analysis.measurements, analysis.selection, montage_config, config.selection
        )
        if not found:
            warnings.append(f"{name}: no flip, spin or close pass to build a shot around")
        candidates.extend(found)
        clips.append(Clip(index, info, times, analysis.analysed_duration))

    shots = pick_shots(candidates, len(infos), montage_config)
    if not shots:
        raise NoFlightContentError(
            "none of the recordings has a moment worth a shot (flip, spin or close pass "
            "in clean flying)"
        )
    by_index = {clip.index: clip for clip in clips}
    used = [by_index[index] for index in dict.fromkeys(shot.clip for shot in shots)]
    fit = montage_fit(used, shots)
    parts = [
        (
            clip,
            snap_to_frames(
                [s.interval for s in shots if s.clip == clip.index], output_rate(clip.info)
            ),
        )
        for clip in used
    ]
    frames = sum(_fitted_frames(clip, segments, fit) for clip, segments in parts)

    if not analyze_only:
        temporary = temporary_output(final)
        try:
            with tempfile.TemporaryDirectory(prefix="fpv-editor-") as work:
                work_dir = Path(work)
                files = []
                for number, (clip, segments) in enumerate(parts):
                    part = work_dir / f"part{number:03d}.mp4"
                    render(
                        tools,
                        clip.info,
                        segments,
                        part,
                        encode=config.encode,
                        work_dir=work_dir,
                        fit=fit,
                    )
                    files.append(part)
                join_parts(tools, files, temporary, work_dir)
            validate_output(
                tools,
                temporary,
                expected_duration=frames / float(fit.rate),
                rate=fit.rate,
                expected_frames=frames,
                segments=len(shots),
            )
            finalize_output(temporary, final, force=force)
        finally:
            temporary.unlink(missing_ok=True)
    if plan_file is not None:
        if plan_file.exists() and not force and not analyze_only:
            warnings.append(f"{plan_file.name} already exists; not replacing it (use --force)")
            plan_file = None
        else:
            document = montage_document(infos, clips, shots, fit, montage_config, warnings)
            plan_file.write_text(json.dumps(document, indent=1), encoding="utf-8")

    return MontageResult(
        output=plan_file if analyze_only and plan_file is not None else final,
        plan_file=plan_file,
        clips=tuple(info.path for info in infos),
        shots=shots,
        footage_duration=sum(clip.analysed_duration for clip in clips),
        edited_duration=frames / float(fit.rate),
        elapsed=time.monotonic() - started,
        warnings=tuple(warnings),
    )


def montage_fit(clips: Sequence[Clip], shots: Sequence[Shot]) -> Fit:
    """The largest frame size of the clips used (even, square pixels), so no clip loses
    detail, at the frame rate of the clip that fills most of the montage."""
    share: dict[int, float] = {}
    for shot in shots:
        share[shot.clip] = share.get(shot.clip, 0.0) + shot.duration
    main = max(clips, key=lambda clip: (share.get(clip.index, 0.0), -clip.index))
    width, height = max((_display_size(clip.info) for clip in clips), key=lambda s: s[0] * s[1])
    return Fit(width - width % 2, height - height % 2, output_rate(main.info))


def _display_size(info: MediaInfo) -> tuple[int, int]:
    width, height = decoded_size(info)
    aspect = float(info.video.sample_aspect_ratio) or 1.0
    if info.video.rotation % 180 == 90:
        return width, round(height * aspect)
    return round(width * aspect), height


def _fitted_frames(clip: Clip, segments: Sequence[Interval], fit: Fit) -> int:
    rate = output_rate(clip.info)
    frames = expected_frame_count(segments, rate, clip.times)
    if rate == fit.rate:
        return frames
    return sum(
        round(Fraction(s.end) * fit.rate) - round(Fraction(s.start) * fit.rate) for s in segments
    )


def montage_document(
    infos: Sequence[MediaInfo],
    clips: Sequence[Clip],
    shots: Sequence[Shot],
    fit: Fit,
    config: MontageConfig,
    warnings: Sequence[str],
) -> dict[str, Any]:
    """JSON-ready description of a montage: the clips, and the shots in playing order."""
    usable = {clip.index for clip in clips}
    return {
        "fpv_editor_version": __version__,
        "config": dataclasses.asdict(config),
        "output": {"width": fit.width, "height": fit.height, "fps": str(fit.rate)},
        "clips": [
            {"path": str(info.path), "duration": info.duration, "usable": index in usable}
            for index, info in enumerate(infos)
        ],
        "shots": [
            {
                "clip": shot.clip,
                "source": infos[shot.clip].path.name,
                "start": round(shot.start, 3),
                "end": round(shot.end, 3),
                "score": round(shot.score, 2),
                "moments": [{"time": round(m.time, 2), "kind": m.kind.value} for m in shot.moments],
            }
            for shot in shots
        ],
        "warnings": list(warnings),
    }


def _render_and_validate(
    tools: Tools,
    info: MediaInfo,
    segments: tuple[Interval, ...],
    final: Path,
    *,
    force: bool,
    config: PipelineConfig,
    times: NDArray[np.float64] | None = None,
) -> None:
    rate = output_rate(info)
    frames = expected_frame_count(segments, rate, times)
    temporary = temporary_output(final)
    try:
        with tempfile.TemporaryDirectory(prefix="fpv-editor-") as work:
            render(tools, info, segments, temporary, encode=config.encode, work_dir=Path(work))
        validate_output(
            tools,
            temporary,
            expected_duration=frames / float(rate),
            rate=rate,
            expected_frames=frames,
            segments=len(segments),
        )
        finalize_output(temporary, final, force=force)
    finally:
        temporary.unlink(missing_ok=True)


def _gap_warnings(info: MediaInfo, times: NDArray[np.float64] | None) -> list[str]:
    """Mention missing frames (signal loss): the edit plays straight across them."""
    if times is None or info.video.nominal_rate is None:
        return []
    gaps = frame_gaps(times, info.video.nominal_rate)
    if not gaps:
        return []
    start, longest = max(gaps, key=lambda gap: gap[1])
    return [
        f"{info.path.name} has {len(gaps)} gap(s) with missing frames (longest {longest:.2f} s "
        f"at {start:.1f} s, e.g. video signal loss); the edit skips over them"
    ]


def _write_analysis(path: Path, analysis: Analysis, tools: Tools, config: PipelineConfig) -> None:
    document = analysis_document(analysis, tools, config.analysis)
    document["config"]["encode"] = {
        "video_codec": config.encode.video_codec,
        "preset": config.encode.preset,
        "crf": config.encode.crf,
    }
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trim and len(args.inputs) != 1:
        parser.error("--trim edits exactly one recording")
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
                )
            )
        result = run(
            args.inputs[0],
            output=args.output,
            force=args.force,
            keep_intermediates=args.keep_intermediates,
            analyze_only=args.analyze_only,
            cache_dir=args.cache_dir,
        )
    except FPVEditError as exc:
        print(f"fpv-editor: error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # Disk full, permission denied, output open in a player on Windows, ...
        print(f"fpv-editor: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("fpv-editor: interrupted; no output written", file=sys.stderr)
        return 130

    retained = result.edited_duration / result.original_duration if result.original_duration else 0
    print(f"Output:    {result.output}")
    print(f"Original:  {format_duration(result.original_duration)}")
    print(f"Edited:    {format_duration(result.edited_duration)} ({retained:.0%} retained)")
    print(f"Segments:  {len(result.selection.intervals)}")
    print(f"Time:      {result.elapsed:.1f} s")
    for warning in result.warnings:
        print(f"Warning:   {warning}", file=sys.stderr)
    return 0


def _report_montage(result: MontageResult) -> int:
    used = len({shot.clip for shot in result.shots})
    print(f"Output:    {result.output}")
    print(f"Clips:     {len(result.clips)} given, shots from {used}")
    print(f"Footage:   {format_duration(result.footage_duration)} of usable recordings")
    print(f"Montage:   {format_duration(result.edited_duration)} in {len(result.shots)} shots")
    print(f"Time:      {result.elapsed:.1f} s")
    for warning in result.warnings:
        print(f"Warning:   {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
