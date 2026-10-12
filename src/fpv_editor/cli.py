"""Command-line entry point: probe → analyse → render → validate → move into place."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fpv_editor import __version__
from fpv_editor.analysis import Analysis, AnalysisConfig, analysis_document, analyze
from fpv_editor.errors import FPVEditError, InvalidInputError, OutputExistsError
from fpv_editor.features import FeatureConfig
from fpv_editor.highlights import Interval, Selection, SelectionConfig
from fpv_editor.media import MediaInfo, Tools, find_tools, is_variable_rate, probe
from fpv_editor.picture import PictureConfig
from fpv_editor.render import (
    EncodeConfig,
    expected_frame_count,
    finalize_output,
    output_rate,
    render,
    resolve_output_path,
    snap_to_frames,
    temporary_output,
    validate_output,
)
from fpv_editor.sampling import SamplingConfig
from fpv_editor.scoring import ScoringConfig


@dataclass(frozen=True)
class PipelineConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    picture: PictureConfig = field(default_factory=PictureConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)

    @property
    def analysis(self) -> AnalysisConfig:
        return AnalysisConfig(
            sampling=self.sampling,
            picture=self.picture,
            features=self.features,
            scoring=self.scoring,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpv-editor",
        description="Turn a raw FPV freestyle recording into one highlight MP4.",
    )
    parser.add_argument("input", type=Path, help="recording to edit (.ts or .mp4)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .mp4 (default: <input>_highlight.mp4 beside the input)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace the output file if it already exists"
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="also write <output name>.analysis.json with per-sample scores and the edit list",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="write only an analysis file and no video (default: <input name>.analysis.json)",
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
    if is_variable_rate(info.video):
        raise InvalidInputError(
            f"{info.path.name} has a variable frame rate (average "
            f"{info.video.frame_rate:.2f} fps, nominal {info.video.nominal_rate} fps), which is "
            "not supported yet. DVR recordings are constant-rate; for phone or screen "
            "captures, convert first, e.g. ffmpeg -i in.mp4 -fps_mode cfr -r 60 out.mp4"
        )
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
    warnings = list(analysis.warnings)
    rate = output_rate(info)
    segments = snap_to_frames(analysis.selection.intervals, rate)
    edited_duration = sum(segment.duration for segment in segments)

    if not analyze_only:
        _render_and_validate(tools, info, segments, final, force=force, config=config)
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


def _render_and_validate(
    tools: Tools,
    info: MediaInfo,
    segments: tuple[Interval, ...],
    final: Path,
    *,
    force: bool,
    config: PipelineConfig,
) -> None:
    rate = output_rate(info)
    temporary = temporary_output(final)
    try:
        with tempfile.TemporaryDirectory(prefix="fpv-editor-") as work:
            render(tools, info, segments, temporary, encode=config.encode, work_dir=Path(work))
        validate_output(
            tools,
            temporary,
            expected_duration=sum(segment.duration for segment in segments),
            rate=rate,
            expected_frames=expected_frame_count(segments, rate),
            segments=len(segments),
        )
        finalize_output(temporary, final, force=force)
    finally:
        temporary.unlink(missing_ok=True)


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
    args = build_parser().parse_args(argv)
    try:
        result = run(
            args.input,
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


if __name__ == "__main__":
    raise SystemExit(main())
