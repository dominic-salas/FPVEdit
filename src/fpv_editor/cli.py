"""Command-line entry point: probe → sample → features → score → select → render → validate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fpv_editor import __version__
from fpv_editor.errors import FPVEditError
from fpv_editor.features import FeatureConfig, FeatureSeries, compute_features
from fpv_editor.highlights import Selection, SelectionConfig, select_highlights
from fpv_editor.media import MediaInfo, Tools, find_tools, probe
from fpv_editor.render import (
    EncodeConfig,
    finalize_output,
    render,
    resolve_output_path,
    temporary_output,
    validate_output,
)
from fpv_editor.sampling import SamplingConfig, SamplingReport, iter_gray_frames, sample_duration
from fpv_editor.scoring import ScoringConfig, idle_mask, interest_score


@dataclass(frozen=True)
class PipelineConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)


@dataclass(frozen=True)
class Result:
    output: Path
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
        help="also write <output>.analysis.json with per-sample scores and the edit list",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run(
    source: Path,
    *,
    output: Path | None = None,
    force: bool = False,
    keep_intermediates: bool = False,
    config: PipelineConfig | None = None,
) -> Result:
    config = config or PipelineConfig()
    started = time.monotonic()
    tools = find_tools()
    info = probe(tools, source)
    final = resolve_output_path(info.path, output, force)

    report = SamplingReport()
    with closing(iter_gray_frames(tools, info, config.sampling, report)) as frames:
        features = compute_features(frames, config.sampling.fps, config.features)
    analysed_duration = sample_duration(report, config.sampling)
    score = interest_score(features, config.scoring)
    selection = select_highlights(
        score, idle_mask(features, config.scoring), config.sampling.fps, config.selection
    )
    warnings = [*report.warnings, *selection.warnings]

    temporary = temporary_output(final)
    try:
        with tempfile.TemporaryDirectory(prefix="fpv-editor-") as work:
            render(
                tools,
                info,
                selection.intervals,
                temporary,
                encode=config.encode,
                work_dir=Path(work),
            )
        validate_output(
            tools,
            temporary,
            expected_duration=selection.kept_duration,
            expect_audio=info.audio is not None,
        )
        finalize_output(temporary, final, force=force)
    finally:
        temporary.unlink(missing_ok=True)

    if keep_intermediates:
        analysis = final.with_name(f"{final.stem}.analysis.json")
        if analysis.exists() and not force:
            warnings.append(f"{analysis.name} already exists; not replacing it (use --force)")
        else:
            _write_analysis(analysis, info, tools, features, score.tolist(), selection, config)

    return Result(
        output=final,
        original_duration=info.duration or analysed_duration,
        edited_duration=selection.kept_duration,
        elapsed=time.monotonic() - started,
        selection=selection,
        warnings=tuple(warnings),
    )


def _write_analysis(
    path: Path,
    info: MediaInfo,
    tools: Tools,
    features: FeatureSeries,
    score: list[float],
    selection: Selection,
    config: PipelineConfig,
) -> None:
    data = {
        "fpv_editor_version": __version__,
        "ffmpeg_major": tools.ffmpeg_major,
        "source": str(info.path),
        "config": asdict(config),
        "fps": features.fps,
        "samples": {
            "motion": features.motion.round(5).tolist(),
            "change": features.change.round(5).tolist(),
            "novelty": features.novelty.round(5).tolist(),
            "brightness": features.brightness.round(4).tolist(),
            "score": [round(value, 5) for value in score],
        },
        "selection": {
            "mode": selection.mode.value,
            "retention": selection.retention,
            "threshold": selection.threshold,
            "intervals": [[i.start, i.end] for i in selection.intervals],
            "idle": [[i.start, i.end] for i in selection.idle],
            "warnings": list(selection.warnings),
        },
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


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
