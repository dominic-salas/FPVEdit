"""Command-line entry point: probe → analyse → pick → render → validate → move into place.

By default the clips given become one montage of their best shots (`montage.py`). With
``--trim`` a single recording is edited as a whole flight instead (`highlights.py`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import queue
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpv_editor import __version__
from fpv_editor.analysis import Analysis, AnalysisConfig, analysis_document, analyze
from fpv_editor.errors import (
    CancelledError,
    FPVEditError,
    InvalidInputError,
    NoFlightContentError,
    OutputExistsError,
)
from fpv_editor.flight import FlightConfig
from fpv_editor.highlights import Interval, Selection, SelectionConfig
from fpv_editor.measure import MeasureConfig
from fpv_editor.media import (
    AudioFile,
    MediaInfo,
    Tools,
    find_tools,
    frame_gaps,
    frame_times,
    is_variable_rate,
    probe,
    probe_audio,
)
from fpv_editor.montage import (
    MontageConfig,
    Shot,
    candidate_shots,
    leave_out_weak_indoor,
    pick_shots,
)
from fpv_editor.picture import PictureConfig, decoded_size
from fpv_editor.render import (
    DEVICES,
    NVENC,
    EncodeConfig,
    Fit,
    add_music,
    encoders,
    expected_frame_count,
    finalize_output,
    join_parts,
    output_rate,
    render,
    resolve_output_path,
    snap_to_frames,
    temporary_output,
    validate_output,
    with_fallback,
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
    encoder: str | None = None  # the video codec used; None when no video was written


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
    encoder: str | None = None  # the video codec used; None when no video was written


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
        "--music",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="background music for the montage (repeat for more tracks, played in order; "
        "cut and faded at the end, silence after if they are shorter)",
    )
    parser.add_argument(
        "--no-scenery",
        action="store_true",
        help="keep indoor-looking shots that are weaker than the outdoor ones (by default "
        "the image model leaves them out when the ml extra is installed)",
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
    device: str = "cpu",
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

    encoder = None
    if not analyze_only:
        _, used = with_fallback(
            encoders(tools, device, config.encode),
            lambda encode: _render_and_validate(
                tools, info, segments, final, force=force, config=config, times=times, encode=encode
            ),
        )
        encoder = used.video_codec
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
        encoder=encoder,
    )


Progress = Callable[[str, float], None]
"""Called with what is happening and the share done (0-1); may raise `CancelledError`."""


@dataclass(frozen=True)
class MontagePlan:
    """Clips analysed for a montage, every candidate shot found in them and the shots picked."""

    infos: tuple[MediaInfo, ...]  # every input, in the order given
    clips: tuple[Clip, ...]  # inputs with usable footage
    candidates: tuple[Shot, ...]
    shots: tuple[Shot, ...]  # in montage order
    config: MontageConfig
    warnings: tuple[str, ...]
    left_out: tuple[Shot, ...] = ()  # weak indoor shots the image model left out

    @property
    def footage_duration(self) -> float:
        return sum(clip.analysed_duration for clip in self.clips)


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
    device: str = "cpu",
    music: Sequence[Path] = (),
    scenery: bool = True,
) -> MontageResult:
    config = config or PipelineConfig()
    started = time.monotonic()
    if not sources:
        raise InvalidInputError("no input recordings given")
    if length is not None and length <= 0:
        raise InvalidInputError("--length must be a positive number of seconds")
    tools = find_tools()
    infos = [probe(tools, source) for source in sources]  # fail early on a bad path
    tracks = [probe_audio(tools, track) for track in music]
    final = resolve_output_path(
        [info.path for info in infos], output, force or analyze_only, suffix=MONTAGE_SUFFIX
    )
    plan_file = plan_path(final) if analyze_only or keep_intermediates else None
    if analyze_only and plan_file is not None and plan_file.exists() and not force:
        raise OutputExistsError(f"{plan_file} already exists; use --force to replace it")

    plan = plan_montage(tools, infos, length=length, cache_dir=cache_dir, config=config)
    if scenery and length is None:
        plan = judge_scenery(tools, plan, cache_dir=cache_dir)
    warnings = list(plan.warnings)
    if not plan.shots:
        raise NoFlightContentError(
            "none of the recordings has a moment worth a shot (flip, spin or close pass "
            "in clean flying)"
        )
    fit = montage_fit(plan.clips, plan.shots)
    encoder = None
    if analyze_only:
        parts = montage_parts(plan.clips, plan.shots)
        edited_duration = sum(_fitted_frames(c, s, fit) for c, s in parts) / float(fit.rate)
    else:
        edited_duration, used = with_fallback(
            encoders(tools, device, config.encode),
            lambda encode: render_montage(
                tools, plan.clips, plan.shots, final, force=force, encode=encode, music=tracks
            ),
        )
        encoder = used.video_codec
    if plan_file is not None:
        if plan_file.exists() and not force and not analyze_only:
            warnings.append(f"{plan_file.name} already exists; not replacing it (use --force)")
            plan_file = None
        else:
            document = montage_document(
                plan.infos, plan.clips, plan.shots, fit, plan.config, warnings
            )
            plan_file.write_text(json.dumps(document, indent=1), encoding="utf-8")

    return MontageResult(
        output=plan_file if analyze_only and plan_file is not None else final,
        plan_file=plan_file,
        clips=tuple(info.path for info in plan.infos),
        shots=plan.shots,
        footage_duration=plan.footage_duration,
        edited_duration=edited_duration,
        elapsed=time.monotonic() - started,
        warnings=tuple(warnings),
        encoder=encoder,
    )


def plan_montage(
    tools: Tools,
    infos: Sequence[MediaInfo],
    *,
    length: float | None = None,
    cache_dir: Path | None = None,
    config: PipelineConfig | None = None,
    progress: Progress | None = None,
    workers: int | None = None,
) -> MontagePlan:
    """Analyse every clip and pick the montage's shots. ``plan.shots`` may be empty.

    Clips are analysed in parallel processes (``workers``, default from the CPU count);
    each one's result is the same as when analysed alone.
    """
    config = config or PipelineConfig()
    montage_config = config.montage
    if length is not None:
        if length <= 0:
            raise InvalidInputError("--length must be a positive number of seconds")
        montage_config = dataclasses.replace(montage_config, length_s=length)
    warnings: dict[int, list[str]] = {}
    jobs: list[ClipJob] = []
    seen: set[Path] = set()
    for index, info in enumerate(infos):
        if info.path.resolve() in seen:
            warnings[index] = [f"{info.path.name} was given more than once; using it once"]
            continue
        seen.add(info.path.resolve())
        jobs.append(ClipJob(tools, info, index, config, montage_config, cache_dir))
    count = min(len(jobs), workers if workers is not None else default_workers())
    if count > 1:
        results = _plan_in_processes(jobs, count, progress)
    else:
        results = []
        for number, job in enumerate(jobs):
            text = f"Analysing {job.info.path.name}"
            results.append(_plan_clip(job, _clip_progress(progress, text, number, len(jobs))))
    return combine_clips(infos, results, montage_config, warnings)


def combine_clips(
    infos: Sequence[MediaInfo],
    results: Sequence[ClipResult],
    config: MontageConfig,
    warnings: dict[int, list[str]] | None = None,
) -> MontagePlan:
    """The montage plan from clips analysed one by one; ``results[i].index`` is the
    clip's place in ``infos``. ``warnings`` adds notes per clip index."""
    notes = {index: list(lines) for index, lines in (warnings or {}).items()}
    clips: list[Clip] = []
    candidates: list[Shot] = []
    for result in results:
        notes[result.index] = list(result.warnings)
        if result.clip is not None:
            clips.append(result.clip)
        candidates.extend(result.candidates)
    clips.sort(key=lambda clip: clip.index)
    return MontagePlan(
        infos=tuple(infos),
        clips=tuple(clips),
        candidates=tuple(candidates),
        shots=pick_shots(candidates, len(infos), config),
        config=config,
        warnings=tuple(note for index in sorted(notes) for note in notes[index]),
    )


def judge_scenery(
    tools: Tools,
    plan: MontagePlan,
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
) -> MontagePlan:
    """Leave out indoor-looking shots that are weaker than the session's best outdoor
    ones (`montage.leave_out_weak_indoor`), using the image model in its own process.

    Without the ``ml`` extra, or if the model fails, the plan is returned with a note.
    """
    from fpv_editor import scenery

    if not plan.shots:
        return plan
    if not scenery.available():
        return dataclasses.replace(
            plan,
            warnings=(
                *plan.warnings,
                "the image model is not installed (uv sync --extra "
                "ml), so indoor shots were not compared with outdoor ones",
            ),
        )
    try:
        indoorness = scenery.score_shots_in_process(
            tools, plan.infos, plan.shots, cache_dir=cache_dir, device=device
        )
    except FPVEditError as exc:
        return dataclasses.replace(
            plan, warnings=(*plan.warnings, f"the image model could not run: {exc}")
        )
    kept, left_out = leave_out_weak_indoor(plan.shots, indoorness, plan.config)
    notes = plan.warnings
    if left_out:
        notes = (
            *notes,
            f"left out {len(left_out)} indoor-looking shot(s) weaker than the best outdoor "
            "shots; they can be added back",
        )
    return dataclasses.replace(plan, shots=kept, left_out=left_out, warnings=notes)


# Each analysis runs one FFmpeg decode (multi-threaded) and one Python process; a quarter
# of the logical CPUs keeps the decoders fed without oversubscribing.
CPUS_PER_ANALYSIS = 4


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1) // CPUS_PER_ANALYSIS)


@dataclass(frozen=True)
class ClipJob:
    tools: Tools
    info: MediaInfo
    index: int
    config: PipelineConfig
    montage: MontageConfig
    cache_dir: Path | None


@dataclass(frozen=True)
class ClipResult:
    index: int
    clip: Clip | None  # None when the clip has nothing usable
    candidates: tuple[Shot, ...]
    warnings: tuple[str, ...]

    def moved_to(self, index: int) -> ClipResult:
        """The same result for the clip at ``index`` of a montage's clip list."""
        return ClipResult(
            index,
            dataclasses.replace(self.clip, index=index) if self.clip is not None else None,
            tuple(dataclasses.replace(shot, clip=index) for shot in self.candidates),
            self.warnings,
        )


def _plan_clip(job: ClipJob, step: Callable[[float], None]) -> ClipResult:
    """Analyse one clip and find its candidate shots (runs in a worker process)."""
    tools, info, name = job.tools, job.info, job.info.path.name
    step(0.0)
    times = frame_times(tools, info)
    if is_variable_rate(info.video, times):
        warning = (
            f"{name} has a variable frame rate and was left out; convert it first, e.g. "
            "ffmpeg -i in.mp4 -fps_mode cfr -r 60 out.mp4"
        )
        return ClipResult(job.index, None, (), (warning,))
    try:
        analysis = analyze(tools, info, job.config.analysis, cache_dir=job.cache_dir, progress=step)
    except NoFlightContentError as exc:
        return ClipResult(job.index, None, (), (f"{name}: nothing usable ({exc})",))
    warnings = [f"{name}: {note}" for note in (*_gap_warnings(info, times), *analysis.warnings)]
    found = candidate_shots(
        job.index, analysis.measurements, analysis.selection, job.montage, job.config.selection
    )
    if not found:
        warnings.append(f"{name}: no flip, spin or close pass to build a shot around")
    clip = Clip(job.index, info, times, analysis.analysed_duration)
    return ClipResult(job.index, clip, tuple(found), tuple(warnings))


def plan_in_worker(job: ClipJob, shares: Any, stop: Any) -> ClipResult:
    """`_plan_clip` in a worker process: progress goes to ``shares``, ``stop`` cancels."""

    def step(share: float) -> None:
        if stop.is_set():
            raise CancelledError("cancelled")
        shares.put((job.index, share))

    return _plan_clip(job, step)


def _plan_in_processes(
    jobs: Sequence[ClipJob], workers: int, progress: Progress | None
) -> list[ClipResult]:
    """Run `_plan_clip` for every job in ``workers`` processes, reporting their progress
    weighted by clip length. If ``progress`` raises, the workers stop and it propagates."""
    weights = {job.index: job.info.duration or 1.0 for job in jobs}
    total = sum(weights.values())
    done = dict.fromkeys(weights, 0.0)
    context = multiprocessing.get_context("spawn")  # the same on Windows and Linux
    with context.Manager() as manager:
        shares, stop = manager.Queue(), manager.Event()
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            futures = [pool.submit(plan_in_worker, job, shares, stop) for job in jobs]
            try:
                while not all(future.done() for future in futures):
                    for future in futures:
                        if future.done() and future.exception() is not None:
                            future.result()  # raise a worker's error now
                    try:
                        index, share = shares.get(timeout=0.25)
                        done[index] = share
                    except queue.Empty:
                        pass
                    if progress is not None:
                        running = sum(1 for f in futures if f.running())
                        text = f"Analysing {len(jobs)} clips ({running} at a time)"
                        progress(text, sum(weights[i] * done[i] for i in done) / total)
                return [future.result() for future in futures]
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise


def _clip_progress(
    progress: Progress | None, text: str, index: int, count: int
) -> Callable[[float], None]:
    def step(share: float) -> None:
        if progress is not None:
            progress(text, (index + share) / max(1, count))

    return step


# Shots of one clip further apart than this are rendered as separate parts (each seeks).
PART_GAP_S = 20.0


def montage_parts(
    clips: Sequence[Clip], shots: Sequence[Shot]
) -> list[tuple[Clip, tuple[Interval, ...]]]:
    """Shots grouped into render parts, in montage order.

    A part reads its clip once and keeps frames in recording order, so it is a run of
    consecutive shots from one clip that go forward in time without overlapping. Shots
    far apart get parts of their own, since seeking to a shot is faster than decoding up
    to it. Reordered shots make more parts.
    """
    by_index = {clip.index: clip for clip in clips}
    runs: list[tuple[Clip, list[Shot]]] = []
    for shot in shots:
        if shot.clip not in by_index:
            raise InvalidInputError(f"shot {shot.start:.1f}-{shot.end:.1f} s has no clip")
        previous = runs[-1][1][-1] if runs and runs[-1][0].index == shot.clip else None
        if previous is not None and 0 <= shot.start - previous.end < PART_GAP_S:
            runs[-1][1].append(shot)
        else:
            runs.append((by_index[shot.clip], [shot]))
    return [
        (clip, snap_to_frames([s.interval for s in run], output_rate(clip.info)))
        for clip, run in runs
    ]


def render_montage(
    tools: Tools,
    clips: Sequence[Clip],
    shots: Sequence[Shot],
    final: Path,
    *,
    force: bool,
    encode: EncodeConfig,
    progress: Progress | None = None,
    music: Sequence[AudioFile] = (),
) -> float:
    """Render ``shots`` in the order given to ``final`` and validate it; returns its duration.

    ``music`` tracks play one after another as the soundtrack (the recordings' own audio
    is never used)."""
    if not shots:
        raise InvalidInputError("no shots to render")
    fit = montage_fit(clips, shots)
    parts = montage_parts(clips, shots)
    sizes = [_fitted_frames(clip, segments, fit) for clip, segments in parts]
    frames = sum(sizes)
    temporary = temporary_output(final)
    try:
        with tempfile.TemporaryDirectory(prefix="fpv-editor-", ignore_cleanup_errors=True) as work:
            work_dir = Path(work)
            files = []
            for number, (clip, segments) in enumerate(parts):
                text = f"Rendering part {number + 1} of {len(parts)} ({clip.info.path.name})"
                part = work_dir / f"part{number:03d}.mp4"
                render(
                    tools,
                    clip.info,
                    segments,
                    part,
                    encode=encode,
                    work_dir=work_dir,
                    fit=fit,
                    on_frames=_frame_progress(progress, text, sum(sizes[:number]), frames),
                )
                files.append(part)
            if progress is not None:
                progress("Joining and checking the montage", 1.0)
            if music:
                video = work_dir / "video.mp4"
                join_parts(tools, files, video, work_dir)
                if progress is not None:
                    progress("Adding the music", 1.0)
                add_music(tools, video, music, temporary, frames / float(fit.rate))
            else:
                join_parts(tools, files, temporary, work_dir)
        validate_output(
            tools,
            temporary,
            expected_duration=frames / float(fit.rate),
            rate=fit.rate,
            expected_frames=frames,
            segments=len(shots),
            music=bool(music),
        )
        if progress is not None:
            progress("Checked", 1.0)  # last chance to cancel before the file appears
        finalize_output(temporary, final, force=force)
    finally:
        temporary.unlink(missing_ok=True)
    return frames / float(fit.rate)


def _frame_progress(
    progress: Progress | None, text: str, done: int, total: int
) -> Callable[[int], None] | None:
    if progress is None:
        return None
    report = progress
    report(text, done / max(1, total))
    return lambda frames: report(text, min(1.0, (done + frames) / max(1, total)))


def montage_fit(clips: Sequence[Clip], shots: Sequence[Shot]) -> Fit:
    """The largest frame size of the clips the shots use (even, square pixels), so no clip
    loses detail, at the frame rate of the clip that fills most of the montage."""
    used = {shot.clip for shot in shots}
    clips = [clip for clip in clips if clip.index in used]
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
    # The render renumbers the kept frames without gaps, then converts the rate.
    return round(frames * fit.rate / rate)


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
                "moments": [
                    {"time": round(m.time, 2), "kind": m.kind.value, "trick": m.name}
                    for m in shot.moments
                ],
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
    encode: EncodeConfig | None = None,
) -> None:
    rate = output_rate(info)
    frames = expected_frame_count(segments, rate, times)
    temporary = temporary_output(final)
    try:
        with tempfile.TemporaryDirectory(prefix="fpv-editor-", ignore_cleanup_errors=True) as work:
            render(
                tools,
                info,
                segments,
                temporary,
                encode=encode or config.encode,
                work_dir=Path(work),
            )
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
    multiprocessing.freeze_support()  # analysis workers in a standalone build
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trim and len(args.inputs) != 1:
        parser.error("--trim edits exactly one recording")
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
                )
            )
        result = run(
            args.inputs[0],
            output=args.output,
            force=args.force,
            keep_intermediates=args.keep_intermediates,
            analyze_only=args.analyze_only,
            cache_dir=args.cache_dir,
            device=args.device,
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
