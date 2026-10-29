"""The editing pipeline: probe, analyse, pick the shots, rate them, render and validate.

A montage (`run_montage`) is made of the best shots of several recordings (`montage.py`);
a trim (`run_trim`) edits one recording as a whole flight (`highlights.py`). The command
line (`cli.py`) and the desktop app (`app/`) both run it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import multiprocessing
import os
import queue
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpvedit import __version__, attitude, osd
from fpvedit.analysis import Analysis, AnalysisConfig, analysis_document, analyze, cache_path
from fpvedit.errors import (
    CancelledError,
    FPVEditError,
    InvalidInputError,
    NoFlightContentError,
    OutputExistsError,
)
from fpvedit.flight import FlightConfig
from fpvedit.highlights import Interval, Selection, SelectionConfig
from fpvedit.measure import MeasureConfig
from fpvedit.media import (
    AudioFile,
    MediaInfo,
    Tools,
    find_tools,
    frame_gaps,
    frame_times,
    is_variable_rate,
    output_rate,
    probe,
    probe_audio,
)
from fpvedit.montage import (
    CutMap,
    Moment,
    MomentKind,
    MontageConfig,
    Shot,
    candidate_shots,
    cut_map,
    pick_shots,
)
from fpvedit.picture import PictureConfig, decoded_size
from fpvedit.rating import rate_shot
from fpvedit.render import (
    EncodeConfig,
    Fit,
    add_music,
    encoders,
    expected_frame_count,
    finalize_output,
    join_parts,
    render,
    resolve_output_path,
    snap_to_frames,
    temporary_output,
    validate_output,
    with_fallback,
)
from fpvedit.taste import RatedShot, Ratings, Taste
from fpvedit.tricks import name_measured

DEFAULT_MIN_STARS = 3  # shots rated below this are left out of a montage (can be added)


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
    of clips. ``cuts`` keeps the little a shot built by hand needs (about 2.6 MB per ten
    minutes at 60 fps).
    """

    index: int
    info: MediaInfo
    times: NDArray[np.float64] | None
    analysed_duration: float
    cuts: CutMap | None = field(default=None, compare=False, repr=False)


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


def analysis_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}.analysis.json")


def plan_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}.montage.json")


def run_trim(
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
            lambda encode: render_trim(
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
    left_out: tuple[Shot, ...] = ()  # shots rated below the minimum, offered to add back

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
    min_stars: int = DEFAULT_MIN_STARS,
    vertical: bool = False,
) -> MontageResult:
    config = config or PipelineConfig()
    started = time.monotonic()
    if not sources:
        raise InvalidInputError("no input recordings given")
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
    if length is None:
        plan = keep_rated(tools, plan, min_stars=min_stars, scenery=scenery, cache_dir=cache_dir)
    warnings = list(plan.warnings)
    if not plan.shots:
        raise NoFlightContentError(
            "none of the recordings has a moment worth a shot (flip, spin or close pass "
            "in clean flying)"
        )
    fit = montage_fit(plan.clips, plan.shots, vertical=vertical)
    encoder = None
    if analyze_only:
        parts = montage_parts(plan.clips, plan.shots)
        edited_duration = sum(_fitted_frames(c, s, fit) for c, s in parts) / float(fit.rate)
    else:
        edited_duration, used = with_fallback(
            encoders(tools, device, config.encode),
            lambda encode: render_montage(
                tools,
                plan.clips,
                plan.shots,
                final,
                force=force,
                encode=encode,
                music=tracks,
                vertical=vertical,
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


def is_cached(tools: Tools, info: MediaInfo, cache_dir: Path, config: PipelineConfig) -> bool:
    """Whether ``info`` has measurements in ``cache_dir`` for the current settings."""
    from fpvedit.analysis import cache_path

    try:
        return cache_path(cache_dir, tools, info, config.analysis).is_file()
    except OSError:
        return False


def _rated_without_picture(
    tools: Tools, infos: Sequence[MediaInfo], rated: Sequence[RatedShot]
) -> tuple[list[MediaInfo], dict[int, Shot]]:
    """The recordings and shots (by index in ``rated``) whose picture is still unseen."""
    infos = list(infos)
    where = {info.path.resolve(): index for index, info in enumerate(infos)}
    extra: dict[int, Shot] = {}
    for index, shot in enumerate(rated):
        if shot.quality.picture is not None or shot.recording is None:
            continue
        path = shot.recording
        try:
            key = path.resolve()
            if key not in where:
                infos.append(probe(tools, path))
                where[key] = len(infos) - 1
        except (FPVEditError, OSError):
            continue  # moved or deleted since it was rated: learned without its picture
        extra[index] = Shot(where[key], shot.start, shot.start + shot.duration, 0.0, ())
    return infos, extra


def taste_predictor(
    infos: Sequence[MediaInfo], ratings: Ratings, model: Taste
) -> Callable[[Shot], float | None]:
    def stars(shot: Shot) -> float | None:
        given = ratings.stars(infos[shot.clip].path, shot)
        if given is not None:
            return float(given)
        if shot.quality is None:
            return None
        return model.predict(shot.quality, [m.name for m in shot.moments], shot.duration)

    return stars


def keep_rated(
    tools: Tools,
    plan: MontagePlan,
    *,
    min_stars: int = DEFAULT_MIN_STARS,
    scenery: bool = True,
    cache_dir: Path | None = None,
    device: str = "auto",
    ratings: Ratings | None = None,
    progress: Progress | None = None,
) -> MontagePlan:
    """Keep the shots rated ``min_stars`` or more (`rating.py`); the rest are left out and
    can be added back. If no shot reaches ``min_stars``, the best-rated ones are kept.

    With ``scenery`` and the ``ml`` extra, the image model (in its own process) first adds
    its small indoor/outdoor nudge and what is in the picture to each rating. With the
    pilot's own ``ratings``, a taste is learned from them (`taste.py`): shots they rated
    keep their own stars, the others get the stars it predicts. Their rated shots' pictures
    are seen in the same pass (once; the frames are cached).
    """
    from fpvedit import scenery as scenery_model

    if not plan.shots:
        return plan
    notes = list(plan.warnings)
    if scenery and attitude.available():
        try:
            plan = measure_turns(tools, plan, cache_dir=cache_dir, device=device, progress=progress)
        except FPVEditError as exc:
            notes.append(f"the tricks' turns could not be measured: {exc}")
    elif scenery:
        notes.append(
            "the gravity model is not installed (uv sync --extra ml), so trick names come "
            "from the flow alone"
        )
    shots = plan.shots
    rated = ratings.rated_shots() if ratings is not None else []
    if scenery and scenery_model.available():
        try:
            infos, extra = _rated_without_picture(tools, plan.infos, rated)
            scenes = scenery_model.score_shots_in_process(
                tools,
                infos,
                [*shots, *extra.values()],
                cache_dir=cache_dir,
                device=device,
                progress=progress,
            )
            shots = tuple(
                dataclasses.replace(
                    s,
                    quality=dataclasses.replace(
                        s.quality, indoorness=scene.indoorness, picture=scene.picture
                    ),
                )
                if s.quality is not None
                else s
                for s, scene in zip(shots, scenes[: len(shots)], strict=True)
            )
            seen = dict(zip(extra, scenes[len(shots) :], strict=True))
            rated = [
                dataclasses.replace(
                    r, quality=dataclasses.replace(r.quality, picture=seen[i].picture)
                )
                if i in seen
                else r
                for i, r in enumerate(rated)
            ]
        except FPVEditError as exc:
            notes.append(f"the image model could not run: {exc}")
    elif scenery:
        notes.append(
            "the image model is not installed (uv sync --extra ml), so the ratings do not "
            "include scenery"
        )
    candidates = plan.candidates
    if ratings is not None:
        stars_for = taste_predictor(plan.infos, ratings, Taste.fit(rated))

        def tasted(found: tuple[Shot, ...]) -> tuple[Shot, ...]:
            return tuple(
                dataclasses.replace(s, quality=dataclasses.replace(s.quality, taste=stars_for(s)))
                if s.quality is not None
                else s
                for s in found
            )

        # The candidates too, so the shots offered to add are ordered by the same taste.
        shots, candidates = tasted(shots), tasted(candidates)
    stars = [s.quality.stars if s.quality is not None else 3 for s in shots]
    bar = min(min_stars, max(stars))
    kept = tuple(s for s, n in zip(shots, stars, strict=True) if n >= bar)
    left_out = tuple(s for s, n in zip(shots, stars, strict=True) if n < bar)
    if left_out:
        notes.append(
            f"left out {len(left_out)} shot(s) rated below {bar} stars; they can be added back"
        )
    return dataclasses.replace(
        plan, candidates=candidates, shots=kept, left_out=left_out, warnings=tuple(notes)
    )


def measure_turns(
    tools: Tools,
    plan: MontagePlan,
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
    progress: Progress | None = None,
) -> MontagePlan:
    """Measure each clip's fast turns with gravity (`attitude.py`, in its own process), then
    name the flips by what was measured and rate their shots again."""
    clips = [(c, c.cuts) for c in plan.clips if c.cuts is not None]
    wanted = []
    for clip, cuts in clips:
        spans = attitude.fast_turns(cuts.signals.rotation, cuts.fps)
        wanted.append((clip.info, attitude.anchor_times(spans, cuts.fps)))
    if not any(times for _, times in wanted):
        return plan
    readings = attitude.read_gravity_in_process(
        tools, wanted, cache_dir=cache_dir, device=device, progress=progress
    )
    measured: dict[int, CutMap] = {}
    for (clip, cuts), gravity in zip(clips, readings, strict=True):
        found = attitude.measure(cuts.signals.rotation, cuts.fps, gravity)
        signals = dataclasses.replace(cuts.signals, attitude=found)
        measured[clip.index] = dataclasses.replace(cuts, signals=signals)

    def named(moment: Moment, found: attitude.Attitude) -> Moment:
        if moment.kind is not MomentKind.FLIP or moment.trick == "power loop":
            return moment
        turn = found.turn_at(moment.time)
        if turn is None or not turn.checked:
            return moment
        return dataclasses.replace(moment, trick=name_measured(turn, found))

    def again(shots: tuple[Shot, ...]) -> tuple[Shot, ...]:
        result = []
        for shot in shots:
            cuts = measured.get(shot.clip)
            found = cuts.signals.attitude if cuts is not None else None
            if cuts is None or found is None or shot.quality is None:
                result.append(shot)
                continue
            moments = tuple(named(m, found) for m in shot.moments)
            quality = rate_shot(
                shot.start, shot.end, [(m.time, m.name) for m in moments], cuts.signals
            )
            result.append(dataclasses.replace(shot, moments=moments, quality=quality))
        return tuple(result)

    return dataclasses.replace(
        plan,
        clips=tuple(
            dataclasses.replace(c, cuts=measured[c.index]) if c.index in measured else c
            for c in plan.clips
        ),
        shots=again(plan.shots),
        candidates=again(plan.candidates),
        left_out=again(plan.left_out),
    )


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


def _throttle(
    tools: Tools,
    info: MediaInfo,
    analysis: Analysis,
    cache_dir: Path | None,
    config: AnalysisConfig,
) -> osd.Throttle | None:
    """The OSD throttle (`osd.py`), cached beside the recording's measurements with its own
    version, so reading it never invalidates them. None when it is not on screen."""
    path = None
    if cache_dir is not None:
        try:
            measured = cache_path(cache_dir, tools, info, config)
            path = measured.with_name(
                measured.name.replace(".measurements.npz", f".throttle-v{osd.VERSION}.npz")
            )
        except OSError:
            path = None
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["values"], dtype=np.float64)
                return osd.Throttle(float(data["fps"]), values) if values.size else None
        except (OSError, ValueError, KeyError):
            pass  # damaged: read it again
    throttle = osd.read_throttle(tools, info, analysis.area)
    if path is not None:
        empty = np.zeros(0)
        with contextlib.suppress(OSError):  # an unwritable cache only costs reading it again
            np.savez(path, fps=osd.FPS, values=throttle.values if throttle else empty)
    return throttle


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
    throttle = _throttle(tools, info, analysis, job.cache_dir, job.config.analysis)
    cuts = cut_map(
        analysis.measurements, analysis.selection, job.montage, job.config.selection, throttle
    )
    found = candidate_shots(
        job.index,
        analysis.measurements,
        analysis.selection,
        job.montage,
        job.config.selection,
        cuts,
    )
    if not found:
        warnings.append(f"{name}: no flip, spin or close pass to build a shot around")
    clip = Clip(job.index, info, times, analysis.analysed_duration, cuts)
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
    vertical: bool = False,
) -> float:
    """Render ``shots`` in the order given to ``final`` and validate it; returns its duration.

    ``music`` tracks play one after another as the soundtrack, in place of the
    recordings' own sound (kept otherwise). ``vertical`` makes a 9:16 video for phones
    (`montage_fit`)."""
    if not shots:
        raise InvalidInputError("no shots to render")
    fit = montage_fit(clips, shots, vertical=vertical)
    parts = montage_parts(clips, shots)
    sizes = [_fitted_frames(clip, segments, fit) for clip, segments in parts]
    frames = sum(sizes)
    temporary = temporary_output(final)
    try:
        with tempfile.TemporaryDirectory(prefix="fpvedit-", ignore_cleanup_errors=True) as work:
            work_dir = Path(work)
            files, sounds = [], []
            for number, (clip, segments) in enumerate(parts):
                text = f"Rendering part {number + 1} of {len(parts)} ({clip.info.path.name})"
                part = work_dir / f"part{number:03d}.mp4"
                sound = None if music else work_dir / f"part{number:03d}.wav"
                render(
                    tools,
                    clip.info,
                    segments,
                    part,
                    encode=encode,
                    work_dir=work_dir,
                    fit=fit,
                    on_frames=_frame_progress(progress, text, sum(sizes[:number]), frames),
                    sound=sound,
                    length=sizes[number] / float(fit.rate),
                )
                files.append(part)
                if sound is not None:
                    sounds.append(sound)
            if progress is not None:
                progress("Joining and checking the montage", 1.0)
            if music:
                video = work_dir / "video.mp4"
                join_parts(tools, files, video, work_dir)
                if progress is not None:
                    progress("Adding the music", 1.0)
                # Muxed on the local temporary disk, then copied: the montage's own folder
                # may be a FAT32 memory card (the DVR's), where FFmpeg's MP4 muxer failed
                # with "Invalid argument" while muxing the soundtrack.
                with_music = work_dir / "montage.mp4"
                add_music(tools, video, music, with_music, frames / float(fit.rate))
                shutil.copyfile(with_music, temporary)
            else:
                join_parts(tools, files, temporary, work_dir, sounds)
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


def montage_fit(clips: Sequence[Clip], shots: Sequence[Shot], *, vertical: bool = False) -> Fit:
    """The largest frame size of the clips the shots use (even, square pixels), so no clip
    loses detail, at the frame rate of the clip that fills most of the montage.

    ``vertical``: a 9:16 frame as tall as the tallest clip, filled edge to edge with the
    middle of each picture (for phones), instead of the clips' own shape."""
    used = {shot.clip for shot in shots}
    clips = [clip for clip in clips if clip.index in used]
    share: dict[int, float] = {}
    for shot in shots:
        share[shot.clip] = share.get(shot.clip, 0.0) + shot.duration
    main = max(clips, key=lambda clip: (share.get(clip.index, 0.0), -clip.index))
    width, height = max((_display_size(clip.info) for clip in clips), key=lambda s: s[0] * s[1])
    if vertical:
        height = max(_display_size(clip.info)[1] for clip in clips)
        height -= height % 2
        return Fit(round(height * 9 / 32) * 2, height, output_rate(main.info), fill=True)
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
    measured = {
        clip.index: clip.cuts.signals.attitude.to_json()
        for clip in clips
        if clip.cuts is not None and clip.cuts.signals.attitude is not None
    }
    return {
        "fpvedit_version": __version__,
        "config": dataclasses.asdict(config),
        "output": {"width": fit.width, "height": fit.height, "fps": str(fit.rate)},
        "clips": [
            {
                "path": str(info.path),
                "duration": info.duration,
                "usable": index in usable,
                **({"turns": measured[index]} if index in measured else {}),
            }
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


def render_trim(
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
        with tempfile.TemporaryDirectory(prefix="fpvedit-", ignore_cleanup_errors=True) as work:
            work_dir = Path(work)
            picture, sound = work_dir / "edit.mp4", work_dir / "edit.wav"
            render(
                tools,
                info,
                segments,
                picture,
                encode=encode or config.encode,
                work_dir=work_dir,
                sound=sound,
                length=frames / float(rate),
            )
            join_parts(tools, [picture], temporary, work_dir, [sound])
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
