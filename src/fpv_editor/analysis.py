"""Analysis without rendering: picture area, OSD layout, per-frame measurements, flight
state and the choice of what to keep.

`analyze` is what the CLI runs before rendering, and what tuning scripts run on their
own. Measuring every frame is the slow part, so the measurements can be cached: with a
cache directory, a second analysis of an unchanged recording with unchanged measuring
code and settings reads them back instead of decoding the video again. Flight-state
detection and selection always run, so their settings can be tuned against the cache.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
import zlib
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpv_editor import __version__
from fpv_editor.flight import FlightConfig, FlightState, detect_flight
from fpv_editor.highlights import Selection, SelectionConfig, select_highlights
from fpv_editor.measure import (
    DecodeReport,
    GrayFrame,
    MeasureConfig,
    Measurements,
    analysis_frame_size,
    decode_frames,
    measure_frames,
    plan_layout,
)
from fpv_editor.media import MediaInfo, Tools
from fpv_editor.motion import SPEED
from fpv_editor.picture import Area, PictureConfig, decoded_size, detect_active_area, sample_frames
from fpv_editor.render import output_rate

CACHE_FORMAT = 3
# Code whose changes alter cached measurements; its bytes are part of the cache key, so
# tuning these modules never silently reuses measurements made by older code.
MEASUREMENT_MODULES = ("analysis.py", "measure.py", "motion.py", "picture.py", "media.py")
# Per-frame signals in the analysis JSON are summarised to this rate to keep it readable.
DOCUMENT_RATE = 10
_ARRAYS = ("brightness", "contrast", "noise", "change", "motion", "layout", "text", "thumbnails")


@dataclass(frozen=True)
class AnalysisConfig:
    picture: PictureConfig = field(default_factory=PictureConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass(frozen=True)
class Analysis:
    info: MediaInfo
    area: Area
    overlay_fraction: float  # share of the analysed picture ignored as OSD/overlay
    measurements: Measurements
    flight: FlightState
    selection: Selection
    analysed_duration: float
    from_cache: bool
    warnings: tuple[str, ...]


def analyze(
    tools: Tools,
    info: MediaInfo,
    config: AnalysisConfig | None = None,
    *,
    cache_dir: Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> Analysis:
    """Measure (or read from the cache), find the flight state and choose what to keep.

    ``progress`` is called with the share of the recording measured so far (0-1); if it
    raises, measuring stops and nothing is cached.
    """
    config = config or AnalysisConfig()
    cache_file = _cache_path(cache_dir, tools, info, config) if cache_dir is not None else None
    cached = _read_cache(cache_file) if cache_file is not None else None
    notes: tuple[str, ...] = ()
    if cached is not None:
        area, overlay_fraction, measurements, warnings = cached
        from_cache = True
    else:
        area, overlay_fraction, measurements, warnings = _measure(tools, info, config, progress)
        from_cache = False
        if cache_file is not None:
            try:
                _write_cache(cache_file, area, overlay_fraction, measurements, warnings)
            except OSError as exc:
                notes = (f"could not write the measurement cache ({exc}); continuing without it",)

    flight = detect_flight(measurements, config.flight)
    selection = select_highlights(measurements, flight, config.selection)
    return Analysis(
        info=info,
        area=area,
        overlay_fraction=overlay_fraction,
        measurements=measurements,
        flight=flight,
        selection=selection,
        analysed_duration=len(measurements) / measurements.fps,
        from_cache=from_cache,
        warnings=(*warnings, *notes, *selection.warnings),
    )


def _measure(
    tools: Tools,
    info: MediaInfo,
    config: AnalysisConfig,
    progress: Callable[[float], None] | None = None,
) -> tuple[Area, float, Measurements, tuple[str, ...]]:
    samples = sample_frames(tools, info, count=config.measure.layout_samples)
    area = detect_active_area(samples, decoded_size(info), config.picture)
    size = analysis_frame_size(info, area, config.measure.height)
    layout = plan_layout(samples, area, size, config.measure, config.picture)
    fps = float(output_rate(info))
    report = DecodeReport()
    with closing(decode_frames(tools, info, area, size, fps, report)) as decoded:
        frames: Iterator[GrayFrame] = decoded
        if progress is not None:
            frames = _reporting(decoded, round(fps * (info.duration or 0.0)), progress)
        measurements = measure_frames(frames, fps, layout, config.measure)
    overlay_fraction = float(layout.overlay.mean()) if layout.overlay is not None else 0.0
    return area, overlay_fraction, measurements, tuple(report.warnings)


def _reporting(
    frames: Iterator[GrayFrame], expected: int, progress: Callable[[float], None]
) -> Iterator[GrayFrame]:
    """Pass ``frames`` through, reporting progress every 30 frames."""
    for index, frame in enumerate(frames):
        if index % 30 == 0:
            progress(min(1.0, index / expected) if expected > 0 else 0.0)
        yield frame


def _code_fingerprint() -> str:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for name in MEASUREMENT_MODULES:
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def _cache_path(cache_dir: Path, tools: Tools, info: MediaInfo, config: AnalysisConfig) -> Path:
    """One file per recording version, measuring code, FFmpeg version and settings."""
    stat = info.path.stat()
    key = json.dumps(
        {
            "format": CACHE_FORMAT,
            "version": __version__,
            "code": _code_fingerprint(),
            "ffmpeg_major": tools.ffmpeg_major,
            "source": str(info.path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "picture": asdict(config.picture),
            "measure": asdict(config.measure),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{info.path.stem}.{digest}.measurements.npz"


def _read_cache(path: Path) -> tuple[Area, float, Measurements, tuple[str, ...]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta: dict[str, Any] = json.loads(str(data["meta"]))
            arrays = {name: data[name] for name in _ARRAYS}
        measurements = Measurements(
            fps=float(meta["fps"]), thumbnail_every=int(meta["thumbnail_every"]), **arrays
        )
        return (
            Area(*(int(v) for v in meta["area"])),
            float(meta["overlay_fraction"]),
            measurements,
            tuple(str(w) for w in meta["warnings"]),
        )
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error):
        return None  # damaged, truncated or from an older layout: measure again


def _write_cache(
    path: Path,
    area: Area,
    overlay_fraction: float,
    measurements: Measurements,
    warnings: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "fps": measurements.fps,
        "thumbnail_every": measurements.thumbnail_every,
        "area": [area.x, area.y, area.width, area.height],
        "overlay_fraction": overlay_fraction,
        "warnings": list(warnings),
    }
    temporary = path.with_name(f"{path.name}.partial.npz")
    arrays = {name: getattr(measurements, name) for name in _ARRAYS}
    np.savez_compressed(temporary, meta=np.array(json.dumps(meta)), **arrays)
    temporary.replace(path)


def _summary(values: NDArray[np.floating], step: int, how: str = "mean") -> list[float]:
    """Mean (or max) over every ``step`` frames, NaN as 0, rounded for the JSON file."""
    usable = len(values) // step * step
    if usable == 0:
        return []
    blocks = np.nan_to_num(np.asarray(values[:usable], dtype=np.float64)).reshape(-1, step)
    result = blocks.max(axis=1) if how == "max" else blocks.mean(axis=1)
    return [round(float(v), 4) for v in result]


def analysis_document(analysis: Analysis, tools: Tools, config: AnalysisConfig) -> dict[str, Any]:
    """JSON-ready description of an analysis (written by --keep-intermediates)."""
    m = analysis.measurements
    selection = analysis.selection
    area = analysis.area
    step = max(1, round(m.fps / DOCUMENT_RATE))
    rotation_rate = np.linalg.norm(np.nan_to_num(m.motion[:, :3]), axis=1)
    return {
        "fpv_editor_version": __version__,
        "ffmpeg_major": tools.ffmpeg_major,
        "source": str(analysis.info.path),
        "config": asdict(config),
        "picture": {
            "active_area": [area.x, area.y, area.width, area.height],
            "overlay_fraction": round(analysis.overlay_fraction, 4),
        },
        "fps": m.fps,
        "flight": {
            "has_osd": analysis.flight.has_osd,
            "armed": [[round(a, 3), round(b, 3)] for a, b in analysis.flight.armed],
            "screens": [
                {"kind": e.kind.value, "start": round(e.start, 3), "end": round(e.end, 3)}
                for e in analysis.flight.events
            ],
        },
        "signals": {
            "rate_hz": m.fps / step,
            "rotation_deg_s_max": _summary(rotation_rate, step, "max"),
            "image_speed": _summary(m.motion[:, SPEED], step),
            "osd_layout": _summary(m.layout, step),
            "noise": _summary(m.noise, step),
            "brightness": _summary(m.brightness, step),
        },
        "selection": {
            "mode": selection.mode.value,
            "retention": selection.retention,
            "intervals": [[i.start, i.end] for i in selection.intervals],
            "cuts": [
                {"start": round(c.start, 3), "end": round(c.end, 3), "reason": c.reason.value}
                for c in selection.cuts
            ],
            "warnings": list(selection.warnings),
        },
    }
