"""Analysis without rendering: picture area, OSD mask, features, score and selection.

`analyze` is what the CLI runs before rendering, and what tuning scripts run on their
own. Decoding every frame is the slow part, so the per-frame features can be cached:
with a cache directory, a second analysis of an unchanged recording with unchanged
analysis settings reads the features back instead of decoding the video again.
Scoring and selection always run, so their settings can be tuned against the cache.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
import zlib
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpv_editor import __version__
from fpv_editor.features import FeatureConfig, FeatureSeries, compute_features
from fpv_editor.highlights import Selection, SelectionConfig, select_highlights
from fpv_editor.media import MediaInfo, Tools
from fpv_editor.picture import (
    Area,
    PictureConfig,
    decoded_size,
    detect_active_area,
    detect_overlay,
    fit_frames,
    sample_frames,
)
from fpv_editor.sampling import (
    SamplingConfig,
    SamplingReport,
    analysis_size,
    iter_gray_frames,
    sample_duration,
)
from fpv_editor.scoring import ScoringConfig, idle_mask, interest_score

FloatArray = NDArray[np.float64]
CACHE_FORMAT = 2
# Code whose changes alter cached measurements; its bytes are part of the cache key, so
# tuning these modules never silently reuses features measured by older code.
MEASUREMENT_MODULES = ("analysis.py", "features.py", "picture.py", "sampling.py", "media.py")


@dataclass(frozen=True)
class AnalysisConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    picture: PictureConfig = field(default_factory=PictureConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass(frozen=True)
class Analysis:
    info: MediaInfo
    area: Area
    overlay_fraction: float  # share of the analysed picture ignored as OSD/overlay
    features: FeatureSeries
    score: FloatArray
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
) -> Analysis:
    config = config or AnalysisConfig()
    cache_file = _cache_path(cache_dir, tools, info, config) if cache_dir is not None else None
    cached = _read_cache(cache_file) if cache_file is not None else None
    notes: tuple[str, ...] = ()
    if cached is not None:
        area, overlay_fraction, features, frame_count, warnings = cached
        from_cache = True
    else:
        area, overlay_fraction, features, frame_count, warnings = _measure(tools, info, config)
        from_cache = False
        if cache_file is not None:
            try:
                _write_cache(cache_file, area, overlay_fraction, features, frame_count, warnings)
            except OSError as exc:
                notes = (f"could not write the measurement cache ({exc}); continuing without it",)

    score = interest_score(features, config.scoring)
    selection = select_highlights(
        score, idle_mask(features, config.scoring), config.sampling.fps, config.selection
    )
    report = SamplingReport(frames=frame_count)
    return Analysis(
        info=info,
        area=area,
        overlay_fraction=overlay_fraction,
        features=features,
        score=score,
        selection=selection,
        analysed_duration=sample_duration(report, config.sampling),
        from_cache=from_cache,
        warnings=(*warnings, *notes, *selection.warnings),
    )


def _measure(
    tools: Tools, info: MediaInfo, config: AnalysisConfig
) -> tuple[Area, float, FeatureSeries, int, tuple[str, ...]]:
    samples = sample_frames(tools, info)
    area = detect_active_area(samples, decoded_size(info), config.picture)
    size = analysis_size(info, config.sampling.width, area)
    overlay = detect_overlay(fit_frames(samples, area, size), config.picture) if samples else None
    report = SamplingReport()
    with closing(iter_gray_frames(tools, info, config.sampling, report, area)) as frames:
        features = compute_features(frames, config.sampling.fps, config.features, ignore=overlay)
    overlay_fraction = float(overlay.mean()) if overlay is not None else 0.0
    return area, overlay_fraction, features, report.frames, tuple(report.warnings)


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
            "sampling": asdict(config.sampling),
            "picture": asdict(config.picture),
            "features": asdict(config.features),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{info.path.stem}.{digest}.features.npz"


def _read_cache(
    path: Path,
) -> tuple[Area, float, FeatureSeries, int, tuple[str, ...]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta: dict[str, Any] = json.loads(str(data["meta"]))
            features = FeatureSeries(
                fps=float(meta["fps"]),
                motion=data["motion"],
                change=data["change"],
                novelty=data["novelty"],
                brightness=data["brightness"],
                contrast=data["contrast"],
            )
            return (
                Area(*(int(v) for v in meta["area"])),
                float(meta["overlay_fraction"]),
                features,
                int(meta["frames"]),
                tuple(str(w) for w in meta["warnings"]),
            )
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error):
        return None  # damaged, truncated or from an older layout: measure again


def _write_cache(
    path: Path,
    area: Area,
    overlay_fraction: float,
    features: FeatureSeries,
    frame_count: int,
    warnings: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "fps": features.fps,
        "area": [area.x, area.y, area.width, area.height],
        "overlay_fraction": overlay_fraction,
        "frames": frame_count,
        "warnings": list(warnings),
    }
    temporary = path.with_name(f"{path.name}.partial.npz")
    np.savez_compressed(
        temporary,
        meta=np.array(json.dumps(meta)),
        motion=features.motion,
        change=features.change,
        novelty=features.novelty,
        brightness=features.brightness,
        contrast=features.contrast,
    )
    temporary.replace(path)


def analysis_document(analysis: Analysis, tools: Tools, config: AnalysisConfig) -> dict[str, Any]:
    """JSON-ready description of an analysis (written by --keep-intermediates)."""
    features = analysis.features
    selection = analysis.selection
    area = analysis.area
    return {
        "fpv_editor_version": __version__,
        "ffmpeg_major": tools.ffmpeg_major,
        "source": str(analysis.info.path),
        "config": asdict(config),
        "picture": {
            "active_area": [area.x, area.y, area.width, area.height],
            "overlay_fraction": round(analysis.overlay_fraction, 4),
        },
        "fps": features.fps,
        "samples": {
            "motion": features.motion.round(5).tolist(),
            "change": features.change.round(5).tolist(),
            "novelty": features.novelty.round(5).tolist(),
            "brightness": features.brightness.round(4).tolist(),
            "score": analysis.score.round(5).tolist(),
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
