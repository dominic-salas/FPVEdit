"""How far each fast turn really turned, and about which of the quad's axes.

The flow rates (`motion.py`) under-count fast turns: blur, and a lens field of view that
is only assumed, leave a full flip measuring 150-380 degrees. Gravity settles it. In the
frames just before and just after a turn a single-image model (GeoCalib, optional) sees
which way is down. The turn about the flow's own axis that carries the first gravity onto
the second is known modulo a whole turn; the flow reading picks how many whole turns, the
fewest that reach `over_read` of it, since the flow rarely reads more than was flown. A
turn that comes back to where it started is a whole flip, roll or spin; one that does not
is a part turn (a half roll, a three-quarter flip), measured rather than rounded up.

Gravity cannot see a turn about the vertical (a flat yaw spin) and is unsure in dark or
broken-up frames; those turns keep the flow reading and are marked unchecked.

The camera is tilted up on the quad, so the camera sees the quad's roll axis tilted (a
roll shows some yaw). Rolls and yaw spins lie 90 degrees apart about the camera's x axis,
and the tilt is the angle that fits both (`estimate_uptilt`), in the flow's own degrees.

GeoCalib (Veicht et al., ECCV 2024): code Apache-2.0, weights CC-BY-4.0, downloaded on
first use and never shipped. It needs the ``ml`` extra. Reading runs in its own process.
"""

from __future__ import annotations

import importlib.util
import json
import math
import multiprocessing
import queue
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from fpvedit.errors import FPVEditError
from fpvedit.media import NO_WINDOW, MediaInfo, Tools
from fpvedit.osd import glyph_pixels
from fpvedit.seeking import file_seconds

VERSION = 1
GRAVITY_CACHE_NAME = "gravity.json"
WIDTH = 640  # frames are read at this width for the gravity model

FloatArray = NDArray[np.float64]
Vector = tuple[float, float, float]
Gravity = tuple[Vector, float]  # unit "up" in camera axes (x right, y down, z forward), deg
Progress = Callable[[str, float], None]


class AttitudeError(FPVEditError):
    """The gravity model could not be loaded or run."""


@dataclass(frozen=True)
class AttitudeConfig:
    fast_rate: float = 150.0  # deg/s over `fast_window_s`: inside a fast turn
    fast_window_s: float = 0.15
    min_turn_deg: float = 90.0  # a fast stretch turning less is not a turn
    min_turn_s: float = 0.05
    pads_s: tuple[float, ...] = (0.2, 0.35, 0.5)  # where to read gravity outside a turn
    max_uncertainty: float = 10.0  # deg: a gravity reading less sure is not used
    min_observable: float = 0.5  # sine of the angle between the turn's axis and gravity
    max_residual: float = 20.0  # deg: gravity misfit allowed after the best turn
    whole_tol_deg: float = 40.0  # from a whole number of turns: it came back round
    over_read: float = 0.85  # the flow reads at most about 1/0.85 of what was flown
    uptilt_prior: float = 15.0  # deg, in flow degrees, when a recording has too few turns
    uptilt_min_turns: int = 8
    uptilt_min_strength: float = 0.5
    uptilt_max_pitch: float = 0.35  # turns with more pitch than this say nothing of tilt


@dataclass(frozen=True)
class Turn:
    """One fast turn. ``axis`` is the camera's (pitch, yaw, roll) direction it turned."""

    start: float  # seconds
    end: float
    axis: Vector
    flow_deg: float  # what the flow measured
    degrees: float  # the best estimate: resolved by gravity, else the flow reading
    checked: bool = False  # gravity on both sides fixed `degrees`
    whole: int = 0  # whole turns when it came back round (checked only), else 0
    residual: float = math.nan  # deg, gravity misfit of `degrees`


@dataclass(frozen=True)
class Attitude:
    uptilt: float  # deg, in flow degrees
    turns: tuple[Turn, ...]

    def turn_at(self, seconds: float, reach: float = 0.5) -> Turn | None:
        """The turn at ``seconds``, or the nearest one within ``reach`` of it."""
        best, distance = None, reach
        for turn in self.turns:
            gap = max(turn.start - seconds, seconds - turn.end, 0.0)
            if gap <= distance:
                best, distance = turn, gap
        return best

    def body_axis(self, turn: Turn) -> Vector:
        """The turn's axis in the quad's own (pitch, yaw, roll), the up-tilt removed."""
        tilt = math.radians(self.uptilt)
        x, y, z = turn.axis
        roll = y * math.sin(tilt) + z * math.cos(tilt)
        yaw = y * math.cos(tilt) - z * math.sin(tilt)
        return (x, yaw, roll)

    def to_json(self) -> dict[str, Any]:
        return {
            "uptilt": round(self.uptilt, 1),
            "turns": [
                {
                    "start": round(t.start, 3),
                    "end": round(t.end, 3),
                    "axis": [round(v, 3) for v in t.axis],
                    "flow_deg": round(t.flow_deg, 1),
                    "degrees": round(t.degrees, 1),
                    "checked": t.checked,
                    "whole": t.whole,
                    "residual": None if math.isnan(t.residual) else round(t.residual, 1),
                }
                for t in self.turns
            ],
        }


def fast_turns(
    rotation: FloatArray, fps: float, config: AttitudeConfig | None = None
) -> list[tuple[int, int]]:
    """Frame spans [start, end) of the fast turns in ``rotation`` (deg/s: pitch, yaw, roll)."""
    config = config or AttitudeConfig()
    window = max(1, round(config.fast_window_s * fps))
    rate = np.linalg.norm(np.nan_to_num(rotation), axis=1)
    smooth = np.convolve(rate, np.ones(window) / window, mode="same")
    fast = np.concatenate([[False], smooth >= config.fast_rate, [False]])
    edges = np.flatnonzero(np.diff(fast.astype(np.int8)))
    spans = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        turned = float(np.linalg.norm(np.nan_to_num(rotation[start:end]).sum(axis=0))) / fps
        if end - start >= config.min_turn_s * fps and turned >= config.min_turn_deg:
            spans.append((int(start), int(end)))
    return spans


def anchor_times(
    spans: Sequence[tuple[int, int]], fps: float, config: AttitudeConfig | None = None
) -> list[float]:
    """The times (seconds) to read gravity at, before and after each turn."""
    config = config or AttitudeConfig()
    times = set()
    for start, end in spans:
        for pad in config.pads_s:
            times.add(round(start / fps - pad, 2))
            times.add(round(end / fps + pad, 2))
    return sorted(t for t in times if t >= 0)


def estimate_uptilt(axes: Sequence[Vector], config: AttitudeConfig | None = None) -> float:
    """The camera's up-tilt (deg) from turn axes: rolls sit at ``u`` and yaw spins at
    ``u + 90`` about the camera's x axis, so folding the axis angle by 90 degrees (four-fold
    circular mean) puts both at ``u``, whatever their direction."""
    config = config or AttitudeConfig()
    angles = [math.atan2(y, z) for x, y, z in axes if abs(x) < config.uptilt_max_pitch]
    if len(angles) < config.uptilt_min_turns:
        return config.uptilt_prior
    mean = complex(np.mean(np.exp(4j * np.array(angles))))
    if abs(mean) < config.uptilt_min_strength:
        return config.uptilt_prior
    return math.degrees(math.atan2(mean.imag, mean.real) / 4) % 90


def _turned(axis: FloatArray, degrees: float) -> FloatArray:
    """How a world direction, seen in camera axes, moves when the camera turns ``degrees``
    about ``axis`` (scene points move the opposite way, as in `motion.py`)."""
    matrix, _ = cv2.Rodrigues(-axis * math.radians(degrees))
    return np.asarray(matrix, dtype=np.float64)


def resolve(
    flow: FloatArray, before: Gravity, after: Gravity, config: AttitudeConfig | None = None
) -> tuple[float, float] | None:
    """The turn's angle (deg) about the axis of ``flow`` (deg: pitch, yaw, roll) that carries
    gravity ``before`` onto ``after``, and the misfit left (deg). None when gravity cannot
    tell: unsure readings, or a turn about the vertical."""
    config = config or AttitudeConfig()
    total = float(np.linalg.norm(flow))
    if total <= 0 or max(before[1], after[1]) > config.max_uncertainty:
        return None
    axis = np.asarray(flow, dtype=np.float64) / total
    g0, g1 = np.asarray(before[0], dtype=np.float64), np.asarray(after[0], dtype=np.float64)
    p0, p1 = g0 - (g0 @ axis) * axis, g1 - (g1 @ axis) * axis
    if min(np.linalg.norm(p0), np.linalg.norm(p1)) < config.min_observable:
        return None
    # The scene turns the other way: the camera turned by minus the angle from p0 to p1.
    base = math.degrees(-math.atan2(float(axis @ np.cross(p0, p1)), float(p0 @ p1))) % 360
    count = max(0, math.ceil((config.over_read * total - base) / 360))
    degrees = base + 360 * count
    misfit = math.degrees(math.acos(float(np.clip(_turned(axis, degrees) @ g0 @ g1, -1, 1))))
    return degrees, misfit


def measure(
    rotation: FloatArray,
    fps: float,
    gravity: Mapping[float, Gravity | None],
    config: AttitudeConfig | None = None,
) -> Attitude:
    """Every fast turn of a clip, resolved by the ``gravity`` read at `anchor_times`."""
    config = config or AttitudeConfig()
    rotation = np.nan_to_num(rotation)
    spans = fast_turns(rotation, fps, config)
    turns = []
    for start, end in spans:
        first = _best(gravity, [start / fps - pad for pad in config.pads_s])
        last = _best(gravity, [end / fps + pad for pad in config.pads_s])
        flow = rotation[start:end].sum(axis=0) / fps
        if first is not None and last is not None:
            # From one reading to the other: the flow just outside the turn counts too.
            low, high = max(0, round(first[1] * fps)), round(last[1] * fps)
            flow = rotation[low:high].sum(axis=0) / fps
        total = float(np.linalg.norm(flow))
        axis = tuple(float(v) for v in flow / max(total, 1e-9))
        turn = Turn(start / fps, end / fps, (axis[0], axis[1], axis[2]), total, total)
        found = None if first is None or last is None else resolve(flow, first[0], last[0], config)
        if found is not None and found[1] <= config.max_residual:
            degrees, misfit = found
            count = round(degrees / 360)
            whole = (
                count if count >= 1 and abs(degrees - 360 * count) <= config.whole_tol_deg else 0
            )
            turn = Turn(turn.start, turn.end, turn.axis, total, degrees, True, whole, misfit)
        turns.append(turn)
    uptilt = estimate_uptilt([t.axis for t in turns], config)
    return Attitude(uptilt, tuple(turns))


def _best(
    gravity: Mapping[float, Gravity | None], times: Sequence[float]
) -> tuple[Gravity, float] | None:
    """The surest gravity reading among ``times``, with its time."""
    found = [(g, t) for t in times if (g := gravity.get(round(t, 2))) is not None]
    return min(found, key=lambda item: item[0][1]) if found else None


# Reading gravity (the ml extra) -------------------------------------------------------


def available() -> bool:
    """Whether the ``ml`` extra with the gravity model is installed."""
    names = ("torch", "kornia", "torchvision", "geocalib")
    return all(importlib.util.find_spec(name) is not None for name in names)


def grab_picture(tools: Tools, info: MediaInfo, seconds: float) -> NDArray[np.uint8] | None:
    """The frame at ``seconds`` (analysis timeline), RGB, `WIDTH` wide, the black bars
    beside a 4:3 picture cut off and the OSD painted out. None if unreadable."""
    args = [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error"),
        *("-ss", f"{file_seconds(info, seconds):.3f}", "-i", str(info.path)),
        *("-map", f"0:{info.video.index}", "-frames:v", "1", "-vf", f"scale={WIDTH}:-2"),
        *("-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"),
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False, creationflags=NO_WINDOW)
    except OSError:
        return None
    height = len(result.stdout) // (WIDTH * 3)
    if result.returncode != 0 or height < 16 or len(result.stdout) != height * WIDTH * 3:
        return None
    frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, WIDTH, 3)
    columns = np.flatnonzero(frame.mean(axis=(0, 2)) > 12)
    if len(columns) < WIDTH // 4:
        return None  # black or nearly
    frame = np.ascontiguousarray(frame[:, columns[0] : columns[-1] + 1])
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    osd = glyph_pixels(np.asarray(gray, dtype=np.uint8))
    mask = cv2.dilate(osd.astype(np.uint8), np.ones((7, 7), np.uint8))
    return np.asarray(cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA), dtype=np.uint8)


class GravityModel:
    """GeoCalib, "distorted" weights (FPV lenses), with a radial-distortion camera."""

    def __init__(self, device: str = "auto") -> None:
        try:
            import torch
            from geocalib import GeoCalib
        except ImportError as exc:
            raise AttitudeError("the gravity model needs the ml extra: uv sync --extra ml") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = GeoCalib(weights="distorted").to(device)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AttitudeError(f"could not load the gravity model: {exc}") from exc
        self._torch, self._device = torch, device

    def __call__(self, frame: NDArray[np.uint8]) -> Gravity:
        torch = self._torch
        image = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(self._device)
        with torch.inference_mode():
            out = self._model.calibrate(image, camera_model="simple_radial")
        up = out["gravity"].vec3d[0].detach().cpu().numpy().astype(np.float64)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        unsure = math.degrees(float(out["gravity_uncertainty"].reshape(-1)[0]))
        return (float(up[0]), float(up[1]), float(up[2])), unsure


def _key(info: MediaInfo, seconds: float) -> str:
    stat = info.path.stat()
    return f"{info.path}|{stat.st_size}|{stat.st_mtime_ns}|{seconds:.2f}|v{VERSION}"


def read_gravity(
    tools: Tools,
    wanted: Sequence[tuple[MediaInfo, Sequence[float]]],
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
    progress: Progress | None = None,
) -> list[dict[float, Gravity | None]]:
    """Gravity at each recording's times (None where a frame could not be read). Readings
    are cached in ``cache_dir`` (`GRAVITY_CACHE_NAME`); the model loads only if needed."""
    cache_file = cache_dir / GRAVITY_CACHE_NAME if cache_dir is not None else None
    cache = _load(cache_file)
    todo = [(info, t) for info, times in wanted for t in times if _key(info, t) not in cache]
    if todo:
        model = GravityModel(device)
        for done, (info, seconds) in enumerate(todo):
            if progress is not None and done % 8 == 0:
                progress("Measuring the tricks' turns", done / len(todo))
            if done and done % 50 == 0:
                _save(cache_file, cache)  # a cancelled run keeps what it read
            frame = grab_picture(tools, info, seconds)
            if frame is None:
                cache[_key(info, seconds)] = None
                continue
            vector, unsure = model(frame)
            cache[_key(info, seconds)] = [*vector, unsure]
        _save(cache_file, cache)
    result = []
    for info, times in wanted:
        found: dict[float, Gravity | None] = {}
        for seconds in times:
            value = cache.get(_key(info, seconds))
            found[round(seconds, 2)] = (
                ((float(value[0]), float(value[1]), float(value[2])), float(value[3]))
                if isinstance(value, list) and len(value) == 4
                else None
            )
        result.append(found)
    return result


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}  # damaged: read again
    return data if isinstance(data, dict) else {}


def _save(path: Path | None, cache: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass  # an unwritable cache only costs reading again


def _read_in_child(
    tools: Tools,
    wanted: Sequence[tuple[MediaInfo, Sequence[float]]],
    cache_dir: Path | None,
    device: str,
    reports: Any = None,
) -> list[dict[float, Gravity | None]]:
    progress = None if reports is None else lambda text, share: reports.put((text, share))
    return read_gravity(tools, wanted, cache_dir=cache_dir, device=device, progress=progress)


def read_gravity_in_process(
    tools: Tools,
    wanted: Sequence[tuple[MediaInfo, Sequence[float]]],
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
    progress: Progress | None = None,
) -> list[dict[float, Gravity | None]]:
    """`read_gravity` in a child process, so its GPU memory is released when it ends."""
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager, ProcessPoolExecutor(1, mp_context=context) as pool:
        reports = manager.Queue() if progress is not None else None
        future = pool.submit(_read_in_child, tools, list(wanted), cache_dir, device, reports)
        while True:
            try:
                result: list[dict[float, Gravity | None]] = future.result(timeout=0.1)
            except FuturesTimeout:
                result = []
                pending = True
            else:
                pending = False
            while reports is not None and progress is not None:
                try:
                    text, share = reports.get_nowait()
                except (queue.Empty, EOFError, OSError):
                    break
                progress(str(text), float(share))
            if not pending:
                return result
