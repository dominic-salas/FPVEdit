"""What a shot shows: indoors or outdoors, from an image model (SigLIP 2).

Motion alone cannot tell a lame indoor trick from weaving through trees outside: on the
test recordings both have the same speed, rotation and close-pass rate. SigLIP 2
(Apache-2.0, `MODEL_ID` at `MODEL_REVISION`) compares three frames of each shot with two
sets of short descriptions, rooms and halls against trees, streets and parks. The
*indoorness* of a frame is its best indoor similarity minus its best outdoor one; a
shot's is the median of its frames. It is a relative signal for comparing shots of one
session, not a probability, and the model never names tricks.

In the same pass each frame is also compared with the `PICTURE` descriptions (a person,
legs, a narrow gap, trees, a car close up...); a shot's *picture* is the mean over its
frames. The learned taste (`taste.py`) uses it to follow what a pilot likes to see.

Optional: needs the ``ml`` extra (PyTorch, transformers, Pillow) and downloads about
1.5 GB of weights on first use. Scoring runs in its own process, so the GPU memory it
used (under 1 GB) is free again as soon as it is done.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpv_editor.errors import FPVEditError
from fpv_editor.media import NO_WINDOW, MediaInfo, Tools
from fpv_editor.montage import Shot
from fpv_editor.seeking import file_seconds

MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
INSIDE = (
    "a photo of a room inside a house",
    "an indoor hallway",
    "inside a garage",
    "inside a warehouse",
    "furniture in a living room",
)
OUTSIDE = (
    "a photo outdoors",
    "trees and sky",
    "buildings and a street outside",
    "a parking lot outside",
    "a park with grass and trees",
)
# What a shot shows, for learning a pilot's taste. Chosen before testing them against
# Dom's ratings; changing them renews the picture cache and needs re-checking (PLAN.md).
PICTURE = (
    "a person standing",
    "a person's legs close up",
    "flying through a narrow gap",
    "flying between trees",
    "a car close up",
    "a parking lot at night",
    "an empty open field",
    "a building close up",
    "blurry motion",
    "the ground very close",
    "the sky",
    "a dark night scene",
    "a bright sunny day",
    "indoors",
    "a bridge",
    "water",
)
FRAME_SHARES = (0.25, 0.5, 0.75)  # where in a shot its frames are taken
SIZE = 224  # the model's input size; frames are scaled to it (as the model was tested)
BATCH = 16  # small batches keep the GPU memory low; it is shared with other work
CACHE_NAME = "scenery.json"
PICTURE_CACHE_NAME = "pictures.json"

Frame = NDArray[np.uint8]  # height x width x 3, RGB
Progress = Callable[[str, float], None]


class SceneryError(FPVEditError):
    """The image model could not be loaded or run."""


def available() -> bool:
    """Whether the ``ml`` extra is installed."""
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def grab_rgb(tools: Tools, info: MediaInfo, seconds: float) -> Frame | None:
    """The frame at ``seconds`` (analysis timeline), RGB at SIZE x SIZE; None if unreadable."""
    args = [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error"),
        *("-ss", f"{file_seconds(info, seconds):.3f}", "-i", str(info.path)),
        *("-map", f"0:{info.video.index}", "-frames:v", "1", "-vf", f"scale={SIZE}:{SIZE}"),
        *("-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"),
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False, creationflags=NO_WINDOW)
    except OSError:
        return None
    if result.returncode != 0 or len(result.stdout) != SIZE * SIZE * 3:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(SIZE, SIZE, 3)


class SceneryModel:
    """SigLIP 2 with the two sets of descriptions embedded once."""

    def __init__(self, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise SceneryError("the image model needs the ml extra: uv sync --extra ml") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            self._model = (
                AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
                .to(device)
                .eval()
            )
            # Untyped in transformers, missing without the ml extra: Any either way, so
            # the type check passes with and without it installed.
            processor: Any = AutoProcessor
            self._processor = processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        except (OSError, ValueError) as exc:
            raise SceneryError(f"could not load {MODEL_ID}: {exc}") from exc
        self._torch, self._device, self._dtype = torch, device, dtype
        with torch.inference_mode():
            text = self._processor(
                text=[*INSIDE, *OUTSIDE, *PICTURE],
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(device)
            self._text = self._normalised(self._model.get_text_features(**text))

    def _normalised(self, output: Any) -> Any:
        tensor = output.pooler_output if hasattr(output, "pooler_output") else output
        return self._torch.nn.functional.normalize(tensor.float(), dim=-1)

    def describe(self, frames: Sequence[Frame]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per frame: indoorness (best indoor minus best outdoor similarity), and the
        similarity to each `PICTURE` description (frames x descriptions)."""
        scores: list[NDArray[np.float64]] = []
        pictures: list[NDArray[np.float64]] = []
        split = len(INSIDE) + len(OUTSIDE)
        with self._torch.inference_mode():
            for first in range(0, len(frames), BATCH):
                batch = self._processor(
                    images=list(frames[first : first + BATCH]), return_tensors="pt"
                )
                pixels = batch["pixel_values"].to(self._device, self._dtype)
                images = self._normalised(self._model.get_image_features(pixel_values=pixels))
                similarity = (images @ self._text.T).cpu().numpy().astype(np.float64)
                inside = similarity[:, : len(INSIDE)].max(axis=1)
                outside = similarity[:, len(INSIDE) : split].max(axis=1)
                scores.append(inside - outside)
                pictures.append(similarity[:, split:])
        if not scores:
            return np.zeros(0), np.zeros((0, len(PICTURE)))
        return np.concatenate(scores), np.concatenate(pictures)

    def indoorness(self, frames: Sequence[Frame]) -> NDArray[np.float64]:
        """Per frame: best indoor similarity minus best outdoor similarity."""
        return self.describe(frames)[0]


def _prompts_key() -> str:
    text = json.dumps([MODEL_ID, MODEL_REVISION, SIZE, INSIDE, OUTSIDE])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _picture_key() -> str:
    text = json.dumps([MODEL_ID, MODEL_REVISION, SIZE, PICTURE])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _frame_key(info: MediaInfo, seconds: float) -> str:
    stat = info.path.stat()
    return f"{info.path}|{stat.st_size}|{stat.st_mtime_ns}|{seconds:.2f}|{_prompts_key()}"


def _picture_frame_key(info: MediaInfo, seconds: float) -> str:
    stat = info.path.stat()
    return f"{info.path}|{stat.st_size}|{stat.st_mtime_ns}|{seconds:.2f}|{_picture_key()}"


@dataclass(frozen=True)
class ShotScene:
    """What the image model saw in one shot (None where no frame could be read)."""

    indoorness: float | None
    picture: tuple[float, ...] | None


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path: Path | None, data: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass  # only a cache


def frame_times(shot: Shot) -> list[float]:
    return [round(shot.start + share * shot.duration, 2) for share in FRAME_SHARES]


def score_shots(
    tools: Tools,
    infos: Sequence[MediaInfo],
    shots: Sequence[Shot],
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
    progress: Progress | None = None,
) -> list[ShotScene]:
    """What each shot shows: its indoorness and its picture descriptions.

    Frame scores are cached in ``cache_dir`` per recording version, time and model, so
    picking again, or the same shots in a later session, needs no model at all.
    """
    cache_file = cache_dir / CACHE_NAME if cache_dir is not None else None
    picture_file = cache_dir / PICTURE_CACHE_NAME if cache_dir is not None else None
    cache: dict[str, float] = {}
    for k, v in _load(cache_file).items():
        try:
            cache[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    seen: dict[str, list[float]] = {}
    for k, v in _load(picture_file).items():
        if isinstance(v, list) and len(v) == len(PICTURE):
            seen[str(k)] = [float(x) for x in v]
    wanted = [(shot.clip, t) for shot in shots for t in frame_times(shot)]
    keys = {(clip, t): _frame_key(infos[clip], t) for clip, t in wanted}
    picture_keys = {(clip, t): _picture_frame_key(infos[clip], t) for clip, t in wanted}
    missing = sorted(
        {item for item in wanted if keys[item] not in cache or picture_keys[item] not in seen}
    )
    if missing:
        if progress is not None:
            progress("Reading frames of the shots", 0.0)
        with ThreadPoolExecutor(4) as pool:
            frames = list(pool.map(lambda item: grab_rgb(tools, infos[item[0]], item[1]), missing))
        readable = [
            (item, frame) for item, frame in zip(missing, frames, strict=True) if frame is not None
        ]
        if readable:
            if progress is not None:
                progress("Comparing the scenery of the shots", 0.5)
            model = SceneryModel(device)
            values, pictures = model.describe([frame for _, frame in readable])
            for (item, _), value, picture in zip(readable, values, pictures, strict=True):
                cache[keys[item]] = round(float(value), 5)
                seen[picture_keys[item]] = [round(float(x), 5) for x in picture]
            _save(cache_file, cache)
            _save(picture_file, seen)
    result: list[ShotScene] = []
    for shot in shots:
        times = frame_times(shot)
        scores = [cache[keys[(shot.clip, t)]] for t in times if keys[(shot.clip, t)] in cache]
        views = [seen[k] for t in times if (k := picture_keys[(shot.clip, t)]) in seen]
        result.append(
            ShotScene(
                float(np.median(scores)) if scores else None,
                tuple(round(float(x), 5) for x in np.mean(views, axis=0)) if views else None,
            )
        )
    return result


def _score_in_child(
    tools: Tools,
    infos: Sequence[MediaInfo],
    shots: Sequence[Shot],
    cache_dir: Path | None,
    device: str,
) -> list[ShotScene]:
    return score_shots(tools, infos, shots, cache_dir=cache_dir, device=device)


def score_shots_in_process(
    tools: Tools,
    infos: Sequence[MediaInfo],
    shots: Sequence[Shot],
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
) -> list[ShotScene]:
    """`score_shots` in a child process: its GPU memory is released when it ends."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=context) as pool:
        future = pool.submit(_score_in_child, tools, list(infos), list(shots), cache_dir, device)
        return future.result()
