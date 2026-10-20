"""What a shot shows: indoors or outdoors, from an image model (SigLIP 2).

Motion alone cannot tell a lame indoor trick from weaving through trees outside: both
can have the same speed, rotation and close-pass rate. SigLIP 2
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
import math
import multiprocessing
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fpvedit.errors import FPVEditError
from fpvedit.media import NO_WINDOW, MediaInfo, Tools
from fpvedit.montage import Shot
from fpvedit.seeking import file_seconds

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
# What a shot shows, for learning a pilot's taste. Changing them renews the picture cache.
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

    def describe(
        self, frames: Sequence[Frame], done: Callable[[int], None] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per frame: indoorness (best indoor minus best outdoor similarity), and the
        similarity to each `PICTURE` description (frames x descriptions). ``done`` is told
        how many frames are compared after each batch."""
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
                if done is not None:
                    done(min(len(frames), first + BATCH))
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


def sample_times(shot: Shot) -> list[float]:
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
    wanted = [(shot.clip, t) for shot in shots for t in sample_times(shot)]
    keys = {(clip, t): _frame_key(infos[clip], t) for clip, t in wanted}
    picture_keys = {(clip, t): _picture_frame_key(infos[clip], t) for clip, t in wanted}
    missing = sorted(
        {item for item in wanted if keys[item] not in cache or picture_keys[item] not in seen}
    )

    def report(text: str, share: float) -> None:
        if progress is not None:
            progress(text, share)

    if missing:
        # Shares of the whole: reading frames 0-30 %, loading the model 30-60 % (it gives
        # no progress of its own, so the bar creeps on while it loads, often 5-25 s),
        # comparing the frames 60-100 %.
        report(f"Reading {len(missing)} frames of the shots", 0.0)
        with ThreadPoolExecutor(4) as pool:
            futures = {
                pool.submit(grab_rgb, tools, infos[item[0]], item[1]): item for item in missing
            }
            grabbed: dict[tuple[int, float], Frame | None] = {}
            for done, future in enumerate(as_completed(futures), start=1):
                grabbed[futures[future]] = future.result()
                if done % 4 == 0 or done == len(missing):
                    report(
                        f"Reading frames of the shots ({done} of {len(missing)})",
                        0.3 * done / len(missing),
                    )
        readable = [(item, frame) for item in missing if (frame := grabbed[item]) is not None]
        if readable:
            loaded = threading.Event()

            def creep() -> None:
                started = time.monotonic()
                while not loaded.wait(0.5):
                    waited = time.monotonic() - started
                    report("Loading the image model", 0.3 + 0.29 * (1 - math.exp(-waited / 10)))

            report("Loading the image model", 0.3)
            ticker = threading.Thread(target=creep, daemon=True)
            ticker.start()
            try:
                model = SceneryModel(device)
            finally:
                loaded.set()
                ticker.join()
            report(f"Comparing {len(readable)} frames", 0.6)
            values, pictures = model.describe(
                [frame for _, frame in readable],
                lambda done: report(
                    f"Comparing frames ({done} of {len(readable)})",
                    0.6 + 0.4 * done / len(readable),
                ),
            )
            for (item, _), value, picture in zip(readable, values, pictures, strict=True):
                cache[keys[item]] = round(float(value), 5)
                seen[picture_keys[item]] = [round(float(x), 5) for x in picture]
            _save(cache_file, cache)
            _save(picture_file, seen)
    result: list[ShotScene] = []
    for shot in shots:
        times = sample_times(shot)
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
    reports: Any = None,
) -> list[ShotScene]:
    progress = None if reports is None else lambda text, share: reports.put((text, share))
    return score_shots(tools, infos, shots, cache_dir=cache_dir, device=device, progress=progress)


def score_shots_in_process(
    tools: Tools,
    infos: Sequence[MediaInfo],
    shots: Sequence[Shot],
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
    progress: Progress | None = None,
) -> list[ShotScene]:
    """`score_shots` in a child process: its GPU memory is released when it ends. Its
    progress reports come back through a queue and are passed to ``progress`` here."""
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager, ProcessPoolExecutor(1, mp_context=context) as pool:
        reports = manager.Queue() if progress is not None else None
        future = pool.submit(
            _score_in_child, tools, list(infos), list(shots), cache_dir, device, reports
        )
        while True:
            try:
                result = future.result(timeout=0.1)
            except FuturesTimeout:
                result = None
            while reports is not None and progress is not None:
                try:
                    text, share = reports.get_nowait()
                except (queue.Empty, EOFError, OSError):
                    break
                progress(str(text), float(share))
            if result is not None:
                return result
