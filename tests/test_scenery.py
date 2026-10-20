"""Keeping shots by rating, with and without the image model's nudge.

The model is replaced by fixed scores here; `-m model` runs the real one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpvedit import scenery
from fpvedit.media import Tools
from fpvedit.montage import Moment, MomentKind, MontageConfig, Shot
from fpvedit.pipeline import MontagePlan, keep_rated
from fpvedit.rating import Quality

TOOLS = Tools(ffmpeg="ffmpeg", ffprobe="ffprobe", ffmpeg_major=7)


def shot(clip: int, start: float, level: float) -> Shot:
    """A shot whose rating parts are all ``level`` (0-1)."""
    quality = Quality(level, level, level, level, ("test",))
    moment = Moment(start + 3, MomentKind.FLIP, 3.0, "back flip")
    return Shot(clip, start, start + 9, 3.0, (moment,), quality)


def plan(shots: list[Shot]) -> MontagePlan:
    return MontagePlan((), (), tuple(shots), tuple(shots), MontageConfig(), ())


def test_shots_below_the_minimum_are_left_out() -> None:
    great, fine, weak = shot(0, 0, 0.9), shot(0, 20, 0.66), shot(1, 0, 0.3)
    assert (great.quality.stars, fine.quality.stars, weak.quality.stars) == (5, 3, 1)  # type: ignore[union-attr]
    judged = keep_rated(TOOLS, plan([great, fine, weak]), min_stars=3)
    assert judged.shots == (great, fine) and judged.left_out == (weak,)
    assert any("not installed" in note for note in judged.warnings)
    everything = keep_rated(TOOLS, plan([great, fine, weak]), min_stars=1)
    assert everything.left_out == ()


def test_when_nothing_reaches_the_minimum_the_best_shots_stay() -> None:
    a, b = shot(0, 0, 0.3), shot(0, 20, 0.6)  # 1 and 2 stars
    judged = keep_rated(TOOLS, plan([a, b]), min_stars=4)
    assert judged.shots == (b,) and judged.left_out == (a,)


def test_the_image_model_nudges_ratings(monkeypatch: pytest.MonkeyPatch) -> None:
    edge = shot(0, 0, 0.64)  # 3 stars by itself
    other = shot(1, 0, 0.9)
    assert keep_rated(TOOLS, plan([edge, other]), min_stars=3).left_out == ()  # no model
    monkeypatch.setattr(scenery, "available", lambda: True)
    for indoorness, kept in ((0.06, False), (-0.05, True)):  # cramped: down; open: up
        monkeypatch.setattr(
            scenery,
            "score_shots_in_process",
            lambda *a, v=indoorness, **k: [
                scenery.ShotScene(v, None),
                scenery.ShotScene(-0.05, None),
            ],
        )
        judged = keep_rated(TOOLS, plan([edge, other]), min_stars=3)
        assert (len(judged.shots) == 2) is kept, indoorness
        assert judged.shots[0].quality.indoorness == indoorness or not kept  # type: ignore[union-attr]


def test_a_failing_model_keeps_the_plain_ratings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> list[float | None]:
        raise scenery.SceneryError("no weights")

    monkeypatch.setattr(scenery, "available", lambda: True)
    monkeypatch.setattr(scenery, "score_shots_in_process", fail)
    judged = keep_rated(TOOLS, plan([shot(0, 0, 0.9)]))
    assert len(judged.shots) == 1 and any("could not run" in n for n in judged.warnings)


@pytest.mark.model
def test_the_model_tells_a_room_from_a_forest() -> None:
    """Real SigLIP 2 inference on two drawn pictures (downloads the weights once)."""
    model = scenery.SceneryModel("auto")
    room = np.full((224, 224, 3), 200, dtype=np.uint8)
    room[120:, :] = (120, 80, 50)  # a wooden floor under a white wall
    room[60:120, 40:180] = (90, 60, 40)  # a sofa
    forest = np.zeros((224, 224, 3), dtype=np.uint8)
    forest[:90] = (120, 170, 235)  # sky
    forest[90:] = (40, 110, 40)  # trees and grass
    for x in range(10, 224, 30):
        forest[60:200, x : x + 8] = (70, 50, 30)  # trunks
    values = model.indoorness([room, forest])
    assert values[0] > values[1]


def test_progress_follows_the_real_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reading frames, loading the model and each batch move the bar; it never sits still
    at one value until the end, and never goes back."""
    from fpvedit.media import MediaInfo

    class Model:
        def __init__(self, device: str) -> None:
            pass

        def describe(self, frames: list[object], done: object = None) -> tuple[object, object]:
            for end in range(scenery.BATCH, len(frames) + scenery.BATCH, scenery.BATCH):
                done(min(end, len(frames)))  # type: ignore[operator]
            n = len(frames)
            return np.zeros(n), np.zeros((n, len(scenery.PICTURE)))

    recording = tmp_path / "flight.mp4"
    recording.write_bytes(b"x")
    info: MediaInfo = type("Info", (), {"path": recording})()  # only its path is read
    monkeypatch.setattr(scenery, "SceneryModel", Model)
    monkeypatch.setattr(scenery, "grab_rgb", lambda *a: np.zeros((2, 2, 3), np.uint8))
    shots = [shot(0, 10 * i, 0.5) for i in range(20)]  # 60 frames
    reports: list[tuple[str, float]] = []
    scenes = scenery.score_shots(
        TOOLS,
        [info],
        shots,
        cache_dir=tmp_path,
        progress=lambda t, s: reports.append((t, s)),
    )
    shares = [s for _, s in reports]
    assert len(scenes) == 20 and all(s.picture is not None for s in scenes)
    assert shares == sorted(shares) and shares[-1] == 1.0
    assert len(set(shares)) >= 10  # many steps, not one jump from 50 % to done
    assert any("Loading the image model" in t for t, _ in reports)
    # Seen once: the next time everything comes from the cache, with no work to report.
    reports.clear()
    scenery.score_shots(
        TOOLS, [info], shots, cache_dir=tmp_path, progress=lambda t, s: reports.append((t, s))
    )
    assert reports == []
