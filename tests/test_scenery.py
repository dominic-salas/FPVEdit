"""The image-model step around the model itself: notes, fallbacks, what is left out.

The model is replaced by fixed scores here; `-m model` runs the real one."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fpv_editor import scenery
from fpv_editor.cli import MontagePlan, judge_scenery
from fpv_editor.media import Tools
from fpv_editor.montage import Moment, MomentKind, MontageConfig, Shot

TOOLS = Tools(ffmpeg="ffmpeg", ffprobe="ffprobe", ffmpeg_major=7)


def shot(clip: int, start: float, score: float) -> Shot:
    return Shot(clip, start, start + 9, score, (Moment(start + 3, MomentKind.FLIP, 3.0),))


def plan(shots: list[Shot]) -> MontagePlan:
    return MontagePlan((), (), tuple(shots), tuple(shots), MontageConfig(), ())


def test_without_the_ml_extra_the_plan_is_kept_with_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scenery, "available", lambda: False)
    shots = [shot(0, 0, 3), shot(1, 0, 9)]
    judged = judge_scenery(TOOLS, plan(shots))
    assert judged.shots == tuple(shots) and judged.left_out == ()
    assert any("not installed" in note for note in judged.warnings)


def test_weak_indoor_shots_are_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    outdoor = [shot(0, 20 * i, 3.0 * (i + 1)) for i in range(4)]
    weak, strong = shot(1, 0, 3.0), shot(1, 20, 12.0)
    scores = [-0.05] * 4 + [0.06, 0.07]
    monkeypatch.setattr(scenery, "available", lambda: True)
    monkeypatch.setattr(scenery, "score_shots_in_process", lambda *a, **k: scores)
    judged = judge_scenery(TOOLS, plan([*outdoor, weak, strong]))
    assert judged.left_out == (weak,) and judged.shots == (*outdoor, strong)
    assert any("left out 1 indoor" in note for note in judged.warnings)


def test_a_failing_model_keeps_every_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> list[float | None]:
        raise scenery.SceneryError("no weights")

    monkeypatch.setattr(scenery, "available", lambda: True)
    monkeypatch.setattr(scenery, "score_shots_in_process", fail)
    shots = [shot(0, 0, 3)]
    judged = judge_scenery(TOOLS, plan(shots))
    assert judged.shots == tuple(shots) and any("could not run" in n for n in judged.warnings)


@pytest.mark.model
def test_the_model_tells_a_room_from_a_forest(tmp_path: Path) -> None:
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
    assert dataclasses.is_dataclass(MontageConfig())
