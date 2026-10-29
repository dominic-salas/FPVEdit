"""Measuring the tricks' turns with gravity: synthetic turns whose true angle is known.

The flow is made to under-read (as blur does) and gravity before and after the turn is
what a real turn of the true angle would leave; the gravity model itself is replaced,
and `-m model` runs the real one."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fpvedit import attitude
from fpvedit.attitude import (
    Attitude,
    Gravity,
    Turn,
    anchor_times,
    estimate_uptilt,
    fast_turns,
    measure,
    resolve,
)
from fpvedit.media import Tools
from fpvedit.montage import CutMap, Moment, MomentKind, MontageConfig, Shot
from fpvedit.pipeline import Clip, MontagePlan, keep_rated, measure_turns
from fpvedit.rating import ClipSignals, rate_shot
from fpvedit.tricks import name_measured

FPS = 60.0
UP = np.array([0.0, -math.cos(0.3), math.sin(0.3)])  # level flight, camera tilted up
TOOLS = Tools(ffmpeg="ffmpeg", ffprobe="ffprobe", ffmpeg_major=7)
PITCH, ROLL, YAW = np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])


def after(axis: np.ndarray, degrees: float, up: np.ndarray = UP) -> np.ndarray:
    """Where "up" is seen once the camera has turned ``degrees`` about ``axis``."""
    return np.asarray(attitude._turned(axis / np.linalg.norm(axis), degrees) @ up)


def reading(vector: np.ndarray, unsure: float = 4.0) -> Gravity:
    return (float(vector[0]), float(vector[1]), float(vector[2])), unsure


def test_a_camera_rolling_right_sees_up_turn_to_its_left() -> None:
    # Rolled 90 degrees right, the camera's left (-x) points up: worked out by hand.
    seen = attitude._turned(ROLL, 90.0) @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(seen, [-1.0, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize(
    ("axis", "true", "flow"),
    [
        (PITCH, 360.0, 280.0),  # a back flip the blur cut short
        (-PITCH, 360.0, 380.0),  # a front flip read a little long
        (ROLL, 360.0, 150.0),  # rolls are read shortest
        (ROLL, 180.0, 140.0),  # a half roll
        (PITCH, 90.0, 85.0),
        (PITCH, 720.0, 640.0),  # a double flip
    ],
)
def test_gravity_resolves_the_true_angle(axis: np.ndarray, true: float, flow: float) -> None:
    found = resolve(axis * flow, reading(UP), reading(after(axis, true)))
    assert found is not None
    degrees, misfit = found
    assert degrees == pytest.approx(true, abs=0.5) and misfit < 0.5


def test_gravity_never_adds_a_turn_the_flow_did_not_see() -> None:
    # A double flip read as 400: the fewest whole turns that reach 85% of it is one.
    found = resolve(PITCH * 400.0, reading(UP), reading(UP))
    assert found is not None and found[0] == pytest.approx(360.0, abs=0.5)


def test_a_turn_about_the_vertical_or_an_unsure_reading_is_not_resolved() -> None:
    up = np.array([0.0, -1.0, 0.0])
    assert resolve(YAW * 300.0, reading(up), reading(up)) is None  # gravity cannot see it
    assert resolve(PITCH * 300.0, reading(UP, 15.0), reading(UP)) is None


def series(seconds: float, turns: list[tuple[float, np.ndarray, float]]) -> np.ndarray:
    """Rates (deg/s) with each (start s, axis, measured degrees) turned over 0.4 s."""
    rotation = np.zeros((round(seconds * FPS), 3))
    for start, axis, degrees in turns:
        rotation[round(start * FPS) : round((start + 0.4) * FPS)] = axis * degrees / 0.4
    return rotation


def test_measure_resolves_whole_and_part_turns() -> None:
    rotation = series(10.0, [(2.0, PITCH, 270.0), (6.0, ROLL, 150.0)])
    spans = fast_turns(rotation, FPS)
    assert len(spans) == 2
    gravity: dict[float, Gravity | None] = {}
    for t in anchor_times(spans, FPS):
        up = UP if t < 6.2 else after(ROLL, 180.0)  # whole flip, then a half roll
        gravity[round(t, 2)] = reading(up)
    found = measure(rotation, FPS, gravity)
    flip, roll = found.turns
    assert flip.checked and flip.whole == 1 and flip.degrees == pytest.approx(360, abs=1)
    assert roll.checked and roll.whole == 0 and roll.degrees == pytest.approx(180, abs=1)
    assert name_measured(flip, found) == "back flip"
    assert name_measured(roll, found) == "half roll"


def test_without_gravity_a_turn_keeps_the_flow_reading() -> None:
    rotation = series(5.0, [(2.0, PITCH, 270.0)])
    (turn,) = measure(rotation, FPS, {}).turns
    assert not turn.checked and turn.whole == 0 and turn.degrees == pytest.approx(270, abs=1)


def test_the_uptilt_comes_from_rolls_and_yaw_spins() -> None:
    u = math.radians(20.0)
    roll = (0.0, math.sin(u), math.cos(u))
    yaw = (0.0, math.cos(u), -math.sin(u))
    axes = [roll, yaw, tuple(-v for v in roll), tuple(-v for v in yaw)] * 3
    assert estimate_uptilt(axes) == pytest.approx(20.0, abs=0.1)  # type: ignore[arg-type]
    assert estimate_uptilt(axes[:3]) == attitude.AttitudeConfig().uptilt_prior  # type: ignore[arg-type]


def checked(axis: tuple[float, float, float], degrees: float, whole: int) -> Turn:
    return Turn(1.0, 1.4, axis, degrees * 0.7, degrees, True, whole, 1.0)


def test_measured_turns_are_named_on_the_quads_own_axes() -> None:
    u = math.radians(20.0)
    tilted = Attitude(20.0, ())
    assert name_measured(checked((0.0, math.sin(u), math.cos(u)), 360, 1), tilted) == "roll"
    assert name_measured(checked((0.0, math.cos(u), -math.sin(u)), 360, 1), tilted) == "yaw spin"
    assert name_measured(checked((-1.0, 0.0, 0.0), 720, 2), tilted) == "double front flip"
    assert name_measured(checked((1.0, 0.0, 0.0), 270, 0), tilted) == "3/4 back flip"
    assert name_measured(checked((1.0, 0.0, 0.0), 540, 0), tilted) == "1 1/2 back flips"


def signals(rotation: np.ndarray) -> ClipSignals:
    count = len(rotation)
    return ClipSignals(
        fps=FPS,
        rate=np.linalg.norm(rotation, axis=1),
        rotation=rotation,
        speed=np.full(count, 0.5),
        approach=np.zeros(count),
        confidence=np.ones(count),
        brightness=np.full(count, 0.5),
        crashes=(),
    )


def a_plan(rotation: np.ndarray) -> MontagePlan:
    cuts = CutMap(
        np.ones(len(rotation), bool), np.zeros(len(rotation)), np.zeros(len(rotation), bool),
        signals(rotation),
    )  # fmt: skip
    info = object()  # only passed through to the (replaced) reader
    clip = Clip(0, info, None, len(rotation) / FPS, cuts)  # type: ignore[arg-type]
    moment = Moment(2.2, MomentKind.FLIP, 3.0, "flip")
    quality = rate_shot(0.0, 5.0, [(moment.time, moment.name)], cuts.signals)
    shot = Shot(0, 0.0, 5.0, 3.0, (moment,), quality)
    return MontagePlan((), (clip,), (shot,), (shot,), MontageConfig(), ())


def test_measure_turns_renames_and_rates_the_flips_again(monkeypatch: pytest.MonkeyPatch) -> None:
    rotation = series(6.0, [(2.0, PITCH, 150.0)])  # a flip the flow read as 150 degrees
    plan = a_plan(rotation)
    before = plan.shots[0].quality
    assert before is not None and before.clean < 1.0  # the flow made it look short

    def fake(
        tools: Tools, wanted: list[tuple[object, list[float]]], **_: object
    ) -> list[dict[float, Gravity | None]]:
        return [{round(t, 2): reading(UP) for t in times} for _, times in wanted]

    monkeypatch.setattr(attitude, "read_gravity_in_process", fake)
    measured = measure_turns(TOOLS, plan)
    shot = measured.shots[0]
    assert shot.moments[0].name == "back flip"
    assert shot.quality is not None and shot.quality.clean > before.clean  # it came round
    cuts = measured.clips[0].cuts
    assert cuts is not None and cuts.signals.attitude is not None
    # --keep-intermediates writes what was measured, for anyone tuning the names.
    turns = cuts.signals.attitude.to_json()["turns"]
    assert turns[0]["checked"] and turns[0]["whole"] == 1


def test_without_the_gravity_model_names_stay_and_a_note_says_why() -> None:
    plan = a_plan(series(6.0, [(2.0, PITCH, 150.0)]))
    judged = keep_rated(TOOLS, plan, min_stars=1)
    assert judged.shots[0].moments[0].name == "flip"
    assert any("gravity model is not installed" in note for note in judged.warnings)


def test_gravity_readings_are_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    info = type("Info", (), {"path": video})()
    calls: list[float] = []

    class Model:
        def __init__(self, device: str) -> None:
            pass

        def __call__(self, frame: np.ndarray) -> Gravity:
            calls.append(1.0)
            return reading(UP)

    monkeypatch.setattr(attitude, "GravityModel", Model)
    monkeypatch.setattr(attitude, "grab_picture", lambda *_: np.zeros((4, 4, 3), np.uint8))
    wanted = [(info, [1.0, 2.0])]
    first = attitude.read_gravity(TOOLS, wanted, cache_dir=tmp_path)
    again = attitude.read_gravity(TOOLS, wanted, cache_dir=tmp_path)
    assert first == again and len(calls) == 2  # the second answer came from the cache
    assert first[0][1.0] is not None


@pytest.mark.model
def test_the_real_gravity_model_reads_a_level_and_a_rolled_picture() -> None:
    import cv2

    if not attitude.available():
        pytest.skip("the ml extra is not installed")
    rng = np.random.default_rng(0)
    # A floor below a bright sky with vertical posts: a scene with an obvious horizon.
    frame = np.full((360, 480, 3), 200, np.uint8)
    frame[180:] = rng.integers(60, 110, (180, 480, 3), dtype=np.uint8)
    for x in range(40, 480, 80):
        frame[60:300, x : x + 12] = 30
    model = attitude.GravityModel()
    level = np.array(model(frame)[0], dtype=np.float64)
    turned: np.ndarray = cv2.warpAffine(
        frame, cv2.getRotationMatrix2D((240, 180), 30, 1.0), (480, 360),
        borderMode=cv2.BORDER_REFLECT,
    )  # fmt: skip
    rolled = np.array(model(np.asarray(turned, np.uint8))[0], dtype=np.float64)
    assert np.degrees(np.arccos(np.clip(after(ROLL, 30.0, level) @ rolled, -1, 1))) < 12
