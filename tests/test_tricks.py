"""Naming fast turns from the camera's pitch, yaw and roll (degrees)."""

from __future__ import annotations

import numpy as np

from fpv_editor.tricks import name_fast_turn, name_turn, power_loop_frames


def test_turns_are_named_by_their_axis() -> None:
    # Measured values of Dom's labelled tricks (the backflip, side flip and trippy spin).
    assert name_turn(232, 13, 3) == "back flip"
    assert name_turn(-226, 0, -23) == "front flip"
    assert name_turn(0, -28, -194) == "roll"
    assert name_turn(12, 167, 104) == "roll"  # yaw and roll the same way: up-tilted roll
    assert name_turn(-50, -153, 91) == "yaw spin"  # opposite ways: a yaw spin
    assert name_turn(500, 10, 10) == "double back flip"
    assert name_turn(0, 300, 340) == "double roll"
    assert name_turn(139, 239, 41) == "flip"  # mixed axes: no claim about which


def test_an_unsure_fit_keeps_the_plain_name() -> None:
    fps = 30
    rotation = np.zeros((60, 3))
    rotation[15:45, 0] = 300  # a clear back flip ...
    confident = np.ones(60)
    assert name_fast_turn(rotation, confident, 30, fps) == "back flip"
    assert name_fast_turn(rotation, confident * 0.2, 30, fps) == "flip"  # ... seen through blur


def test_a_slow_full_pitch_turn_is_a_power_loop_a_fast_one_is_not() -> None:
    fps = 30
    usable = np.ones(300, dtype=bool)
    slow = np.zeros((300, 3))
    slow[100:175, 0] = 130  # 325 degrees over 2.5 s, never 180 in one second
    assert float(np.max(power_loop_frames(slow, usable, fps))) > 0
    fast = np.zeros((300, 3))
    fast[100:130, 0] = 400  # a flip
    assert float(np.max(power_loop_frames(fast, usable, fps))) == 0
    usable[120] = False
    assert power_loop_frames(slow, usable, fps)[137] == 0  # not across unusable footage
