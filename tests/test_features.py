from __future__ import annotations

import numpy as np
import pytest

from fpv_editor.features import compute_features

FPS = 8.0


def texture(width: int = 320, height: int = 180, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(height // 8, width // 8), dtype=np.uint8)
    return np.kron(small, np.ones((8, 8), dtype=np.uint8)).astype(np.uint8)


def test_known_shift_gives_expected_speed_in_frame_widths_per_second() -> None:
    base = texture(width=400)
    frames = [np.ascontiguousarray(base[:, x : x + 320]) for x in range(0, 80, 8)]
    features = compute_features(frames, FPS)
    expected = 8 / 320 * FPS  # 8 px per sample at 320 px wide, 8 samples per second
    assert np.median(features.motion[1:]) == pytest.approx(expected, rel=0.25)


def test_static_frames_have_no_motion_change_or_novelty() -> None:
    frame = texture()
    features = compute_features([frame] * 10, FPS)
    assert np.allclose(features.motion, 0.0, atol=1e-3)
    assert np.allclose(features.change, 0.0)
    assert np.allclose(features.novelty, 0.0, atol=1e-6)


def test_first_sample_copies_its_neighbour() -> None:
    frames = [texture(seed=i) for i in range(4)]
    features = compute_features(frames, FPS)
    for values in (features.motion, features.change, features.novelty):
        assert values[0] == values[1]


def test_novelty_and_change_are_bounded() -> None:
    frames = [np.zeros((180, 320), np.uint8), np.full((180, 320), 255, np.uint8)] * 3
    features = compute_features(frames, FPS)
    assert np.max(features.novelty) <= 1.0 + 1e-6
    assert np.max(features.change) <= 1.0
    assert np.max(features.novelty) > 0.9


def test_blank_frames_report_low_brightness_and_contrast() -> None:
    features = compute_features([np.zeros((180, 320), np.uint8)] * 3, FPS)
    assert np.max(features.brightness) == 0.0
    assert np.max(features.contrast) == 0.0


def test_empty_input_gives_empty_series() -> None:
    features = compute_features([], FPS)
    assert len(features) == 0
