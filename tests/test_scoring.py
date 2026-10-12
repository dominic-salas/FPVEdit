from __future__ import annotations

import numpy as np
import pytest

from fpv_editor.features import FeatureSeries
from fpv_editor.scoring import (
    ScoringConfig,
    gaussian_smooth,
    idle_mask,
    interest_score,
    normalize,
)


def series(n: int, **values: float) -> FeatureSeries:
    def column(name: str, default: float) -> np.ndarray:
        return np.full(n, values.get(name, default), dtype=np.float64)

    return FeatureSeries(
        fps=8.0,
        motion=column("motion", 0.5),
        change=column("change", 0.05),
        novelty=column("novelty", 0.1),
        brightness=column("brightness", 0.5),
        contrast=column("contrast", 0.2),
    )


def test_normalize_clips_to_unit_range() -> None:
    result = normalize(np.array([-1.0, 0.0, 0.5, 2.0, 10.0]), reference=2.0)
    assert result.tolist() == [0.0, 0.0, 0.25, 1.0, 1.0]


def test_normalize_rejects_non_positive_reference() -> None:
    with pytest.raises(ValueError):
        normalize(np.ones(3), 0.0)


def test_weights_must_sum_to_one_and_be_non_negative() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ScoringConfig(motion_weight=0.9)
    with pytest.raises(ValueError, match="negative"):
        ScoringConfig(motion_weight=1.2, change_weight=-0.5, novelty_weight=0.3)


def test_gaussian_smooth_preserves_length_constant_and_mass() -> None:
    constant = np.full(20, 0.3)
    assert gaussian_smooth(constant, 2.0) == pytest.approx(constant)
    spike = np.zeros(41)
    spike[20] = 1.0
    smoothed = gaussian_smooth(spike, 3.0)
    assert len(smoothed) == 41
    assert smoothed.sum() == pytest.approx(1.0)
    assert int(np.argmax(smoothed)) == 20
    assert np.max(smoothed) < 0.2


def test_gaussian_smooth_handles_tiny_inputs() -> None:
    assert gaussian_smooth(np.array([0.4]), 5.0).tolist() == pytest.approx([0.4])
    assert len(gaussian_smooth(np.array([]), 5.0)) == 0


def test_motion_alone_cannot_saturate_the_score() -> None:
    config = ScoringConfig()
    fast_but_static_scene = series(40, motion=50.0, change=0.0, novelty=0.0)
    score = interest_score(fast_but_static_scene, config)
    assert np.max(score) == pytest.approx(config.motion_weight)


def test_score_is_bounded_between_zero_and_one() -> None:
    extreme = series(40, motion=99.0, change=99.0, novelty=99.0)
    assert np.max(interest_score(extreme, ScoringConfig())) == pytest.approx(1.0)
    still = series(40, motion=0.0, change=0.0, novelty=0.0)
    assert np.max(interest_score(still, ScoringConfig())) == pytest.approx(0.0)


def test_idle_requires_both_low_motion_and_low_change() -> None:
    config = ScoringConfig()
    assert idle_mask(series(5, motion=0.001, change=0.001), config).all()
    assert not idle_mask(series(5, motion=0.001, change=0.2), config).any()
    assert not idle_mask(series(5, motion=0.5, change=0.001), config).any()


def test_dark_flat_frames_are_idle_even_with_noise_motion() -> None:
    blank = series(5, motion=0.5, change=0.05, brightness=0.02, contrast=0.01)
    assert idle_mask(blank, ScoringConfig()).all()
