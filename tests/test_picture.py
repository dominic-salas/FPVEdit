from __future__ import annotations

import numpy as np
import pytest

from fpv_editor.picture import Area, detect_active_area, detect_overlay, fit_frames

WIDTH, HEIGHT = 320, 180
BAR = 40  # 4:3 picture (240 px) between two 40 px bars


def scene(seed: int, *, level: int = 120, spread: int = 60) -> np.ndarray:
    """Blocky texture, so optical flow and differences are well defined."""
    rng = np.random.default_rng(seed)
    small = rng.integers(level - spread, level + spread + 1, size=(HEIGHT // 6, WIDTH // 6))
    big = np.kron(small, np.ones((6, 6))).astype(np.uint8)
    return np.pad(big, ((0, HEIGHT - big.shape[0]), (0, WIDTH - big.shape[1])), mode="edge")


def pillarboxed(frame: np.ndarray, black: int = 4) -> np.ndarray:
    out = frame.copy()
    out[:, :BAR] = black
    out[:, WIDTH - BAR :] = black
    return out


def with_osd(frame: np.ndarray, seed: int) -> np.ndarray:
    """White text-like boxes with a dark outline; the 'digits' change between frames."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for x, y in ((60, 10), (200, 10), (60, 150), (190, 150)):
        out[y : y + 14, x : x + 40] = 0
        strokes = rng.integers(0, 2, size=(5, 18)).astype(bool)
        glyph = np.kron(strokes, np.ones((2, 2), dtype=bool))
        out[y + 2 : y + 12, x + 2 : x + 38][glyph] = 255
    return out


def test_finds_the_4_3_picture_between_bars() -> None:
    frames = [pillarboxed(scene(i)) for i in range(8)]
    area = detect_active_area(frames, (WIDTH, HEIGHT))
    assert abs(area.x - BAR) <= 3 and abs(area.x + area.width - (WIDTH - BAR)) <= 3
    assert (area.y, area.height) == (0, HEIGHT)
    assert area.x % 2 == 0 and area.width % 2 == 0


def test_dark_night_picture_still_counts_as_picture() -> None:
    # Analog night footage is dark but noisy; the bars are flat.
    frames = [pillarboxed(scene(i, level=14, spread=10)) for i in range(8)]
    area = detect_active_area(frames, (WIDTH, HEIGHT))
    assert abs(area.x - BAR) <= 3 and abs(area.width - (WIDTH - 2 * BAR)) <= 6


def test_a_few_broken_frames_do_not_move_the_bars() -> None:
    frames = [pillarboxed(scene(i)) for i in range(8)]
    frames[2] = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)  # grey decode garbage
    area = detect_active_area(frames, (WIDTH, HEIGHT))
    assert abs(area.x - BAR) <= 3


@pytest.mark.parametrize("count", [0, 1])
def test_without_bars_or_frames_the_whole_frame_is_used(count: int) -> None:
    frames = [scene(i) for i in range(count)]
    assert detect_active_area(frames, (WIDTH, HEIGHT)) == Area(0, 0, WIDTH, HEIGHT)
    assert detect_active_area([scene(i) for i in range(8)], (WIDTH, HEIGHT)) == Area(
        0, 0, WIDTH, HEIGHT
    )


def test_overlay_mask_covers_osd_boxes_and_little_else() -> None:
    frames = [with_osd(scene(i), seed=i) for i in range(12)]
    mask = detect_overlay(frames)
    assert mask is not None
    osd = np.zeros_like(mask)
    for x, y in ((60, 10), (200, 10), (60, 150), (190, 150)):
        osd[y : y + 14, x : x + 40] = True
    assert mask[osd].mean() > 0.95
    assert mask[~osd].mean() < 0.05


def test_overlay_needs_enough_frames() -> None:
    assert detect_overlay([scene(0), scene(1)]) is None


def test_fit_frames_crops_then_scales() -> None:
    frame = pillarboxed(scene(0))
    (fitted,) = fit_frames([frame], Area(BAR, 0, WIDTH - 2 * BAR, HEIGHT), (120, 90))
    assert fitted.shape == (90, 120)
    assert int(np.min(fitted)) > 4  # no bar pixels left
