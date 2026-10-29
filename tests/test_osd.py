"""Reading the throttle from a burned-in OSD: synthetic frames drawn with OpenCV's own font
(Betaflight's fonts are not shipped), on noisy scenery, with a timer counting seconds."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from fpvedit import osd
from fpvedit.osd import (
    FPS,
    HEIGHT,
    WIDTH,
    Throttle,
    find_elements,
    find_throttle,
    read_throttle,
    strip_of,
)
from fpvedit.picture import Area
from fpvedit.rating import throttle_notes

PITCH = 32  # analog OSD: 30 characters across the 960 px picture


def draw(frame: np.ndarray, text: str, right: int, baseline: int) -> None:
    """Right-aligned OSD text: white glyphs with a black outline, one per 32 px."""
    for k, ch in enumerate(reversed(text)):
        if ch == " ":
            continue
        x = right - (k + 1) * PITCH + 7
        for colour, thickness in ((0, 7), (255, 2)):
            cv2.putText(frame, ch, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, thickness)


def throttle_values(n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    value, out = 30.0, []
    for i in range(n):
        if i % 4 == 0:
            value = float(np.clip(value + rng.normal(0, 18), 0, 100))
        out.append(round(value))
    return out


def osd_frames(
    seconds: int = 40, *, timer: bool = True, seed: int = 0
) -> tuple[list[np.ndarray], list[int]]:
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    values = throttle_values(n, seed)
    frames = []
    for i in range(n):
        base = 90 + 40 * np.sin(np.linspace(0, 6, WIDTH) + i / 15)[None, :]
        frame = np.clip(base + rng.normal(0, 12, (HEIGHT, WIDTH)), 0, 255).astype(np.uint8)
        draw(frame, f"{values[i]:3d}", 920, 590)  # the throttle
        if timer:
            second = 5 + i // int(FPS)
            draw(frame, f"{second // 60:02d}:{second % 60:02d}", 920, 650)
        draw(frame, "4.2V", 920, 250)  # the voltage: static
        frames.append(frame)
    return frames, values


def throttle_of(frames: list[np.ndarray]) -> np.ndarray | None:
    """`read_throttle` without FFmpeg: elements from samples, strips, the throttle."""
    elements = find_elements(frames[:: max(1, len(frames) // 150)])
    strips = {box: np.stack([strip_of(f, box) for f in frames]) for box in elements}
    found = find_throttle(elements, strips, int(FPS))
    return None if found is None else found[1]


def test_the_throttle_is_found_and_read() -> None:
    frames, truth = osd_frames()
    values = throttle_of(frames)
    assert values is not None
    read = np.isfinite(values)
    assert read.mean() >= 0.8
    assert np.mean(values[read] == np.array(truth)[read]) >= 0.97


def test_without_a_running_timer_nothing_is_guessed() -> None:
    frames, _ = osd_frames(timer=False)
    assert throttle_of(frames) is None


def test_scenery_without_an_osd_has_no_elements() -> None:
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8) for _ in range(30)]
    assert find_elements(frames) == []


@pytest.mark.integration
def test_read_throttle_on_an_encoded_recording(tmp_path: Path) -> None:
    from fpvedit.media import find_tools, probe
    from media_fixtures import ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")
    frames, truth = osd_frames(seconds=30, seed=2)
    path = tmp_path / "osd.mp4"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray"),
            *("-s", f"{WIDTH}x{HEIGHT}", "-r", f"{FPS:g}", "-i", "-"),
            *("-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p"),
            str(path),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True,
    )
    tools = find_tools()
    throttle = read_throttle(tools, probe(tools, path), Area(0, 0, WIDTH, HEIGHT))
    assert throttle is not None
    n = min(len(truth), len(throttle.values))
    read = np.isfinite(throttle.values[:n])
    assert read.mean() >= 0.8
    assert np.mean(throttle.values[:n][read] == np.array(truth[:n])[read]) >= 0.95


def series(values: list[float], fps: float = FPS) -> Throttle:
    return Throttle(fps, np.array(values, dtype=np.float64))


def test_throttle_notes_name_how_a_trick_was_flown() -> None:
    fps = int(FPS)
    cruise = [40.0] * (2 * fps)
    # A flip at 2 s: cut to 5% for half a second, then punched to 90%.
    punch = series(cruise + [5.0] * (fps // 2) + [90.0] * fps + [50.0] * (2 * fps))
    assert throttle_notes(0.0, 5.0, [2.2], punch) == ["throttle punch"]
    # A dive: off the throttle through the turn and after it.
    dive = series(cruise + [2.0] * (3 * fps))
    assert throttle_notes(0.0, 5.0, [2.5], dive) == ["throttle off"]
    # Flown on the throttle the whole time.
    assert throttle_notes(0.0, 4.0, [], series([75.0] * (4 * fps))) == ["high throttle"]


def test_throttle_notes_need_a_readable_throttle() -> None:
    assert throttle_notes(0.0, 4.0, [1.0], None) == []
    mostly_unread = series([np.nan] * (4 * int(FPS) - 3) + [80.0] * 3)
    assert throttle_notes(0.0, 4.0, [1.0], mostly_unread) == []


def test_the_throttle_is_cached_beside_the_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fpvedit import pipeline
    from fpvedit.analysis import analyze
    from fpvedit.media import find_tools, probe
    from media_fixtures import Span, ffmpeg_available, make_flight

    if not ffmpeg_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")
    tools = find_tools()
    info = probe(tools, make_flight(tmp_path / "f.ts", [Span("active", 12)]))
    config = pipeline.PipelineConfig().analysis
    analysis = analyze(tools, info, config, cache_dir=tmp_path)
    calls: list[int] = []
    original = osd.read_throttle

    def counted(*args: object, **kwargs: object) -> Throttle | None:
        calls.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(osd, "read_throttle", counted)
    first = pipeline._throttle(tools, info, analysis, tmp_path, config)
    again = pipeline._throttle(tools, info, analysis, tmp_path, config)
    assert first is None and again is None  # no OSD in this recording
    assert len(calls) == 1  # the second answer came from the cache
    assert list(tmp_path.glob(f"*.throttle-v{osd.VERSION}.npz"))
