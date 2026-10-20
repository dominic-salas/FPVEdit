"""Montage renders measured from the output file itself, never from the renderer's own sums.

Each source frame carries its own number, drawn as 12 black or white columns (bit k of
the frame number in column k), wide enough to survive H.264. Decoding the montage and
reading the numbers back shows exactly which source frames it holds: none dropped or
repeated inside a shot, nothing extra at the joins, shots in the order given.
"""

from __future__ import annotations

import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from fpvedit.errors import FPVEditError
from fpvedit.media import Tools, find_tools, frame_times, probe
from fpvedit.montage import Shot
from fpvedit.pipeline import Clip, render_montage
from fpvedit.render import EncodeConfig
from media_fixtures import ffmpeg_available

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH"),
]

FAST = EncodeConfig(preset="ultrafast")
BITS, COLUMN = 12, 26  # 12 bits x 26 px = 312 of the 320 px width


def numbered_clip(
    path: Path, seconds: float, fps: int, width: int = 320, height: int = 240
) -> Path:
    """Frame N shows N in binary across the top half; the bottom half is mid grey."""
    bit = f"mod(floor(N/pow(2,floor(X/{COLUMN}))),2)"
    lum = f"if(lt(Y,{height // 2})*lt(X,{BITS * COLUMN}),16+219*{bit},128)"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi"),
            *("-i", f"color=c=gray:s={width}x{height}:r={fps}:d={seconds}"),
            *("-vf", f"geq=lum='{lum}':cb=128:cr=128,format=yuv420p"),
            *("-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-g", str(2 * fps)),
            str(path),
        ],
        check=True,
    )
    return path


def gray_frames(path: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width)


def frame_numbers(frames: np.ndarray, row: int, left: int = 0, scale: float = 1.0) -> list[int]:
    centres = [left + round((k + 0.5) * COLUMN * scale) for k in range(BITS)]
    bits = frames[:, row, centres] > 128
    return [int(sum(1 << k for k in range(BITS) if b[k])) for b in bits]


def clip(tools: Tools, index: int, path: Path) -> Clip:
    info = probe(tools, path)
    times = frame_times(tools, info)
    assert times is not None
    return Clip(index, info, times, float(times[-1]))


def shot(index: int, start: float, end: float) -> Shot:
    return Shot(index, start, end, 1.0, ())


def runs(numbers: list[int]) -> list[list[int]]:
    """Split into runs of consecutive frame numbers."""
    out = [[numbers[0]]]
    for previous, number in pairwise(numbers):
        if number == previous + 1:
            out[-1].append(number)
        else:
            out.append([number])
    return out


def test_every_shot_holds_exactly_its_frames_and_nothing_more(tmp_path: Path) -> None:
    tools = find_tools()
    source = numbered_clip(tmp_path / "numbered.mp4", 20, 60)
    clips = [clip(tools, 0, source)]
    # Out of time order on purpose: the montage plays them in the order given.
    shots = [shot(0, 10.5, 12.0), shot(0, 2.0, 5.0), shot(0, 15.25, 16.0)]
    output = tmp_path / "montage.mp4"
    render_montage(tools, clips, shots, output, force=False, encode=FAST)

    numbers = frame_numbers(gray_frames(output, 320, 240), row=60)
    pieces = runs(numbers)
    assert len(pieces) == len(shots), "a frame dropped, repeated or added inside a shot"
    for piece, s in zip(pieces, shots, strict=True):
        assert len(piece) == round((s.end - s.start) * 60)
        assert abs(piece[0] - round(s.start * 60)) <= 1
    measured = probe(tools, output, count_frames=True).video.frame_count
    assert measured == len(numbers) == sum(round((s.end - s.start) * 60) for s in shots)


def test_a_montage_of_two_rates_keeps_time_and_order(tmp_path: Path) -> None:
    # 30 fps and 60 fps clips: the montage takes the rate of the clip with most screen time
    # (here the 60 fps one), so each 30 fps frame is shown twice and nothing else repeats.
    tools = find_tools()
    slow = numbered_clip(tmp_path / "slow.mp4", 10, 30)
    fast = numbered_clip(tmp_path / "fast.mp4", 10, 60)
    clips = [clip(tools, 0, slow), clip(tools, 1, fast)]
    shots = [shot(0, 2.0, 4.0), shot(1, 1.0, 5.0)]
    output = tmp_path / "montage.mp4"
    render_montage(tools, clips, shots, output, force=False, encode=FAST)

    info = probe(tools, output, count_frames=True)
    assert info.video.nominal_rate == 60
    numbers = frame_numbers(gray_frames(output, 320, 240), row=60)
    first, second = numbers[:120], numbers[120:]
    assert len(second) == 240 and info.video.frame_count == 360
    assert first == [n for n in range(60, 120) for _ in (0, 1)]  # 2 s of 30 fps, doubled
    assert second == list(range(60, 300))  # 4 s of 60 fps, untouched


def test_a_different_aspect_ratio_is_letterboxed_not_stretched(tmp_path: Path) -> None:
    tools = find_tools()
    wide = numbered_clip(tmp_path / "wide.mp4", 6, 30, width=320, height=180)  # 16:9
    square = numbered_clip(tmp_path / "square.mp4", 6, 30, width=480, height=360)  # 4:3
    clips = [clip(tools, 0, wide), clip(tools, 1, square)]
    shots = [shot(0, 1.0, 2.0), shot(1, 1.0, 2.0)]
    output = tmp_path / "montage.mp4"
    render_montage(tools, clips, shots, output, force=False, encode=FAST)

    frames = gray_frames(output, 480, 360)
    assert frames.shape[1:] == (360, 480)
    wide_part, square_part = frames[:30], frames[30:]
    # 320x180 fits 480x270: 45 px black bars above and below, picture not stretched.
    assert wide_part[:, :40].mean() < 24 and wide_part[:, 320:].mean() < 24
    assert wide_part[:, 50:310].mean() > 60
    assert frame_numbers(wide_part, row=45 + 45, scale=1.5) == list(range(30, 60))
    assert square_part[:, :40].mean() > 24  # the 4:3 clip fills the frame
    assert frame_numbers(square_part, row=90) == list(range(30, 60))


def test_unusual_paths_render(tmp_path: Path) -> None:
    tools = find_tools()
    folder = tmp_path / "vol 1" / "flüge ✈"
    folder.mkdir(parents=True)
    source = numbered_clip(folder / "flug nr. 1 ✈.mp4", 4, 30)
    output = folder / "montage ✈.mp4"
    render_montage(
        tools, [clip(tools, 0, source)], [shot(0, 1.0, 2.0)], output, force=False, encode=FAST
    )
    assert probe(tools, output, count_frames=True).video.frame_count == 30


def test_an_unwritable_output_fails_clearly_and_leaves_nothing(tmp_path: Path) -> None:
    tools = find_tools()
    source = numbered_clip(tmp_path / "numbered.mp4", 4, 30)
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("a file where the output folder should be")
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(FPVEditError):
        render_montage(
            tools,
            [clip(tools, 0, source)],
            [shot(0, 1.0, 2.0)],
            blocker / "montage.mp4",
            force=False,
            encode=FAST,
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert blocker.read_text() == "a file where the output folder should be"


def test_vertical_export_fills_a_9_by_16_frame_with_the_middle(tmp_path: Path) -> None:
    tools = find_tools()
    source = numbered_clip(tmp_path / "numbered.mp4", 4, 30)  # 320x240
    output = tmp_path / "vertical.mp4"
    render_montage(
        tools,
        [clip(tools, 0, source)],
        [shot(0, 1.0, 2.0)],
        output,
        force=False,
        encode=FAST,
        vertical=True,
    )
    info = probe(tools, output, count_frames=True)
    assert (info.video.width, info.video.height) == (136, 240)  # 9:16, as tall as the clip
    assert info.video.frame_count == 30
    frames = gray_frames(output, 136, 240)
    # Edge to edge: no black bars anywhere (the bottom half of the source is mid grey).
    assert frames[:, 130:, :].min() > 100
    # The middle 136 px of the source (x 92-228), not squeezed: bit columns 4-7 of each
    # frame number sit at their source places, shifted by 92 px.
    expected = list(range(30, 60))
    for bit in range(4, 8):
        x = 26 * bit + 13 - 92
        shown = (frames[:, 60, x] > 128).tolist()
        assert shown == [bool(n >> bit & 1) for n in expected], bit
