"""Montage renders measured from the output file itself, never from the renderer's own sums.

Each source frame carries its own number, drawn as 12 black or white columns (bit k of
the frame number in column k), wide enough to survive H.264. Decoding the montage and
reading the numbers back shows exactly which source frames it holds: none dropped or
repeated inside a shot, nothing extra at the joins, shots in the order given.
"""

from __future__ import annotations

import json
import re
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from fpvedit.errors import FPVEditError
from fpvedit.media import Tools, find_tools, frame_times, probe, probe_audio
from fpvedit.montage import Shot
from fpvedit.pipeline import Clip, render_montage
from fpvedit.render import MAX_BOOST, EncodeConfig, loudness_gain
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


def test_vertical_export_fills_1080x1920_with_the_middle(tmp_path: Path) -> None:
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
    assert (info.video.width, info.video.height) == (1080, 1920)  # what phone platforms expect
    assert info.video.frame_count == 30
    frames = gray_frames(output, 1080, 1920)
    # Edge to edge: no black bars anywhere (the bottom half of the source is mid grey).
    assert frames[:, 1040:, :].min() > 100
    # Scaled 8x to fill the height and the middle 1080 of 2560 px kept (x 740-1820), not
    # squeezed: bit columns 4-7 of each frame number sit at their source places.
    expected = list(range(30, 60))
    for bit in range(4, 8):
        x = (26 * bit + 13) * 8 - 740
        shown = (frames[:, 480, x] > 128).tolist()
        assert shown == [bool(n >> bit & 1) for n in expected], bit


def test_the_video_has_the_structure_youtube_asks_for(tmp_path: Path) -> None:
    tools = find_tools()
    source = numbered_clip(tmp_path / "numbered.mp4", 4, 60)
    output = tmp_path / "out.mp4"
    # The default encoder: FAST's preset turns off what High profile adds.
    render_montage(
        tools,
        [clip(tools, 0, source)],
        [shot(0, 0.5, 3.5)],
        output,
        force=False,
        encode=EncodeConfig(),
    )
    stream = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json"),
            *("-show_entries", "stream=profile,has_b_frames:packet=flags,pts_time"),
            *("-show_packets", str(output)),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found = json.loads(stream)
    assert found["streams"][0]["profile"] == "High"
    assert found["streams"][0]["has_b_frames"] == 2
    keys = sorted(float(p["pts_time"]) for p in found["packets"] if "K" in p["flags"])
    assert len(keys) >= 6
    assert max(b - a for a, b in pairwise(keys)) <= 0.5 + 1e-3  # half the frame rate


def loudness_of(path: Path) -> tuple[float, float]:
    """Integrated loudness and true peak of a file, read by FFmpeg's EBU R128 meter."""
    log = subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0"),
            *("-af", "ebur128=peak=true", "-f", "null", "-"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stderr
    summary = log[log.rindex("Summary:") :]
    integrated = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    assert integrated is not None and peak is not None
    return float(integrated.group(1)), float(peak.group(1))


def tone_clip(path: Path, seconds: float, volume_db: float) -> Path:
    picture = numbered_clip(path.with_name(f"{path.stem}-picture.mp4"), seconds, 60)
    tone = f"sine=f=440:r=48000:d={seconds},volume={volume_db}dB"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(picture), "-f", "lavfi", "-i", tone),
            *("-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k"),
            str(path),
        ],
        check=True,
    )
    return path


def test_a_loud_track_is_turned_down_to_the_target(tmp_path: Path) -> None:
    tools = find_tools()
    source = numbered_clip(tmp_path / "numbered.mp4", 8, 60)
    song = tmp_path / "song.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=8,volume=14dB", str(song)],
        check=True,
    )
    assert loudness_of(song)[0] > -12  # louder than the target before
    output = tmp_path / "out.mp4"
    render_montage(
        tools,
        [clip(tools, 0, source)],
        [shot(0, 0.5, 7.5)],
        output,
        force=False,
        encode=FAST,
        music=[probe_audio(tools, song)],
    )
    integrated, peak = loudness_of(output)
    assert integrated == pytest.approx(-14, abs=1)
    assert peak <= -0.5


def test_quiet_recorded_sound_is_raised_but_not_past_the_limit(tmp_path: Path) -> None:
    tools = find_tools()
    source = tone_clip(tmp_path / "quiet.mp4", 8, -40)
    before = loudness_of(source)[0]
    output = tmp_path / "out.mp4"
    render_montage(
        tools, [clip(tools, 0, source)], [shot(0, 0.5, 7.5)], output, force=False, encode=FAST
    )
    assert loudness_of(output)[0] == pytest.approx(before + MAX_BOOST, abs=1)


@pytest.mark.parametrize(
    ("integrated", "peak", "gain"),
    [
        (-8.0, -0.2, -6.0),  # a loud master comes down to the target
        (-20.0, -10.0, 6.0),  # a quiet one comes up to it
        (-20.0, -3.0, 2.0),  # peaks stop it first
        (-60.0, -50.0, MAX_BOOST),  # hiss is not raised to full level
        (float("-inf"), float("-inf"), 0.0),  # silence is left alone
    ],
)
def test_loudness_gain(integrated: float, peak: float, gain: float) -> None:
    assert loudness_gain(integrated, peak) == pytest.approx(gain)


def beeping_clip(path: Path, seconds: float, beep: float, audio_late: float) -> Path:
    """Numbered frames at 60 fps, and a 1 kHz beep at ``beep`` s of the picture's time in
    audio that starts ``audio_late`` s after the picture (MPEG-TS, as DVRs record)."""
    picture = numbered_clip(path.with_name("picture.mp4"), seconds, 60)
    on = f"between(t,{beep - audio_late},{beep - audio_late + 0.1})"
    tone = f"sine=f=1000:r=48000:d={seconds},volume='{on}':eval=frame"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(picture)),
            *("-itsoffset", f"{audio_late}", "-f", "lavfi", "-i", tone),
            *("-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(path)),
        ],
        check=True,
    )
    return path


def sound_of(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype="<i2")


@pytest.mark.parametrize("audio_late", [0.1, -0.1])
def test_the_recordings_sound_stays_with_its_picture(tmp_path: Path, audio_late: float) -> None:
    tools = find_tools()
    source = beeping_clip(tmp_path / "beep.ts", 12, beep=5.0, audio_late=audio_late)
    clips = [clip(tools, 0, source)]
    shots = [shot(0, 2.0, 4.0), shot(0, 4.8, 6.0)]  # the beep is 0.2 s into the second
    output = tmp_path / "montage.mp4"
    render_montage(tools, clips, shots, output, force=False, encode=FAST)

    # Where the beep really is in the recording, on the picture's timeline (the tone is
    # switched on per block of samples, so a few ms after 5.0 s).
    source_info = probe(tools, source)
    assert source_info.audio is not None and source_info.audio.start_time is not None
    assert source_info.video.start_time is not None
    late = source_info.audio.start_time - source_info.video.start_time
    beep = late + onset(sound_of(source))
    info = probe(tools, output)
    assert info.audio is not None and info.audio.codec == "aac"
    sound = sound_of(output)
    expected = 2.0 + (beep - 4.8)
    heard = onset(sound)
    assert abs(heard - expected) <= 0.002, f"the beep is at {heard:.4f} s, expected {expected:.4f}"
    assert abs(len(sound) / 48000 - 3.2) <= 0.03  # as long as the picture


def onset(sound: np.ndarray) -> float:
    # The tone is about 4100 loud; mono to stereo and back makes it about 2900.
    loud = np.flatnonzero(np.abs(sound) > 1000)
    assert loud.size, "the beep is missing"
    return float(loud[0] / 48000)
