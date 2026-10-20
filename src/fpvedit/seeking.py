"""Where to seek in a recording for a time on the analysis timeline."""

from __future__ import annotations

from fpvedit.media import MediaInfo


def file_seconds(info: MediaInfo, seconds: float) -> float:
    """``seconds`` on the analysis timeline (seconds since the first video frame) as a
    position for FFmpeg's input ``-ss``, which counts from the start of the file."""
    video_start = info.video.start_time if info.video.start_time is not None else info.start_time
    quarter_frame = 0.25 / float(info.video.nominal_rate or 30)
    return max(0.0, seconds + video_start - info.start_time - quarter_frame)
