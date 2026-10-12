"""Turn a finished edit plus its raw recordings into reference-edit annotations.

Usage:
    python scripts/match_edit.py EDIT.mp4 RAW1.ts [RAW2.ts ...] --out-dir eval/annotations

Every frame of the edit is matched to the most similar frame of any raw recording
(small, contrast-normalised greyscale fingerprints of the active picture). Runs of edit
frames that map to consecutive raw frames become segments; each raw recording then
gets a ``"kind": "reference_edit"`` annotation, ``<raw name>.reference.json``, listing
what the editor kept (include) and left out (editor_cut), which ``scripts/evaluate.py``
scores against.

The match assumes hard cuts at normal speed, which is how Dom edits. Titles,
transitions, speed ramps, crops or colour grades show up as weak or short matches.
They are reported, and the raw time they point to is marked ``ignore`` (unknown)
instead of being guessed at. An existing reference edit is only replaced with
``--force``, and hand-made annotations are never replaced. Footage is only read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fpv_editor.media import MediaInfo, Tools, find_tools, probe
from fpv_editor.picture import decoded_size, detect_active_area, sample_frames

FingerprintArray = NDArray[np.float32]

FINGERPRINT_SIZE = (48, 36)
MAX_RATE = 60.0
MIN_CORRELATION = 0.9  # a real match of the same frame scores ~0.99
MIN_SEGMENT_S = 0.5
JUMP_S = 0.75  # an offset change larger than this starts a new segment
CHUNK = 1000


@dataclass(frozen=True)
class Segment:
    raw: int  # index into the raw recordings
    edit_start: float
    edit_end: float
    raw_start: float
    raw_end: float
    mean_correlation: float

    @property
    def duration(self) -> float:
        return self.edit_end - self.edit_start


def fingerprints(tools: Tools, info: MediaInfo, rate: float) -> FingerprintArray:
    """One zero-mean, unit-length fingerprint per frame at ``rate`` frames per second."""
    area = detect_active_area(sample_frames(tools, info), decoded_size(info))
    width, height = FINGERPRINT_SIZE
    args = [
        *(tools.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-i", str(info.path)),
        *("-map", f"0:{info.video.index}", "-an", "-sn", "-dn"),
        *("-vf", f"{area.crop_filter},fps={rate},scale={width}:{height}:flags=area,format=gray"),
        *("-f", "rawvideo", "-pix_fmt", "gray", "-"),
    ]
    raw = subprocess.run(args, check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width * height).astype(np.float32)
    frames -= frames.mean(axis=1, keepdims=True)
    frames /= np.linalg.norm(frames, axis=1, keepdims=True) + 1e-6
    return frames


def best_matches(
    edit: FingerprintArray, raws: list[FingerprintArray]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    """For each edit frame: which raw recording, which frame, and the correlation."""
    library = np.concatenate(raws)
    starts = np.cumsum([0] + [len(r) for r in raws])
    index = np.empty(len(edit), dtype=np.int64)
    score = np.empty(len(edit), dtype=np.float32)
    for begin in range(0, len(edit), CHUNK):
        similarity = edit[begin : begin + CHUNK] @ library.T
        best = similarity.argmax(axis=1)
        index[begin : begin + len(best)] = best
        score[begin : begin + len(best)] = similarity[np.arange(len(best)), best]
    which = np.searchsorted(starts, index, side="right") - 1
    return which, index - starts[which], score


def segments_from_matches(
    which: NDArray[np.int64], frame: NDArray[np.int64], score: NDArray[np.float32], rate: float
) -> list[Segment]:
    """Split the edit where the source recording or the time offset jumps."""
    offset = frame - np.arange(len(frame))
    jump = max(2, round(JUMP_S * rate))
    confirm = max(2, round(0.5 * rate))
    bounds = []
    start = 0
    for i in range(1, len(frame) + 1):
        if i < len(frame):
            same_raw = which[i] == which[start]
            reference = np.median(offset[start:i])
            if same_raw and abs(offset[i] - reference) <= jump:
                continue
            # A single noisy match is not a cut: the next frames must agree.
            following = offset[i : i + confirm]
            if same_raw and np.median(np.abs(following - reference)) <= jump:
                continue
        bounds.append((start, i))
        start = i
    result = []
    for begin, end in bounds:
        shift = int(np.median(offset[begin:end]))
        result.append(
            Segment(
                raw=int(which[begin]),
                edit_start=begin / rate,
                edit_end=end / rate,
                raw_start=(begin + shift) / rate,
                raw_end=(end + shift) / rate,
                mean_correlation=float(score[begin:end].mean()),
            )
        )
    return result


def annotation(
    raw: MediaInfo,
    kept: list[Segment],
    edit: MediaInfo,
    rate: float,
    unsure: list[Segment] | None = None,
) -> dict[str, object]:
    """Reference-edit annotation; ``unsure`` segments (weak matches) become ``ignore``."""
    duration = raw.duration or 0.0
    marked = [(s, True) for s in kept] + [(s, False) for s in unsure or []]
    events: list[dict[str, object]] = []
    cursor = 0.0
    for segment, used in sorted(marked, key=lambda item: item[0].raw_start):
        start = max(cursor, segment.raw_start, 0.0)
        end = min(duration, segment.raw_end)
        if end <= start:
            continue
        if start > cursor + 1e-3:
            events.append(_cut(cursor, start))
        note = (
            f"edit {segment.edit_start:.2f}-{segment.edit_end:.2f} s, "
            f"match {segment.mean_correlation:.3f}"
        )
        if used:
            event: dict[str, object] = {
                "label": "editor_kept",
                "keep": "include",
                "continuous": True,
            }
        else:
            event = {"label": "unmatched", "keep": "ignore"}
            note += " (weak or short match: unknown whether it was used)"
        events.append({"start": round(start, 2), "end": round(end, 2), **event, "note": note})
        cursor = end
    if duration > cursor + 1e-3:
        events.append(_cut(cursor, duration))
    return {
        "flight": raw.path.stem,
        "source_file": raw.path.name,
        "duration_s": round(duration, 2),
        "annotator": f"derived from the finished edit {edit.path.name}",
        "annotated_on": date.today().isoformat(),
        "kind": "reference_edit",
        "complete": not unsure,
        "notes": (
            f"Made by scripts/match_edit.py: every edit frame matched to the raw recordings "
            f"with {FINGERPRINT_SIZE[0]}x{FINGERPRINT_SIZE[1]} normalised greyscale "
            f"fingerprints at {rate:g} fps. 'include' = used in the edit, 'editor_cut' = left "
            "out (reason unknown)."
        ),
        "events": events,
    }


def _cut(start: float, end: float) -> dict[str, object]:
    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "label": "editor_cut",
        "keep": "editor_cut",
        "note": "not used in the edit",
    }


def _is_reference_edit(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("kind") == "reference_edit")
    except (OSError, ValueError, AttributeError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("edit", type=Path, help="the finished edit")
    parser.add_argument("raws", nargs="+", type=Path, help="raw recordings the edit was cut from")
    parser.add_argument("--out-dir", type=Path, required=True, help="where annotations go")
    parser.add_argument("--force", action="store_true", help="replace existing annotations")
    args = parser.parse_args(argv)

    tools = find_tools()
    edit = probe(tools, args.edit)
    raws = [probe(tools, path) for path in args.raws]
    targets = [args.out_dir / f"{raw.path.stem}.reference.json" for raw in raws]
    if len(set(targets)) != len(targets):
        print("two raw recordings share a file name; rename one first", file=sys.stderr)
        return 1
    for target in targets:
        if not target.exists():
            continue
        if not args.force:
            print(f"refusing to replace {target} (use --force)", file=sys.stderr)
            return 1
        if not _is_reference_edit(target):
            print(f"refusing to replace {target}: it is not a reference edit", file=sys.stderr)
            return 1

    rates = [r.video.frame_rate or MAX_RATE for r in (edit, *raws)]
    rate = min(MAX_RATE, *rates)
    print(f"fingerprinting at {rate:g} fps ...", file=sys.stderr)
    try:
        edit_prints = fingerprints(tools, edit, rate)
        raw_prints = [fingerprints(tools, raw, rate) for raw in raws]
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
        print(f"decoding failed: {detail or exc}", file=sys.stderr)
        return 1
    which, frame, score = best_matches(edit_prints, raw_prints)
    found = segments_from_matches(which, frame, score, rate)

    kept: list[list[Segment]] = [[] for _ in raws]
    unsure: list[list[Segment]] = [[] for _ in raws]
    for segment in found:
        good = segment.duration >= MIN_SEGMENT_S and segment.mean_correlation >= MIN_CORRELATION
        status = "kept" if good else "SKIPPED (short or weak match)"
        print(
            f"edit {segment.edit_start:7.2f}-{segment.edit_end:7.2f}  "
            f"{raws[segment.raw].path.name} {segment.raw_start:7.2f}-{segment.raw_end:7.2f}  "
            f"match {segment.mean_correlation:.3f}  {status}"
        )
        (kept if good else unsure)[segment.raw].append(segment)
    median = float(np.median(score)) if len(score) else float("nan")
    print(f"median frame match {median:.3f}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for raw, segments, weak, target in zip(raws, kept, unsure, targets, strict=True):
        document = annotation(raw, segments, edit, rate, weak)
        target.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        used = sum(min(s.raw_end, raw.duration or s.raw_end) - s.raw_start for s in segments)
        print(f"wrote {target} ({len(segments)} segments, {used:.1f} s used)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
