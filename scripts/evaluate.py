"""Score an edit (kept intervals) against hand annotations of the same flight.

Usage:
    python scripts/evaluate.py ANNOTATION.json ANALYSIS.json [ANNOTATION.json ANALYSIS.json ...]

ANALYSIS.json is the ``--keep-intermediates`` file written by fpv-editor; only its
``selection.intervals`` list is used. Annotation times are m:ss strings (1 s resolution) or
seconds, so event edges get ``EDGE_TOLERANCE_S`` of slack before a cut counts as inside an event.

Two kinds of annotation file exist. Hand-labelled moments (``keep``: include, optional,
exclude) are scored per event. A ``"kind": "reference_edit"`` file lists the stretches a human
editor used (include) and left out (editor_cut); it is scored as time overlap with that edit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

EDGE_TOLERANCE_S = 0.5
HIT_COVERAGE = 0.8


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    label: str
    keep: str
    continuous: bool
    note: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_time(text: str | float) -> float:
    if isinstance(text, int | float):
        return float(text)
    parts = [float(part) for part in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def load_events(path: Path) -> tuple[str, float, list[Event]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind", "moments") not in ("moments", "reference_edit"):
        raise ValueError(f"{path.name}: unknown annotation kind {data['kind']!r}")
    events = []
    for raw in data["events"]:
        start, end = parse_time(raw["start"]), parse_time(raw["end"])
        if end <= start:
            raise ValueError(f"{path.name}: event {raw['start']}-{raw['end']} ends too early")
        events.append(
            Event(
                start=start,
                end=end,
                label=raw["label"],
                keep=raw["keep"],
                continuous=bool(raw.get("continuous", False)),
                note=raw.get("note", ""),
            )
        )
    return data["flight"], float(data["duration_s"]), events


def load_intervals(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(float(start), float(end)) for start, end in data["selection"]["intervals"]]


def overlap(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


def cuts_inside(intervals: list[tuple[float, float]], start: float, end: float) -> int:
    """Segment boundaries strictly inside the event (ignoring the edge tolerance)."""
    lo, hi = start + EDGE_TOLERANCE_S, end - EDGE_TOLERANCE_S
    return sum(lo < edge < hi for interval in intervals for edge in interval)


def fmt(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:04.1f}"


def evaluate_reference_edit(
    flight: str, duration: float, events: list[Event], intervals: list[tuple[float, float]]
) -> dict[str, float]:
    reference = [(e.start, e.end) for e in events if e.keep == "include"]
    kept = sum(b - a for a, b in intervals)
    ref_total = sum(b - a for a, b in reference)
    common = sum(overlap(intervals, a, b) for a, b in reference)
    union = kept + ref_total - common
    edges = [edge for a, b in intervals for edge in (a, b) if 0.5 < edge < duration - 0.5]
    ref_edges = [edge for a, b in reference for edge in (a, b) if 0.5 < edge < duration - 0.5]
    errors = [min((abs(edge - other) for other in edges), default=duration) for edge in ref_edges]
    print(
        f"\n== {flight} (reference edit): kept {fmt(kept)} of {fmt(duration)} "
        f"({kept / duration:.0%}) in {len(intervals)} segments; editor kept {fmt(ref_total)} "
        f"({ref_total / duration:.0%}) in {len(reference)} segments"
    )
    for a, b in reference:
        print(
            f"  editor {fmt(a)}-{fmt(b)} kept {overlap(intervals, a, b) / (b - a):4.0%}, "
            f"{cuts_inside(intervals, a, b)} cut(s) inside"
        )
    summary = {
        "retention": kept / duration,
        "precision": common / kept if kept else float("nan"),
        "recall": common / ref_total if ref_total else float("nan"),
        "iou": common / union if union else float("nan"),
        "median_cut_error_s": sorted(errors)[len(errors) // 2] if errors else float("nan"),
        "segments": float(len(intervals)),
    }
    print("  " + ", ".join(f"{k}={v:.2f}" for k, v in summary.items()))
    return summary


def evaluate(annotation: Path, analysis: Path) -> dict[str, float]:
    flight, duration, events = load_events(annotation)
    intervals = load_intervals(analysis)
    if json.loads(annotation.read_text(encoding="utf-8")).get("kind") == "reference_edit":
        return evaluate_reference_edit(flight, duration, events, intervals)
    kept = sum(b - a for a, b in intervals)
    print(
        f"\n== {flight}: kept {fmt(kept)} of {fmt(duration)} ({kept / duration:.0%}), "
        f"{len(intervals)} segments"
    )
    include = [e for e in events if e.keep in ("include", "optional")]
    exclude = [e for e in events if e.keep == "exclude"]
    hits = include_time = include_kept = 0.0
    broken = 0
    for event in include:
        covered = overlap(intervals, event.start, event.end)
        fraction = covered / event.duration
        cuts = cuts_inside(intervals, event.start, event.end)
        hit = fraction >= HIT_COVERAGE
        hits += hit
        include_time += event.duration
        include_kept += covered
        flag = "ok " if hit and not cuts else "MISS" if not hit else "CUT"
        if event.continuous and cuts:
            broken += 1
        details = f", {cuts} cut(s) inside" if cuts else ""
        details += " (continuous)" if event.continuous else ""
        details += " (optional)" if event.keep == "optional" else ""
        print(
            f"  {flag} {fmt(event.start)}-{fmt(event.end)} {event.label:<9} "
            f"kept {fraction:4.0%}{details}"
        )
    exclude_time = exclude_kept = 0.0
    for event in exclude:
        covered = overlap(intervals, event.start, event.end)
        exclude_time += event.duration
        exclude_kept += covered
        flag = "ok " if covered <= EDGE_TOLERANCE_S * 2 else "LEAK"
        print(
            f"  {flag} {fmt(event.start)}-{fmt(event.end)} {event.label:<9} kept "
            f"{covered / event.duration:4.0%} (should be cut)"
        )
    labelled = sum(e.duration for e in events)
    unlabelled_kept = kept - include_kept - exclude_kept
    summary = {
        "retention": kept / duration,
        "event_recall": hits / len(include) if include else float("nan"),
        "include_time_recall": include_kept / include_time if include_time else float("nan"),
        "exclude_time_kept": exclude_kept / exclude_time if exclude_time else 0.0,
        "continuous_broken": float(broken),
        "unlabelled_kept_share": unlabelled_kept / max(1e-9, duration - labelled),
        "segments": float(len(intervals)),
    }
    print("  " + ", ".join(f"{k}={v:.2f}" for k, v in summary.items()))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pairs", nargs="+", type=Path, help="ANNOTATION ANALYSIS pairs")
    args = parser.parse_args(argv)
    if len(args.pairs) % 2:
        parser.error("pass annotation and analysis files in pairs")
    for annotation, analysis in zip(args.pairs[::2], args.pairs[1::2], strict=True):
        evaluate(annotation, analysis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
