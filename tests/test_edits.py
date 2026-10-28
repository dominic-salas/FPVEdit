"""Learning from the pilot's edits: what an export says, how it is kept, and its weight."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpvedit.app.model import ShotList
from fpvedit.app.myratings import MyRatings
from fpvedit.montage import Moment, MomentKind, Shot
from fpvedit.rating import Quality
from fpvedit.taste import EDITS, RatedShot, Taste, _fit, _left_out

DURATIONS = {0: 100.0, 1: 60.0}


def shot(clip: int, start: float, end: float, trick: float = 0.5) -> Shot:
    quality = Quality(trick, 1.0, 0.5, 0.9, ("flip",))
    return Shot(clip, start, end, 5.0, (Moment(start + 3, MomentKind.FLIP, 3.0),), quality)


A, B, C, D = shot(0, 10, 20), shot(0, 40, 50), shot(1, 5, 15), shot(1, 30, 40)


def test_an_untouched_montage_is_all_exported() -> None:
    shots = ShotList([A, B, C], [A, B, C, D], DURATIONS)
    assert sorted(what for _, what in shots.outcome()) == ["exported"] * 3


def test_outcome_names_each_edit() -> None:
    shots = ShotList([A, B, C], [A, B, C, D], DURATIONS)
    shots.remove(1)  # B
    shots.add(D)
    shots.trim(0, end=-2.0)  # A, shorter
    said = {(s.clip, s.start): what for s, what in shots.outcome()}
    assert said == {(0, 10): "trimmed", (0, 40): "removed", (1, 5): "exported", (1, 30): "added"}


def test_an_undone_edit_says_nothing() -> None:
    shots = ShotList([A, B, C], [A, B, C, D], DURATIONS)
    shots.remove(0)
    shots.undo()
    assert {what for _, what in shots.outcome()} == {"exported"}


def test_a_swap_removes_one_and_adds_the_other() -> None:
    other = shot(1, 32, 44)
    shots = ShotList([A, C, D], [A, C, D, other], DURATIONS)
    shots.replace(2, other)
    said = {(s.clip, s.start): what for s, what in shots.outcome()}
    assert said[(1, 30)] == "removed" and said[(1, 32)] == "added"


def test_edits_are_kept_and_learned_from_and_stars_win(tmp_path: Path) -> None:
    clip = tmp_path / "a.ts"
    mine = MyRatings(tmp_path / "my-ratings.jsonl")
    mine.note_edits([(clip, A, "removed"), (clip, B, "added"), (clip, C, "trimmed")])
    mine.rate(clip, B, 1)
    again = MyRatings(tmp_path / "my-ratings.jsonl")
    assert (tmp_path / "my-edits.jsonl").is_file()
    assert again.edit_count == 2  # the trim is only kept
    rated = {r.start: r for r in again.rated_shots()}
    assert rated[A.start].stars == EDITS["removed"][0]
    assert rated[A.start].weight == EDITS["removed"][1]
    assert rated[B.start].stars == 1 and rated[B.start].weight == 1.0  # own stars win
    assert C.start not in rated
    again.clear()
    assert not (tmp_path / "my-edits.jsonl").exists() and not again.rated_shots()


def test_keeping_a_shot_again_does_not_undo_an_earlier_add(tmp_path: Path) -> None:
    clip = tmp_path / "a.ts"
    mine = MyRatings(tmp_path / "my-ratings.jsonl")
    mine.note_edits([(clip, A, "added")])
    mine.note_edits([(clip, A, "exported")])
    assert MyRatings(tmp_path / "my-ratings.jsonl").rated_shots()[0].weight == EDITS["added"][1]


def test_weights_count_like_repeated_samples() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.random((12, 4)), rng.integers(1, 6, 12).astype(float)
    weighted = _fit(x, y, np.r_[2.0, np.ones(11)], 3.0)
    repeated = _fit(np.r_[x[:1], x], np.r_[y[:1], y], np.ones(13), 3.0)
    assert weighted.predict(x[5]) == pytest.approx(repeated.predict(x[5]), abs=0.05)


def test_the_closed_form_leave_one_out_matches_refitting() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((90, 6))
    y = np.clip(1 + 4 * x[:, 0] + rng.normal(0, 0.3, 90), 1, 5)
    w = rng.choice(np.array([1.0, 0.3, 0.6]), 90).astype(np.float64)
    fast = _left_out(x, y, w, 10.0)
    keep = ~np.eye(90, dtype=bool)
    slow = np.array(
        [_fit(x[keep[i]], y[keep[i]], w[keep[i]], 10.0).predict(x[i]) for i in range(90)]
    )
    assert np.mean(np.abs(fast - slow)) < 0.02


def test_edits_alone_teach_a_taste() -> None:
    # This pilot removes every shot without a double and adds every one with one.
    rng = np.random.default_rng(2)
    rated = []
    for _ in range(60):
        double = bool(rng.random() < 0.5)
        q = Quality(float(rng.random()), 1.0, 0.5, 0.9, ("flip",))
        tricks = ("double back flip",) if double else ("back flip",)
        what = "added" if double else "removed"
        rated.append(RatedShot(q, tricks, 10.0, EDITS[what][0], weight=EDITS[what][1]))
    taste = Taste.fit(rated)
    assert taste.learned and taste.error < taste.base_error
    plain = Quality(0.9, 1.0, 0.5, 0.9, ("flip",))
    assert taste.predict(plain, ("double roll",), 10.0) > taste.predict(plain, ("roll",), 10.0)


def test_the_primal_and_dual_fits_agree() -> None:
    rng = np.random.default_rng(5)
    x, y = rng.random((40, 6)), rng.integers(1, 6, 40).astype(np.float64)
    w = rng.random(40) + 0.2
    big = _fit(x, y, w, 3.0)  # 40 samples > 6 features: primal
    small = _fit(x[:, :6], y, w, 3.0)
    wide = _fit(np.hstack([x, np.zeros((40, 40))]), y, w, 3.0)  # 40 <= 46 features: dual
    assert np.allclose(big.weights, small.weights)
    assert np.allclose(wide.weights[:6], big.weights)


def test_a_damaged_edits_line_is_skipped(tmp_path: Path) -> None:
    clip = tmp_path / "a.ts"
    MyRatings(tmp_path / "my-ratings.jsonl").note_edits([(clip, A, "removed")])
    with (tmp_path / "my-edits.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n[1, 2]\n")
    assert MyRatings(tmp_path / "my-ratings.jsonl").edit_count == 1
