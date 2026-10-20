from __future__ import annotations

from pathlib import Path

import numpy as np

from fpv_editor.app.myratings import MyRatings
from fpv_editor.montage import Moment, MomentKind, Shot
from fpv_editor.rating import Quality
from fpv_editor.taste import MIN_RATINGS, RatedShot, Taste


def quality(trick: float, close: bool, crash: bool) -> Quality:
    reasons = (
        "flip",
        *(("very close",) if close else ()),
        *(("crash right after",) if crash else ()),
    )
    return Quality(trick, 0.6 if crash else 1.0, 0.9 if close else 0.4, 0.9, reasons)


def shots(count: int, seed: int = 0) -> list[tuple[Quality, tuple[str, ...]]]:
    rng = np.random.default_rng(seed)
    return [
        (quality(rng.random(), rng.random() < 0.5, rng.random() < 0.3), ("flip",))
        for _ in range(count)
    ]


def test_too_few_ratings_keep_the_apps_own_stars() -> None:
    rated = [RatedShot(q, t, 10.0, 5) for q, t in shots(MIN_RATINGS - 1)]
    taste = Taste.fit(rated)
    assert not taste.learned
    q, t = shots(1, seed=9)[0]
    assert taste.predict(q, t, 10.0) == q.stars


def test_a_consistent_taste_is_learned() -> None:
    # This pilot loves close passes and hates crashes, whatever the app thinks.
    def stars(q: Quality) -> int:
        return (
            3
            + (2 if "very close" in q.reasons else 0)
            - (2 if "crash right after" in q.reasons else 0)
        )

    rated = [RatedShot(q, t, 10.0, stars(q)) for q, t in shots(40)]
    taste = Taste.fit(rated)
    assert taste.learned and taste.error < taste.base_error
    close = quality(0.2, close=True, crash=False)
    crashed = quality(0.9, close=False, crash=True)
    assert taste.predict(close, ("flip",), 10.0) > 4.0
    assert taste.predict(crashed, ("flip",), 10.0) < 2.0


def test_random_ratings_teach_nothing() -> None:
    rng = np.random.default_rng(3)
    rated = [RatedShot(q, t, 10.0, int(rng.integers(1, 6))) for q, t in shots(40, seed=4)]
    taste = Taste.fit(rated)
    q, t = shots(1, seed=5)[0]
    # Whatever it chose, it never moves far from the app's own stars on noise.
    assert abs(taste.predict(q, t, 10.0) - q.stars) < 1.5


def test_taste_replaces_the_stars() -> None:
    q = quality(0.9, close=True, crash=False)
    from dataclasses import replace

    assert replace(q, taste=1.4).stars == 1
    assert replace(q, taste=4.6).stars == 5


def test_ratings_are_learned_from_and_can_be_cleared(tmp_path: Path) -> None:
    file = tmp_path / "my-ratings.jsonl"
    mine = MyRatings(file)
    recording = tmp_path / "flight.mp4"
    for i in range(3):
        shot = Shot(
            0,
            i * 10.0,
            i * 10.0 + 8,
            1.0,
            (Moment(i * 10.0 + 2, MomentKind.FLIP, 3.0, "back flip"),),
            quality(0.5, close=i == 1, crash=False),
        )
        mine.rate(recording, shot, i + 2)
    mine.unrate(recording, shot)  # the last one undone
    again = MyRatings(file)
    rated = again.rated_shots()
    assert sorted(r.stars for r in rated) == [2, 3]
    assert rated[0].tricks == ("back flip",) and rated[0].duration == 8.0
    again.clear()
    assert again.count == 0 and not file.exists()
    assert MyRatings(file).rated_shots() == []
