from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvedit.app.myratings import MyRatings
from fpvedit.montage import Moment, MomentKind, Shot
from fpvedit.rating import Quality
from fpvedit.taste import MIN_RATINGS, RatedShot, Taste


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
    # Stars that nothing explains: the best guess is the pilot's average, so predictions
    # stay close to it for every kind of shot instead of following the measurements.
    average = float(np.mean([r.stars for r in rated]))
    predictions = [taste.predict(q, t, 10.0) for q, t in shots(30, seed=5)]
    assert max(abs(p - average) for p in predictions) < 0.6
    assert taste.error >= 1.0  # and it does not claim to predict noise well


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


def test_a_taste_for_what_is_in_view_is_learned_from_the_picture() -> None:
    # This pilot likes shots where the picture shows a person (the 2nd description),
    # whatever was measured; the measurements alone cannot tell those shots apart.
    rng = np.random.default_rng(7)
    rated = []
    for i in range(40):
        person = i % 2 == 0
        picture = tuple(float(v) for v in rng.normal(0.05, 0.01, 16))
        picture = (picture[0], 0.12 if person else 0.02, *picture[2:])
        q = Quality(0.5, 1.0, 0.5, 0.9, ("flip",), picture=picture)
        rated.append(RatedShot(q, ("flip",), 10.0, 5 if person else 2))
    taste = Taste.fit(rated)
    assert taste.uses_picture and taste.error < 0.5 < taste.base_error
    seen = Quality(0.5, 1.0, 0.5, 0.9, ("flip",), picture=(0.05, 0.12, *([0.05] * 14)))
    empty = Quality(0.5, 1.0, 0.5, 0.9, ("flip",), picture=(0.05, 0.02, *([0.05] * 14)))
    assert taste.predict(seen, ("flip",), 10.0) > 4.0
    assert taste.predict(empty, ("flip",), 10.0) < 3.0
    # Without the image model the shot has no picture: the plain model (or the app) answers.
    blind = Quality(0.5, 1.0, 0.5, 0.9, ("flip",))
    assert 1.0 <= taste.predict(blind, ("flip",), 10.0) <= 5.0


def test_keep_rated_uses_own_stars_and_the_learned_taste(tmp_path: Path) -> None:
    from fpvedit.media import Tools
    from fpvedit.montage import MontageConfig
    from fpvedit.pipeline import MontagePlan, keep_rated

    class Fixed:
        """Ratings that like close passes and dislike everything else, plus one own rating."""

        def __init__(self, own: Shot) -> None:
            self.own = own

        def stars(self, path: Path, shot: Shot) -> int | None:
            return 1 if shot == self.own else None

        def rated_shots(self) -> list[RatedShot]:
            return [
                RatedShot(quality(t, close=c, crash=False), ("flip",), 10.0, 5 if c else 1)
                for t, c in [(x / 20, x % 2 == 0) for x in range(20)]
            ]

    close = Shot(0, 0, 9, 3.0, (), quality(0.2, close=True, crash=False))
    far = Shot(0, 20, 29, 3.0, (), quality(0.9, close=False, crash=False))
    own = Shot(0, 40, 49, 3.0, (), quality(0.9, close=True, crash=False))
    plan = MontagePlan((), (), (close, far, own), (close, far, own), MontageConfig(), ())
    tools = Tools(ffmpeg="ffmpeg", ffprobe="ffprobe", ffmpeg_major=7)

    class Info:
        path = tmp_path / "flight.mp4"

    plan = MontagePlan((Info(),), (), plan.candidates, plan.shots, plan.config, ())  # type: ignore[arg-type]
    judged = keep_rated(tools, plan, min_stars=3, scenery=False, ratings=Fixed(own))
    assert [s.start for s in judged.shots] == [0]  # the close pass, predicted 4-5 stars
    assert sorted(s.start for s in judged.left_out) == [20, 40]  # far: predicted low; own: 1
    assert next(s for s in judged.left_out if s.start == 40).quality.stars == 1  # type: ignore[union-attr]
