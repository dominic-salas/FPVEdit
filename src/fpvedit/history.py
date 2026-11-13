"""The pilot's saved flight scores, kept on this PC, and what they say over time.

Many recordings carry no real date, so a flight's place in the history is the order it was
first scored in, by file name within one import. The first import has no trustworthy order
and is the starting level, not progress. A flight is identified by its content, so renaming
a file or adding it twice never counts it twice.
"""

from __future__ import annotations

import hashlib
import sqlite3
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from fpvedit.scoring import FlightScore, ScoringConfig, Status, disqualify

SAMPLE_BYTES = 1 << 20
LEVEL_FLIGHTS = 10  # the current level is the median of this many flights
COMPARE_FLIGHTS = 5  # recent flights compared with the ones before them
NOTABLE_CHANGE = 5.0  # points a part must move to be worth mentioning

COLUMNS = (
    "conditions",
    "flying_s",
    "freestyle",
    "flow",
    "difficulty",
    "variety",
    "linking",
    "continuity",
    "smoothness",
    "crashes",
    "turns",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    batch INTEGER NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    conditions TEXT NOT NULL,
    flying_s REAL NOT NULL,
    freestyle REAL NOT NULL,
    flow REAL NOT NULL,
    difficulty REAL NOT NULL,
    variety REAL NOT NULL,
    linking REAL NOT NULL,
    continuity REAL NOT NULL,
    smoothness REAL NOT NULL,
    crashes INTEGER NOT NULL,
    turns INTEGER NOT NULL,
    disqualified TEXT NOT NULL,
    count_anyway INTEGER NOT NULL DEFAULT 0,
    longest_line_s REAL
)
"""


@dataclass(frozen=True)
class Flight:
    id: int
    key: str
    name: str
    batch: int
    version: int
    status: str
    conditions: str
    flying_s: float
    freestyle: float
    flow: float
    difficulty: float
    variety: float
    linking: float
    continuity: float
    smoothness: float
    crashes: int
    turns: int
    disqualified: str
    count_anyway: bool
    longest_line_s: float | None = None  # the longest line found in this recording

    @property
    def counts(self) -> bool:
        """Whether this flight belongs in levels, bests and trends."""
        return self.status == Status.SCORED.value and (not self.disqualified or self.count_anyway)


def flight_key(path: Path) -> str:
    """Identify a recording by its size and the bytes at both ends."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as file:
        digest.update(file.read(SAMPLE_BYTES))
        file.seek(max(0, size - SAMPLE_BYTES))
        digest.update(file.read(SAMPLE_BYTES))
    return digest.hexdigest()


class FlightHistory:
    def __init__(self, path: Path | str, config: ScoringConfig | None = None) -> None:
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._config = config or ScoringConfig()
        with self._db:
            self._db.execute(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add the columns a database made by an older version lacks."""
        columns = {row["name"] for row in self._db.execute("PRAGMA table_info(flights)")}
        if "longest_line_s" not in columns:
            self._db.execute("ALTER TABLE flights ADD COLUMN longest_line_s REAL")

    def close(self) -> None:
        self._db.close()

    def flights(self) -> list[Flight]:
        """Every saved flight, oldest first."""
        rows = self._db.execute(f"SELECT {', '.join(_FIELDS)} FROM flights ORDER BY id")  # noqa: S608 - fixed column names
        return [_flight(row) for row in rows]

    def record(self, scored: Iterable[tuple[Path, FlightScore]]) -> int:
        """Save one import of scored recordings; returns how many were new or re-scored.

        A flight already saved by this scorer is left alone, and a re-scored one keeps its
        place in the history.
        """
        batch = int(
            self._db.execute("SELECT COALESCE(MAX(batch), 0) + 1 FROM flights").fetchone()[0]
        )
        changed = 0
        with self._db:
            for path, score in sorted(scored, key=lambda item: item[0].name):
                key = flight_key(path)
                row = self._db.execute(
                    "SELECT id, version FROM flights WHERE key = ?", (key,)
                ).fetchone()
                if row is not None and int(row["version"]) == score.version:
                    continue
                before = int(row["id"]) if row is not None else None
                reason = disqualify(score, self._earlier(before), self._config) or ""
                self._save(key, path.name, batch, score, reason, before)
                changed += 1
        return changed

    def count_anyway(self, flight_id: int, counted: bool) -> None:
        """The pilot's own call on a disqualified flight."""
        with self._db:
            self._db.execute(
                "UPDATE flights SET count_anyway = ? WHERE id = ?", (int(counted), flight_id)
            )

    def set_longest_line(self, path: Path, seconds: float) -> None:
        """Note the longest line found in a recording; nothing happens if it was never scored.

        Scores, and whether the flight counts, are left as they are.
        """
        with self._db:
            self._db.execute(
                "UPDATE flights SET longest_line_s = ? WHERE key = ?", (seconds, flight_key(path))
            )

    def _earlier(self, before: int | None) -> list[float]:
        return [
            f.freestyle for f in self.flights() if f.counts and (before is None or f.id < before)
        ]

    def _save(
        self, key: str, name: str, batch: int, score: FlightScore, reason: str, existing: int | None
    ) -> None:
        values = (
            score.conditions,
            score.flying_s,
            score.freestyle,
            score.flow,
            score.difficulty,
            score.variety,
            score.linking,
            score.continuity,
            score.smoothness,
            score.crashes,
            len(score.turns),
        )
        if existing is None:
            names = ", ".join(COLUMNS)
            marks = ", ".join("?" * len(COLUMNS))
            self._db.execute(
                f"INSERT INTO flights (key, name, batch, version, status, disqualified, {names}) "  # noqa: S608 - fixed column names
                f"VALUES (?, ?, ?, ?, ?, ?, {marks})",
                (key, name, batch, score.version, score.status.value, reason, *values),
            )
        else:
            sets = ", ".join(f"{column} = ?" for column in COLUMNS)
            self._db.execute(
                f"UPDATE flights SET name = ?, version = ?, status = ?, disqualified = ?, "  # noqa: S608 - fixed column names
                f"count_anyway = 0, {sets} WHERE id = ?",
                (name, score.version, score.status.value, reason, *values, existing),
            )


_FIELDS = tuple(Flight.__dataclass_fields__)


def best_line(flights: Sequence[Flight]) -> float | None:
    """The longest line, in seconds, among the flights that count; None when there is none."""
    lengths = [f.longest_line_s for f in flights if f.counts and f.longest_line_s is not None]
    return max(lengths) if lengths else None


def _flight(row: sqlite3.Row) -> Flight:
    values = {name: row[name] for name in _FIELDS}
    values["count_anyway"] = bool(values["count_anyway"])
    return Flight(**values)


def leaderboard(flights: Sequence[Flight], score: str) -> list[Flight]:
    """The flights that count, best first by ``score`` ("freestyle" or "flow")."""
    return sorted((f for f in flights if f.counts), key=lambda f: -float(getattr(f, score)))


def rolling_level(values: Sequence[float], window: int = LEVEL_FLIGHTS) -> list[float]:
    """At each flight, the median of it and the flights just before it."""
    return [statistics.median(values[max(0, i + 1 - window) : i + 1]) for i in range(len(values))]


@dataclass(frozen=True)
class Summary:
    flights: int
    freestyle: list[float]  # level after each counted flight
    flow: list[float]
    start_freestyle: float
    start_flow: float
    baseline: int  # how many of the first points are the starting import
    best_freestyle: float
    best_flow: float


def summarise(flights: Sequence[Flight]) -> Summary | None:
    """Levels and bests of the flights that count, or None when none do."""
    counted = [f for f in flights if f.counts]
    if not counted:
        return None
    first = counted[0].batch
    start = [f for f in counted if f.batch == first]
    return Summary(
        flights=len(counted),
        freestyle=rolling_level([f.freestyle for f in counted]),
        flow=rolling_level([f.flow for f in counted]),
        start_freestyle=statistics.median(f.freestyle for f in start),
        start_flow=statistics.median(f.flow for f in start),
        baseline=len(start),
        best_freestyle=max(f.freestyle for f in counted),
        best_flow=max(f.flow for f in counted),
    )


PARTS: tuple[tuple[str, str], ...] = (
    ("difficulty", "bigger, faster turns"),
    ("variety", "more varied turns"),
    ("linking", "linking turns together"),
    ("continuity", "keeping the flight moving"),
    ("smoothness", "smoother flying"),
)


def insight(flights: Sequence[Flight]) -> str:
    """What moved most between the latest flights and the ones before them."""
    counted = [f for f in flights if f.counts]
    if len(counted) < 2 * COMPARE_FLIGHTS:
        left = 2 * COMPARE_FLIGHTS - len(counted)
        return f"{left} more scored flights until the app can say what is changing."
    recent, before = counted[-COMPARE_FLIGHTS:], counted[-2 * COMPARE_FLIGHTS : -COMPARE_FLIGHTS]

    def change(name: str) -> float:
        return float(
            statistics.median(getattr(f, name) for f in recent)
            - statistics.median(getattr(f, name) for f in before)
        )

    moves = sorted(((change(name), text) for name, text in PARTS), key=lambda move: move[0])
    crashes = change("crashes")
    lines: list[str] = []
    if moves[-1][0] >= NOTABLE_CHANGE:
        lines.append(f"Improving: {moves[-1][1]}.")
    if moves[0][0] <= -NOTABLE_CHANGE:
        lines.append(f"Slipping: {moves[0][1]}.")
    if crashes <= -1:
        lines.append("Crashing less.")
    elif crashes >= 1:
        lines.append("Crashing more.")
    return " ".join(lines) or "Steady over the last 10 flights."
