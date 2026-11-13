"""App pieces without a window: settings, time left, the cache, the icon (offscreen Qt)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from fpvedit.app.dialogs import cache_size, clear_cache
from fpvedit.app.settings import AppSettings
from fpvedit.app.window import time_left, use_folders_first


def settings_in(folder: Path) -> AppSettings:
    return AppSettings(QSettings(str(folder / "settings.ini"), QSettings.Format.IniFormat))


def test_settings_are_remembered_and_bad_values_fall_back(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    assert settings.encoder == "auto"
    assert settings.output_folder is None
    settings.encoder = "cpu"
    settings.output_folder = tmp_path
    again = settings_in(tmp_path)
    assert (again.encoder, again.output_folder) == ("cpu", tmp_path)
    again.encoder = "quantum"
    assert again.encoder == "auto"
    again.output_folder = None
    assert again.output_folder is None


def test_time_left() -> None:
    assert time_left(1, 0.5) == ""  # too early to say
    assert time_left(30, 0.01) == ""
    assert time_left(30, 0.5) == "less than a minute left"
    assert time_left(120, 0.5) == "about 2 min left"


def test_clearing_the_cache_deletes_only_measurement_files(tmp_path: Path) -> None:
    (tmp_path / "clip.abc.measurements.npz").write_bytes(b"x" * 10)
    keep = tmp_path / "flight.mp4"
    keep.write_bytes(b"footage")
    assert cache_size(tmp_path) == 10
    assert clear_cache(tmp_path) == 1
    assert keep.read_bytes() == b"footage"
    assert cache_size(tmp_path) == 0 and clear_cache(tmp_path / "missing") == 0


def test_ffmpeg_folders_go_first_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "elsewhere")
    use_folders_first([tmp_path])
    assert os.environ["PATH"] == os.pathsep.join([str(tmp_path), "elsewhere"])


def test_icon_draws_at_every_size() -> None:
    from fpvedit.app.look import SIZES, app_icon, icon_pixmap

    _ = QApplication.instance() or QApplication([])  # widgets need QApplication later
    for size in SIZES:
        image = icon_pixmap(size).toImage()
        assert image.width() == size
        assert image.pixelColor(size // 2, size // 2).alpha() == 255  # the tile is opaque
        assert image.pixelColor(0, 0).alpha() == 0  # rounded corner
    assert not app_icon().isNull()


def test_recording_time_comes_from_the_name_or_the_file(tmp_path: Path) -> None:
    from datetime import datetime

    from fpvedit.app.clips import clock, recorded_at

    named = recorded_at(Path("D:/clips/20261010-203859.ts"))
    assert named is not None and named.from_name
    assert named.when == datetime(2026, 10, 10, 20, 38, 59)
    assert named.label() == "REC 10 OCT 2026 20:38"
    other = tmp_path / "hdz_0061.mp4"
    other.write_bytes(b"")
    found = recorded_at(other)
    assert found is not None and not found.from_name and found.label().startswith("FILE ")
    not_a_date = tmp_path / "20261399-999999.mp4"  # month 13: not a date
    not_a_date.write_bytes(b"")
    fallback = recorded_at(not_a_date)
    assert fallback is not None and not fallback.from_name
    assert recorded_at(tmp_path / "missing.mp4") is None
    assert (clock(189.9), clock(59.4), clock(3723)) == ("3:10", "0:59", "1:02:03")


def test_timeline_click_picks_and_drag_moves_a_shot() -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from fpvedit.app.timeline import RULER, Block, Timeline

    QApplication.instance() or QApplication([])
    timeline = Timeline()
    timeline.resize(306, 90)
    timeline.set_blocks([Block(10.0, i, f"{i}", "", None) for i in range(3)])
    picked: list[tuple[int, float]] = []
    moved: list[tuple[int, int]] = []
    seeked: list[float] = []
    timeline.picked.connect(lambda i, share: picked.append((i, share)))
    timeline.moved.connect(lambda i, gap: moved.append((i, gap)))
    timeline.seeked.connect(seeked.append)
    y = RULER + 20
    QTest.mouseClick(timeline, Qt.MouseButton.LeftButton, pos=QPoint(150, y))
    assert picked and picked[0][0] == 1 and 0.4 < picked[0][1] < 0.6
    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, pos=QPoint(20, y))
    QTest.mouseMove(timeline, QPoint(120, y))
    QTest.mouseMove(timeline, QPoint(290, y))
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, pos=QPoint(290, y))
    assert moved == [(0, 3)]  # the first shot dropped after the last
    QTest.mouseClick(timeline, Qt.MouseButton.LeftButton, pos=QPoint(250, 4))  # the ruler
    assert len(seeked) == 1 and 24 < seeked[0] < 26


def test_dragging_the_playhead_scans_through_the_montage() -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from fpvedit.app.timeline import RULER, Block, Timeline

    QApplication.instance() or QApplication([])
    timeline = Timeline()
    timeline.resize(306, 90)
    timeline.set_blocks([Block(10.0, i, f"{i}", "", None) for i in range(3)])
    scrubbed: list[float] = []
    seeked: list[float] = []
    moved: list[tuple[int, int]] = []
    timeline.scrubbed.connect(scrubbed.append)
    timeline.seeked.connect(seeked.append)
    timeline.moved.connect(lambda i, gap: moved.append((i, gap)))
    y = RULER + 20
    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, pos=QPoint(2, y))  # the playhead
    QTest.mouseMove(timeline, QPoint(100, y))
    QTest.mouseMove(timeline, QPoint(200, y))
    timeline.set_position(1.0)  # a frame of an earlier seek must not move the needle back
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, pos=QPoint(200, y))
    assert moved == []  # dragging the playhead never moves a shot
    assert len(scrubbed) == 3 and scrubbed == sorted(scrubbed) and 9 < scrubbed[1] < 11
    assert len(seeked) == 1 and 19 < seeked[0] < 21  # play on from where it was let go


def test_dot_letters_share_the_x_height_and_baseline() -> None:
    from fpvedit.app.look import BASELINE, GLYPHS, ROWS, X_TOP

    ascenders, descenders = set("bdfhiklt"), set("gjpqy")
    for letter, rows in GLYPHS.items():
        assert len(rows) == ROWS and all(len(row) == 5 for row in rows), letter
        lit = [y for y, row in enumerate(rows) if "#" in row]
        assert max(lit) == (ROWS - 1 if letter in descenders else BASELINE), letter
        if letter not in ascenders:
            assert min(lit) == X_TOP, letter


def test_montages_default_to_the_videos_folder_not_the_clips_card(tmp_path: Path) -> None:
    from fpvedit.app.clips import default_output_folder

    card = tmp_path / "DCIM"
    card.mkdir()
    videos = tmp_path / "Videos" / "FPVEdit"
    other = tmp_path / "montages"
    other.mkdir()
    clips = [card / "a.ts", card / "b.ts"]
    assert default_output_folder(None, clips, videos) == videos
    assert default_output_folder(card, clips, videos) == videos  # never the clips' folder
    assert default_output_folder(other, clips, videos) == other  # the folder used last
    assert default_output_folder(tmp_path / "gone", clips, videos) == videos


def test_my_ratings_are_saved_and_read_back(tmp_path: Path) -> None:
    import json

    from fpvedit.app.myratings import MyRatings
    from fpvedit.montage import Moment, MomentKind, Shot
    from fpvedit.rating import Quality

    quality = Quality(0.8, 0.7, 0.5, 1.0, ("back flip", "smooth recovery"))
    shot = Shot(0, 10.0, 19.0, 3.0, (Moment(13.0, MomentKind.FLIP, 3.0, "back flip"),), quality)
    file = tmp_path / "my-ratings.jsonl"
    clip = tmp_path / "flight.ts"
    mine = MyRatings(file)
    assert mine.stars(clip, shot) is None
    mine.rate(clip, shot, 2)
    mine.rate(clip, shot, 4)  # changed his mind: the later one counts
    again = MyRatings(file)
    assert again.stars(clip, shot) == 4 and again.count == 1
    last = json.loads(file.read_text(encoding="utf-8").splitlines()[-1])
    assert last["app_stars"] == quality.stars and last["tricks"] == ["back flip"]
    with pytest.raises(ValueError):
        mine.rate(clip, shot, 6)


def test_rating_screen_mixes_clips_and_skips_rated_shots(tmp_path: Path) -> None:
    from fpvedit.app.myratings import MyRatings
    from fpvedit.app.ratepage import interleave
    from fpvedit.montage import Shot

    a = [Shot(0, 10.0 * i, 10.0 * i + 9, 1.0, ()) for i in range(3)]
    b = [Shot(1, 10.0 * i, 10.0 * i + 9, 1.0, ()) for i in range(2)]
    order = interleave([a, b])
    assert sorted(order, key=lambda s: (s.clip, s.start)) == sorted(
        [*a, *b], key=lambda s: (s.clip, s.start)
    )
    assert [s.clip for s in order[:4]] == [0, 1, 0, 1]  # clips take turns
    mine = MyRatings(tmp_path / "r.jsonl")
    mine.rate(tmp_path / "x.ts", a[0], 3)
    assert mine.stars(tmp_path / "x.ts", a[0]) == 3 and mine.stars(tmp_path / "x.ts", a[1]) is None


def test_an_undone_rating_is_forgotten(tmp_path: Path) -> None:
    from fpvedit.app.myratings import MyRatings
    from fpvedit.montage import Shot

    shot = Shot(0, 1.0, 10.0, 1.0, ())
    file = tmp_path / "r.jsonl"
    mine = MyRatings(file)
    mine.rate(tmp_path / "a.ts", shot, 3)
    mine.unrate(tmp_path / "a.ts", shot)
    assert MyRatings(file).count == 0 and len(file.read_text(encoding="utf-8").splitlines()) == 2


def test_settings_are_stored_between_runs() -> None:
    # Without an organisation name Qt refuses to store settings (an AccessError), so every
    # setting was lost when the app closed; run() sets one.
    import uuid

    from PySide6.QtCore import QSettings

    app = QApplication.instance() or QApplication([])
    names = app.organizationName(), app.applicationName()
    app.setOrganizationName(f"fpvedit-test-{uuid.uuid4().hex}")
    app.setApplicationName("FPVEdit")
    try:
        settings = AppSettings()
        settings.learn_taste = True
        assert QSettings().status() == QSettings.Status.NoError
        assert AppSettings().learn_taste  # a fresh object reads it back
    finally:
        QSettings().clear()
        app.setOrganizationName(names[0])
        app.setApplicationName(names[1])


def test_right_name_answers_are_kept_apart_from_edits(tmp_path: Path) -> None:
    from fpvedit.app.myratings import MyRatings

    clip = tmp_path / "a.ts"
    mine = MyRatings(tmp_path / "my-ratings.jsonl")
    mine.note_name(clip, 12.3, "back flip", "yes")
    mine.note_name(clip, 40.0, "flip", "power loop")
    again = MyRatings(tmp_path / "my-ratings.jsonl")
    assert again.names_right() == (1, 2)
    assert again.name(clip, 40.0, "flip") == "power loop"  # the pilot's name is shown
    assert again.name(clip, 12.3, "back flip") == "back flip"
    assert again.edit_count == 0 and not (tmp_path / "my-edits.jsonl").exists()
    again.clear()
    assert not (tmp_path / "my-names.jsonl").exists()


def test_the_shot_panel_asks_right_name_and_shows_the_answer(tmp_path: Path) -> None:
    from fpvedit.app.model import ShotList
    from fpvedit.app.review import ReviewPage
    from fpvedit.media import Tools
    from fpvedit.montage import Moment, MomentKind, Shot
    from fpvedit.rating import Quality

    QApplication.instance() or QApplication([])
    info = type("Info", (), {"path": tmp_path / "a.ts", "duration": 60.0})()
    quality = Quality(0.8, 0.8, 0.8, 0.8, ("test",))
    flip = Moment(5.0, MomentKind.FLIP, 3.0, "back flip")
    shot = Shot(0, 2.0, 12.0, 3.0, (flip,), quality)
    page = ReviewPage(Tools("ffmpeg", "ffprobe", 7), ratings_file=tmp_path / "my-ratings.jsonl")
    try:
        page.set_montage([info], ShotList((shot,), (), {0: 60.0}))
        page._picked(0, 0.0)
        assert page._name_row.isVisibleTo(page)
        assert "BACK FLIP" in page._name_question.text()
        page._answer_name("roll")
        assert page._shot_title.text() == "Roll"
        assert "YOU SAID: ROLL" in page._name_question.text()
        assert page.ratings.names_right() == (0, 1)
    finally:
        page.shutdown()


def test_export_warns_only_when_a_vertical_montage_is_too_long(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QWidget

    from fpvedit.app.dialogs import ExportDialog

    QApplication.instance() or QApplication([])
    parent = QWidget()
    settings = settings_in(tmp_path)
    long = ExportDialog(tmp_path / "m.mp4", settings, "No music", parent, duration=200.0)
    short = ExportDialog(tmp_path / "m.mp4", settings, "No music", parent, duration=120.0)
    shape = long._shape
    assert long._length_note.isHidden()  # landscape: no limit applies
    shape.setCurrentIndex(1)
    assert not long._length_note.isHidden()
    short._shape.setCurrentIndex(1)
    assert short._length_note.isHidden()


def test_a_shot_title_names_only_its_biggest_kind_of_moment() -> None:
    from fpvedit.app.review import headline
    from fpvedit.montage import Moment, MomentKind, Shot

    def moment(kind: MomentKind, name: str) -> Moment:
        return Moment(1.0, kind, 1.0, name)

    flips = (moment(MomentKind.FLIP, "back flip"), moment(MomentKind.SPIN, "spin"))
    assert headline(Shot(0, 0.0, 9.0, 1.0, flips)) == "back flip"
    both = (moment(MomentKind.FLIP, "back flip"), moment(MomentKind.FLIP, "flip"))
    assert headline(Shot(0, 0.0, 9.0, 1.0, both)) == "back flip"
    spins = (moment(MomentKind.SPIN, "spin"), moment(MomentKind.SPIN, "spin"))
    assert headline(Shot(0, 0.0, 9.0, 1.0, spins)) == "2 spins"
    close = (moment(MomentKind.PROXIMITY, "close pass"),)
    assert headline(Shot(0, 0.0, 9.0, 1.0, close)) == "close pass"


def test_steps_light_none_on_a_side_page() -> None:
    from fpvedit.app.window import Steps

    QApplication.instance() or QApplication([])
    steps = Steps()
    steps.show_state(-1, {0})
    assert [b.property("state") for b in steps._buttons] == ["off"] * 4


def test_pilot_page_shows_the_best_line_only_when_there_is_one(tmp_path: Path) -> None:
    from fpvedit.app.pilotpage import PilotPage
    from fpvedit.history import FlightHistory
    from fpvedit.scoring import SCORER_VERSION, FlightScore, Status

    _ = QApplication.instance() or QApplication([])
    good = FlightScore(Status.SCORED, SCORER_VERSION, 60.0, "daylight", freestyle=40.0, flow=50.0)
    history = FlightHistory(":memory:")
    flight = tmp_path / "a.mp4"
    flight.write_bytes(b"1")
    history.record([(flight, good)])
    page = PilotPage(history, show_back=False)
    page.show()
    page.refresh()
    assert not page._columns["line"].isVisible()
    history.set_longest_line(flight, 83.0)
    page.refresh()
    assert page._columns["line"].isVisible()
    assert page._stats["line"].text() == "1:23"


def lines_page(tmp_path: Path):  # type: ignore[no-untyped-def]
    from fpvedit.app.model import ShotList
    from fpvedit.app.review import ReviewPage
    from fpvedit.lines import Line, LineEnd
    from fpvedit.media import Tools
    from fpvedit.montage import Moment, MomentKind
    from fpvedit.pipeline import lines_as_shots

    QApplication.instance() or QApplication([])
    info = type("Info", (), {"path": tmp_path / "a.ts", "duration": 60.0})()
    moments = tuple(Moment(t, MomentKind.FLIP, 3.0, "roll") for t in (6.0, 9.0, 12.0))
    crash = Line(0, 4.0, 20.0, 3, 12.0, LineEnd.CRASH, moments)
    landed = Line(0, 30.0, 44.0, 0, 9.0, LineEnd.LANDED)
    shots = lines_as_shots([crash, landed])
    page = ReviewPage(Tools("ffmpeg", "ffprobe", 7), ratings_file=tmp_path / "my-ratings.jsonl")
    return page, info, ShotList(shots, shots, {0: 60.0}), (crash, landed)


def test_a_line_panel_shows_length_events_and_how_it_ended(tmp_path: Path) -> None:
    page, info, shots, lines = lines_page(tmp_path)
    try:
        page.set_montage([info], shots, "", lines=lines)
        assert page.lines_mode
        page._picked(0, 0.0)
        assert page._shot_number.text() == "LINE 01 OF 02"
        assert page._shot_title.text() == "Ended in a crash"
        assert [v.text() for v in page._fact_values] == ["16.0 s", "3", "12 s"]
        assert page._loop_button.text() == "↻  LOOP THIS LINE"
        assert page._line_facts.isVisibleTo(page) and not page._name_row.isVisibleTo(page)
        page._picked(1, 0.0)
        assert page._shot_title.text() == "Landed"
        assert page._fact_values[1].text() == "0"
        page.remove_selected()  # a line goes like a shot, and comes back with undo
        assert len(page._current()) == 1
        page.undo()
        assert len(page._current()) == 2
    finally:
        page.shutdown()


def test_the_tabs_and_hints_follow_the_mode_and_back(tmp_path: Path) -> None:
    page, info, shots, lines = lines_page(tmp_path)
    try:
        heard: list[bool] = []
        page.modeChosen.connect(heard.append)
        page._tabs._tabs[1].click()
        page._tabs._tabs[0].click()
        assert heard == [True, False]
        page.set_montage([info], shots, "", lines=lines)
        assert page._heading.text() == "Review your longest lines"
        assert page._add_button.text() == "+  ADD A LINE"
        assert any(t.text() == "EDIT A LINE" for t, _ in page._hints._actions)
        assert not page._find_button.isVisibleTo(page)
        page.set_montage([info], shots, "")
        assert not page.lines_mode
        assert page._heading.text() == "Review your montage"
        assert page._loop_button.text() == "↻  LOOP THIS SHOT"
        assert any(t.text() == "EDIT A SHOT" for t, _ in page._hints._actions)
    finally:
        page.shutdown()


def test_no_lines_shows_the_reason(tmp_path: Path) -> None:
    from fpvedit.app.model import ShotList
    from fpvedit.lines import LinesConfig, no_lines_message

    page, info, _, _ = lines_page(tmp_path)
    try:
        message = no_lines_message(LinesConfig(fast_flying=False))
        page.set_montage([info], ShotList((), (), {0: 60.0}), "", lines=(), no_lines=message)
        assert page._heading_summary.text() == message
        assert message.upper() in page._message.text()
        assert "fast flying" not in message
    finally:
        page.shutdown()
