"""What the app remembers between runs (Qt's per-user settings store)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings

ENCODERS = ("auto", "cpu")  # the GPU when it works, or always the CPU


class AppSettings:
    def __init__(self, store: QSettings | None = None) -> None:
        self._store = store if store is not None else QSettings()

    def _path(self, key: str) -> Path | None:
        value = self._store.value(key, "")
        return Path(str(value)) if value else None

    def _set_path(self, key: str, path: Path | None) -> None:
        self._store.setValue(key, str(path) if path is not None else "")

    @property
    def encoder(self) -> str:
        value = str(self._store.value("render/encoder", "auto"))
        return value if value in ENCODERS else "auto"

    @encoder.setter
    def encoder(self, value: str) -> None:
        self._store.setValue("render/encoder", value if value in ENCODERS else "auto")

    @property
    def output_folder(self) -> Path | None:
        """Where montages go; None writes them beside the first clip."""
        return self._path("render/output_folder")

    @output_folder.setter
    def output_folder(self, folder: Path | None) -> None:
        self._set_path("render/output_folder", folder)

    @property
    def clip_folder(self) -> Path | None:
        """The folder the last clips were added from."""
        return self._path("clips/last_folder")

    @clip_folder.setter
    def clip_folder(self, folder: Path | None) -> None:
        self._set_path("clips/last_folder", folder)

    @property
    def ffmpeg_folder(self) -> Path | None:
        """A folder with ffmpeg and ffprobe chosen by the user when they are not on PATH."""
        return self._path("tools/ffmpeg_folder")

    @ffmpeg_folder.setter
    def ffmpeg_folder(self, folder: Path | None) -> None:
        self._set_path("tools/ffmpeg_folder", folder)

    @property
    def theme(self) -> str:
        value = str(self._store.value("window/theme", "dark"))
        return value if value in ("dark", "light") else "dark"

    @theme.setter
    def theme(self, value: str) -> None:
        self._store.setValue("window/theme", value if value in ("dark", "light") else "dark")

    @property
    def geometry(self) -> QByteArray | None:
        value = self._store.value("window/geometry")
        return value if isinstance(value, QByteArray) else None

    @geometry.setter
    def geometry(self, value: QByteArray) -> None:
        self._store.setValue("window/geometry", value)

    @property
    def music_folder(self) -> Path | None:
        return self._path("folders/music")

    @music_folder.setter
    def music_folder(self, folder: Path | None) -> None:
        self._set_path("folders/music", folder)

    @property
    def vertical_export(self) -> bool:
        """Export 9:16 for phones (the middle of the picture) instead of as recorded."""
        return str(self._store.value("export/vertical", "false")).lower() == "true"

    @vertical_export.setter
    def vertical_export(self, value: bool) -> None:
        self._store.setValue("export/vertical", "true" if value else "false")

    @property
    def open_folder_after_export(self) -> bool:
        return str(self._store.value("export/open_folder", "true")).lower() == "true"

    @open_folder_after_export.setter
    def open_folder_after_export(self, value: bool) -> None:
        self._store.setValue("export/open_folder", "true" if value else "false")

    @property
    def colour_by_rating(self) -> bool:
        """Timeline blocks coloured by their 1-5 rating instead of by recording."""
        return str(self._store.value("review/colour_by_rating", "true")).lower() == "true"

    @colour_by_rating.setter
    def colour_by_rating(self, value: bool) -> None:
        self._store.setValue("review/colour_by_rating", "true" if value else "false")

    @property
    def learn_taste(self) -> bool:
        """Learn from the pilot's ratings and leave out shots predicted below `min_stars`
        (when shots are next found). Off: every shot found is kept. Rating shots is always
        possible either way."""
        return str(self._store.value("review/learn_taste", "false")).lower() == "true"

    @learn_taste.setter
    def learn_taste(self, value: bool) -> None:
        self._store.setValue("review/learn_taste", "true" if value else "false")

    @property
    def min_stars(self) -> int:
        """With `learn_taste`, shots predicted below this many stars are left out."""
        try:
            value = int(str(self._store.value("review/min_stars", 3)))
        except ValueError:
            return 3
        return min(5, max(1, value))

    @min_stars.setter
    def min_stars(self, value: int) -> None:
        self._store.setValue("review/min_stars", min(5, max(1, int(value))))
