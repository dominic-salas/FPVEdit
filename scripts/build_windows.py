"""Build the standalone Windows app: one folder with FPV Editor.exe, FFmpeg and licences.

    uv sync --group build
    uv run python scripts/build_windows.py --ffmpeg C:/path/to/ffmpeg/bin

The folder runs without Python, uv or the project. `packaging/installer.iss` (Inno Setup)
turns it into an installer; the release workflow does both on GitHub's runners.
``--ffmpeg`` is a folder with ffmpeg.exe and ffprobe.exe (and their DLLs for a shared
build); without it the app uses the FFmpeg on PATH.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "FPV Editor"
CLI_NAME = "fpv-editor"
LICENSED_PACKAGES = ("PySide6-Essentials", "shiboken6", "numpy", "opencv-python-headless")


def write_icon(path: Path) -> None:
    """A multi-size .ico with PNG images (Windows Vista and later), drawn by the app."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtWidgets import QApplication

    from fpv_editor.app.look import SIZES, icon_pixmap

    _ = QApplication.instance() or QApplication([])
    images = []
    for size in SIZES:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        icon_pixmap(size).save(buffer, "PNG")
        images.append((size, bytes(buffer.data().data())))
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, blobs = b"", b""
    for size, png in images:
        side = 0 if size >= 256 else size  # 0 means 256 in the ICO format
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(png), offset)
        blobs += png
        offset += len(png)
    path.write_bytes(header + entries + blobs)


def copy_ffmpeg(source: Path, target: Path) -> str:
    """Copy ffmpeg, ffprobe (and ffplay, which plays the music in the app's preview, when
    the build has it) and their DLLs; return the build's licence ("GPL" or "LGPL")."""
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        if not (source / name).is_file():
            raise SystemExit(f"{source} has no {name}")
    target.mkdir(parents=True, exist_ok=True)
    optional = [source / "ffplay.exe"] if (source / "ffplay.exe").is_file() else []
    for path in [source / "ffmpeg.exe", source / "ffprobe.exe", *optional, *source.glob("*.dll")]:
        shutil.copy2(path, target / path.name)
    for name in ("LICENSE", "LICENSE.txt", "COPYING.GPLv3", "COPYING.LGPLv2.1"):
        for folder in (source, source.parent):
            if (folder / name).is_file():
                shutil.copy2(folder / name, target / f"FFmpeg-{name}")
    version = subprocess.run(
        [str(target / "ffmpeg.exe"), "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return "GPL" if "--enable-gpl" in version else "LGPL"


def copy_licences(target: Path, ffmpeg_licence: str | None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "LICENSE", target / "FPV-Editor-LICENSE.txt")
    for package in LICENSED_PACKAGES:
        distribution = importlib.metadata.distribution(package)
        for file in distribution.files or []:
            name = file.name.upper()
            if "LICEN" in name or "COPYING" in name:
                source = Path(str(file.locate()))
                if source.is_file():
                    shutil.copy2(source, target / f"{package}-{file.name}")
    ffmpeg = (
        f"FFmpeg ({ffmpeg_licence} build) is included in the ffmpeg folder and runs as a "
        "separate program. Its source code is available from https://ffmpeg.org/download.html "
        "(see the version printed by ffmpeg -version)."
        if ffmpeg_licence
        else "FFmpeg is not included; the FFmpeg installed on this computer is used."
    )
    (target / "THIRD-PARTY-NOTICES.txt").write_text(
        "FPV Editor is MIT licensed (FPV-Editor-LICENSE.txt). It includes:\n"
        "- Python (PSF License) and the PyInstaller bootloader (GPL-2.0 with an exception "
        "that allows distribution with any program)\n"
        "- Qt for Python: PySide6 and shiboken6 (LGPL-3.0), used unmodified as shared "
        "libraries; you may replace them with your own build of the same version\n"
        "- NumPy (BSD-3-Clause), OpenCV (Apache-2.0)\n"
        f"- {ffmpeg}\n"
        "The licence texts of these packages are in this folder.\n",
        encoding="utf-8",
    )


def pyinstaller(
    build: Path, out: Path, name: str, module: str, icon: Path, *, windowed: bool
) -> Path:
    """One PyInstaller folder build of ``module``'s main(); returns the folder."""
    launcher = build / f"{name.replace(' ', '_')}_main.py"
    launcher.write_text(
        f"import sys\n\nfrom {module} import main\n\nsys.exit(main())\n", encoding="utf-8"
    )
    subprocess.run(
        [
            *(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean"),
            "--windowed" if windowed else "--console",
            *("--name", name, "--icon", str(icon)),
            *("--distpath", str(out), "--workpath", str(build / "work" / name)),
            *("--specpath", str(build), "--exclude-module", "tkinter"),
            str(launcher),
        ],
        check=True,
        cwd=ROOT,
    )
    return out / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ffmpeg", type=Path, help="folder with ffmpeg.exe and ffprobe.exe")
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="output folder")
    args = parser.parse_args()
    build = ROOT / "build" / "windows"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    icon = build / "fpv-editor.ico"
    write_icon(icon)
    app = pyinstaller(build, args.out, APP_NAME, "fpv_editor.app", icon, windowed=True)
    # The command-line tool, in the same folder: its own small exe, the app's libraries.
    cli = pyinstaller(build, build / "cli", CLI_NAME, "fpv_editor.cli", icon, windowed=False)
    shutil.copy2(cli / f"{CLI_NAME}.exe", app / f"{CLI_NAME}.exe")
    shutil.copytree(cli / "_internal", app / "_internal", dirs_exist_ok=True)
    # OpenCV's own video I/O (an FFmpeg build of ~30 MB) is never used: FFmpeg does that.
    for unused in (app / "_internal" / "cv2").glob("opencv_videoio_ffmpeg*.dll"):
        unused.unlink()
    licence = copy_ffmpeg(args.ffmpeg, app / "ffmpeg") if args.ffmpeg else None
    copy_licences(app / "licenses", licence)
    size = sum(p.stat().st_size for p in app.rglob("*") if p.is_file()) / 1e6
    print(f"Built {app} ({size:.0f} MB, FFmpeg: {licence or 'not included'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
