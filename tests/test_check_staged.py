from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_staged.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_staged", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_staged"] = module
    spec.loader.exec_module(module)
    return module


check_staged = _load()


@pytest.mark.parametrize(
    "path",
    [
        "flight.ts",
        "clips/FLIGHT_0001.MP4",
        "out/flight_highlight.mp4",
        "weights/model.safetensors",
        "pytorch_model.bin",
        "models\\siglip.onnx",
        ".env",
        "config/.env.local",
        "deploy.pem",
    ],
)
def test_prohibited_paths_are_refused(path: str) -> None:
    assert check_staged.classify(path, size=10) is not None


@pytest.mark.parametrize(
    "path",
    ["src/fpv_editor/cli.py", "README.md", "tests/fixtures/annotations.json", "docs/env.md"],
)
def test_ordinary_source_files_are_allowed(path: str) -> None:
    assert check_staged.classify(path, size=10) is None


def test_oversized_file_is_refused() -> None:
    reason = check_staged.classify("data/table.csv", size=check_staged.MAX_FILE_BYTES + 1)
    assert reason is not None and "larger" in reason


def test_main_reports_every_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "flight.ts").write_bytes(b"\x47" * 188)
    (tmp_path / "big.csv").write_bytes(b"0" * (check_staged.MAX_FILE_BYTES + 1))
    monkeypatch.chdir(tmp_path)

    code = check_staged.main(["ok.py", "flight.ts", "big.csv"])

    err = capsys.readouterr().err
    assert code == 1
    assert "flight.ts (video file)" in err
    assert "big.csv" in err
    assert "ok.py" not in err


def test_main_passes_clean_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    assert check_staged.main(["ok.py"]) == 0
