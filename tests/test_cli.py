from __future__ import annotations

import pytest

from fpvedit import __version__
from fpvedit.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--vertical", "--music=song.mp3", "--length=60"])
def test_montage_only_options_are_refused_with_trim(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["flight.ts", "--trim", flag])
    assert exc.value.code == 2
    assert "montage" in capsys.readouterr().err
