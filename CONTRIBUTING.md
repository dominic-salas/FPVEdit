# Contributing to FPVEdit

Thanks for helping. Bug reports, ideas and pull requests are all welcome.

## Reporting a bug or asking for a feature

Use the [issue templates](https://github.com/dominic-salas/FPVEdit/issues/new/choose). For
bugs, say what you did, what you expected and what happened, and attach the log (in the
app: Menu > Show the log file). Please do not attach your recordings unless you are happy
for them to be public; a description of the footage (goggles/DVR, resolution, frame rate,
.ts or .mp4, with or without OSD) is usually enough.

## Setting up

Requires Python 3.12+, FFmpeg 6+ on PATH and [uv](https://docs.astral.sh/uv/).

```
uv sync --extra app            # add --extra ml for the optional image model
uv run fpvedit-app             # the desktop app
uv run fpvedit a.ts b.ts       # the command line
```

How it fits together: `src/fpvedit/pipeline.py` runs the steps (`cli.py` is the command line). `media` probes and decodes
with FFmpeg; `analysis` measures every frame (`picture`, `measure`, `motion`), reads the
flight state from the OSD (`flight`) and cleans the flight (`highlights`); `montage` finds
the moments and builds and picks the shots; `rating` and `taste` rate them; `render`
writes and validates the video. The desktop app is in `src/fpvedit/app` (PySide6), with its
editing rules in `app/model.py`. Known limitations are in `docs/limitations.md`.

## Making a change

- Keep changes small and focused, with tests. Tests generate their own clips with FFmpeg at
  test time; never commit footage, renders, model weights or credentials
  (`uv run python scripts/check_staged.py` checks what is staged).
- Tests must not need a GPU or a model download; mark those that do with `@pytest.mark.model`.
- Prefer tests that measure the output (frames, durations, pixels) over tests that repeat the
  code's own arithmetic.
- Run the same checks as CI before opening a pull request:

  ```
  uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
  ```

- Code style: type hints everywhere (mypy strict), `pathlib`, frozen dataclasses for
  configuration and results, explicit error types, subprocess calls with argument lists
  (never `shell=True`).
- A new dependency needs a reason (licence, size, why it is needed) in the pull request
  and a comment next to it in `pyproject.toml`.
- Do not claim more than the measurements support: the motion signals and the image model
  do not recognise named tricks, and image-text similarity is not a probability.

## Pull requests

Describe what changes for someone using FPVEdit, and how you checked it (commands and
results). CI must be green. By contributing you agree that your work is released under the
[MIT License](LICENSE). Please follow the [code of conduct](CODE_OF_CONDUCT.md).
