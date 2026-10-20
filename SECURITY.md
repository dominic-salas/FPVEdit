# Security policy

## Supported versions

Security fixes go into the latest release only. Please update before reporting.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private
[vulnerability reporting](https://github.com/dominic-salas/FPVEdit/security/advisories/new)
for this repository, with the steps to reproduce and the version (Menu > About). You will get
an answer as soon as the maintainer can; this is a one-person hobby project, so there is no
fixed response time.

## What FPVEdit does on your computer

- It reads the recordings you give it and never changes, moves or deletes them; it writes
  only the montage you ask for (never over an existing file unless you choose to replace it),
  temporary files, and its own data folder (`%LOCALAPPDATA%\FPVEdit`).
- It makes no network connections, except that the optional image model (only when running
  from source with the `ml` extra) downloads its weights from Hugging Face once.
- It runs FFmpeg (included in the Windows build) as a separate program, with arguments passed
  as a list, never through a shell.
- The installer is not code-signed. Download it only from this repository's
  [releases](https://github.com/dominic-salas/FPVEdit/releases).
