# Windows portability log

A fresh install on the Windows machine at each milestone (see `CLAUDE.md`, "Milestone checks"). CI covers
code regressions, installing prerequisites, first-run weight download,
the app on a real display, playback speed on ordinary hardware, drag-and-drop, and later SmartScreen.

## Checklist (PowerShell, from a clean clone)

```powershell
# 1. prerequisites
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # uv
winget install Gyan.FFmpeg                                                             # ffmpeg (GPL build: fine to use locally, never to bundle)
git clone <repo-url> pt-vision; cd pt-vision

# 2. install and self-check
uv sync --extra app --group dev --group qt
uv run ptv version
uv run ptv models pull --mode lightweight
uv run pytest -m "not model"
uv run pytest                                  # includes the model test

# 3. real behaviour
uv run ptv pose tests\fixtures\clip_2s_480p.mp4                 # note the fps line
uv run ptv analyze tests\fixtures\clip_2s_480p.mp4 --protocol sts_5x --age 34   # expect quality FAIL, no crash
uv run ptv app tests\fixtures\clip_2s_480p.mp4                  # drop a second video onto the window too
uv run ptv bench tests\fixtures\clip_2s_480p.mp4 --modes lightweight,balanced --devices cpu
```

Record: Windows build, CPU/GPU, Python version uv picked, wall time for `uv sync`, pose fps per mode,
anything that needed a workaround. Every friction point becomes a backlog entry.

## Milestones that trigger a check

| # | milestone | why it matters on Windows |
|---|---|---|
| M1 | Baseline (now: Slices 0–1 + desktop viewer) | first data point; install path, Qt, drag-drop, playback speed |
| M2 | ffmpeg decision implemented (B34 / B45) | the install step most likely to break for a PT |
| M3 | Slice 2 gait done | new segmenter and scaling code paths |
| M4 | Before the first PT partner hands-on session | rehearsal of exactly what they will do |
| M5 | Distribution route (B46 `uv tool install`, later B17 bundles) | installer scripts, SmartScreen, bundle size |
| any | a dependency added or bumped, or changes in `io/video.py`, `pose/models.py`, `app/` | these are where platform differences live |

## Log

| date | milestone | commit | Windows / hardware | uv sync time | pose fps (light / balanced) | result | notes → backlog |
|---|---|---|---|---|---|---|---|
| | M1 | | | | | | |
