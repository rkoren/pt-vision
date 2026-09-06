# pt-vision

Building CV measurement tools to assist physical therapists: functional tests,
gait, and range-of-motion analysis from ordina phone video

## In Progress

Current:
- `ptv pose`: RTMPose Halpe-26 keypoints from video, to Parquet plus an overlay
- `ptv analyze --protocol sts_5x | sts_30s`: Five-Times Sit-to-Stand and 30-second chair stand:
  phase segmentation, timing and count metrics with error bars, trunk/knee angles at seat-off,
  comparison to age-band reference values, capture-quality gating, HTML + JSON report.
- `ptv app`: desktop viewer. Drop a video in, watch progress, then play with a skeleton overlay
  coloured by tracking confidence or by the protocol's joint-angle bands

## Quickstart

Requires Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), and `ffmpeg` on your PATH.

```sh
uv sync --extra app --group qt           # create .venv and install (app extra = PySide6 viewer)
uv run ptv models pull --mode balanced   # download RTMPose-m + YOLOX-m ONNX weights (~100 MB)
uv run ptv pose path/to/clip.mov         # extract Halpe-26 keypoints -> Parquet + overlay video
uv run ptv analyze clip.mov --protocol sts_5x --age 71   # Five-Times Sit-to-Stand report
uv run ptv protocols list                # available protocols
uv run ptv app clip.mov                  # desktop viewer: pose + colored skeleton overlay
```

Outputs land in `./ptv_out/<clip-name>/` unless a patient and visit are given, in which
case they go to the data store (`$PTV_DATA_DIR`, default `~/ptvision-data`).

## What it measures

| Tier | Examples | Typical error (literature) |
|---|---|---|
| 1 | sit-to-stand time and count, TUG time, gait speed, cadence, stance/swing time, symmetry | 0.01–0.04 s, 0.02–0.04 m/s |
| 2 | peak sagittal knee/hip flexion, bout-mean step length, knee/elbow/shoulder ROM | 4–6° RMSE, <2 cm |
| 3 | ankle angles, frontal-plane hip/pelvis, absolute upper-limb ROM | not reported as values |

## Layout

```
src/ptvision/
  io/          video probe + ffmpeg normalization, Parquet/JSON helpers
  pose/        PoseBackend protocol, rtmlib (RTMPose Halpe-26) backend, tracking, overlay
  kinematics/  filtering, joint angles, TRC/MOT export
  clinical/    protocols (TOML), segmenters, metric registry, normative tables
  quality/     capture-quality gating
  report/      JSON + HTML reports
  viz/         shared status → color semantics and skeleton segments (CLI overlay and app)
  app/         PySide6 desktop viewer (optional `app` extra)
  data/        patient / visit / trial store, capture and run provenance
  cli/         `ptv` command line
  _vendor/     BSD-3 code vendored from Pose2Sim (see NOTICE)
experiments/   separate uv project for research-licensed models and training; never imported by ptvision
```

## Licensing
Public domain (Unlicense)