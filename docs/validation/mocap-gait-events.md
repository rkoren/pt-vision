# Gait-event detection against public motion-capture datasets

Run on 2026-09-22 with `ptv datasets eval-gait <name> --method velocity` (Zeni 2008 velocity method,
30 fps virtual side camera, edge peaks on). Marker trajectories were projected onto an orthographic
sagittal camera (`ptvision.datasets.mocap_projection`), the same heel-strike / toe-off detector used
on video was run, and each labelled event was matched to the nearest detection of the same side and
kind within 0.25 s. This tests the event algorithm and the cycle logic, not the pose model: the
inputs are marker positions, so the numbers are an upper bound on what real video will give.

Reference: Zeni et al. 2008 reported 94 % of events within one frame (healthy, treadmill) and 89 %
within two frames (impaired) at 60 Hz.

| dataset | license | trials | labelled events | within 1 frame | within 2 frames | MAE (frames) | missed | extra |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Schreiber & Moissenet 2019 | CC BY 4.0 | 1,143 | 7,994 | 78.6 % | 94.2 % | 0.93 | 16 | 18 |
| Van Criekinge 2023, able-bodied | CC0 | 454 | 5,900 | 89.5 % | 98.4 % | 0.78 | 16 | 7 |
| Van Criekinge 2023, post-stroke | CC0 | 152 | 1,534 | 73.1 % | 87.8 % | 1.21 | 25 | 128 |
| Fukuchi 2018 | CC BY 4.0 | 1,969 | 53,964 | 80.4 % | 97.1 % | 0.94 | 1,830 | 2,407 |

"Extra" excludes detections within 0.25 s of a clip boundary, where the true event lies outside the
recording and is unlabelled (2,130 such in Schreiber, whose trials start on a foot strike).

## By walking speed (Schreiber 2019)

| condition | speed | within 1 frame | within 2 frames | MAE |
| --- | --- | --- | --- | --- |
| C1 | 0–0.4 m/s | 41.1 % | 75.4 % | 2.00 |
| C2 | 0.4–0.8 m/s | 72.6 % | 97.1 % | 1.05 |
| C3 | 0.8–1.2 m/s | 91.4 % | 99.9 % | 0.58 |
| C4 | self-selected | 93.8 % | 99.9 % | 0.53 |
| C5 | fast | 96.8 % | 99.9 % | 0.44 |

## Fukuchi 2018 by condition

| condition | trials | within 1 frame | within 2 frames | MAE | missed | extra |
| --- | --- | --- | --- | --- | --- | --- |
| overground comfortable | 534 | 79.8 % | 93.2 % | 1.04 | 223 | 263 |
| overground fast | 572 | 84.5 % | 95.2 % | 0.91 | 170 | 315 |
| overground slow | 535 | 72.4 % | 90.1 % | 1.29 | 646 | 522 |
| treadmill speeds 1–3 (slowest) | 126 | 67–74 % | 96–98 % | 1.0–1.2 | 669 | 838 |
| treadmill speeds 4–8 | 202 | 79–92 % | ≥ 99.8 % | 0.7–0.9 | 122 | 469 |

## What the numbers say

- At comfortable and fast walking the detector matches the published Zeni accuracy (91–97 % within
  one frame, MAE about half a frame at 30 fps) on 1,143 + 454 trials from two labs.
- Post-stroke gait: 73 % within one frame and 88 % within two, in line with Zeni's 89 % within two
  frames for impaired walkers. Extra detections (128 over 152 trials) come from shuffling steps and
  hesitations that produce additional velocity zero-crossings; the cycle validator drops most of
  them before metrics, but this is the case to improve (BACKLOG B61).
- Slow walking (below 0.8 m/s) is the weak spot everywhere: at 30 fps the heel's relative velocity
  crosses zero slowly, so the crossing frame is poorly defined. Two frames of tolerance (67 ms)
  recovers most of it. Clinical populations walk in this range, so this matters (B61, B53 for 60 fps).
- Heel strikes are detected about one frame early on average and toe-offs about one frame late on
  Fukuchi; a fixed offset could be calibrated per method once real-video results exist.
- One Van Criekinge stroke trial has left and right labelled the other way round in the dataset; the
  harness detects and reports this rather than scoring it as zero.
- Fukuchi labels every toe-off twice in the c3d event block; duplicates are removed before scoring.

## Reproduce

```sh
uv sync --extra datasets
uv run ptv datasets pull schreiber2019 vancriekinge2023 fukuchi2018
uv run ptv datasets eval-gait schreiber2019 --method velocity
```

Outputs land in `~/ptvision-data/datasets/<name>/derived/eval_gait/velocity_30fps/`
(`summary.json`, `trials.csv`, `events.csv`).
