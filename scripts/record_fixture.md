# Recording fixture and validation clips

## Pose fixture (already provided)
`tests/fixtures/clip_2s_480p.mp4` is derived from Sports2D's BSD-3 demo video; see tests/fixtures/README.md.

## Five-Times Sit-to-Stand validation clips (Slice 1)
Record 3–5 clips of healthy volunteers plus one deliberately bad clip.

Setup (matches Hwang 2026 and Bertrand 2026):
- Phone in landscape on a tripod, lens about 0.8 m above the floor, about 3 m from the chair,
  perpendicular to the person's side (sagittal view). Whole body, feet, and chair seat in frame.
- Standard armless chair, seat height 43–45 cm, arms crossed over the chest.
- 30 fps or 60 fps, 1080p, good light, plain background if possible. Lock exposure/focus.
- Start recording, show a running stopwatch (second phone) to the camera for 2 s, then perform
  five sit-to-stands as fast as safely possible. Keep the stopwatch visible in a corner of the
  frame the whole time so timing can be checked frame by frame.
- Bad clip: film from the front (frontal view), or let the subject walk out of frame, or do 4 reps.

Then:
```sh
uv run ptv pose clip.mov                    # keypoints + overlay to inspect tracking
uv run python scripts/make_pose_fixture.py ptv_out/clip/pose/cam0.parquet tests/fixtures/sts_5x_pose_cam0.parquet
```
and annotate seat-off / stand-reached / seated-return frames in `tests/fixtures/sts_5x_events_manual.csv`.
