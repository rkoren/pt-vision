# Recording session 1: shot list

Goal: first real clips to validate sit-to-stand (B03) and walkway gait (B52) and to see how the whole
pipeline and the app behave on real footage. Two or three healthy volunteers, about 45 minutes.

## Before you start

- **Phone settings**: 1080p, 30 fps for most clips (one or two at 60 fps), landscape on a tripod.
  iPhone: Settings → Camera → Formats → "Most Compatible" (8-bit H.264) and turn off "Auto FPS" /
  auto low-light frame rate. Long-press the preview to lock exposure and focus once the subject is in
  frame. Stabilization can stay on since the phone is on a tripod.
- **Subject**: fitted clothes (no baggy trousers), shoes that show the heel and toe, colours that contrast
  with the background. Measure height against a wall (barefoot or note shoe type) and record age and sex.
- **Scene**: plain background if you can, even light, nothing else moving. Only the subject in frame
  except for the deliberate "bad" clips.
- **Files**: keep everything outside the repo, e.g. `~/ptvision-data/raw/session1/`. Name clips
  `s1_<subject>_<protocol>_<take>_<fps>.mov` (e.g. `s1_A_sts5x_02_30.mov`). Keep a `notes.csv` next to
  them: `file, subject, height_m, age, sex, protocol, fps, ground_truth, notes`.

## Priority 1: Five-Times Sit-to-Stand (about 15 min)

Setup per `docs/capture-guides/sts.md`: tripod lens ~0.8 m high, ~3 m from the chair, perpendicular
to the person's side; whole body, feet and seat in frame; armless chair against a wall; arms crossed.

| clip | what | ground truth |
|---|---|---|
| 3 takes per subject at 30 fps | sit still 2 s, "go", 5 stands as fast as safe, hold standing 2 s | time it yourself with a stopwatch app (first seat-off to fifth full stand), write the time in `notes.csv`; say "go" out loud so the start is on the audio track. No stopwatch needs to be in frame: the app shows frame and time, and video timing integrity is checked separately (B55). |
| 1 take per subject at 60 fps | same | same |
| 1 take from the other side | same, camera on the subject's other side | tests near/far side selection |

Your hand-timed total is the clinician-style ground truth for the primary metric; frame-level
events are annotated afterwards by stepping through the clip in `ptv app` (status bar shows the frame).

## Priority 2: Walkway gait (about 15 min)

Setup per `docs/capture-guides/gait.md`: tripod lens ~1 m high, 4–5 m back from the walking line,
perpendicular, at the middle of the walkway. Put two tape marks on the floor, measured, ideally 3 m
apart, both inside the frame. Start and finish points at least 2 m outside the frame on each side.

| clip | what | ground truth |
|---|---|---|
| 3 passes each direction per subject at 30 fps | comfortable pace, walk through without stopping | the tape distance; count the steps between the marks if you can |
| 1 pass each direction at 60 fps | same | same |
| 1 pass at a deliberately slow pace | slow but steady | tests the bout-detection threshold |

Enter each subject's height when analysing (`--height-m`), otherwise speed and step length are reported
as unavailable.

## Priority 3: deliberately bad clips (about 10 min)

These test the quality gating, which is as important as the metrics.

| clip | what should happen |
|---|---|
| sit-to-stand filmed from the front | `camera_view` fails |
| sit-to-stand with only 4 repetitions | `repetition_count` fails |
| walk with a second person crossing behind | second person tracked in grey, `other_people_in_frame` warns, metrics unchanged |
| one clip in **portrait** orientation | ingest rotates it; everything else normal |
| one clip **handheld** (no tripod) | `camera_motion` warns |
| subject partly out of frame (feet cut off) | `subject_in_frame` / foot confidence warns |

## Optional if time allows

- A 30-second chair stand (`sts_30s`): count the stands yourself.
- One sit-to-stand and one walk in a cluttered room with mixed lighting.

## Then

```sh
uv run ptv analyze <clip> --protocol sts_5x --age <a> --height-m <h> --sex <f|m>
uv run ptv analyze <clip> --protocol gait_sagittal --age <a> --height-m <h> --sex <f|m>
uv run ptv app <trial folder>          # step through events with the arrow keys
```

Fill `docs/validation/sts-v0.md` and start a `gait-v0.md` with the same table shape. Anything that looks
wrong becomes a backlog entry; fixes go on a branch.
