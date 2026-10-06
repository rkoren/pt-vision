# Validation log: sit-to-stand v0

Goal: on 3–5 healthy-volunteer clips, total time within ±0.2 s of a stopwatch visible in frame, and
seat-off / stand events within ±2 frames of manual annotation. Record everything here.

## Clips

| clip | date | subject | fps | view side | reps | notes |
|---|---|---|---|---|---|---|
| sts1.MOV | 2026-09-22 | A (m, 26, 1.80 m) | 30 | left, three-quarter, portrait | 5 | starts standing; recording ends on the 5th rise |
| sts2.MOV | 2026-09-22 | A | 30 | right, three-quarter, portrait | 5 | camera_view fail at 0.37; recording ends on the 5th rise |
| sts34.MOV | 2026-09-22 | A | 59.94 VFR, 4K | frontal | 10 | deliberate view-check failure; two sets of 5 |

## Total time vs stopwatch

| clip | stopwatch (s) | ptvision (s) | diff (s) | start rule | end rule |
|---|---|---|---|---|---|
| sts1.MOV | 7.01 | 6.90 | −0.11 | first_seat_off | fifth_stand (truncated: frame 235/238) |
| sts_side.MOV (bundled sample, `src/ptvision/samples/sts_side.mp4`) | 10.75 | 10.40 | −0.35 | first_seat_off | fifth_stand; recording continues after the test; view ratio 0.45 (warn) |
| sts2.MOV | 10.2 | 9.67 | −0.53 | first_seat_off | fifth_stand (truncated: frame 334/335) |

## Event frames vs manual annotation

Manual annotation: step through `video/cam0.mp4` frame by frame; seat-off = first frame the buttocks
leave the seat; stand = first frame the hips stop rising; seated = first frame fully seated.

| clip | rep | manual seat-off | ptv seat-off | manual stand | ptv stand | manual seated | ptv seated |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Decisions
- start/end rule agreed with PT partner: (pending)
- observed failure modes and parameter changes: both takes stopped recording during the fifth rise, so the
  end event is clipped; ghost tracks on furniture; oblique-view hard fail; pre-test
  descent; camera-motion warning from subject motion. Hand-timed comparison is not yet a
  valid accuracy figure until re-recorded with 2 s of tail.
