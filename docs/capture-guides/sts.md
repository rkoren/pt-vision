# Capture guide: sit-to-stand tests (5xSTS, 30-second chair stand)

Matches the setups validated in Bertrand et al. 2026 (n=228, ICC 0.995 vs clinician) and
Hwang et al. 2026 (2D RGB, knee RMSE 5.9°, trunk RMSE 4.0°).

**Camera**
- Phone in landscape on a tripod, lens about 0.8 m above the floor.
- About 3 m from the chair, perpendicular to the person's side (sagittal view). Either side is fine;
  the report says which side was analysed.
- Whole body, both feet, and the chair seat in frame for the entire test. Nothing between the camera
  and the hips/knees (no table, no therapist).
- 30 fps or 60 fps, 1080p. Lock exposure and focus if the app allows. Avoid "auto frame rate" or
  "auto low light" modes (they produce variable frame rate; ptvision resamples but timing is slightly worse).
- Good, even light. A plain background helps. Only one person in frame if possible.

**Chair and patient**
- Standard armless chair, seat height 43–45 cm, against a wall so it cannot slide.
- Arms crossed over the chest. Feet flat, about shoulder width.

**Procedure (5xSTS)**
1. Start recording. If you want an independent timing check, hold a running stopwatch in view for a
   moment, then keep it in a corner of the frame.
2. Patient sits, still, for at least 2 s.
3. "Ready, go": stand fully and sit back down five times as fast as is safe. The timed interval ends
   at the fifth full stand (protocol `end_rule = "fifth_stand"`; use `fifth_sit` if your clinic times to
   the final sit).
4. Keep recording for 2 s after the last movement, then stop.

**Procedure (30-second chair stand)**
Same setup. On "go", stand and sit as many times as possible in 30 s. ptvision counts full stands whose
top is inside the 30 s window starting at the first seat-off (or at `--t0` if you give the start time).

**What ptvision refuses or flags**
- Frontal view, subject leaving the frame, fewer/more than the expected repetitions: report marked `fail`.
- Another person in frame, camera moving, low resolution, variable frame rate: `warn`.

Run: `ptv analyze clip.mov --protocol sts_5x --age 71 --height-m 1.68 --sex f`
