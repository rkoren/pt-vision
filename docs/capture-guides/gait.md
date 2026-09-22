# Capture guide: walkway gait (sagittal)

Protocol `gait_sagittal`. Matches the single-camera setups validated in Stenum et al. 2021 (PLoS Comput
Biol) and 2024 (PLOS Digital Health): timing errors around 0.02 s, gait speed within 0.02–0.04 m/s of
motion capture, in healthy adults, post-stroke and Parkinson's.

**Camera**
- Phone in landscape on a tripod, lens about 1 m above the floor.
- 4–5 m from the walking line, perpendicular to it, at the middle of the walkway. The frame should hold
  at least 3 m of walkway so several full steps are visible.
- Whole body, including both feet, visible for the entire crossing. 30 fps minimum, 60 fps better.
- Even light, plain background if possible, only one person walking.

**Walkway and patient**
- Mark a start point at least 2 m before the frame edge and an end point 2 m past the other edge, so the
  frame holds steady-state walking rather than acceleration or slowing. Reference values for normal speed
  are only valid for steady comfortable walking (Bohannon & Williams Andrews 2011).
- "Walk at your usual comfortable pace to the far mark." Several passes in each direction; each pass is
  a bout and the report pools their steady cycles.
- Record the subject's height (`--height-m`); without it speed and step length are not reported.

**What the report gives you**
- Tier 1 (timing): gait speed, cadence, stride time, stance/swing/double-support percentages, step-time
  and stance-time asymmetry, number of steady cycles.
- Tier 2 (angles, near side): peak knee flexion in swing and peak hip flexion, with the published
  single-camera error. Step and stride length as bout means only; per-step lengths are biased by where
  the person is in the frame.
- Not reported: ankle angles, anything frontal-plane. Foot drop should be looked for through hip and knee
  compensation, not ankle numbers.

**What ptvision refuses or flags**
- Fewer than 3 steady gait cycles, or the subject not crossing the frame (walking toward the camera):
  `fail`.
- Weak far-side foot keypoints, camera motion, another person in frame: `warn`.

Run: `ptv analyze walk.mov --protocol gait_sagittal --height-m 1.72 --age 68 --sex m`
