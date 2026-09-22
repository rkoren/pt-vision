# Backlog

The single list of things we are not doing right now. See `CLAUDE.md` for the practice: add thoroughly,
re-assess critically on pick-up, drop stale items with a reason. Sections: Now, Next (ordered), Later,
Ideas, Done, Dropped. Effort: S (hours), M (a day or two), L (several days).

Entry template:

```
### Bnn Title
Area · Added YYYY-MM-DD · Effort S/M/L · Depends on: …
Why: …
What: …
Refs: …
Pick-up check: …
```

---

## Now

_(nothing in progress)_

## Next (ordered)

### B02 Angle-band rules in protocol TOML
Clinical · Added 2026-09-02 · Effort S · Depends on: B01 (folded in)
Why: The viewer's "rules" color mode needs declarative bands a PT can read and edit.
What: `[[rules]]` entries (`angle`, `ok`, `warn`, `segments`, `note`) validated by `RuleSpec`; first bands in
`sts_5x.toml` / `sts_30s.toml` marked provisional. Confirm bands with the PT partner and record the source.
Refs: plan Part 3 schema.
Pick-up check: after real 5xSTS clips exist (B03), replace guessed bands with observed healthy ranges.

### B03 Real 5xSTS validation clips and committed fixture
Validation · Added 2026-09-01 · Effort M · Depends on: recording sessions with volunteers
Why: Slice 1 is validated on synthetic data only. Tier-1 claims need real clips against a stopwatch.
What: 3–5 clips per `docs/capture-guides/sts.md` with a stopwatch in frame plus one bad clip; annotate
seat-off/stand/seated frames; `scripts/make_pose_fixture.py` → `tests/fixtures/sts_5x_pose_cam0.parquet`
plus `sts_5x_events_manual.csv`; integration test asserting events within 2 frames, time within 0.1 s;
fill `docs/validation/sts-v0.md`; settle `start_rule`/`end_rule` with the PT partner.
Refs: Bertrand 2026 (ICC 0.995), Hwang 2026 (RGB STS).
Pick-up check: verify the capture guide still matches the segmenter parameters.

### B06 Left/right swap mitigation in sagittal view
Pose · Added 2026-09-01 · Effort M–L · Depends on: real clips showing the problem
Why: The #1 failure mode for side-specific angles (Sports2D issue #39); unsolved in open source.
What: first a pre-filter: detect trajectory jumps between paired keypoints and reassign by temporal
continuity (velocity-consistent assignment); expose a swap-rate metric in quality checks (exists as a
warn). Later: OpenSim inverse kinematics via the `opensim` extra (maintainer's recommended fix).
Refs: Sports2D #39, Pose2Sim #102.
Pick-up check: measure swap rate on real clips first; if <5 % of frames on the near side, deprioritise.

### B07 Manual event correction in the app
UI · Added 2026-09-02 · Effort M · Depends on: B01
Why: Validation needs ground-truth frames; clinicians will want to nudge a mis-detected seat-off.
What: draggable event markers on the timeline; save corrected events as a new run artifact (never
overwrite auto events); export the validation CSV format from `docs/validation/sts-v0.md`.
Pick-up check: make sure provenance records "events corrected manually".

### B08 Angle arcs and numeric labels at joints
UI · Added 2026-09-01 · Effort S · Depends on: none
Why: PTs read angles off the joint, not off a plot (Sports2D-style overlays).
What: draw an arc and value at knee/hip/trunk on the near side in `viz/skeleton.py` for both the CLI overlay
and the app; toggle in the panel.
Pick-up check: keep text legible on Retina and in the H.264 overlay export.

### B09 Longitudinal comparison with MDC
Clinical · Added 2026-09-01 · Effort M · Depends on: B03 (measurement error), data store
Why: "Is this change real?" is the product; nothing open-source does it.
What: `clinical/longitudinal.py`: `compare_runs(prev, cur)` with minimal detectable change thresholds from a
`norms/mdc.csv`; `ptv compare --patient --episode`; report section "change since last visit"; warn when
method versions differ between runs.
Refs: ADR-0007.
Pick-up check: needs an MDC per metric from our own test-retest data or literature.

## Later

### B10 Slice 3: range-of-motion protocols
Clinical · Added 2026-09-01 · Effort M
What: knee, hip, elbow, shoulder flexion/abduction protocols; rep detection by peak finding; report whole
degrees ± MDD (~5° shoulder); no ankle (unreliable). Goniometer self-validation.
Refs: Sabo 2023 (ceiling is goniometry's own reliability), van den Hoorn 2025.

### B11 Slice 4: Timed Up and Go segmentation
Clinical · Added 2026-09-01 · Effort L · Depends on: B04
What: reuse STS for sit→stand; walk out, turn (hip-x reversal + shoulder-width change), walk back, turn, sit;
total time tier 1 (Bohannon 2006 norms); phases tier 2. Study `tugturn` (arXiv 2602.21425) for design.

### B12 Reference movement envelopes and "deviation" color mode
Clinical · Added 2026-09-02 · Effort L · Depends on: B04, public datasets
Why: The clinically interesting color signal is "how far from expected at this phase", not a fixed band.
What: normative gait-cycle envelopes from Fukuchi 2018 (CC BY) and Van Criekinge 2023 (CC0, incl. stroke);
patient-own baseline envelopes for STS; a third color mode in `viz/status.py`.

### B13 Left/right symmetry color mode; `sides = "both"` rule option
UI/Clinical · Added 2026-09-02 · Effort S–M
What: color the lagging side when both sides are visible (frontal/gait); allow rules to mirror onto the
far side explicitly instead of leaving it neutral.

### B14 Side-by-side visit playback
UI · Added 2026-09-02 · Effort M · Depends on: B09
What: two sessions in the viewer synced by event (first seat-off / first heel strike).

### B15 Annotated video/GIF and PDF export
UI · Added 2026-09-02 · Effort S–M
What: export the current view with overlay for the patient; PDF of the report.

### B16 Face blurring and de-identified trial bundles
Data · Added 2026-09-02 · Effort M
What: blur faces in overlays/thumbnails (keypoints give the face box); export/import a trial bundle (zip)
without the source video for sharing with the PT partner.

### B17 macOS `.app` packaging
Infra · Added 2026-09-02 · Effort L · Depends on: B34
What: PyInstaller `--onedir`; arm64-only, macOS 14+ (onnxruntime and opencv ship no universal2 wheels);
move `base_library.zip` to Resources for notarization; `CFBundleDocumentTypes` so drag-drop works after
packaging; LGPL notice, Qt source offer and relink instructions in `NOTICE`; ship opencv's third-party
license file. Expect 500–700 MB. Windows later.
Refs: pyinstaller #8927, Qt LGPL obligations page.

### B18 MediaPipe backend
Pose · Added 2026-09-01 · Effort S
What: `mediapipe` extra with the Tasks API `PoseLandmarker` (BlazePose-33, heels/foot index) remapped to
Halpe-26; use for cross-checks and as a fallback.

### B19 ViTPose++ backend
Pose · Added 2026-09-01 · Effort M
What: rtmlib `ViTPose` class with HF weights (verifiably Apache-2.0) as the license fallback for the
Halpe-26 training-data concern (ADR-0002); needs a foot-keypoint strategy.

### B20 GPU box support
Infra · Added 2026-09-01 · Effort S
What: `onnxruntime-gpu` extra with uv `conflicts` against `onnxruntime`; benchmark `performance` mode on the
RTX 5070 Ti; pin its archive sha256 in `models.toml`.

### B21 CoreML execution provider revisit
Pose · Added 2026-09-01 · Effort S
What: CoreML EP fails at inference on ORT 1.29 for these models; re-test on each ORT release or re-export
the models with static shapes.

### B22 30-s chair stand norms and SPPB balance protocol
Clinical · Added 2026-09-01 · Effort S–M
What: Rikli & Jones 1999 / CDC STEADI norms after PT verification; SPPB side-by-side/semi-tandem/tandem hold
timing (Bertrand 2026 ICC 0.81–1.0).

### B23 STS peak power and foot-drop compensation flag
Clinical · Added 2026-09-01 · Effort M · Depends on: B05, subject mass
What: Hwang 2026 power surrogate; foot drop detected via hip/knee swing compensation, never via ankle angle.

### B24 Live webcam mode
UI · Added 2026-09-01 · Effort L
What: streaming pipeline with the lightweight model (86 fps on M4 Pro CPU); deferred by design decision.

### B25 Camera calibration and two-iPhone triangulation
Pose · Added 2026-09-01 · Effort L
What: intrinsics/distortion from a checkerboard clip; Pose2Sim-style sync + triangulation producing
`pose/world.parquet` (`coord_space=world_m`); OpenCap-style capture guide.

### B26 More filters and derived signals
Kinematics · Added 2026-09-01 · Effort S
What: Kalman and GCV-spline options; angular velocity/acceleration outputs.

### B27 Person selection by click in the app
UI · Added 2026-09-02 · Effort S · Depends on: B01
What: click a skeleton to make it the analysed subject; re-run downstream stages.

### B28 Data-store index and clinician annotations
Data · Added 2026-09-01 · Effort S–M
What: SQLite index rebuilt from the JSON tree when listings slow down; free-text annotations per run.

### B29 Validation harness on public datasets
Validation · Added 2026-09-01 · Effort L
What: OpenCap SimTK dataset (registration) replicating Horsak 2025's single-view design; Fukuchi 2018 and
Van Criekinge 2023 for normative bands; UI-PRMD (public domain) for rehab exercises.

### B30 `experiments/` research project
Research · Added 2026-09-01 · Effort L
What: separate uv project with torch cu128 for the 5070 Ti; evaluations of SAM 3D Body (posture),
MotionBERT lifting, learned STS-phase / gait-event detectors, Godosim synthetic data, GAVD pathological
gait; watch OnlineHMR (CVPR 2026) and BiomechGPT releases.

### B31 LLM interpretation over computed metrics
Clinical · Added 2026-09-01 · Effort M
What: grounded plain-language summaries from metrics + norms only; never from video (VLMs on raw video
score near chance in 2026 studies, arXiv 2604.08294, 2511.17727).

### B32 Robustness pass
Infra · Added 2026-09-01 · Effort M
What: HDR 10-bit inputs test, corrupt-file handling, structured logging and `--verbose`, Windows path pass,
batched inference for speed.

### B33 Browser-based viewer / shareable HTML bundle
UI · Added 2026-09-02 · Effort M
What: FastAPI + canvas with `requestVideoFrameCallback` (frame-accurate in all 2026 browsers); Range-request
video serving. Rejected for v1 in favour of the desktop app; keep as the sharing story.

### B34 ffmpeg distribution story
Infra · Added 2026-09-02 · Effort S–M · Depends on: decide before B17 · Priority raised 2026-09-03 (Windows portability)
Why: The Homebrew ffmpeg we develop against is a GPL build (`--enable-gpl --enable-libx264`) and cannot be
redistributed inside a commercial-friendly app.
What: either keep "ffmpeg on PATH" as a documented prerequisite, or bundle an LGPL-only build and switch
macOS encoding to `h264_videotoolbox` (verified available here).

### B35 Bundle-size diet
Infra · Added 2026-09-02 · Effort M
What: pyarrow is 123 MB for Parquet I/O only; consider a lighter Parquet path for the app build.

### B36 Subprocess pipeline runner
UI · Added 2026-09-02 · Effort M
What: `QProcess` running `ptv … --progress json` for crash isolation and hard kill, if the in-process
QThread runner ever stalls or crashes the GUI (ORT releases the GIL, so this is not expected).

### B37 Colour-blind-safe palette toggle
UI · Added 2026-09-02 · Effort S
What: alternative to the red–green status gradient (e.g. blue–orange), selectable in the panel.

### B38 Re-normalize old trials on open
UI · Added 2026-09-02 · Effort S
What: trials normalized before the 15-frame GOP change seek slowly; offer to re-encode on open.

### B39 `opencv-python-headless` swap
Infra · Added 2026-09-02 · Effort S
What: if the Linux opencv wheel's bundled Qt plugin path fights PySide6's offscreen platform in CI, switch
to headless via the existing uv override pattern (nothing in `src/` uses highgui).

### B41 Manual QA of the viewer on a real display
UI · Added 2026-09-03 · Effort S
What: 1080p/30 playback smoothness and Retina line crispness; Finder drag-drop; menu shortcuts incl.
⌘←/→; real RTMPose run with cancel and close mid-run; palette legibility on real footage; spot-check
three frames of `ptv analyze --color-by rules` overlay against the app.

### B42 Pose-only view polish
UI · Added 2026-09-03 · Effort S
What: hide the empty Measurements/Events groups when a trial has no protocol; show the tracker's
primary-person choice and let the user switch (overlaps B27); remember splitter sizes.

### B43 Ingest option for keyframe interval on existing trials
Infra · Added 2026-09-03 · Effort S · Depends on: B38
What: `ptv trial reencode` to bring pre-GOP-change trials up to date; record `gop` in capture.json.

### B45 OpenCV-only ingest fallback (no external ffmpeg)
Infra · Added 2026-09-03 · Effort M · Depends on: decide with B34
Why: ffmpeg on the PATH is the biggest install hurdle on Windows and the Homebrew build is GPL.
What: when ffmpeg is absent, decode with OpenCV's bundled LGPL FFmpeg (rotation metadata is applied
automatically), resample to constant frame rate by index, and write with `cv2.VideoWriter` (MPEG-4 Part 2,
short GOP). Record `ingest.method` in capture.json. Trade-off: MPEG-4 Part 2 plays in the app but is less
universal than H.264 for shared overlay clips; keep ffmpeg (or a bundled LGPL build) as the preferred path.
Pick-up check: verify `cv2.VideoWriter` H.264 availability in the current opencv-python wheels first.

### B46 Cross-platform distribution via `uv tool install`
Infra · Added 2026-09-03 · Effort S–M
Why: PyInstaller bundles are 500–700 MB and unsigned Windows executables trigger SmartScreen; early PT
partners need something lighter and identical on Mac and Windows.
What: publish `ptvision` (with `app` extra) to a package index or Git tag; one-line installer scripts
(PowerShell and bash) that install `uv`, then `uv tool install "ptvision[app]"`, then ffmpeg via
winget/brew (until B45); `ptv doctor` to check ffmpeg, weights, and Qt. Bundles/signing remain B17.
Pick-up check: uv tool install of an extras-bearing package must be supported by the uv version in use.

### B47 Portable GPU acceleration: DirectML (Windows) and OpenVINO (Intel) providers
Pose · Added 2026-09-03 · Effort S–M · Depends on: Windows test machine
Why: mid-range Windows laptops run the balanced model at roughly a third of the M4 Pro's 28 fps;
DirectML uses any DirectX 12 GPU including Intel integrated graphics without a CUDA install.
What: `onnxruntime-directml` extra on Windows (uv `conflicts` with `onnxruntime`), `--device dml`;
benchmark lightweight/balanced on the 5070 Ti box with DirectML and CUDA; OpenVINO EP via rtmlib's
`openvino` backend for Intel CPUs/iGPUs. Extends B20/B21.
Pick-up check: confirm rtmlib's device map exposes DmlExecutionProvider or add it in our backend.

### B48 Cross-platform numeric consistency test
Validation · Added 2026-09-03 · Effort S · Depends on: B44
Why: ONNX Runtime CPU kernels can differ slightly between macOS arm64 and Windows x86_64; longitudinal
comparisons across a PT's machines need a stated tolerance.
What: CI artifact of the fixture clip's keypoints per platform; test asserting max keypoint deviation
< 1 px and identical primary-person selection; record tolerance in docs.

### B50 Revisit the project license before first public release
Infra · Added 2026-09-06 · Effort S · Depends on: first release or first outside contributor
Why: The Unlicense (ADR-0009) has no patent grant and no contributor terms; some corporate legal teams
reject public-domain dedications. Released snapshots stay public domain regardless of later changes.
What: decide between keeping the Unlicense, 0BSD, or Apache-2.0 (patent grant, contributor clarity);
update LICENSE, pyproject, NOTICE, README, CLAUDE.md, and add a CONTRIBUTING note.
Pick-up check: has anything been published or contributed since ADR-0009?

### B51 Pose2Sim parity test is referenced by CI but does not exist
Infra · Added 2026-09-06 · Effort S · Depends on: `opensim` extra installable in CI
Why: `.github/workflows/ci.yml` has a manual `opensim-parity` job running
`tests/integration/test_pose2sim_parity.py`, planned in ADR-0002 (compare our angles with
`Pose2Sim.common.points_to_angles` on a fixture) but never written. The job would fail if dispatched.
What: write the parity test (skip when Pose2Sim is not importable), confirm the `opensim` extra resolves
on ubuntu-24.04, run the job once by hand.
Pick-up check: Pose2Sim version pinned in pyproject still matches the vendored PROVENANCE commit.

### B52 Real walkway gait validation clips
Validation · Added 2026-09-06 · Effort M · Depends on: volunteers, a measured walkway
Why: Slice 2 is validated on synthetic walking only. Speed and step length need a ground truth.
What: film 3+ passes each direction per `docs/capture-guides/gait.md` with a tape measure or two floor
marks a known distance apart in frame (so speed can be checked without a stopwatch), 30 and 60 fps;
annotate heel strikes on one pass; compare timing (target ±2 frames), speed (target ±0.05 m/s) and
step length; commit a pose Parquet fixture; record results in `docs/validation/gait-v0.md`.
Pick-up check: confirm the Bohannon 2011 table transcription against the paper at the same time.

### B53 Gait timing at 60 fps and the asymmetry noise floor
Clinical · Added 2026-09-06 · Effort S
Why: at 30 fps a step is ~16 frames, so one frame of event jitter is ~6 % apparent asymmetry; the
symmetry metrics now carry that as their error, but 60 fps halves it.
What: recommend 60 fps in the capture guide once real clips confirm event accuracy; consider sub-frame
event refinement (parabolic peak interpolation on the Zeni signals).

### B54 Depth-perspective correction for walkway gait
Kinematics · Added 2026-09-06 · Effort M · Depends on: B52
Why: Sports2D corrects the far-limb-looks-smaller effect using camera distance; we dropped that term in
v1 (subject assumed at constant depth). Stenum reports step length bias vs position in frame.
What: add optional `distance_m`/`fov_deg` to the capture and the perspective term to `ScaleModel`;
quantify the gain on B52 clips before keeping it.

### B55 Video timing-integrity check at ingest
Infra · Added 2026-09-09 · Effort S
Why: Timing metrics assume every frame is real and evenly spaced. A dropped or duplicated frame from
variable frame rate, resampling, or a bad container silently shifts every event. A stopwatch in frame
would reveal this, but we are replacing that with a hand-timed total (session 1 shot list), so the
check must come from the file itself.
What: in `pipeline.ingest`/`quality.check_capture`, compare `n_frames / fps` with the container
duration for both the source and the normalized file (fail if they differ by > 0.05 s or by more than
one frame per 10 s); record both in `capture.json`; warn when the source was VFR with a large spread
between nominal and average frame rate. Optionally decode the normalized file once and count frames
to catch a wrong `nb_frames` header.
Pick-up check: confirm whether ffprobe's `nb_frames` is reliable for the phone codecs actually seen in
session 1 (HEVC .mov); if not, count packets.

### B56 Spurious short-lived tracks from background objects
Pose · Added 2026-09-22 · Effort S · Found on: session-1 clips
Why: On every real clip the detector briefly fires on furniture, a plush toy, a mirror reflection
(4–16 frames each, scores 0.2–0.5). They get person ids, clutter the overlay ("id 3" on a toy) and
inflate `n_persons`; on the frontal clip the subject was assigned id 8 among 11 ghosts.
What: after tracking, drop tracks present < `min_track_s` (0.5 s) or with mean score < 0.4 unless
they are the only candidate; keep the raw detections in Parquet but mark tracks `spurious` so the
overlay hides them and quality checks ignore them; primary-person selection unchanged.
Pick-up check: confirm no legitimate short-lived second person (therapist stepping in) is dropped.

### B57 `camera_view` check is a hard fail at a borderline ratio
Quality · Added 2026-09-22 · Effort S · Found on: session-1 sts2 (ratio 0.37 vs threshold 0.35)
Why: A three-quarter view failed the sagittal check by 0.02 while all five reps and timings came out
clean. Timing metrics degrade gracefully with view angle; angles do not.
What: make it a band: ratio < 0.35 pass, 0.35–0.5 warn ("oblique view; angles less reliable"),
> 0.5 fail; record the ratio in the report as an estimated view angle; capture guide already says
perpendicular.
Pick-up check: re-derive the thresholds from the session-1 clips (pure sagittal ≈ ?, frontal 0.63).

### B58 Test-start rule when the recording begins standing
Clinical · Added 2026-09-22 · Effort S · Found on: session-1 sts1/sts2
Why: Both sagittal clips start with the subject standing and sitting down before the test; the
segmenter correctly ignores that descent (start = first seat-off) but the trajectory figure shows a
large pre-test excursion and normalisation uses it. Also the first stand overshoots to 1.1 because
the 95th percentile is set by later, lower stands.
What: normalise on the post-`test_start` window (or on seated/standing plateaus) and note "recording
began standing" in the report; confirm with the PT partner whether the clinic protocol starts seated.

### B59 Camera-motion warning on tripod clips
Quality · Added 2026-09-22 · Effort S · Found on: session-1 (3–5 px on every clip)
Why: Every clip warned about camera motion; the subject's own movement dominates the phase
correlation when the person fills much of a portrait frame.
What: mask the subject's bounding box out of the correlation (keypoints are known) or correlate on
the frame border; re-check the 2 px threshold at 1080p portrait.

### B60 Warn when a timed event is truncated by the clip boundary
Quality · Added 2026-09-22 · Effort S · Found on: session-1 sts1/sts2
Why: In both real clips the fifth stand fell on the last 1–3 frames of the recording, so
`stand_reached` was really "recording stopped"; measured times were 0.11 s and 0.53 s short of the
hand-timed 7.01 s and 10.2 s. The report gave a confident ± 0.07 s.
What: in `segment_sts` (and gait), if `test_start` or `test_end` is within `edge_margin_s` (0.3 s)
of the clip start/end, add a warning and a `truncated` flag on the metric; quality check
`test_truncated` → fail for the primary metric; note in the capture guide (already says record 2 s
after the last movement).
Pick-up check: apply the same idea to gait bouts that touch the clip edge.

### B40 Recent-trials list on the drop page
UI · Added 2026-09-02 · Effort S
What: QSettings-backed list of recently opened trials.

## Ideas

- Exercise rep counting with form feedback (Good-GYM-style state machines) once ROM protocols exist.
- Patient-facing summary language reviewed by the PT partner.
- Remote therapeutic monitoring (RTM) documentation export; commercial products lead with it.

## Done

- 2026-09-06 B04 + B05 Slice 2 gait (`gait_sagittal`): Zeni 2008 heel-strike/toe-off events, walking bouts,
  steady-state cycles, cadence, stance/swing/double support, step- and stance-time asymmetry with an error
  floor, gait speed and bout-mean step/stride length in metres from subject height (Sports2D-style height
  and floor-line estimate), sagittal knee/hip angles per cycle, Bohannon 2011 speed norms, gait report,
  app support. Synthetic-validated only (B52 for real clips).
- 2026-09-06 B44 Windows CI: `windows-latest` in both matrices, ffmpeg via Chocolatey, bash for every step,
  portable model cache (`PTV_MODEL_DIR`), `.gitattributes` enforcing LF, `workflow_dispatch` trigger.
  First green run on the `ci-windows` PR. Surfaced B51 (parity test referenced by CI but missing).
- 2026-09-03 B01 Desktop viewer app (`ptv app`): drop → progress with cancel → overlay playback with
  rule/confidence colouring, scrubbing, synced timeline, metrics panel; shared `viz/status` gradient
  also drives CLI overlays; angle side persisted; 15-frame GOP ingest; Qt tests offscreen in CI.
  Manual checks still pending on a real display: 1080p/30 playback smoothness, Retina crispness,
  Finder drag-drop, menu shortcuts (added as B41).
- 2026-09-01 Slice 0: scaffold, `ptv pose`, rtmlib Halpe-26 backend, Parquet + provenance, CI, benchmark.
- 2026-09-01 Slice 1: `ptv analyze` for 5xSTS / 30-s chair stand, synthetic-validated (real clips: B03).

## Dropped

- Streamlit dashboard: frame-synced overlays and scrubbing are awkward.
- OpenPose-based tooling: noncommercial license.
- QMediaPlayer for playback: millisecond-only positions, no frame index, undocumented seek tolerance, +332 MB.
- PyAV `av` wheels: bundle a GPL FFmpeg (use `basswood-av` if PyAV is ever needed).
