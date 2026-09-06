# Biomechanics and clinical-analysis layer: research report (September 2026)

Produced 2026-09-01 by a research agent during planning. Reproduced as delivered. Package status was
verified against PyPI / GitHub / anaconda.org / figshare APIs on 2026-09-01. Numbers come from fetched
abstracts or full texts unless explicitly flagged as unverified.

---

## A. Open-source biomechanics software: status and macOS arm64 / Python 3.11–3.12 compatibility

### A1. OpenSim: the situation changed in 2026; the docs are stale
**`pip install opensim` now works, and it ships a macOS arm64 (universal2) wheel.**
- PyPI `opensim` 4.6, Apache-2.0, uploaded 2026-06-23 (build 2). Wheels: `opensim-4.6-2-cp311-cp311-macosx_15_0_universal2.whl` (36.5 MB), same for cp312 and cp313; plus `manylinux_2_27_x86_64` and `win_amd64`. `requires_dist: numpy>=2.1`. Wheel tag `macosx_15_0` → requires macOS 15 or newer. No Linux aarch64 wheel.
- conda `opensim-org` channel also has osx-arm64 builds for 4.4–4.6 (py311/312/313), latest upload 2026-04-17.
- **Stale documentation trap:** the official Confluence "Conda Package" page still says macOS only supports x86_64 and lists Python 3.7–3.10. Wrong as of 2026.
- There is no `opensim` package on conda-forge.
- `opencap-processing`'s README says macOS is x86_64 only; that concerns their muscle-driven simulation stack (CasADi/Rosetta), not OpenSim IK.

### A2. AddBiomechanics: effectively web-service-only
- https://github.com/keenon/AddBiomechanics, 65 stars, last push 2026-08-03. README: not supportable for non-Stanford devs to build from source; the cloud application is coupled to their AWS resources; self-hosting would "effectively just join our cluster." Uploaded data is part of a public data-sharing effort. **Not self-hostable in practice.**
- nimblephysics (dependency): PyPI latest 0.10.52.2 (2024-12-02) publishes exactly one wheel (cp39 macOS universal2). Version 0.10.52.1 (uploaded out of order 2025-09-14) includes cp311 macOS universal2 and Linux wheels. No cp312/cp313 macOS wheel. Roughly a year dormant.

### A3. SKEL / HSMR: technically excellent, license kills commercial use
- SKEL (Keller et al., SIGGRAPH Asia 2023). License (skel.is.tue.mpg.de/license.html): Max-Planck-Gesellschaft; non-commercial scientific research, education, or artistic projects only; explicitly prohibits incorporation in a commercial product or service and training for commercial purposes.
- HSMR (CVPR 2025 Oral): MIT code, 630 stars, pushed 2026-03-06, but requires SKEL weights, so the pipeline inherits the MPG restriction. Successor SKEL-CF (arXiv 2511.20157).
- No published validation of HSMR joint angles against goniometry, marker-based IK, or OpenSim was found.

### A4. Lighter-weight Python kinematics libraries (all verified)

| Package | Latest / date | License | Py | macOS arm64 | GitHub last push | Verdict |
|---|---|---|---|---|---|---|
| ezc3d | 1.7.2 / 2026-08-07 | MIT | ≥3.10 | cp310–cp314 arm64 wheels | 2026-08-23 | Actively maintained; use for C3D I/O |
| kineticstoolkit | 0.17.0 / 2025-09-29 | Apache-2.0 | ≥3.10 | pure Python | 2026-08-03 | Maintained; no gait-event module |
| pyomeca | 2026.0.2 / 2026-08-11 | Apache-2.0 | ≥3.6 | pure Python | 2026-08-11 | Dormant 2024→2026, just revived |
| Pose2Sim | 0.10.49 / 2026-07-10 | BSD-3 | ≥3.11 | pure Python | 2026-08-24 | Very active |
| Sports2D | 0.8.34 / 2026-07-10 | BSD-3 | ≥3.11 | pure Python | 2026-08-25 | Very active |
| gaitalytics | 0.2.2 / 2025-02-07 | MIT | ≥3.11 | pure Python | 2026-04-22 | Semi-dormant; operates on C3D from marker mocap, not video |
| biorbd | — | — | — | — | — | Not on PyPI (conda only) |
| biomechzoo | — | — | — | — | — | MATLAB |

- OpenSim-free angle computation directly from keypoints is well trodden and validated (Stenum 2021, Sports2D). For sagittal-plane clinical work it is the pragmatic default; OpenSim IK buys anatomical constraints and 3D at the cost of a scaling step and a heavy dependency.

### A5. Sports2D / Pose2Sim angles, conventions, filters (from source)
- Sports2D default joint angles: ankle dorsiflexion, knee flexion, hip flexion, shoulder flexion, elbow flexion, left and right. Conventions (README): ankle dorsiflexion −90° when the foot is aligned with the shank; knee flexion 0° when the shank is aligned with the thigh. Segment angles measured anticlockwise from horizontal.
- Filtering (Config_demo.toml): default Butterworth, 6 Hz cut-off, order 4, zero-phase. Alternatives: `acc_minimizing` (Whittaker–Henderson, inspired by AddBiomechanics), `kalman` (trust_ratio 500), `one_euro` (4 Hz, beta 1.5; blunts RoM), `gcv_spline` (auto), gaussian, LOESS, median, butterworth_on_speed. Pre-filtering: Hampel outlier rejection (window 7), gap interpolation for gaps <100 frames, `min_chunk_size = 10`.
- Pixel→metres: ratio of participant height in metres to height in pixels; camera angle and floor level auto-estimated or manual.
- Pose2Sim multi-camera accuracy (Pagnon et al., Sensors 2022;22(7):2712): 4th-order 6 Hz Butterworth on 3D points; CMC > 0.9 sagittal; mean errors 3.0° walking, 4.1° running, 4.0° cycling; stride-to-stride SD 1.7–3.2°.
- Sports2D ROM reliability (Korea J Sports Med 2025;43(4):254–264): reliable for elbow, neck, knee, hip, trunk; unreliable for ankle. *Abstract-level claim; full text not fetched.*

---

## B. Video-based gait analysis: algorithms and validation

### B1. Gait event detection without force plates
**Zeni et al. 2008** (Gait Posture 27(4):710–714, PMID 17723303, PMC2384115), the reference method.
- Coordinate method: heel strike = frame where the heel marker reaches maximum anterior displacement relative to the sacrum/pelvis; toe-off = maximum posterior displacement of the toe marker relative to the pelvis. Velocity method = zero-crossings of the same relative signals.
- Accuracy vs vertical GRF (verbatim): 94% of treadmill events from healthy subjects within one frame (0.0167 s); in impaired populations 89% within two frames; overground 98% within two frames. Populations: healthy young n=7, MS n=7, stroke n=4 (treadmill); healthy n=5 overground.
- Why it matters: needs only heel/toe/pelvis positions in the direction of travel, exactly what a sagittal 2D pose model gives. **The single highest-value algorithm to port.**

From 2D video specifically:
- Stenum 2021 achieved MAE 0.02 s for step/stance/swing/double-support time from a single sagittal camera with OpenPose (≈½–1 frame at 30 fps).
- `tugturn` (arXiv 2602.21425, Feb 2026): Python markerless pipeline using a relative-distance strategy for heel strike / toe off within valid gait windows; a reusable open reference for windowed event detection during TUG.
- ML approaches: LSTM vs kinematic comparison (arXiv 2503.00794) reports LSTMs performing comparably to kinematic methods; two-step deep-learning heel-keypoint detection (Med Biol Eng Comput 2025, PMC11695559).
- Could not verify the "14 ms / 17 ms RMSE" figure that circulates for Zeni; use the within-N-frames figures from the primary paper.

### B2. Spatiotemporal parameters from single-camera video vs gold standard
**Stenum, Rossi & Roemmich 2021, PLoS Comput Biol 17(4):e1008935** (PMC8099131): single sagittal camera, OpenPose BODY_25, n=31, vs marker-based mocap. Step/stance/swing/double-support time MAE 0.02 s each (0.01 s at participant-mean level). Step length mean absolute difference 0.049 m per step, <0.020 m for participant means. Gait speed up to 0.04 m/s (greatest 0.09). Sagittal kinematics MAE: hip 4.0°, knee 5.6°, ankle 7.4°. Systematic bias: step-length estimation depends on position along the camera's field of view; errors offset when averaged across the bout. Hip and ankle angles worst at the periphery; knee invariant to position. Code: https://github.com/janstenum/GaitAnalysis-PoseEstimation.

**Stenum, Hsu, Pantelyat & Roemmich 2024, PLOS Digit Health 3:e0000467**: n=32 controls, 44 post-stroke, 19 Parkinson's; sagittal and frontal; OpenPose. Stroke sagittal: step time diff/error 0 and 1 mocap frame; step length ~1 and 3 cm; speed 0.02 and 0.04 m/s. PD sagittal: step time 0 and 1 frame; step length −1 and 2 cm; speed −0.02 and 0.03 m/s. Sensitive to within-participant change, r ≥ 0.949.

**Barzyk et al. 2024, Sensors** (PMC11644854): n=8 stroke patients, single 2D smartphone vs 12-camera Vicon. Sagittal hip/knee/ankle r ≥ 0.79, RMSE ≤ 4.6°, MAE ≤ 3.2°. Spatiotemporal ICC 0.78–0.99: speed 0.997, cadence 0.987, step length 0.781. Underpowered.

**OpenCap (Uhlrich et al. 2023, PLoS Comput Biol 19(10):e1011462)**: two or more smartphones; n=10 healthy; joint angle MAE 4.5°; joint moment MAE 1.2% bodyweight × height. Local run needs Windows/Ubuntu and a CUDA GPU with 4–24 GB. Dataset (raw video + mocap + GRF + EMG) on SimTK.

**OpenCap validation scoping review, Front Digit Health 2026** (doi 10.3389/fdgth.2026.1882536): 51 studies, 23 quantitative. RMSE: hip sagittal 5.99° (IQR 4.60–6.90), frontal 3.93°, transverse 5.22°; knee sagittal 6.07° (4.93–7.36); ankle sagittal 7.30° (5.99–10.40); upper limb sagittal 13.81–30.02°, frontal 23.6–52.17°. Accuracy "pathological < physiological."

**Cheng et al. 2025, Gait Posture 120:150–160 (PMID 40250127)**: OpenCap MAE 4.1° for 3D joint angles, higher for rotations; OpenPose ICCs 0.89–0.994 for spatiotemporal parameters and MAE < 5.2° for 2D hip and knee sagittal angles; ankle poor (ICCs 0.37–0.57, MAE 3.1–9.77°).

**OpenCap vs OptoGait, Sensors 2026;26(4):1234**: r = 0.951 speed, 0.864 stride length; systematic bias, not interchangeable without correction. *Abstract-level.*

### B3. Kinematic errors: 2D sagittal vs 3D monocular vs multi-camera

| Approach | Reported error | Source |
|---|---|---|
| 2D single sagittal camera, direct keypoint angles | hip 4.0°, knee 5.6°, ankle 7.4° MAE | Stenum 2021 |
| 2D single smartphone, stroke | RMSE ≤ 4.6° sagittal (n=8) | Barzyk 2024 |
| Monocular 3D (CameraHMR/SMPL + OpenSim IK) | validity RMSD 5.5 ± 1.1°; test-retest 3.0 ± 1.0° | Horsak 2025 |
| Monocular 3D + physics (OpenCap Monocular) | 4.8° MAE rotational, 3.4 cm pelvis translation | arXiv 2603.24733 |
| Two-camera OpenCap | 4.1–4.5° MAE overall; 6.0–7.3° RMSE per lower-limb joint | Uhlrich 2023 / reviews |
| Common clinical acceptability threshold | 2–5° | widely cited |

**Horsak et al. 2025, J Biomech 193:112986 (PMID 41046587)**: CameraHMR on single-view OpenCap videos → OpenSim IK; n=19, four gait patterns (physiological, crouch, circumduction, equinus); comparable to OpenCap (p > 0.05); validity RMSD 5.5 ± 1.1°, reliability 3.0 ± 1.0°; "further refinement is needed to reach clinically acceptable accuracy thresholds." Takeaway: one camera ≈ two cameras for sagittal lower-limb angles.

**OpenCap Monocular, arXiv 2603.24733 (2026-03-25)**: WHAM + optimisation + biomechanically constrained skeleton + physics/ML kinetics from a single smartphone: 4.8° MAE rotational, 3.4 cm pelvis; GRF comparable to or better than two-camera OpenCap. No public code repo yet (enumerated `opencap-org` repos). Related: MonoMSK (arXiv 2511.19326).

### B4. Normative reference data
- **Gait speed: Bohannon & Williams Andrews 2011, Physiotherapy 97(3):182–189 (PMID 21820535)**: 41 articles, 23,111 subjects, sex × decade. Range from 143.4 cm/s (men 40–49) to 94.3 cm/s (women 80–99). Limitation: "may not be useful as a standard of normal if gait is measured over short distances from the command 'go' or if a turn is involved." Correct second-author citation is Williams Andrews A. The <0.8 m/s adverse-outcome cut-point is not from this paper.
- Cadence / step length: weaker evidence. Mobbs et al., Sensors 2025 (PMC11768510) normative database across decades, but n = 6 in the 71–80 stratum and not publicly released.
- Public kinematic datasets (licenses verified via figshare API): Fukuchi et al. 2018 (42 subjects, overground + treadmill, CC BY 4.0, doi:10.6084/m9.figshare.5722711); Schreiber & Moissenet 2019 (50 adults, 5 speeds, CC BY 4.0, doi:10.6084/m9.figshare.7734767); Van Criekinge et al. 2023 (138 able-bodied 21–86 y + 50 stroke survivors, full-body kinematics, kinetics, EMG; CC0 on items checked; doi:10.6084/m9.figshare.c.6503791); Camargo et al. 2021 (22 participants; Mendeley Data, check per part). **Fukuchi + Van Criekinge are the two to build normative bands and a validation harness on.**

### B5. Clinical gait deviations → measurable kinematics
Taxonomy used in a video-CV study (Reddy et al. 2025, PLOS Digit Health): normal, circumduction, Trendelenburg, antalgic, crouch, Parkinsonian, vaulting; 86.5% accuracy with frontal and sagittal views combined.

| Deviation | Signature | View | Feasibility from 1 camera |
|---|---|---|---|
| Antalgic | reduced stance time on affected side; step-time/stance-% asymmetry | sagittal | easy (pure timing) |
| Reduced knee flexion / stiff knee | peak swing knee flexion below ~60° | sagittal | good (knee is the most accurate angle and position-invariant) |
| Foot drop / steppage | reduced dorsiflexion at initial contact + hip/knee compensation | sagittal | hard directly (ankle worst joint); detect via compensation |
| Reduced arm swing (PD) | shoulder ROM, L/R asymmetry | sagittal | relative asymmetry only (upper-limb RMSE 13.8–30°) |
| Crouch | persistent stance knee flexion | sagittal | good (errors rise in crouch) |
| Circumduction | lateral hip abduction arc in swing | frontal | worst case for markerless |
| Trendelenburg | contralateral pelvic drop | frontal | hard; second view; false positives |
| Vaulting | contralateral plantarflexion / COM rise | sagittal | depends on ankle |
| General asymmetry | symmetry index on step time, length, stance % | sagittal | easy and robust |

Design principle: prefer ratios and timings over absolute angles, and sagittal over frontal.

---

## C. Functional clinical tests from single-camera video

### C1. 5xSTS / 30-second chair stand: the strongest evidence in this report
**Bertrand et al. 2026, PLOS Digit Health, doi:10.1371/journal.pdig.0001172** (PMC12773808): n = 228 adults with chronic disease; LiDAR-enabled iPad Pro on a tripod ~2.5 m away, lens ~0.5 m high, 34 landmarks at ~30 Hz. ICC vs clinician: 5xSTS 0.995 (0.983–0.998); TUG 0.962 (0.927–0.977); 30s-STS 0.928 (0.895–0.949); gait speed 0.868; tandem stand 0.812; side-by-side / semi-tandem perfect agreement; SPPB total weighted κ = 0.808. Tech-related data loss 3.1%. Caveat: LiDAR depth, not plain RGB; timing/counting tasks degrade gracefully.

**Hwang et al. 2026, J Cachexia Sarcopenia Muscle 17(1):e70208**, plain 2D RGB: n=129 older adults; validation subgroup n=20 vs mocap + force plate. iPad Pro 11 RGB, 3 m perpendicular to the sagittal plane, 0.8 m high, 800×600, 30 fps. Peak STS power ICC 0.94, R² 0.95, RMSE 66.6 W. Knee flexion RMSE 5.9°, trunk flexion RMSE 4.0°.

**Norms: Bohannon 2006, Percept Mot Skills 103(1):215–222.** 5-rep STS worse-than-average cutoffs: >11.4 s (60–69), >12.6 s (70–79), >14.8 s (80–89).

### C2. Timed Up and Go
- Bertrand 2026: ICC 0.962 vs clinician.
- `tugturn` (arXiv 2602.21425): open Python pipeline segmenting stand → first gait → turn → second gait → sit via spatial thresholds; relative-distance heel-strike/toe-off; TOML-configured; HTML reports.
- PMC12762308 (2025): TUG subtask segmentation with AlphaPose + MotionBERT. Electronics 2025;14(23):4650: smartphone TUG phase segmentation, MAE 0.42 s total (abstract-level).
- Norms: Bohannon 2006, J Geriatr Phys Ther 29(2):64–68: mean 9.4 s (≥60 y); 8.1 s (60–69); 9.2 s (70–79); 11.3 s (80–99).

### C3. ROM vs goniometer
**Sabo et al. 2023, IEEE J Transl Eng Health Med** (PMC10712662), n=97, five pose libraries vs goniometry:

| Joint | Best automated (Spearman ρ) | Human manual annotation ρ |
|---|---|---|
| Elbow extension | .722 (Detectron) | .746 |
| Knee extension | .608 (MoveNet-Thunder) | .693 |
| Shoulder flexion | .632 (MoveNet-Thunder) | .661 |
| 5th finger extension | .786 (custom) | .754 |
| Ankle flexion | not significantly correlated | .290 |

The ceiling is goniometry's own reliability, not the pose model.

**van den Hoorn et al. 2025, Shoulder & Elbow** (PMC12331643), n=17, Apple Vision 2D pose vs 3D mocap, shoulder: R² > 0.98 overall; abduction and flexion MDD ≈ 5° but 2D overestimates by 2–25° at greater ranges; extension MDD 3.1°; external rotation MDD 7.9° and 5.4°; internal rotation limited resolution.

Bottom line: elbow flexion, knee flexion, shoulder abduction/flexion from a plane-aligned 2D view; not ankle; not shoulder rotations. Report whole degrees with an explicit ±MDD band.

### C4. Squat / lunge / step-down / single-leg stance / Y-balance / functional reach
- Frontal Plane Projection Angle: r = 0.78 with 3D knee abduction in single-leg squat; inter-tester ICC 0.97–0.99; within/between-day ICC 0.72–0.91; common threshold >10° valgus/varus on lateral step-down. *Search-summary level; verify primary sources.*
- Single-leg stance / tandem stand: Bertrand 2026 ICC 0.812 to perfect agreement.
- Y-balance and functional reach: no validated single-camera implementation; need metric scale, reintroducing perspective error. Do not attempt first.

### C5. Berg Balance Scale
No validated single-RGB-camera scorer; existing automation is Kinect/depth or IMU based. Structural poor fit (14 heterogeneous items, object interaction, 360° turns, rater judgement). Do SPPB instead.

---

## D. Public datasets

### D1. Gait video with clinical labels
- **GAVD** (github.com/Rahmyyy/GAVD; arXiv 2407.04190; IEEE Access 2025): 1,874 sequences, 400+ subjects; **annotations and metadata only, no video files** (YouTube IDs; link rot acknowledged); annotations MIT. Baselines TSN 94%, SlowFast 92%.
- **CARE-PD** (github.com/TaatiTeam/CARE-PD; arXiv 2510.04312, NeurIPS 2025 D&B): 9 cohorts, 363 participants, 3D SMPL mesh gait; MDS-UPDRS gait score benchmarks. Verify license.
- Reddy et al. 2025 simulated gait impairments: 743 videos, 7 gait types; **not public** (faces).
- Van Criekinge 2023: 50 stroke + 138 able-bodied, CC0, mocap only.
- Sci Data 2025 (s41597-025-05959-w): 1,356 trials, 260 participants, 4 IMUs, not video.

### D2. Video + mocap paired
Essentially every large 3D pose corpus is non-commercial: Human3.6M (academic), 3DPW, MPI-INF-3DHP, AMASS (MPG license, forbids commercial training), BEDLAM (MPG academic), MoVi (processed version under AMASS license), Fit3D (request-based non-commercial), CARE-PD (see above). **OpenCap SimTK dataset** (registration): real smartphone video + mocap + GRF + EMG for walking, squats, STS, drop jumps; Horsak 2025 used it as a single-view validation harness.

### D3. Exercise / rehab video datasets
- **UI-PRMD**: 10 rehab exercises × 10 subjects × correct/incorrect; Vicon + Kinect; **Open Data Commons PDDL (public domain)**. The only license-clean option.
- REHAB24-6 (Zenodo 10.5281/zenodo.13305825): 65 recordings, 2 RGB cameras, 41 markers, 1,072 repetitions with correctness labels; academic non-commercial only.
- KIMORE: 78 subjects (34 patients), RGB + depth + joints, clinician scores; request-based.
- IntelliRehabDS: 29 subjects (15 patients), Kinect 25-joint; Zenodo 4610859.
- UCO Physical Rehabilitation (Sensors 2023;23(21):8862); MobiPhysio (Data in Brief 2026, verify license); TheraPose (not yet available).

---

## E. Regulatory / ethical framing (US; not legal advice)
- The four non-device CDS criteria come from 21st Century Cures §3060 → FD&C Act §520(o)(1)(E). FDA's final CDS guidance was revised and reissued January 6, 2026.
- Criterion 1 is the binding constraint: software must not analyse "a pattern or signal from a signal acquisition system." The 2026 guidance describes signal acquisition as continuous or streaming measurement. A tool that ingests continuous video and derives kinematic waveforms reads as processing a streaming signal, so the CDS exemption likely does not apply regardless of criterion 4.
- What loosened in 2026: a single recommendation is acceptable under enforcement discretion; the time-critical limitation moved to criterion 4; emphasis on well-understood sources; silent on patient-facing tools and AI.
- **General Wellness is the realistic lane for v1** (FDA revised the General Wellness policy the same day). Wellness = tracking walking speed and squat depth over time; regulated = flagging probable foot drop for a PT.
- Risk-reduction architecture: no diagnostic or disease claims; expose every raw trace and the full computation; present findings as value vs published normative band with citation and reported error; per-measure accuracy documentation; process video locally by default (PHI). EU MDR Rule 11 is stricter (likely Class IIa).

---

## Recommendation

### Tier 1: build first (timing and counting; event detection, not angle magnitude)
1. 5xSTS time + 30-second chair stand count (ICC 0.995 / 0.928; Bohannon 2006 norms).
2. TUG total time (ICC 0.962; Bohannon 2006 norms); sub-phases second (`tugturn` design).
3. Gait speed over a measured walkway (ICC 0.868–0.997; 0.02–0.04 m/s; Bohannon 2011 norms; enforce accel/decel zones).
4. Cadence and step/stance/swing/double-support time (MAE 0.02 s; Zeni 2008).
5. Temporal symmetry indices (antalgic gait without angles).
6. SPPB balance holds.

### Tier 2: with explicit uncertainty bands
7. Peak sagittal knee flexion (RMSE 5.6–6.1°; position-invariant). 8. Peak sagittal hip flexion (MAE 4.0°). 9. Step/stride length as bout average only. 10. Knee, elbow, shoulder flexion/abduction ROM (ρ 0.61–0.72, human ceiling 0.66–0.75; whole degrees ± MDD). 11. STS peak power (ICC 0.94). 12. FPPA on single-leg squat / lateral step-down, labelled as a 2D projection.

### Tier 3: defer or qualitative flags only
13. Anything ankle. 14. Frontal/transverse hip measures incl. Trendelenburg and circumduction. 15. Absolute upper-limb ROM / arm swing. 16. Shoulder rotations. 17. Y-balance, functional reach. 18. Berg Balance Scale. 19. Kinetics.

### Concrete stack decision
- `pip install opensim` 4.6 works natively on Apple Silicon if IK is wanted; Tier 1 and most of Tier 2 need no OpenSim.
- Sports2D (BSD-3) angle definitions and filter defaults so conventions match published work.
- ezc3d for C3D if ever needed.
- Avoid for v1: AddBiomechanics, nimblephysics, SKEL/HSMR, OpenCap local processing.
- Validation harness: replicate Horsak 2025 on the OpenCap SimTK dataset; normative bands from Fukuchi (CC BY) and Van Criekinge (CC0); UI-PRMD for rehab exercises.
- Reference target: clinically desirable 2–5°; single-camera methods sit at 4.8–5.5° RMSE today. At the edge for sagittal lower-limb angles, well inside for timing measures. Let that asymmetry drive the roadmap.

**Sources:** OpenSim on PyPI · OpenSim conda files · OpenSim Conda Package doc (stale) · AddBiomechanics · nimblephysics · SKEL license · HSMR · SKEL-CF · Sports2D · Pose2Sim · Pose2Sim accuracy (Sensors 2022) · kineticstoolkit · pyomeca · ezc3d · gaitalytics · Zeni 2008 · Stenum 2021 · Stenum code · Stenum 2024 · Barzyk 2024 · Uhlrich 2023 · OpenCap scoping review 2026 · Cheng 2025 · Horsak 2025 · OpenCap Monocular · MonoMSK · OpenCap vs OptoGait · Bohannon & Williams Andrews 2011 · Bohannon 2006 TUG · Bohannon 2006 5xSTS · Mobbs 2025 · Fukuchi 2018 · Schreiber & Moissenet 2019 · Van Criekinge 2023 · Camargo 2021 · Bertrand 2026 · Hwang 2026 · tugturn · TUG segmentation PMC12762308 · Sabo 2023 · van den Hoorn 2025 · Sports2D ROM reliability 2025 · GAVD · CARE-PD · Reddy 2025 · AMASS license · BEDLAM · BML-MoVi · Fit3D · OpenCap dataset (SimTK) · UI-PRMD · REHAB24-6 · IntelliRehabDS · UCO Physical Rehabilitation · MobiPhysio · FDA CDS guidance · Covington, DLA Piper, Troutman analyses.
