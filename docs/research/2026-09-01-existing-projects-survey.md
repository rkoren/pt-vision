# CV / pose estimation for PT, rehab and gait: landscape survey (September 2026)

Produced 2026-09-01 by a research agent during planning. Reproduced as delivered. Facts were checked
against GitHub metadata, source files and papers; commercial pricing is not public for any product listed.

## Headline conclusion
**The clinically validated single-camera work is all copyleft or noncommercial. The permissive,
commercially usable stack is Pose2Sim/Sports2D (BSD-3) + rtmlib (Apache-2.0) + OpenCap
core/processing (Apache-2.0).** That license split, not technical fit, is the most decision-relevant fact.

---

## A. Open-source projects

### A.1 Sports2D + Pose2Sim (David Pagnon), the primary reference

| | Pose2Sim | Sports2D |
|---|---|---|
| URL | github.com/perfanalytics/pose2sim | github.com/davidpagnon/Sports2D |
| Stars / forks | 793 / 110 | 292 / 40 |
| License | BSD-3-Clause | BSD-3-Clause |
| Last commit | 2026-08-24 | 2026-08-25 |
| Language | Python ≥3.11 | Python ≥3.11 |

Sports2D is not standalone: its entire dependency list is `Pose2Sim @ git+...@<sha>`. Pose2Sim carries the weight: `opensim, rtoml, lxml, mpl_interactions, PySide6-essentials, tqdm, anytree, pandas, scipy, statsmodels, filterpy, ipython, c3d, rtmlib, openvino, av, caliscope, bvhsdk`.

Code architecture (verified from source):
1. **Config: cascading TOML** (session → participant → trial; a missing key is looked up one level above). Sections mirror pipeline stages. Worth copying wholesale as a clinic → patient → visit hierarchy.
2. **Keypoint formats: anytree trees.** `skeletons.py` defines each skeleton as an `anytree.Node` hierarchy where the tree is kinematic parenting and `id=` is the model's output index (HALPE_26, COCO_133, COCO_17, BLAZEPOSE, BODY_25, ...). `id=None` marks synthetic nodes reconstructed by `add_neck_hip_data()` etc.
3. **Angle computation: a 4-tuple table.** `angle_dict` maps name → `[keypoint_names, angle_type, offset, scaling]`, e.g. `'right knee': [['RAnkle','RKnee','RHip'], 'flexion', -180, 1]`. 12 joint angles + 16 segment angles; applied in `fixed_angles()` with wrapping. The single most copyable pattern; adding a clinical angle is one row.
4. **Filtering: nine interchangeable 1-D filters** with a uniform signature: butterworth (default 6 Hz / order 4, zero-phase), butterworth_on_speed, kalman (filterpy), one_euro, gcv_spline, acc_minimizing (Whittaker–Henderson), gaussian, loess, median; plus `hampel_filter(col, window_size=7, n_sigma=2)` before the main filter. Pre-filter order: interpolate gaps → chunk selection → Hampel → filter.
5. **Person tracking.** `sort_people_sports2d` (match_by keypoints/centroid/bbox, max distance, grace window), `sort_people_rtmlib`, `sort_people_deepsort` (not recommended). Person *selection* is separate (`on_click` default, `highest_likelihood`, `largest_size`, ...). Quality gates: keypoint likelihood 0.3, average likelihood 0.5, keypoint number 0.3.
6. **Pixels → metres** (Sports2D): `first_person_height` via `compute_height()` on extended frames with a trimmed mean; `floor_angle='auto'` from foot-contact events (`compute_floor_line`, toe speed below a threshold); `xy_origin` at first foot contact; perspective compensation via `distance_m` / `f_px` / `fov_deg`; README notes 1–2% coordinate error at 10 m uncorrected.
7. **OpenSim export**: `read/write_trc`, `read/write_mot`, C3D via `c3d`; `kinematics.py` scaling + IK (`use_simple_model=true` 10× faster).
8. **Marker augmentation**: OpenCap's LSTM (43 markers) shipped as ONNX in `Pose2Sim/MarkerAugmenter/LSTM/`. The cleanest way to get OpenCap's marker enhancer without their cloud.
9. **Outputs**: per person TRC (px and m), MOT of angles, optional C3D and calibration, overlay video, raw-vs-filtered plots.

Also `Pose2Sim_Blender` (MIT) for Blender import. Reusable: essentially all of it, importable from `Pose2Sim.common`.

### A.2 OpenCap ecosystem

| Repo | Stars | License | Last commit | Note |
|---|---|---|---|---|
| opencap-org/opencap-core | 362 | Apache-2.0 | 2026-08-25 | org renamed from stanfordnmbl |
| opencap-org/opencap-processing | 153 | Apache-2.0 | 2026-04-24 | |
| opencap-org/opencap-viewer | 10 | Apache-2.0 | 2026-08-04 | JS viewer |
| Seeeeeyo/opencap-visualizer | 6 | Apache-2.0 | 2026-08-20 | Vue/Three.js |
| utahmobl/opencap-monocular | 213 | PolyForm Noncommercial 1.0.0 | 2026-08-02 | |

- opencap-core: flat module layout; `utilsAPI.py` hardcodes `https://api.opencap.ai/`; fully local processing of non-OpenCap videos is "under development". Local install needs OpenPose (non-commercial) + CUDA. So "reusable offline" collapses to specific pieces: the marker augmenter (take it from Pose2Sim), `opensimPipeline/` XMLs and marker set, `utilsTRC.py`, `utilsCameraPy3.py`, `utilsSync.py`.
- **The reusable gem is `opencap-processing/ActivityAnalyses/`**: `gait_analysis.py` (`segment_walking()`, `get_gait_events()`, `compute_scalars()` over stride_length, step_length, step_length_symmetry, gait_speed, cadence, step_width, stance/swing/single/double support, midswing dorsiflexion, peak_angle, rom, % gait-cycle normalisation) and `sts_analysis.py` (`segment_sts()`, torso orientation, rise_time, sts_time, max knee extension moment). The reflection-based `compute_<name>()` returning `{'value','units'}` is a copyable registry contract.
- opencap-monocular: WHAM + ViTPose + camera optimisation + OpenSim IK + activity classification. Read for architecture, do not ship.

### A.3 FreeMoCap
- 10,123 stars, AGPL-3.0, very active; modular "skelly" packages (camera / tracker / sync / solve / post-process), all AGPL. Multi-camera first; not clinical. Architectural lesson is the decomposition. AGPL is disqualifying as a dependency.

### A.4 Gait-specific
- **OpenGait and friends are gait *recognition*** (biometric re-identification), invariant to exactly the asymmetries we want to measure. Do not build on them.
- `janstenum/GaitAnalysis-PoseEstimation`: GPL-3.0, MATLAB, stale since 2024-04; the most clinically validated open 2D gait pipeline (PLOS Comp Biol 2021, PLOS Digital Health 2024). Read the method, don't depend on the code.
- `IntelligentSensingAndRehabilitation/GaitTransformer`: GPL-3.0, transformer over pose sequences → gait events.
- `Rahmyyy/GAVD`: MIT annotations only (YouTube IDs), 1,874 sequences.
- `batking24/OpenPose-for-2D-Gait-Analysis`: no license; read only.
- **`IntelligentSensingAndRehabilitation/MonocularBiomechanics` ("Portable Biomechanics Laboratory")**: AGPL-3.0, pushed 2026-07-09, arXiv 2507.08268. Biomechanical model fitting to handheld smartphone video; joint-angle errors <3° across neurological injury, prosthesis users, paediatric inpatients; 1,021 videos in prospective deployment, gait metrics ICC > 0.9; responsive to intervention. **This is the technical bar; AGPL means reference, not dependency.**
- `IntelligentSensingAndRehabilitation/PosePipeline`: GPL-3.0; DataJoint schema relating videos, algorithms and outputs. **The provenance idea is the most valuable architectural steal**; mirror the pattern.

### A.5 Rehab / exercise assessment
- Datasets and baselines: UI-PRMD (avakanski baseline, no license), KIMORE, `fokhruli/STGCN-rehab` (MIT, active), `bruceyo/EGCN` (BSD-2), D2STA. Surveys: arXiv 2502.02817, 2403.02772, 2606.30309.
- Practical rep-counting apps: `yo-WASSUP/Good-GYM` (403 stars, MIT, RTMPose-only, active) is the best-maintained angle-threshold state-machine example; the rest are MediaPipe hobby projects with no validation or persistence.
- The gap between these and a PT-usable assessment is entirely the interpretation and longitudinal layers.

### A.6 Clinical test automation
- Sit-to-stand: `opencap-processing/ActivityAnalyses/sts_analysis.py` is the only real open implementation.
- TUG: no maintained open repo; PMC12762308 (AlphaPose + MotionBERT) is the literature reference. A genuine build.
- ROM / goniometer: nothing credible open source; commercial or SDK-gated. ROM is nearly free once angles exist; the hard part is standardising capture.

### A.7 LLM / VLM movement analysis
- **BiomechGPT** (arXiv 2505.18465, Cotton lab): motion-language model trained on 71 h of biomechanics from 750 participants; activity classification, impairment diagnosis, falls history. Repo exists but is empty; code not released.
- Generic VLMs do not work yet: "Can Vision Language Models Judge Action Quality?" (arXiv 2604.08294, Apr 2026): Gemini 3.1 Pro, Qwen3-VL, InternVL3.5 perform "only marginally above random chance." Stroke rehab case study (arXiv 2511.17727 / PubMed 42406872): dose estimates comparable to a vision-free baseline; impairment scores not reliably predicted.
- Non-clinical motion-language models (MotionGPT, MotionGPT-2/3, MG-MotionLLM, ChatMotion) use HumanML3D-style vocabulary, nothing like clinical description.
- Finding: an LLM layer should reason over computed metrics with strict grounding, not over video or raw keypoints.

---

## B. Commercial / closed reference products (pricing not public)

| Product | Measures | Cameras | Notes |
|---|---|---|---|
| Kemtai | 111 body points; assessment + guided exercise with real-time correction | single, browser | markets RTM billing explicitly |
| Exer Health / Exer AI | Exer Scan, Exer Health, Exer Hands, Exer Gait (beta) | single mobile | "RTM Codes" page, automated billing reports |
| Kinetisense | triplanar ROM, movement quality, asymmetry, balance | single | publishes a CPT billing codes PDF |
| Theia3D | 124 keypoints → 3D skeletal model; <1 cm / <3° claim | multi (6–18) | the accuracy benchmark others cite |
| DARI Motion | full-body booth | multi | squat ROM ICC 0.86 |
| Uplift Labs | 30+ joints | two iOS devices | sports, not clinical |
| Kinotek | 64 movements, pre/post comparison | single | peer-reviewed validity (glenohumeral) |
| Physimax, VueMotion, Mobility-Metrics, MotionMetrix | screening / gait | camera | |
| Sword Health / Hinge Health, Kaia | digital MSK care programmes | single | sold to employers/payers |
| OneStep | gait speed, asymmetry, fall risk | **IMU, not camera** | the most deployed "gait app" is not CV |
| Kinovea | manual 2D angles | single | free incumbent PTs already use |

What PTs pay for: the reimbursement pathway, not features. RTM-billable products sell a billable documentation artifact (2026 PFS: new codes 98984/98985 for 2–15 days of device supply, 98979 for 10–19 min management); employer/payer products sell outcomes; cash-pay products sell a client-facing pre/post report. Nobody pays for a better skeleton: they pay for a defensible number in a note, a visual of change over time, and a mapping to billing or a sales conversation. Every open-source project produces the number per trial and stops.

---

## C. Architecture patterns worth copying
- **Storage**: TRC/MOT/C3D are the right *export* formats (OpenSim, Visual3D), the wrong *working* format. Use Parquet (long format: frame, time, person_id, keypoint, x, y, z, score) internally; write both pixel and metre TRC.
- **Skeleton mapping**: Pose2Sim's anytree trees (one per model, `id=` index, `id=None` derived joints, explicit reconstruction functions).
- **Tracking**: two stages, association then selection; `on_click` disambiguation is more robust than heuristics when a therapist stands in frame.
- **Provenance**: PosePipeline's schema treats "which model version produced this measurement" as first-class; longitudinal comparison needs it.
- **Metric registry**: OpenCap's `compute_scalars` contract, extended with confidence, n_cycles, method_version.
- **Config**: cascading TOML maps onto clinic → patient → visit.
- **Visualisation**: keypoints coloured by confidence, angle arcs, raw-vs-processed plots; opencap-visualizer (Apache-2.0) for 3D.

---

## D. Failure modes for single-camera PT use, and mitigations
1. **Depth / perspective**: step length errors grow with distance (11–16 cm), gait speed overestimated 0.13–0.21 m/s walking away from a frontal camera; Sports2D perspective compensation; constrain distance, prefer walking toward.
2. **Left/right leg swap and 180° angle inversion in sagittal view**: the #1 gait failure (Sports2D issue #39). Maintainer: `do_ik true` solves it best; fallbacks are likelihood gating, trajectory-jump detection before filtering, or a temporal reassignment model; the maintainer's own solution is IP-protected. **Unsolved in open source; commercially valuable.**
3. **Ankle is systematically the worst joint**; trunk inclination the best (±1.5°). Build claims on hip/knee/trunk. Halpe-26 feet help.
4. **View-dependent keypoint behaviour** (toward vs away); fix and record capture direction per protocol; Sports2D `visible_side`.
5. **Camera tilt**: floor-angle estimation from foot contacts.
6. **Dropouts and spikes**: layered defence (confidence gating, interpolation with a gap cap, chunk selection, Hampel, Butterworth).
7. **Scale without calibration**: participant height, robustified.
8. **Frame rate**: not the cause of side swaps; 50–60 fps comfortable, 30 workable.
9. **Population shift**: pose models trained on internet imagery underrepresent assistive devices, prostheses, paediatric and geriatric proportions; GAVD for evaluation.
10. **Operator burden**: "requires GPU access, software download, and manual video processing" is the productisation gap in one sentence.

---

## What to reuse vs build

**Depend on directly**: Pose2Sim (BSD-3) pieces, Sports2D (BSD-3) orchestration ideas, rtmlib (Apache-2.0), opencap-processing ActivityAnalyses (Apache-2.0), opencap-visualizer (Apache-2.0), OpenSim (Apache-2.0) for IK later, Pose2Sim_Blender (MIT) optionally. Pin Pose2Sim/Sports2D versions.

**Mirror the pattern, do not import**: MonocularBiomechanics (AGPL; the accuracy target and validation protocol template), opencap-monocular (PolyForm NC), PosePipeline (GPL; provenance schema), Stenum's MATLAB (GPL; validation design), FreeMoCap (AGPL; package decomposition), unlicensed repos (read only). `fokhruli/STGCN-rehab` (MIT) is the one academic rehab-scoring baseline that could be imported.

**Must build**: clinical interpretation (normative comparison, MDC/MCID, plain-language findings); longitudinal patient → episode → visit → trial model with error-aware change detection; capture-quality gating that refuses bad recordings; left/right disambiguation in sagittal view; TUG subtask segmentation; note-ready / RTM-shaped output; grounded LLM interpretation over metrics.

**Structural warning**: anything downstream of OpenPose is commercially unusable; the Pagnon stack's move to RTMPose-via-rtmlib is what makes it the correct foundation.

**Sources:** Pose2Sim · Sports2D · Pose2Sim_Blender · Pose2Sim docs · Sports2D docs · Sports2D #39 · Pose2Sim #102 · opencap-core · opencap-processing · opencap-viewer · opencap-visualizer · opencap-monocular · OpenCap paper · FreeMoCap · PosePipeline · MonocularBiomechanics · GaitTransformer · GaitAnalysis-PoseEstimation · PLOS Digital Health pdig.0000467 · PLOS Comp Biol pcbi.1008935 · GAVD · TUG CV validation PMC12762308 · STGCN-rehab · EGCN · D2STA · Good-GYM · AQA survey · BiomechGPT · VLM AQA arXiv 2604.08294 · VLM stroke rehab arXiv 2511.17727 · MotionGPT · rtmlib · mmpose · Kemtai · Exer AI · Theia Markerless · Kinetisense · Kinotek · Uplift · OneStep · RTM 2026 code analyses · Kinovea validity.
