# Open-source pose estimation landscape (September 2026)

Produced 2026-09-01 by a research agent during planning. Reproduced as delivered. Verification method
stated by the agent: repo metadata via the GitHub API, package versions and wheel tags via the PyPI and
Anaconda JSON APIs, licenses read from raw LICENSE files, rtmlib backend behaviour read from source.
Where a widely repeated claim turned out wrong, the agent flags it.

---

## A. 2D pose (body + feet)

### A1. rtmlib — https://github.com/Tau-J/rtmlib
- **v0.0.16**, released 2026-08-03 (GitHub tag) / PyPI 2026-08-04. **Apache-2.0**. `requires_python>=3.10`, pure-Python wheel (`py3-none-any`), no compiled deps. 660 stars, actively pushed 2026-08-03.
- **Exported classes** (verified in `rtmlib/__init__.py`): detectors `YOLOX`, `RTMDet`, `RFDETR`; estimators `RTMPose`, `RTMO`, `RTMPose3d`, `ViTPose`; solutions `Body`, `BodyWithFeet`, `Wholebody`, `Wholebody3d`, `Hand`, `Animal`, `Custom`, `PoseTracker`. **There is no `RTMW` class**: RTMW is the ONNX checkpoint used inside `Wholebody`.
- **Backends actually implemented** in `rtmlib/tools/base.py`: `opencv`, `onnxruntime`, `openvino` only. **The README's claim of a TensorRT backend is not in the code.** Device map for onnxruntime:
  ```python
  'cpu': 'CPUExecutionProvider', 'cuda': 'CUDAExecutionProvider',
  'rocm': 'ROCMExecutionProvider',
  'mps': 'CoreMLExecutionProvider' if check_mps_support() else 'CPUExecutionProvider'
  ```
  So **`device='mps'` is a misnomer: it selects the CoreML EP**, and silently degrades to plain CPU if CoreML isn't available. `'cuda:N'` is also parsed for multi-GPU.
- **Feet keypoints:** `BodyWithFeet` = **Halpe-26** → `LBigToe=20, RBigToe=21, LSmallToe=22, RSmallToe=23, LHeel=24, RHeel=25`. `Wholebody` = COCO-133 (also has the 6 foot points, indices 17–22). `Body` = COCO-17, **no feet**.
- Checkpoints for `BodyWithFeet` are `rtmpose-{s,m,x}_simcc-body7_pt-body7-halpe26` (lightweight/balanced/performance) + YOLOX {tiny,m,x} detector, pulled from `download.openmmlab.com`, with a HuggingFace mirror.

### A2. RTMPose / RTMW / RTMO (MMPose) — https://github.com/open-mmlab/mmpose
- **Apache-2.0**, not archived, 7,867 stars, **but effectively dormant.** Last release **v1.3.2, 2024-07-12**. Commit history: doc merges on 2025-08-04, before that 2024-08-07. **332 open issues.** Treat MMPose as a frozen model zoo, not a living framework.
- Practical consequence: **consume the exported ONNX via rtmlib rather than installing MMPose** (which needs the mmcv/mmdet/mmengine stack pinned to old torch).
- Accuracy anchors from the MMPose RTMPose README: RTMPose-m Halpe-26 @256×192 = 94.75 PCK@0.1 / 71.91 AUC (Body8); RTMPose-x @384×288 = 95.74 / 74.82. RTMW-l reaches 70.2 whole-body AP on COCO-WholeBody.
- **Weights/training-data caveat (unresolved upstream):** MMPose's LICENSE is plain Apache-2.0 with no dataset carve-out, but the Halpe-26 checkpoints are trained on `body7` and the RTMW ones on `cocktail14` (14 datasets, including Halpe). **`Fang-Haoshu/Halpe-FullBody` has no LICENSE file and no license statement at all.** Code Apache-2.0, weights distributed under the same repo, training-set terms not addressed by anyone. A residual risk to get legal sign-off on.

### A3. ViTPose / Sapiens
- **ViTPose / ViTPose++** — https://github.com/ViTAE-Transformer/ViTPose — **Apache-2.0**, last push 2025-12-25, 2,139 stars. First-class in HuggingFace `transformers`. **Weights are Apache-2.0** (verified via HF API): `usyd-community/vitpose-plus-base` (3.0M downloads), `vitpose-plus-huge`, `vitpose-base-simple`. Also wrapped as `ViTPose` inside rtmlib. **The strongest unambiguously commercial 2D backbone available.**
- **Sapiens v1** — https://github.com/facebookresearch/sapiens — **CC-BY-NC-4.0** (raw LICENSE; HF `facebook/sapiens-pose-1b` cardData confirms). **Non-commercial; disqualified.** 308 keypoints, 0.3B–2B.
- **Sapiens2** — https://github.com/facebookresearch/sapiens2 — **ICLR 2026**, initial release 2026-04-24, arXiv 2604.21681, 930 stars. Sizes 0.4B / 0.8B / 1B / 5B, 308 keypoints, needs a separate detector. Custom "Sapiens2 License" (Meta community style), more permissive than v1's CC-BY-NC but read it before relying on it. Practicality poor here: 0.4B–5B ViT at high resolution is not real-time, PyTorch/CUDA only.

### A4. MediaPipe Pose Landmarker — https://github.com/google-ai-edge/mediapipe
- **Apache-2.0**, extremely active (pushed 2026-09-01), 36,807 stars. **v1.0.0 released 2026-07-28**; PyPI `mediapipe` 1.0.1, 2026-08-14.
- **Breaking change: the legacy `mp.solutions` API was removed in 0.10.31.** Maintainer closing issue #6204: "With the release of MediaPipe 0.10.31, the Solutions API has been removed. Please migrate to MediaPipe Tasks." Use `mediapipe.tasks.python.vision.PoseLandmarker`.
- **macOS arm64: yes.** The 1.0.1 wheel is `py3-none-macosx_11_0_arm64` (Python-version-agnostic since 0.10.30). Intel-mac wheels are gone.
- BlazePose 33 landmarks include `left_heel`=29, `right_heel`=30, `left_foot_index`=31, `right_foot_index`=32, plus pseudo-metric `world_landmarks`. Single-person by default; world landmarks are root-relative and not metrically reliable enough for clinical joint angles.

### A5. Ultralytics YOLO11 / YOLO26
- https://github.com/ultralytics/ultralytics — v8.4.137, 2026-08-31, 61k stars. YOLO26 released Jan 2026.
- **License: AGPL-3.0.** Hard blocker for a commercial product without an Enterprise license.
- **Also disqualifying: YOLO-pose ships COCO-17 only, no heel or toe.**

### A6. Notable 2025–2026 additions
- Sapiens2 (above). SDPose (arXiv 2509.24980), diffusion priors for out-of-domain robustness. RF-DETR now a detector option inside rtmlib. No new open 2D model displaces RTMPose/ViTPose for the real-time + feet + permissive-license combination.

---

## B. Monocular 3D pose / human mesh recovery

The decisive column is not the code license: it is whether the pipeline requires a body model you must register for at MPI.

| Method | Repo | Code license | Body model | Commercial? |
|---|---|---|---|---|
| **SAM 3D Body** | facebookresearch/sam-3d-body | **SAM License** (Meta) | **MHR — Apache-2.0** | **Yes** (only one) |
| HSMR | IsshikiHugh/HSMR | MIT | SKEL (MPI, research) | No |
| 4DHumans / HMR2.0 | shubham-goel/4D-Humans | MIT | SMPL (MPI) | No |
| TRAM | yufu-wang/tram | MIT | SMPL + SMPLify (MPI) | No |
| WHAM | yohanshin/WHAM | MIT | SMPL (MPI) | No |
| NLF | isarandi/nlf | MIT code | — | **No: weights are noncommercial** |
| GVHMR | zju3dv/GVHMR | NC (explicit) | SMPL | No |
| PromptHMR | yufu-wang/PromptHMR | NC (Meshcapade) | SMPL-X | No |
| TokenHMR | saidwivedi/TokenHMR | NC (MPI) | SMPL | No |
| Multi-HMR | naver/multi-hmr | NAVER NC | SMPL-X | No |
| CameraHMR | pixelite1201/CameraHMR | no LICENSE file | SMPL | No (all rights reserved) |
| MotionBERT | Walter0807/MotionBERT | Apache-2.0 | none (H36M skeleton) | Yes-ish |

### B1. The classic stack
- **WHAM**: MIT, last push 2024-04-18, effectively abandoned upstream; requires SMPL.
- **TRAM**: MIT, last push 2025-06-08; README requires SMPLify and SMPL registration. Best-in-class camera trajectory + world grounding via SLAM.
- **GVHMR**: SIGGRAPH Asia 2024, TPAMI 2026, 1,915 stars, pushed 2026-05-21. LICENSE: educational, research and non-profit purposes only. Strongest quality/effort for research; unusable commercially.
- **CameraHMR**: 256 stars, no LICENSE file (default copyright).
- **PromptHMR**: 484 stars, pushed 2026-01-28; "Non-Commercial Scientific Research Use Only", owned by Meshcapade (who also sell SMPL commercial licenses).
- **Multi-HMR**: NAVER Non-Commercial License.
- **NLF**: v0.3.2 (2025-05-22). Code MIT but README: "Models for PyTorch and TensorFlow are available for noncommercial research use."
- **4DHumans / HMR2.0**: MIT, pushed 2026-02-07, needs the SMPL neutral model from MPI.

### B2. Meta SAM 3D Body — https://github.com/facebookresearch/sam-3d-body
- Released 2025-11-19, 3,493 stars, pushed 2026-02-19. Checkpoints: DINOv3-H+ (840M) and ViT-H (631M).
- **License: SAM License** (dated Nov 19 2025): non-exclusive, worldwide, non-transferable, royalty-free rights to use, reproduce, distribute, modify. Commercially permissive with trade-control and attribution conditions. HF `facebook/sam-3d-body-dinov3` is gated (manual access request).
- **Body model: MHR (Momentum Human Rig)** — https://github.com/facebookresearch/MHR — Apache-2.0, pushed 2026-08-31. Decouples skeletal structure and surface shape.
- **Critical limitation for gait: single-image only; no temporal/video model.** Per-frame estimates jitter where gait analysis needs precision (heel strike, toe off).

### B3. HSMR + SKEL — https://github.com/IsshikiHugh/HSMR
- MIT code, CVPR 2025 Oral, 630 stars, pushed 2026-03-06. Uses SKEL (https://github.com/MarilynKeller/SKEL), a biomechanically constrained skeleton, so joint angles respect anatomical DoF limits. **The only one in the list that outputs biomechanically meaningful joint angles directly.**
- **But SKEL is an MPI release under a research-only license**, so the pipeline is research-only despite HSMR's MIT code. Includes SKELify (optimisation fitting SKEL to 2D keypoints), a good architectural reference.

### B4. MotionBERT — https://github.com/Walter0807/MotionBERT
- Apache-2.0, 1,441 stars, pushed 2026-03-14. 2D→3D lifting, no body model dependency. Caveats: H36M 17-joint skeleton (no feet), root-relative, trained on indoor lab data. A component, not a product.

### B5. 2026 releases
- **OnlineHMR** (CVPR 2026): first properly causal/streaming world-grounded HMR (KV-cache design, built on TRAM). No public code yet. Also CVPR 2026: MetricHMSR, SHOW, DanceHMR.
- **SMPL license unchanged:** non-commercial scientific research, education, or artistic projects; explicitly forbids training algorithms for commercial deployment. Commercial licensing via Meshcapade. Same for SMPL-X and SKEL.

---

## C. Markerless biomechanics toolkits

### C1. OpenCap: two different projects
**(a) opencap-core** — https://github.com/opencap-org/opencap-core (moved from `stanfordnmbl`)
- Apache-2.0, active (commits through 2026-08-25), 362 stars. Companion `opencap-processing` (Apache-2.0).
- **Still requires 2+ cameras**; capture via iOS devices through app.opencap.ai. No monocular option in opencap-core.
- Self-hostable in three modes (cloud, local processing of app videos, local processing of own near-synchronous videos). Requirements: Windows 10 or Ubuntu, Python 3.9, conda, CUDA GPU ≥4 GB (24 GB high-res). **No macOS.**
- Commercial blocker is OpenPose (CMU non-commercial), the default detector; `utilsDetector.py` has an `mmpose` branch as the potential commercial route.

**(b) opencap-monocular** — https://github.com/utahmobl/opencap-monocular (University of Utah)
- **PolyForm Noncommercial License 1.0.0.** 213 stars, pushed 2026-08-02. 2026 arXiv preprint.
- Single smartphone video → 3D kinematics and kinetics: WHAM 3D pose (ViTPose 2D) → camera + pose optimisation → OpenSim IK → `.trc`/`.mot` + scaled `.osim`. Ubuntu only. Also inherits SMPL non-commercial via WHAM.
- **The closest thing to what we want, and we cannot ship it.**

### C2. Pose2Sim — https://github.com/perfanalytics/pose2sim
- **BSD-3-Clause.** v0.10.49 released 2026-07-10, pushed 2026-08-24, 793 stars. `requires_python>=3.11`.
- Multi-camera calibration, synchronisation, robust triangulation, filtering → OpenSim scaling and IK without a static trial. Deps: `opensim`, `rtmlib`, `openvino`, `caliscope`, `av`, `PySide6-essentials`.
- Skeletons: HALPE_26 (default), COCO_133, COCO_17, and custom. Modes lightweight / balanced / performance. Backends onnxruntime / openvino / opencv; devices cuda / mps (→CoreML) / rocm / cpu.
- Single-camera explicitly not supported (points to Sports2D). macOS: yes (see D3).

### C3. Sports2D — https://github.com/davidpagnon/Sports2D
- **BSD-3-Clause.** v0.8.34 released 2026-07-10, pushed 2026-08-25, 292 stars. `requires_python>=3.11`; sole dependency is `Pose2Sim>=0.10.49`.
- Single camera or live webcam. 2D joint angles (ankle, knee, hip, shoulder, elbow) and segment angles. Multi-person tracking. Pixel-to-metre via participant height or a calibration file; compensates floor angle, floor height, depth perspective. OpenSim export on by default in recent versions. `--device mps` for Apple Silicon.
- Known constraint: motion must lie roughly in the sagittal or frontal plane.

### C4. Others
- **FreeMoCap**: AGPL-3.0, very active (v2.0.0-alpha.23, 2026-08-27), 10,123 stars. Multi-camera, GUI-driven. AGPL is a commercial blocker.
- **Anipose**: BSD-2-Clause, pushed 2026-06-02. Multi-camera triangulation; reusable calibration/triangulation math.
- **KinePose**: GPL-3.0, pushed 2025-02-25, 29 stars. Temporally optimised IK with biomechanical constraints.
- **Theia3D**: commercial, closed; the clinical accuracy comparator in validation papers.

---

## D. Practical notes

### D1. ONNX Runtime + CoreML on macOS
- `onnxruntime` 1.29.0, MIT, `requires_python>=3.11`. macOS arm64 wheels for cp311–cp314.
- Do not use `onnxruntime-silicon` (stuck at 1.16.3). CoreML support is in mainline.
- CoreML EP partitions the graph and silently falls back to CPU for unsupported ops; for small pose models it is frequently slower than the CPU EP. Dynamic shapes poorly supported. Treat `device='mps'` as "must benchmark".
- `onnxruntime-gpu` 1.29.0 is Linux/Windows only.

### D2. PyTorch on Blackwell (RTX 5070 Ti, sm_120) and MPS
- sm_120 requires cu128 or newer. cu128 carries torch 2.7.0–2.11.0; cu129 to 2.13.0; cu130 to 2.14.0. Torch ≥2.7.0 on cu128+ is the floor; 2.9.1+cu128 or newer is the safe pick.
- The default PyPI `torch` 2.13.0 Linux wheel bundles CUDA 13 (covers sm_120). On Windows the default wheel is CPU-only; use the explicit index.
- The real Blackwell friction is research repos pinning torch 1.13/2.0 + CUDA 11.x (and `chumpy`, broken on modern NumPy).
- PyTorch MPS: default macOS arm64 wheel is the MPS build, but every realistic 3D candidate has CUDA-specific dependencies. In practice: 2D on the Mac via ONNX Runtime, 3D on the 5070 Ti. MPS gotchas: unimplemented ops (`PYTORCH_ENABLE_MPS_FALLBACK=1`), no float64, numerical divergence from CUDA.

### D3. Python version: 3.14 fails
- **Use Python 3.12 or 3.13.** `pose2sim` depends on `opensim`, and `opensim` 4.6 publishes only cp311/cp312/cp313 wheels (`macosx_15_0_universal2`, `manylinux_2_27_x86_64`, `win_amd64`). No cp314 wheel and no sdist, so resolution fails on 3.14. `sports2d` inherits this.
- The OpenSim wheel is `macosx_15_0_universal2`, so it runs natively on Apple Silicon via pip (the widely cited "x86_64-only on Mac" guidance is outdated).
- Other floors: `numpy` 2.5.2 requires ≥3.12; `onnxruntime` ≥3.11; `rtmlib` ≥3.10; `mediapipe` 1.0.1 is `py3-none`.

---

## Recommendation

### First version, runs on the M4 Pro
**2D: rtmlib `BodyWithFeet` (Halpe-26, `mode='balanced'` = RTMPose-m) driving Sports2D-style single-camera analysis.** Halpe-26 gives heels and big toes; Sports2D and Pose2Sim are BSD-3 and actively maintained; runs on macOS arm64 with `uv venv --python 3.12`. Start with CPU, benchmark CoreML rather than assuming. Skip 3D in v1 and sidestep the SMPL licensing minefield.

### Higher accuracy on the 5070 Ti
- 2D: same Halpe-26 format, `mode='performance'` (RTMPose-x @384×288) via the CUDA EP. Keeping the keypoint format fixed makes Mac/desktop and single/multi-camera into config changes.
- Multi-camera later: Pose2Sim, same rtmlib backend and format.
- 3D from one camera, commercially clean: SAM 3D Body (SAM License + MHR Apache-2.0), single-image only, so static posture not gait timing. Request HF access early.
- Internal R&D only (do not ship): GVHMR (world-grounded quality), HSMR (biomechanical SKEL angles), opencap-monocular (closest analogue).

### Two things to decide early
1. Legal comfort on the RTMPose Halpe-26 weights' training data (`Halpe-FullBody` has no license file); fallback ViTPose++ with verifiably Apache-2.0 HF weights via rtmlib.
2. Watch OnlineHMR (CVPR 2026): causal streaming world-grounded HMR is the shape of the right answer for real-time gait.

**Sources:** rtmlib · rtmlib base.py · MMPose · MMPose RTMPose model zoo · Halpe-FullBody · ViTPose · usyd-community/vitpose-plus-base · Sapiens · Sapiens2 · MediaPipe · MediaPipe Pose Landmarker docs · MediaPipe issue 6204 · Ultralytics · YOLO26 · SAM 3D Body · MHR · SAM 3D blog · GVHMR · TRAM · WHAM · PromptHMR · Multi-HMR · NLF · TokenHMR · CameraHMR · 4D-Humans · HSMR · SKEL · MotionBERT · OnlineHMR (CVPR 2026) · SMPL license · opencap-core · opencap-monocular · Pose2Sim · Sports2D · FreeMoCap · Anipose · KinePose · ONNX Runtime CoreML EP · PyTorch cu128 wheel index · opensim on PyPI.
