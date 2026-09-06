# Design review: package scaffold and first vertical slices (September 2026)

Produced 2026-09-01 by a design-review agent during planning, from a brief describing the proposed
architecture and the verified dependency facts. Reproduced as delivered. The project adopted its
recommendations: vendor Pose2Sim pieces, plain JSON on disk, per-run provenance, a decorator metric
registry, protocols as TOML, an `experiments/` firewall, and the reordered slice sequence
(sit-to-stand before gait).

Verified upstream facts that changed the plan:
- Pose2Sim's core deps: `opensim, rtoml, lxml, mpl_interactions, PySide6-essentials, tqdm, anytree, pandas, scipy, statsmodels, filterpy, ipython, c3d, rtmlib, openvino, av, caliscope, bvhsdk`. Depending on it means installing OpenSim, Qt, OpenVINO and a calibration GUI to compute a knee angle.
- `Pose2Sim/filtering.py` imports `plotWindow` (Qt) from `common`, plus `statsmodels`, `filterpy`, and private scipy internals. Not vendorable as a unit; take `hampel_filter` and the TRC/MOT writers. Butterworth is 6 lines of scipy.
- `Pose2Sim/common.py` imports `cv2, c3d, rtoml, matplotlib, anytree` at module level, so importing `angle_dict` from upstream drags them in. `angle_dict`, `points_to_angles`, `fixed_angles`, `read/write_trc/mot`, `sort_people_sports2d` are small pure functions; vendorable.
- Sports2D `process.py` does `from Pose2Sim.common import *`, uses matplotlib widgets for person selection, and its entry point writes files rather than returning objects. Its floor/px→m logic (~150 lines) is only needed from Slice 2 onward.
- rtmlib requires `opencv-python` and `opencv-contrib-python` (two wheels that both install `cv2`), caches weights under `~/.cache/rtmlib`, and `BodyWithFeet.__init__` accepts local file paths, so we can own weight management. `PoseTracker` does IoU tracking internally but does not return track ids or accept local paths.
- Pose2Sim's own `setup_backend_device` only enables CoreML when `onnxruntime <= 1.26.0`, a hint that the CoreML EP regressed in newer ORT. Default to CPU until benchmarked.
- opencap `segment_sts`: pelvis vertical position normalised, `find_peaks` for standing maxima, walk left from max velocity to `velSeated` (0.1 m/s) for lift-off and right to `velStanding` for standing; torso angular velocity crossing `lean_threshold` for lean onset; periodicity refinement using return to within 5 cm of rise height. Directly adaptable to 2D pixel data with normalised units.

---

## 1. Critique of the proposed architecture

| Proposal | Verdict | Why |
|---|---|---|
| Single `ptvision` package, `src/` layout, `uv`, no workspace | Keep | One lockfile, one CI matrix. Split only if a subpackage needs conflicting deps; `experiments/` should be its own project anyway. |
| `PoseBackend.estimate(video) -> PoseTrack` | Keep, but take a frame iterator | Lets tests feed synthetic frames; ingest lives in an `io` layer. |
| Reuse Pose2Sim `skeletons.py` (anytree) | Reduce | You need a flat names + parent map, not a tree library. Vendor the HALPE_26 tree as data; keep the anytree text for parity. |
| Long-format Parquet | Keep, add `camera_id` | 900 frames × 26 kpts = 23k rows; long format makes 2D/3D/multi-cam a column change. Dense `(T,P,K,D)` in memory. |
| One `capture.json` sidecar with camera and model provenance | Split | Capture facts are immutable per trial; processing facts change per run. `capture.json` per trial, `provenance.json` per run. |
| Patient hierarchy: dirs+JSON vs SQLite | Dirs + pydantic JSON; no DB in v1 | Single-user laptop, no real patients, artifacts must be inspectable and local. |
| Delegate filtering to Pose2Sim | Don't | See imports above. Butterworth (scipy), Hampel (vendored), linear gap interpolation. |
| Event detection in `kinematics` | Split | Generic signal tools in `kinematics.signal`; protocol-specific segmenters in `clinical.segmenters`. |
| `compute_<name>` reflection registry | Keep, use a decorator | Explicit `@register_metric(name, version, tier, citation)` lets tests assert every tier-1 metric has a citation. |
| Report: Jinja2 + matplotlib static | Keep; JSON primary | HTML is a rendering of `report.json`. |
| `experiments/` as an extra of the core | Change to a separate uv project | An extra still lands research-licensed packages in the core lockfile; a separate project is a real firewall; add a grep test. |
| Tests with short fixture clips | Refine | Store the pose Parquet of a real clip as the fixture for downstream tests; one 2-second 480p clip for the pose layer under a `model` marker. |
| Missing: video ingest | Add `ptvision.io` | iPhone clips are HEVC (often 10-bit), rotated via metadata, possibly VFR. Normalise once with ffmpeg. |
| Missing: regulatory framing in schema | Add to `Metric` | `value, units, error, error_kind, n_events, tier, method_version, citation, flags` mandatory and tested. |

Over-engineering to cut: sqlmodel, a plugin system, a DataJoint-like DB, Plotly, Pose2Sim as a dependency, a GPU extra on the Mac. Under-engineered in the brief: ingest normalisation, provenance separation, timing start/stop rule as an explicit protocol parameter, a fixture strategy that doesn't need weights in CI.

## 2. Dependency decision: vendor the pure pieces; Pose2Sim only in the `opensim` extra
Reasons: weight (>1 GB and Qt system libs in CI to import a joint-angle table); shape (file-in/file-out monolith with interactive selection); what we reuse is small and BSD-3 (HALPE_26 ids, `angle_dict`, `points_to_angles`/`fixed_angles`, TRC/MOT writers, `hampel_filter`, `sort_people_sports2d`); drift mitigated by `PROVENANCE.md` and a parity test in the `opensim` CI job; OpenSim IK path stays open through Pose2Sim-compatible TRC.

OpenCV wrinkle: accept both wheels or drop contrib with a uv override (`opencv-contrib-python ; sys_platform == 'never'`); Linux CI needs `libgl1 libglib2.0-0`.

## 3. Repository layout
Package by layer: `io/` (video probe/normalise/iterate/write, parquet, json, hashing), `pose/` (layout, track, base, rtmlib backend, tracking, models manifest, overlay), `kinematics/` (preprocess, signal, angles, export), `clinical/` (protocol, protocols TOML, segmenters, metrics registry, norms), `quality/`, `report/`, `data/` (models, store, ids), `pipeline.py`, `cli/`, `_vendor/pose2sim/` with LICENSE and PROVENANCE. `tests/` with unit, integration, fixtures, a license-firewall test. `experiments/` as its own project.

## 4. pyproject sketch
Pins indicative; `uv.lock` carries exact versions. Extras `mediapipe`, `opensim` (opensim + pose2sim); dev group pytest/ruff/mypy/pre-commit/syrupy; `[project.scripts] ptv`; uv override for opencv-contrib; ruff line length 100 excluding `_vendor`; pytest markers `model`, `slow`; mypy strict on `clinical.*` and `data.*`. `gpu` extra deferred (onnxruntime-gpu conflicts on the module name; needs uv `conflicts`).

## 5. Data schemas
- Pose Parquet v1: `camera_id, frame, time_s, person_id, keypoint, kp_index, x, y, z, score`, metadata `layout, coord_space, fps, n_frames, image size, stage`. Raw is canonical; filtered is derived. Multi-camera later = one file per camera plus a triangulated `pose/world.parquet`.
- `PoseTrack` dataclass with `(T,P,K,D)` coords, scores, person ids, Parquet round-trip.
- `capture.json` (trial): source hash, codec, fps, VFR flag, rotation applied, normalised file, dims, view, distance, camera height, subject (age in years, never DOB).
- `runs/<run_id>/provenance.json`: versions, git sha, platform, pose model names + sha256, ORT version and providers, tracker, preprocess params, protocol sha, segmenter, metric versions, quality.
- Data store: `patients/<id>/patient.json`, `episodes.json`, `visits/<id>/visit.json`, `trials/<id>/{capture.json, source/, video/, pose/, runs/, latest.json}`; ids like `T-20260901-140322-a3f9`; episode is a field on the visit. Anonymous trials under `./ptv_out/<stem>/` use the same layout.
- `Metric` pydantic: `name, label, value, units, error, error_kind, n_events, side, tier, method_version, citation, flags, per_event`; `NormComparison` with a comparative statement and citation.

## 6. Protocols: declarative TOML + code registry
TOML declares capture requirements, pose config, preprocess params, segmenter name + params, metrics (with norm table), report template; Python registers segmenters and metrics by name. Example `sts_5x.toml` with `start_rule` / `end_rule` variants and provisional thresholds; `sts_30s.toml` with a 30-s window and partial-rep rule.

## 7. Model weights
Manifest `models.toml` (url, sha256, input size, layout, license), `ModelManager` downloading to a platformdirs cache with sha verification, backend given local paths so provenance records exact weights, own `det_frequency` loop, CI cache keyed on the manifest.

## 8–9. Slice 0 and Slice 1 steps
Slice 0: project files, tooling, CI, `io/video.py`, layout, track, backend protocol, models, rtmlib backend, tracking, overlay, minimal data store, CLI (`pose`, `models pull|list`, `bench`), tests, fixture, benchmark. Done when `uv sync` → `ptv pose` on the fixture writes Parquet, capture, provenance, overlay; CI green; benchmark table committed.

Slice 1 (5xSTS): preprocess (gate, interpolate, Hampel, NaN-aware Butterworth), signal helpers, clinical angle definitions with near-side selection, TRC/MOT export, `segment_sts` adapted from opencap with explicit start/end rules, metrics (`sts_total_time` tier 1 with resolution error, rep count, rise/sit/cycle times, trunk lean and knee flexion at seat-off tier 2 with literature RMSE placeholders, 30-s count), Bohannon 2006 norm table, quality checks (fps, duration, presence, required-keypoint confidence, in-frame, sagittal view, left/right stability, camera motion, rep count), report bundle JSON + HTML, pipeline orchestration writing each artifact before the next stage, CLI, synthetic segmenter tests with known events, integration from a real-clip pose fixture, validation doc.

## 10. Slices 2–4 outline
Gait (Sports2D scale port, Zeni events, tier-1 timing metrics, step length bout-mean only, Bohannon 2011 norms, quality gates), ROM (peak finding, whole degrees ± MDD, no ankle), TUG + longitudinal (segmentation, MDC deltas, provenance mismatch warnings).

## 11. Risks
CoreML EP; duplicate `cv2`; iPhone HEVC/HDR/rotation/VFR; left/right swap; occlusion and multiple people; stopwatch vs algorithm start/stop definition; norms misread as diagnosis; scope creep; vendored code drift; research-license leakage.

## 12. Order of work
Green CI first; io/layout/track/backend; models/backend/tracker/overlay; data store + CLI + benchmark + ADRs; preprocess/signal/angles/export; segmenter/metrics/norms on synthetic signals; real clips → fixture; quality/report/pipeline/CLI; integration tests + validation doc; tag v0.1.0.
