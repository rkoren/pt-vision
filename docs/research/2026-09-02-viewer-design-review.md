# Design review: desktop viewer against the real code (September 2026)

Produced 2026-09-02 by a design-review agent that read the codebase (`PoseTrack`, `HALPE26`, the cv2
overlay, `pipeline.analyze`, `AngleSeries`, `Protocol`, `StsEvents`, the CLI, `normalize()`, the
synthetic test doubles) before critiquing the proposed app design. Reproduced as delivered. The project
adopted nearly all of it, including the in-process `QThread` with cooperative cancellation (the parallel
stack-research report had suggested a subprocess; that variant is backlog B36).

Measured on this machine (M4 Pro, OpenCV 5.0 FFmpeg backend) on the normalised 480p fixture: GOP 60+
(1 I / 15 P / 44 B); sequential decode 0.30 ms/frame; `CAP_PROP_POS_FRAMES` seek+read 8.3 ms mean,
frame-accurate (0/40 mismatches vs sequential). Extrapolated to 1080p: ~1.5 ms/frame sequential; seek
worst case with GOP 250 ≈ 375 ms (unusable for back-stepping); with GOP 15 ≈ 25 ms.

---

## 1. Critique of the proposed design

**Over-engineered (cut or defer)**
- "both" colour mode + generic `combine(modes...)`: drop. Two modes only: *Rules* (bone colour = rule status, confidence shown as dimming) and *Confidence*. "Both" double-encodes what dimming already conveys.
- "Recent trials" on the drop page: defer (QSettings persistence; zero value for the demo).
- Angle labels at joints in the app overlay: already a backlog item.
- `app/model.py`: rename to `app/session.py` (collides with `pose/models.py` and `data/models.py`).

**Under-engineered / will bite (with the fix)**
1. **No cancellation exists in the pipeline.** Fix without touching the pipeline API: the worker's progress callback raises `PipelineCancelled` when a `threading.Event` is set; it propagates out of `backend.estimate()`. Stages without progress take <1 s. On cancel, delete run dirs under `runs/` that lack `provenance.json` and are newer than the job start.
2. **`QThread: Destroyed while thread is still running` aborts the process.** Keep the worker on the window; `finished -> deleteLater`; `closeEvent` must `cancel()` then `wait(5000)`; never `terminate()`.
3. **onnxruntime in a QThread is fine** (ORT releases the GIL in `Run`) but rtmlib's pre/post-processing holds the GIL between frames; the UI stays usable, not silky. Throttle progress emits. Never touch widgets from the worker. Model load (1–3 s, plus a possible ~100 MB download) happens inside the worker under a "loading model" stage.
4. **pytest-qt aborts the whole pytest run** when installed without a Qt binding. Put it in a `qt` dependency group installed alongside `--extra app`, and `importorskip` in `tests/app/conftest.py`.
5. **QImage without copies**: `QImage(frame.data, w, h, frame.strides[0], Format_BGR888)` wraps the cv2 buffer; the ndarray must outlive the QImage. `QPixmap.fromImage` is the one unavoidable copy (~2 ms at 1080p). Do not paint the skeleton onto the 24-bit QImage (slow raster path, blurs when scaled on Retina); draw it as a separate `QGraphicsItem` in image coordinates with cosmetic pens.
6. **Seek performance**: the `-g 15` change in `normalize()` is load-bearing. Add a small LRU frame cache (256 MB ≈ 41 frames at 1080p). Open with `cv2.CAP_FFMPEG` explicitly (AVFoundation frame indexing is less reliable). Old long-GOP trials stay correct, just slower on backward seeks.
7. **pyqtgraph, not matplotlib, for the timeline.** `report/figures.py` calls `matplotlib.use("Agg")` at import and `analyze()` imports it inside the worker; a QtAgg canvas in the same process would fight that. pyqtgraph moves the cursor with one `setPos`. Set `PYQTGRAPH_QT_LIB=PySide6` before importing; disable its context menu and y-mouse.
8. **HiDPI on macOS**: Qt 6 scales automatically; cosmetic pens stay crisp. `fitInView` has the classic 2 px margin bug: `setFrameShape(NoFrame)` and compute the scale manually.
9. **Keyboard focus**: arrows/space die in a focused `QSlider`/`QListWidget`/`QPushButton`. Real `QAction`s in a Playback menu with `Qt.WindowShortcut`; `VideoView` `StrongFocus` and focused when the viewer shows; panel widgets `NoFocus`/`ClickFocus`. macOS Home/End need Fn, so add `Cmd+Left/Right` as second shortcuts. *(Project addition: an application-level key filter as a fallback, because shortcuts only fire for the active window and never on the offscreen platform.)*
10. **Playback timing**: a plain `QTimer` at 1000/fps accumulates lag. Drive from `QElapsedTimer`: `target = base + int(elapsed_ms * fps / 1000)`, drop frames when behind, `Qt.PreciseTimer` at half the frame interval.
11. **Side is not persisted**: rule colouring needs `AngleSeries.side`. Add `AngleSeries.save/load` with Arrow metadata (like `PoseTrack.save`); pipeline switches from `to_parquet` to `save`. Old runs: fall back to `resolve_side(raw_single, "near")`.
12. **The plan's verification step is wrong**: synthetic `trunk_lean` peaks at 30°, so with `ok = [0, 40]` the trunk never leaves green. Keep clinical bands provisional; for tests use a tmp protocol with tighter bands.
13. **Two gradients would drift**: `pose/overlay.py::score_color` must delegate to `viz/status.py`.
14. **Linux CI + cv2**: opencv-python manylinux wheels have bundled Qt5 and set `QT_QPA_PLATFORM_PLUGIN_PATH` on import, which can break PySide6's `offscreen` plugin. Verify on ubuntu; if it bites, switch to `opencv-python-headless` with the existing override trick.
15. **QMediaPlayer** (rejected): needs Addons, seeks by ms to keyframes, no frame index, own clock.

## 2. Modules, signatures, data flow

**Decode: cv2 sequential + seek + LRU cache (recommended).** Memory math kills pre-decode:

| resolution | MB/frame | 60 s | 120 s |
|---|---|---|---|
| 1080p | 6.22 | 11.2 GB | 22.4 GB |
| 720p | 2.76 | 5.0 GB | 9.9 GB |
| 480p | 1.23 | 2.2 GB | 4.4 GB |

A JPEG-in-RAM cache is viable but a second decode path. With GOP 15 the cv2 path is ~1.5 ms sequential, ≤25 ms seek at 1080p, no memory cliff.

- `io/video.py`: `normalize(..., gop=15)` appends `-g 15 -keyint_min 15 -sc_threshold 0` (keep B-frames; ~20–30% larger files at crf 18). `FrameSource(path, cache_bytes=256<<20)` with `n_frames, fps, size, read(index), close()`: cache hit → return; `index == pos` → read; short forward → read-and-discard; else seek; tracks `pos` itself.
- `viz/status.py` (numpy only): `CONF_HIDE 0.3`, `CONF_FULL 0.6`, `CONF_LO/HI 0.3/0.8`; `status_rgb`, `dim_rgb`, `bone_confidence`, `confidence_status`, `Band`, `band_status` (continuous: 1 inside ok, 1→0.5 to the warn edge, 0.5→0 one more band width, NaN passthrough), `rule_status`, `bone_rule_status`, `BoneStatus` with `value(mode)` and `colors(t, mode) -> (rgb, visible)`, `legend(mode)`.
- `viz/skeleton.py`: `SEGMENTS`, `segment_edges(layout, segment, side)`, `bone_segments(coords, edges, colors, visible)` — the one place both renderers loop over.
- `kinematics/angles.py`: `AngleDef.segments`; `AngleSeries.save/load` with `ptvision_angles_schema, side, facing, fps`.
- `pipeline.py`: `angles.save`; `pose_only()` shared by `cli.pose` and the app; `AnalyzeOptions.color_by`; `render_overlay(..., status, mode)`.
- `pose/overlay.py`: `draw_pose(..., bone_colors, visible, joint_mode)`; `render_overlay(..., status, mode, legend)`; `score_color` via `viz.status`.
- `app/session.py` (Qt-free): `TrialSession.load(trial_dir, run_id)` from capture → pose Parquet → run provenance (primary person, protocol) → `angles.parquet` (or computed on the fly for pose-only trials) → `events.json` → markers/phases/window → metrics, norms, quality; `BoneStatus`; `available_modes`, `bone_colors`, `angle_readout`.
- `app/worker.py`: `PipelineWorker(QThread)` with `progress`, `stage`, `finished_ok`, `failed`, `cancelled`; `cancel()`; `overlay=False` always (the app is the overlay).
- `app/player.py`: clock-driven `Player(QObject)` with `frameChanged`, `play/pause/toggle/seek/step`.
- `app/overlay.py`: `SkeletonItem(QGraphicsItem)` with cosmetic pens; `app/view.py`: `VideoView(QGraphicsView)` with `show_frame`, fit/zoom/pan; `app/timeline.py`: `AngleTimeline(pg.GraphicsLayoutWidget)` with x-linked rows, phase shading, ok band, test window, cursor per row, click/drag → `seekRequested`; `app/panel.py`: `SidePanel` with quality banner, metrics table, events list, mode radios, angle checkboxes, `LegendWidget`.
- `app/main.py`: `DropPage`, `ProgressPage`, `MainWindow` with `QStackedWidget`, viewer splitter, Playback menu actions, status bar readout, `closeEvent`, `_handle_paths` as the tested entry behind `dropEvent`; test hook `MainWindow(backend_factory=...)`.

Data flow: `DropPage → MainWindow.start_job → PipelineWorker → finished_ok → TrialSession.load → set_session on view/timeline/panel → Player.frameChanged → show_frame + timeline cursor + status bar`; `timeline.seekRequested / panel.eventActivated → Player.seek`; `panel.modeChanged → SkeletonItem.set_mode + legend`.

## 3. `[[rules]]` schema and bone mapping
`angle` (key of ANGLE_DEFS), `label`, `ok = [lo, hi]`, `warn` (must contain ok; default ok widened by `margin`), `segments` (default `ANGLE_DEFS[angle].segments`), `note`. `RuleSpec` validates (known angle, lo<hi, warn contains ok, known segments) and offers `band()`, `resolved_segments()`. First provisional bands: trunk_lean ok [0,40] warn [−10,55]; knee_flexion ok [0,110] warn [0,130]; hip_flexion ok [0,120] warn [0,140].

Segment → HALPE26 edges: trunk = Hip–Neck, Hip–LHip, Hip–RHip, Neck–LShoulder, Neck–RShoulder (midline; girdle stubs included so the torso lights up as a block); head = Neck–Head, Head–Nose, Nose–eyes, eyes–ears; thigh = {S}Hip–{S}Knee; shank = {S}Knee–{S}Ankle; foot = {S}Ankle–{S}Heel, {S}Ankle–{S}BigToe, {S}BigToe–{S}SmallToe; upper_arm, forearm likewise. Near side from `AngleSeries.side` maps `{S}`; only that side's bones receive rule status, contralateral bones stay NaN → neutral. A future `sides = "both"` option is a backlog item.

## 4. Status → colour gradient and legend
Piecewise linear in sRGB: 0.00 `#D62828`, 0.50 `#F2A900`, 1.00 `#2BA84A`; NaN `#8C9BAB` (not assessed); dim target `#5A5F66` as confidence falls 0.60 → 0.30; hidden below 0.30. Dimming by colour lerp rather than alpha so cv2 and Qt produce the same pixels. Joints: confidence mode = gradient by score; rules mode = small white dots. Other persons `#9E9E9E`. `legend(mode)` feeds both renderers. Red/green legibility for deuteranopes is a known limitation (backlog).

## 5. pyproject, CI, `ptv app`
`app = ["PySide6-Essentials", "pyqtgraph"]`; `qt = ["pytest-qt"]` group installed only with `--extra app`; marker `qt`; `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` before any Qt import; CI installs `--group qt --extra app` with Linux libs `libegl1 libxkbcommon0 libfontconfig1 libdbus-1-3`; `NOTICE` gains PySide6 (LGPL-3, dynamic) and pyqtgraph (MIT). `ptv app [SOURCE] [-p pose|sts_5x|…] [-o DIR] [--run ID]` (function `app_cmd`; lazy import with an install hint): no argument → drop page; video → runs `-p` immediately; trial dir → viewer if it has a run, otherwise a pose-only job.

## 6. Test plan
Refactor `FakeBackend` and the synthetic video helper into `tests/synthetic.py`, with `delay_s` and per-frame `progress` for cancel tests. Unit: status stops and NaN, `band_status` shape, `bone_rule_status` colours exactly the expected edges with min over overlapping rules, every `SEGMENTS` pair is a real edge and the union is all 25 edges, `RuleSpec` validation, `AngleSeries` roundtrip + legacy fallback, `normalize` I-frame count and `FrameSource` seek equality, `draw_pose` pixel colour, `TrialSession.load` on analysed and pose-only trials. pytest-qt smoke: window constructs; open trial shows viewer; step/seek via actions; arrow keys with focus on the view and on the events list; mode switch changes the trunk colour and `grab()` is non-empty; worker completes; cancel leaves no orphan run dirs; close while running; `_handle_paths` with a video and with a trial dir. Manual: 1080p/30 smoothness, Retina crispness, Finder drag-drop, menu shortcuts, real RTMPose run with cancel/close, palette legibility, CLI overlay vs app spot-check.

## 7. Order of implementation
1. `AngleDef.segments`, `AngleSeries.save/load`, `pose_only()`. 2. `viz/status.py`, `viz/skeleton.py` + tests. 3. `RuleSpec`, rules in TOMLs. 4. CLI overlay `--color-by`, legend, `score_color` delegation. 5. `normalize` GOP + `FrameSource` + seek tests. 6. pyproject/CI/conftest/NOTICE, `app/__init__`, first Qt smoke test (proves the cv2/Qt plugin question early). 7. `session.py`. 8. `player.py`, `view.py`, `overlay.py`. 9. `timeline.py`. 10. `panel.py`. 11. `main.py`, `worker.py`, `ptv app`, remaining Qt tests. 12. Manual QA, screenshots, ADR, README, backlog. Roughly 7–8 focused days; steps 1–6 land in CI before any UI work.
