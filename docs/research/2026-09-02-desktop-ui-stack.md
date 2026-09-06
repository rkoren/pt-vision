# Desktop app research for pt-vision (verified September 2026)

Produced 2026-09-02 by a research agent during planning of the desktop viewer. Reproduced as delivered.
Method stated by the agent: every version, license and wheel tag came from `https://pypi.org/pypi/<pkg>/json`
or a page fetched in the session; benchmarks were run on this machine against the 480p normalised fixture.
Where something could not be verified the report says so.

---

## 1. PySide6

| Fact | Value |
|---|---|
| Latest | 6.11.2, uploaded 2026-08-18 |
| `requires_python` | `>=3.10,<3.15` |
| macOS wheel | `cp310-abi3-macosx_13_0_universal2`: one wheel covers arm64 + x86_64 and every Python ≥3.10 |
| License | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (plus commercial) |

`PySide6` is a meta-package depending on `shiboken6`, `PySide6_Essentials`, `PySide6_Addons`. Sizes (macOS universal2): Essentials 110.8 MB, Addons 332.0 MB, full 442.8 MB. **QtMultimedia is in Addons, not Essentials** (verified from `README.pyside6_essentials.md`). Choosing self-decoding therefore saves 332 MB of wheel.

### LGPLv3 obligations for a commercial app (from qt.io's obligations page)
1. Dynamic linking only (the wheels already ship Qt as separate dylibs; never `--onefile`).
2. Ship Qt's corresponding source or a written offer.
3. Allow the user to relink, with sufficient installation information.
4. Include the LGPLv3 text and a prominent notice.
5. No anti-tivoization.

macOS gotcha: relinking conflicts with hardened-runtime notarization (swapping `QtCore.framework` invalidates the signature). Workable answer: keep Qt as replaceable dylibs, document the relink path (replace, then ad-hoc re-sign, accepting loss of the notarization ticket), plus a written source offer, stated in `NOTICE`. pt-vision's Apache-2.0 license does not discharge LGPL obligations; they attach to distributing the binary.

---

## 2. Video playback + overlay: recommend (b), decode with OpenCV

The discriminating requirement is arrow-key frame stepping: land deterministically on frame N, including backward, cheaply.

### (a) QMediaPlayer + QVideoSink: rejected
- Exposes only `position()` in milliseconds; no frame index; seek precision undocumented. On macOS Qt 6.11 defaults to the FFmpeg backend; AVFoundation seek tolerance is not exposed. Pulls the 332 MB Addons wheel. Would require rewriting `draw_pose()` for QPainter.

### (b) cv2.VideoCapture → QImage → QGraphicsView: recommended
Frame-accurate by construction. Measured on this machine (OpenCV 5.0.0, M-series):

```
sequential decode, 854x480       : 3332 fps
1080p cvtColor BGR->RGB          : 0.36 ms
1080p frame .copy()              : 0.22 ms
1080p copy + draw 26-kp skeleton : 0.48 ms
frame budget at 30 fps           : 33.3 ms
```

Caveats: the `QImage`→`QPixmap`→paint leg was not measured (PySide6 not installed in that session). Use `QImage.Format_BGR888` to skip `cvtColor`; construct zero-copy as `QImage(arr.data, w, h, arr.strides[0], Format_BGR888)` and keep a Python reference to the ndarray alive while the widget can repaint, or you get tearing and eventually a segfault. The seek test was degenerate: the 60-frame fixture has exactly one keyframe, so every seek decoded from frame 0; it says nothing about 1,800-frame clips where OpenCV issues #4890, #9053, #23088 bite.

**Required change**: `normalize()` passes no `-g`, so libx264 uses `keyint=250` (a keyframe every ~8 s). Add `-g 15` (or 30). With a 15-frame GOP a worst-case scrub is ≤15 decodes ≈ 5 ms at 1080p and exactness stops depending on OpenCV's PTS bookkeeping. This change is what makes (b) defensible.

Structural advantage: `iter_frames()` already yields BGR numpy frames and `draw_pose()` already renders; the GUI reuses both.

### (c) PyAV: good fallback, licensing trap
`av` 18.1.0 (BSD-3 bindings) has the better seek primitive, **but the binary wheels bundle a GPL FFmpeg** (`--enable-gpl --enable-libx264 --enable-libx265` in `pyav-ffmpeg`'s build script). The LGPL-only alternative is `basswood-av` 15.2.1, three major versions behind.

**Verdict: (b)**, conditional on `-g 15`.

---

## 3. Background work: QProcess for v1 (agent's recommendation)
ONNX Runtime releases the GIL during `session.run()`, so a `QThread` keeps the UI responsive; `multiprocessing` is unnecessary (macOS `spawn` re-imports everything). But a thread blocked inside `session.run()` cannot be interrupted, and users will hit Cancel. The agent recommended `QProcess` running the existing CLI with JSON-lines progress (kill-ability, crash isolation, reuse of tested code), switching to a thread later if in-process access is needed. Either way: cap `intra_op_num_threads`, never touch widgets from the worker.

*(Project decision: an in-process QThread with cooperative cancellation between frames was chosen; the subprocess variant is backlog B36.)*

---

## 4. Timeline plot: pyqtgraph
pyqtgraph 0.14.0, MIT, `py3-none-any` (1.9 MB), uploaded 2025-11-16; supports PySide6. Moving an `InfiniteLine` is a single item transform; matplotlib's QtAgg re-renders the figure. Caveats: pyqtgraph 0.14.0 predates PySide6 6.11.2 by ~9 months (pin and smoke-test); matplotlib stays for static report figures.

---

## 5. Drag-and-drop on macOS
`setAcceptDrops(True)`; `dragEnterEvent` checks `hasUrls()`; `dropEvent` iterates `urls()` → `toLocalFile()`. Security-scoped bookmarks are an App Sandbox mechanism and not applicable. Real gotcha: drag-and-drop that works from `python main.py` can stop working after packaging unless `Info.plist` carries `CFBundleDocumentTypes`; test in the packaged bundle. Filter by extension yourself.

---

## 6. Packaging: PyInstaller, and the headline blocker is architecture
- **You cannot ship universal2.** `onnxruntime` 1.29.0 macOS wheels are `macosx_14_0_arm64` only; `opencv-python` 5.0.0.93 ships arm64 and x86_64 as separate wheels. So: arm64-only `.app`, macOS 14 minimum.
- Bundle size (installed, this venv): pyarrow 123 MB, cv2 120 MB, scipy 83 MB, onnxruntime 77 MB, pandas 48 MB, matplotlib 28 MB, numpy 25 MB, Pillow 13 MB, PySide6-Essentials ~125 MB arm64-thinned (estimate). Realistic `.app`: 500–700 MB plus ~100 MB of weights. **pyarrow is the single largest dependency and is GUI-irrelevant.**
- Tools: PyInstaller 6.22.2 (GPLv2 with an exception permitting non-free programs); use `--onedir`; known notarization issue (`base_library.zip` must move to `Contents/Resources`, pyinstaller #8927, #5112). briefcase 0.4.4 wants to own the layout and is happiest with pure-Python deps. pyside6-deploy wraps Nuitka; Nuitka's tracker documents notarization failures, especially with data files (we ship TOML, CSV, templates). Avoid for v1.
- **The ffmpeg licensing trap**: the Homebrew ffmpeg on this machine is a GPL build (`--enable-gpl --enable-version3 --enable-libx264`) and cannot be redistributed inside a commercial-friendly app. `ffmpeg -encoders` lists `h264_videotoolbox`, so an LGPL-only build can encode H.264 via hardware on macOS. opencv-python's bundled FFmpeg is the LGPL build (separate dylibs, no x264/x265 in `LICENSE-3RD-PARTY.txt`); ship that license file.

---

## 7. Alternatives
- **Dear PyGui 2.3.1** (MIT, 2 MB): no video widget; you upload textures per frame and hand-roll the timeline, slider, drag-drop and menu bar. Not worth it once Addons is already avoided.
- **Flet 0.86.5** (Apache-2.0): pleasant authoring, but needs a Flutter SDK in the build chain and builds universal2 by default, which fights the arm64-only constraint; frame-accurate scrubbing with a per-frame overlay is not its design centre.
- **Local browser (FastAPI + canvas + `requestVideoFrameCallback`)**: technically the most elegant for this problem (rVFC gives `mediaTime` and `presentedFrames`; Baseline across Chrome, Firefox and Safari since October 2024); needs Range-request video serving. Reason not to: it's not a `.app`; wrapping it in pywebview/Tauri trades one packaging problem for two, plus a localhost server that clinic security software will ask about. Keep as the fallback or the future web version.

---

## Recommended v1 architecture (as delivered)

```
MainWindow (QMainWindow)                          — accepts drops; QSplitter layout
├─ DropZone (QWidget, setAcceptDrops)             — dragEnterEvent/dropEvent → toLocalFile()
├─ PipelineRunner (QProcess)                      — runs `ptv pose --progress-json <clip>`
│    stdout JSON-lines → Signal(int,int) → QProgressBar; kill() on Cancel
├─ VideoPane (QGraphicsView + QGraphicsPixmapItem)
│    cv2.VideoCapture on the normalized mp4 (encoded with -g 15, REQUIRED)
│    QTimer(1000/fps) → cap.read() → draw_pose(frame, ...) → QImage(Format_BGR888) → setPixmap
├─ TimelinePlot (pyqtgraph.PlotWidget)            — joint-angle curves + InfiniteLine cursor
├─ MetricsPanel (QWidget / QFormLayout)
└─ AppState (QObject)                             — single source of truth: current_frame:int
```

Packages to add: `PySide6-Essentials>=6.11.2,<7` (not `PySide6`), `pyqtgraph>=0.14,<0.15`, `opencv-python` promoted to direct; `pyinstaller` in a packaging group later. Deliberately not added: `av`, `PySide6-Addons`, `briefcase`, `flet`, `dearpygui`.

Two things to do before writing GUI code: add `-g 15` to `normalize()`; decide the ffmpeg story now (bundle LGPL with VideoToolbox, or keep ffmpeg-on-PATH as a documented prerequisite).

**Sources:** PySide6 / PySide6-Essentials / pyside6-addons on PyPI · README.pyside6_essentials.md · Qt for Python package details · Qt LGPL obligations · QMediaPlayer 6.11.2 docs · Qt Multimedia on Apple · Qt forum seek-accuracy thread · OpenCV #4890, #9053, #23088 · av on PyPI · pyav-ffmpeg build script · basswood-av · FFmpeg legal · onnxruntime Python API and threading docs · pyqtgraph · QDropEvent docs · Qt forum drag-drop after packaging · PyInstaller #8927, #5112 · Nuitka #2232, #2906 · pyside6-deploy docs · DearPyGui · Flet macOS packaging · MDN and web.dev on requestVideoFrameCallback.
