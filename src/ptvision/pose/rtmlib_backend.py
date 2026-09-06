"""RTMPose Halpe-26 backend via rtmlib (YOLOX person detector + RTMPose top-down estimator).

We drive rtmlib's detector and estimator ourselves instead of using `BodyWithFeet.__call__` or
`PoseTracker` so that (a) weights come from our checksummed cache and land in provenance,
(b) the detector runs every `det_frequency` frames with keypoint-derived boxes in between,
and (c) frames with no detection produce no skeleton (rtmlib would otherwise estimate a pose on
the whole image).
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable

import numpy as np

from ptvision.pose.base import PoseModelInfo, ProgressFn
from ptvision.pose.layout import HALPE26
from ptvision.pose.models import ModelManager
from ptvision.pose.track import PoseTrack
from ptvision.pose.tracking import TrackerParams, assign_ids


def bboxes_from_keypoints(
    kpts: np.ndarray,
    scores: np.ndarray,
    image_size: tuple[int, int],
    *,
    kpt_thr: float = 0.3,
    min_keypoints: int = 6,
    pad_frac: float = 0.10,
) -> np.ndarray:
    """xyxy boxes around confidently detected keypoints, padded; persons with too few good
    keypoints are dropped (they will be picked up again at the next detector frame)."""
    w, h = image_size
    boxes = []
    for kp, sc in zip(kpts, scores, strict=True):
        good = sc >= kpt_thr
        if good.sum() < min_keypoints:
            continue
        pts = kp[good]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        boxes.append(
            [
                max(0.0, x0 - pw),
                max(0.0, y0 - ph),
                min(float(w), x1 + pw),
                min(float(h), y1 + ph),
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class RtmlibBackend:
    def __init__(
        self,
        *,
        mode: str = "balanced",
        device: str = "cpu",
        backend: str = "onnxruntime",
        det_frequency: int = 4,
        models: ModelManager | None = None,
        tracker: TrackerParams | None = None,
        verify_checksums: bool = True,
    ):
        self.layout = HALPE26
        self.mode = mode
        self.device = device
        self.backend_name = backend
        self.det_frequency = max(1, int(det_frequency))
        self.tracker = tracker or TrackerParams()

        mm = models or ModelManager()
        paths = mm.ensure_mode(mode, verify=verify_checksums, quiet=True)
        self._paths = paths

        import rtmlib

        with contextlib.redirect_stdout(io.StringIO()):  # rtmlib prints "load ... with ..."
            self._model = rtmlib.BodyWithFeet(
                det=str(paths.det.onnx_path),
                det_input_size=paths.det.spec.input_size,
                pose=str(paths.pose.onnx_path),
                pose_input_size=paths.pose.spec.input_size,
                mode=mode,
                to_openpose=False,
                backend=backend,
                device=device,
            )

        providers: list[str] = []
        runtime: str | None = None
        if backend == "onnxruntime":
            import onnxruntime as ort

            providers = list(self._model.pose_model.session.get_providers())
            runtime = f"onnxruntime {ort.__version__}"
        elif backend == "opencv":
            import cv2

            runtime = f"opencv {cv2.__version__}"
        elif backend == "openvino":
            runtime = "openvino"

        self._warmup()

        self.info = PoseModelInfo(
            backend="rtmlib",
            model_class="BodyWithFeet",
            mode=mode,
            layout=self.layout.name,
            det_name=paths.det.spec.name,
            det_url=paths.det.spec.url,
            det_sha256=paths.det.onnx_sha256,
            det_input_size=paths.det.spec.input_size,
            pose_name=paths.pose.spec.name,
            pose_url=paths.pose.spec.url,
            pose_sha256=paths.pose.onnx_sha256,
            pose_input_size=paths.pose.spec.input_size,
            runtime=runtime,
            providers=providers,
            device=device,
            det_frequency=self.det_frequency,
            tracker=self.tracker.to_dict(),
        )

    def _warmup(self) -> None:
        """Run one detector + pose inference on a blank frame so provider failures surface early
        with a useful message (the ONNX CoreML EP fails on these models with ORT 1.29)."""
        h, w = self._paths.det.spec.input_size
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        try:
            self._model.det_model(blank)
            self._model.pose_model(blank, bboxes=[[0, 0, w // 2, h]])
        except Exception as e:
            if self.device != "cpu":
                raise RuntimeError(
                    f"Inference failed on device={self.device!r} ({self.backend_name}). "
                    "The ONNX Runtime CoreML provider is known to fail for these models; "
                    "use --device cpu (the default)."
                ) from e
            raise

    def detect(self, img: np.ndarray) -> np.ndarray:
        boxes = self._model.det_model(img)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def pose(self, img: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(bboxes) == 0:
            return np.empty((0, self.layout.n, 2), np.float32), np.empty(
                (0, self.layout.n), np.float32
            )
        kpts, scores = self._model.pose_model(img, bboxes=bboxes.tolist())
        return np.asarray(kpts, np.float32), np.asarray(scores, np.float32)

    def estimate(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        *,
        fps: float,
        image_size: tuple[int, int],
        n_frames: int | None = None,
        progress: ProgressFn | None = None,
    ) -> PoseTrack:
        per_kpts: list[np.ndarray] = []
        per_scores: list[np.ndarray] = []
        bboxes = np.empty((0, 4), np.float32)
        for idx, img in frames:
            if idx % self.det_frequency == 0 or len(bboxes) == 0:
                bboxes = self.detect(img)
            kpts, scores = self.pose(img, bboxes)
            per_kpts.append(kpts)
            per_scores.append(scores)
            bboxes = bboxes_from_keypoints(kpts, scores, image_size)
            if progress is not None:
                progress(idx + 1, n_frames)

        if not any(a.shape[0] for a in per_kpts):
            t = len(per_kpts)
            coords = np.full((t, 0, self.layout.n, 2), np.nan, np.float32)
            score = np.zeros((t, 0, self.layout.n), np.float32)
            ids = np.zeros((0,), np.int16)
        else:
            coords, score, ids = assign_ids(
                per_kpts, per_scores, fps=fps, image_size=image_size, params=self.tracker
            )
        return PoseTrack(
            layout=self.layout,
            fps=fps,
            coords=coords,
            score=score,
            person_ids=ids,
            image_size=image_size,
            stage="raw",
        )
