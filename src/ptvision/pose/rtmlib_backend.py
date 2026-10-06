"""RTMPose Halpe-26 backend via rtmlib (YOLOX person detector + RTMPose top-down estimator)"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable

import numpy as np

from ptvision.files import DownloadProgress
from ptvision.pose.base import PoseModelInfo, PreviewFn, ProgressFn
from ptvision.pose.identity import TrackerParams, assign_ids, stitch_tracks
from ptvision.pose.layout import HALPE26
from ptvision.pose.models import ModelManager
from ptvision.pose.track import PoseTrack


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


def _box_area(bx: np.ndarray) -> float:
    return float((bx[2] - bx[0]) * (bx[3] - bx[1]))


def dedupe_persons(
    kpts: np.ndarray, scores: np.ndarray, *, iou_thr: float = 0.5, kpt_thr: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Drop pose estimates that sit on top of another one (two detector boxes on one person)"""
    n = kpts.shape[0]
    if n < 2:
        return kpts, scores
    boxes = np.full((n, 4), np.nan, np.float32)
    for i in range(n):
        good = scores[i] >= kpt_thr
        if good.sum() >= 3:
            pts = kpts[i][good]
            boxes[i] = [*pts.min(axis=0), *pts.max(axis=0)]
    keep = np.ones(n, dtype=bool)
    good = scores >= kpt_thr
    mean_good = np.where(
        good.any(axis=1), (scores * good).sum(axis=1) / np.maximum(good.sum(axis=1), 1), 0.0
    )
    order = np.argsort(-mean_good)
    for a_i, a in enumerate(order):
        if not keep[a] or np.isnan(boxes[a]).any():
            continue
        for b in order[a_i + 1 :]:
            if not keep[b] or np.isnan(boxes[b]).any():
                continue
            ix0, iy0 = max(boxes[a, 0], boxes[b, 0]), max(boxes[a, 1], boxes[b, 1])
            ix1, iy1 = min(boxes[a, 2], boxes[b, 2]), min(boxes[a, 3], boxes[b, 3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            union = _box_area(boxes[a]) + _box_area(boxes[b]) - inter
            if union > 0 and inter / union > iou_thr:
                keep[b] = False
    return kpts[keep], scores[keep]


def _grow_boxes(boxes: np.ndarray, frac: float, image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    out = boxes.copy()
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    out[:, 0] = np.clip(boxes[:, 0] - frac * bw, 0, w)
    out[:, 1] = np.clip(boxes[:, 1] - frac * bh, 0, h)
    out[:, 2] = np.clip(boxes[:, 2] + frac * bw, 0, w)
    out[:, 3] = np.clip(boxes[:, 3] + frac * bh, 0, h)
    return out


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
        download_progress: DownloadProgress | None = None,
        carry_over_s: float = 1.0,
        det_score_thr: float = 0.5,
    ):
        self.carry_over_s = carry_over_s
        self.det_score_thr = det_score_thr
        self.layout = HALPE26
        self.mode = mode
        self.device = device
        self.backend_name = backend
        self.det_frequency = max(1, int(det_frequency))
        self.tracker = tracker or TrackerParams()

        mm = models or ModelManager()
        paths = mm.ensure_mode(
            mode, verify=verify_checksums, quiet=True, progress=download_progress
        )
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

        # rtmlib's YOLOX default of 0.7 misses a person whose head or torso is out of frame;
        # 0.5 keeps them (short false detections are filtered by spurious_slots afterwards)
        det = getattr(self._model, "det_model", None)
        if det is not None and hasattr(det, "score_thr"):
            det.score_thr = self.det_score_thr

        providers: list[str] = []
        runtime: str | None = None
        if backend == "onnxruntime":
            import onnxruntime as ort

            if device == "cpu":
                self._tune_sessions()
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

    def _tune_sessions(self) -> None:
        """rebuild rtmlib's ONNX sessions with explicit CPU options"""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        so.log_severity_level = 3
        for tool, path in (
            (self._model.det_model, self._paths.det.onnx_path),
            (self._model.pose_model, self._paths.pose.onnx_path),
        ):
            tool.session = ort.InferenceSession(
                str(path), sess_options=so, providers=["CPUExecutionProvider"]
            )

    def close(self) -> None:
        """Release the model sessions (call after the last estimate; safe to call twice)."""
        model = getattr(self, "_model", None)
        if model is None:
            return
        for tool in (getattr(model, "det_model", None), getattr(model, "pose_model", None)):
            if tool is not None and hasattr(tool, "session"):
                tool.session = None
        self._model = None

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
        preview: PreviewFn | None = None,
    ) -> PoseTrack:
        per_kpts: list[np.ndarray] = []
        per_scores: list[np.ndarray] = []
        bboxes = np.empty((0, 4), np.float32)
        carry_frames = max(1, round(self.carry_over_s * fps))
        last_boxes = np.empty((0, 4), np.float32)
        unseen = 0
        w, h = image_size
        margin = 0.02 * max(w, h)
        for idx, img in frames:
            if idx % self.det_frequency == 0 or len(bboxes) == 0:
                bboxes = self.detect(img)
                if len(bboxes) == 0 and len(last_boxes) and unseen < carry_frames:
                    bboxes = last_boxes
            kpts, scores = self.pose(img, bboxes)
            if kpts.shape[0]:
                off = (
                    (kpts[..., 0] < -margin)
                    | (kpts[..., 0] > w + margin)
                    | (kpts[..., 1] < -margin)
                    | (kpts[..., 1] > h + margin)
                )
                scores = np.where(off, 0.0, scores).astype(np.float32)
                kpts, scores = dedupe_persons(kpts, scores)
            per_kpts.append(kpts)
            per_scores.append(scores)
            bboxes = bboxes_from_keypoints(kpts, scores, image_size)
            if len(bboxes):
                last_boxes = _grow_boxes(bboxes, 0.15, image_size)
                unseen = 0
            else:
                unseen += 1
            if preview is not None:
                preview(idx, img, kpts, scores)
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
            coords, score = stitch_tracks(coords, score, fps=fps, image_size=image_size)
            ids = np.arange(coords.shape[1], dtype=np.int16)
        return PoseTrack(
            layout=self.layout,
            fps=fps,
            coords=coords,
            score=score,
            person_ids=ids,
            image_size=image_size,
            stage="raw",
        )
