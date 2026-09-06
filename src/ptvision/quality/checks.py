"""Quality checks on capture metadata and the raw pose track. Each check yields pass / warn / fail
with the measured value, the threshold, and a plain-language reason a clinician can act on."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ptvision.clinical.protocol import Protocol
from ptvision.data.models import CameraCapture
from ptvision.pose.track import PoseTrack

Status = Literal["pass", "warn", "fail"]
_ORDER = {"pass": 0, "warn": 1, "fail": 2}


class QualityCheck(BaseModel):
    name: str
    status: Status
    value: float | None
    threshold: float | None
    message: str


class QualityReport(BaseModel):
    status: Status
    checks: list[QualityCheck] = Field(default_factory=list)

    @property
    def failures(self) -> list[QualityCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[QualityCheck]:
        return [c for c in self.checks if c.status == "warn"]


def _mk(
    name: str,
    ok: bool,
    value: float | None,
    threshold: float | None,
    msg_ok: str,
    msg_bad: str,
    *,
    severity: Status = "fail",
) -> QualityCheck:
    return QualityCheck(
        name=name,
        status="pass" if ok else severity,
        value=value,
        threshold=threshold,
        message=msg_ok if ok else msg_bad,
    )


def check_capture(cam: CameraCapture, protocol: Protocol) -> list[QualityCheck]:
    spec = protocol.capture
    out = [
        _mk(
            "frame_rate",
            cam.fps >= spec.min_fps,
            cam.fps,
            spec.min_fps,
            f"{cam.fps:.1f} fps",
            f"{cam.fps:.1f} fps is below the {spec.min_fps:g} fps this protocol needs "
            "for event timing.",
        ),
        _mk(
            "resolution",
            min(cam.width, cam.height) >= 480,
            float(min(cam.width, cam.height)),
            480,
            f"{cam.width}x{cam.height}",
            f"{cam.width}x{cam.height} is low; keypoint precision degrades below 480p.",
            severity="warn",
        ),
    ]
    if cam.duration_s is not None:
        lo, hi = spec.duration_s
        out.append(
            _mk(
                "duration",
                lo <= cam.duration_s <= hi,
                cam.duration_s,
                None,
                f"{cam.duration_s:.1f} s",
                f"{cam.duration_s:.1f} s is outside the expected {lo:g}–{hi:g} s "
                "for this protocol.",
                severity="warn",
            )
        )
    if cam.source_is_vfr:
        out.append(
            QualityCheck(
                name="variable_frame_rate",
                status="warn",
                value=None,
                threshold=None,
                message=(
                    "Source was variable frame rate; it was resampled to constant frame rate on "
                    "ingest. Lock the frame rate in the camera app for best timing accuracy."
                ),
            )
        )
    return out


def check_track(
    track: PoseTrack, protocol: Protocol, *, person: int = 0, min_score: float = 0.5
) -> list[QualityCheck]:
    """Checks on the raw (unfiltered) single-person track."""
    lay = track.layout
    coords = track.coords[:, person]
    score = track.score[:, person]
    present = ~np.isnan(coords).all(axis=(1, 2))
    out: list[QualityCheck] = []

    frac_present = float(present.mean()) if present.size else 0.0
    out.append(
        _mk(
            "subject_present",
            frac_present >= 0.95,
            frac_present,
            0.95,
            f"subject detected in {frac_present:.0%} of frames",
            f"subject detected in only {frac_present:.0%} of frames; keep the whole body "
            "in frame for the entire test.",
        )
    )

    req = [n for n in protocol.capture.required_keypoints if lay.has(n)]
    if req and present.any():
        idx = lay.indices(*req)
        mean_req = float(np.nanmean(score[present][:, idx]))
        out.append(
            _mk(
                "required_keypoint_confidence",
                mean_req >= min_score,
                mean_req,
                min_score,
                f"mean confidence of required keypoints {mean_req:.2f}",
                f"mean confidence of required keypoints is {mean_req:.2f}; hips/knees may be "
                "occluded (chair arms, clothing, another person).",
            )
        )

    if track.image_size is not None and present.any():
        w, h = track.image_size
        bb = track.bboxes()[:, person][present]
        margin = 0.01 * max(w, h)
        touching = (
            (bb[:, 0] <= margin)
            | (bb[:, 1] <= margin)
            | (bb[:, 2] >= w - margin)
            | (bb[:, 3] >= h - margin)
        )
        frac_touch = float(np.mean(touching))
        out.append(
            _mk(
                "subject_in_frame",
                frac_touch <= 0.05,
                frac_touch,
                0.05,
                "subject stays inside the frame",
                f"subject touches the frame edge in {frac_touch:.0%} of frames; "
                "move the camera back or re-aim.",
                severity="warn",
            )
        )

    # View check: in a sagittal view the shoulders overlap, so their horizontal separation is small
    # relative to trunk length.
    if all(lay.has(n) for n in ("LShoulder", "RShoulder", "Neck", "Hip")) and present.any():
        ls, rs, nk, hp = (
            coords[:, lay.index(n)] for n in ("LShoulder", "RShoulder", "Neck", "Hip")
        )
        with np.errstate(invalid="ignore"):
            sep = np.abs(ls[:, 0] - rs[:, 0])
            trunk = np.linalg.norm(nk - hp, axis=1)
            ratio = float(np.nanmedian(sep / trunk))
        want_sagittal = protocol.capture.view in ("sagittal", "sagittal_left", "sagittal_right")
        is_sagittal = ratio < 0.35
        out.append(
            _mk(
                "camera_view",
                is_sagittal == want_sagittal,
                ratio,
                0.35,
                f"view consistent with protocol ({'sagittal' if is_sagittal else 'frontal'})",
                f"protocol expects a {protocol.capture.view} view but the recording looks "
                f"{'sagittal' if is_sagittal else 'frontal'} "
                f"(shoulder separation / trunk length = {ratio:.2f}).",
            )
        )

    # Left/right ordering instability of paired lower-limb keypoints (a symptom of side swaps).
    pairs = [("LHip", "RHip"), ("LKnee", "RKnee"), ("LAnkle", "RAnkle")]
    flips = []
    for ln, rn in pairs:
        if lay.has(ln) and lay.has(rn):
            dx = coords[:, lay.index(ln), 0] - coords[:, lay.index(rn), 0]
            ok = ~np.isnan(dx)
            if ok.sum() > 10:
                sign = np.sign(dx[ok])
                med = np.sign(np.median(dx[ok])) or 1
                flips.append(float(np.mean(sign != med)))
    if flips:
        inst = float(np.mean(flips))
        out.append(
            _mk(
                "left_right_stability",
                inst <= 0.3,
                inst,
                0.3,
                "left/right assignment stable",
                f"left/right keypoints swap order in {inst:.0%} of frames; side-specific "
                "angles are unreliable (timing metrics are unaffected).",
                severity="warn",
            )
        )
    return out


def check_multi_person(full_track: PoseTrack) -> list[QualityCheck]:
    if full_track.n_persons <= 1:
        return []
    present = full_track.present()
    frac_multi = float((present.sum(axis=1) > 1).mean())
    return [
        _mk(
            "other_people_in_frame",
            frac_multi <= 0.2,
            frac_multi,
            0.2,
            f"another person visible in {frac_multi:.0%} of frames "
            "(primary subject selected automatically)",
            f"another person is visible in {frac_multi:.0%} of frames; check the overlay that "
            "the right person was analysed (use --person to override).",
            severity="warn",
        )
    ]


def check_camera_motion(video_path: Path | str) -> QualityCheck:
    from ptvision.quality.camera_motion import estimate_camera_motion

    m = estimate_camera_motion(video_path)
    if m.get("unreliable"):
        return QualityCheck(
            name="camera_motion",
            status="pass",
            value=None,
            threshold=2.0,
            message="camera motion not assessable (too little image texture); assumed steady",
        )
    return _mk(
        "camera_motion",
        m["median_px"] <= 2.0,
        m["median_px"],
        2.0,
        "camera steady",
        f"camera moves about {m['median_px']:.1f} px between sampled frames; use a tripod.",
        severity="warn",
    )


def combine(checks: list[QualityCheck]) -> QualityReport:
    status: Status = "pass"
    for c in checks:
        if _ORDER[c.status] > _ORDER[status]:
            status = c.status
    return QualityReport(status=status, checks=checks)
