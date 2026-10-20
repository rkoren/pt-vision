"""Run our single-video pipeline on lab-validation videos and score it against the lab's
marker mocap and force plates

Per trial + camera):
  1. pose on the synced video (cached in a trial dir under derived/)
  2. our angles -> RMSE, bias, limits of agreement, r, per-rep ROM error
  3. our angles against the lab's model joint angles mapped to our conventions (secondary)
  4. our sit-to-stand events against the force-plate events -> seat-off timing error, rep count,
     first-to-last seat-off span

"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ptvision.datasets.lab_trial import JOINT_ANGLE_COLUMNS, LabTrial
from ptvision.datasets.registry import DatasetDirs, load_manifest
from ptvision.datasets.sts_force import match_reps

ANGLES: tuple[str, ...] = JOINT_ANGLE_COLUMNS
DATASETS: tuple[str, ...] = ("opencap",)
# Segment inclinations (degrees from vertical, forward positive) scored vs lab
SEGMENTS: dict[str, tuple[str, str]] = {
    "thigh_vertical": ("{S}Knee", "{S}Hip"),
    "shank_vertical": ("{S}Ankle", "{S}Knee"),
    "trunk_vertical": ("Hip", "Neck"),
    "trunk_shoulders_vertical": ("Hip", "MidShoulder"),
}


def segment_angles(track: Any, side: str, facing: int) -> dict[str, np.ndarray]:
    from ptvision.kinematics.angles import segment_angle_from_vertical

    prefix = "L" if side == "left" else "R"

    def point(name: str) -> np.ndarray:
        if name == "MidShoulder":
            mid = (track.keypoint("LShoulder") + track.keypoint("RShoulder")) / 2
            return np.asarray(mid, dtype=np.float64)
        return np.asarray(track.keypoint(name.replace("{S}", prefix)), dtype=np.float64)

    out = {
        name: facing * segment_angle_from_vertical(point(base), point(tip))
        for name, (base, tip) in SEGMENTS.items()
    }
    out["thigh_vertical"] = -out["thigh_vertical"]  # knee ahead of the hip (flexed) = positive
    return out


# ---------------------------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------------------------
@dataclass
class Agreement:
    n: int
    rmse: float | None
    bias: float | None
    sd: float | None
    loa_lo: float | None
    loa_hi: float | None
    r: float | None
    slope: float | None  # least-squares slope of ours against ref; 1 = no scaling error

    @classmethod
    def of(cls, ours: np.ndarray, ref: np.ndarray) -> Agreement:
        ok = np.isfinite(ours) & np.isfinite(ref)
        n = int(ok.sum())
        if n < 2:
            return cls(n, None, None, None, None, None, None, None)
        d = ours[ok] - ref[ok]
        bias = float(d.mean())
        sd = float(d.std(ddof=1))
        slope: float | None = None
        with np.errstate(invalid="ignore", divide="ignore"):  # a flat series has no r
            r = float(np.corrcoef(ours[ok], ref[ok])[0, 1]) if n > 2 else None
            var = float(np.var(ref[ok]))
            if n > 2 and var > 0:
                slope = float(np.cov(ours[ok], ref[ok], ddof=0)[0, 1] / var)
        return cls(
            n,
            float(np.sqrt(np.mean(d**2))),
            bias,
            sd,
            bias - 1.96 * sd,
            bias + 1.96 * sd,
            None if r is None or not np.isfinite(r) else r,
            slope,
        )

    def flat(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_{k}": (round(v, 3) if isinstance(v, float) else v)
            for k, v in asdict(self).items()
        }


def estimate_lag_s(
    ours: np.ndarray, ref: np.ndarray, fps: float, max_lag_s: float = 0.5
) -> float | None:
    """Lag (ours later = positive) at the cross-correlation peak within ±max_lag_s."""
    ok = np.isfinite(ours) & np.isfinite(ref)
    if ok.sum() < int(fps):
        return None
    a = np.where(ok, ours, np.nan)
    b = np.where(ok, ref, np.nan)
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    max_lag = round(max_lag_s * fps)
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[: len(b) - lag]
        else:
            x, y = a[: len(a) + lag], b[-lag:]
        if len(x) < int(fps):
            continue
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        if c > best:
            best, best_lag = c, lag
    return best_lag / fps


def _shift(y: np.ndarray, frames: int) -> np.ndarray:
    """`y` moved later by `frames` (earlier when negative); vacated samples become NaN."""
    out = np.full_like(y, np.nan, dtype=np.float64)
    if frames >= 0:
        out[frames:] = y[: len(y) - frames]
    else:
        out[: len(y) + frames] = y[-frames:]
    return out


def resample_to(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(t_src)
    if ok.sum() < 2:
        return np.full_like(t_dst, np.nan, dtype=np.float64)
    out = np.interp(t_dst, t_src[ok], y[ok], left=np.nan, right=np.nan)
    # keep gaps: a destination sample whose nearest source sample is missing stays NaN
    idx = np.clip(np.searchsorted(t_src, t_dst), 0, len(t_src) - 1)
    out[~ok[idx]] = np.nan
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------
@dataclass
class TrialCameraResult:
    dataset: str
    subject: str
    task: str
    camera: int
    view_deg: float
    status: str = "ok"  # ok | skipped | error
    note: str = ""
    fps: float | None = None
    n_frames: int = 0
    side: str = ""
    facing: int = 0
    n_persons: int = 0
    n_persons_real: int = 0
    frac_side_gated: float | None = None  # frames with tracked-side hip/knee/ankle below gate
    camera_view_status: str = ""
    camera_view_value: float | None = None
    cos_theta_nominal: float | None = None  # cos of the camera's nominal angle off the side
    cos_theta_thigh: float | None = None  # kinematics.view estimates from the video itself
    cos_theta_shoulders: float | None = None
    lag_knee_s: float | None = None
    window_s: tuple[float, float] | None = None  # angle statistics are taken inside it
    reps_video: int = 0
    reps_force: int | None = None
    force_rule: str = ""
    seat_off_err_ms_mean: float | None = None
    seat_off_err_ms_sd: float | None = None
    seat_off_abs_err_ms_mean: float | None = None
    seat_off_err_synced_ms_mean: float | None = None  # after removing lag_knee_s
    reps_matched: int = 0
    span_video_s: float | None = None
    span_force_s: float | None = None
    span_err_s: float | None = None
    pose_seconds: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)  # flattened per-angle agreement

    def row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("metrics")
        d.update(self.metrics)
        return d


@dataclass
class EventRow:
    dataset: str
    subject: str
    task: str
    camera: int
    rep: int
    seat_off_video_s: float | None
    seat_off_force_s: float | None
    err_ms: float | None


# ---------------------------------------------------------------------------------------------
# One trial and camera
# ---------------------------------------------------------------------------------------------
def _side_gate_fraction(track: Any, side: str, threshold: float) -> float:
    prefix = "L" if side == "left" else "R"
    sc = np.stack([track.keypoint_score(f"{prefix}{n}") for n in ("Hip", "Knee", "Ankle")], axis=1)
    return float(np.mean(np.any(sc < threshold, axis=1)))


def evaluate_trial_camera(
    trial: LabTrial,
    camera: int,
    *,
    trial_dir: Path,
    backend: Any,
    mode: str | None,
    reuse: bool,
    protocol: Any,
    frames_dir: Path,
) -> tuple[TrialCameraResult, list[EventRow]]:
    from ptvision.clinical.segmenters.sts import StsParams, segment_sts
    from ptvision.kinematics.angles import compute_angles
    from ptvision.kinematics.preprocess import PreprocessConfig, preprocess
    from ptvision.kinematics.view import (
        estimate_view_shoulders,
        estimate_view_thigh_length,
        rectify_sagittal,
    )
    from ptvision.pipeline import AnalyzeOptions, pose_only
    from ptvision.quality import checks as Q
    from ptvision.trials.models import Subject

    res = TrialCameraResult(
        trial.dataset, trial.subject, trial.task, camera, float(trial.view_deg.get(camera, np.nan))
    )
    events: list[EventRow] = []
    video = trial.videos[camera]
    subject = Subject(
        height_m=trial.height_m,
        mass_kg=trial.mass_kg,
        sex="f" if trial.sex == "f" else "m" if trial.sex == "m" else None,
    )
    t0 = time.perf_counter()
    pose = pose_only(
        video,
        AnalyzeOptions(
            out_dir=trial_dir, mode=mode, overlay=False, reuse_pose=reuse, subject=subject
        ),
        backend=backend,
    )
    res.pose_seconds = round(time.perf_counter() - t0, 2)
    track = pose.track
    res.fps = float(track.fps)
    res.n_frames = track.n_frames
    res.n_persons = track.n_persons
    res.n_persons_real = track.n_persons_real
    if pose.primary_person is None:
        res.status, res.note = "error", "no person detected"
        return res, events
    single = track.select_person(pose.primary_person)

    # quality: the camera-view band our tool would report for this clip
    for chk in Q.check_track(single, protocol):
        if chk.name == "camera_view":
            res.camera_view_status = chk.status
            res.camera_view_value = round(float(chk.value), 3) if chk.value is not None else None

    pcfg = PreprocessConfig(
        score_threshold=protocol.preprocess.score_threshold,
        interp_max_gap_s=protocol.preprocess.interp_max_gap_s,
        filter_type=protocol.preprocess.filter_type,
        filter_order=protocol.preprocess.filter_order,
        filter_cutoff_hz=protocol.preprocess.filter_cutoff_hz,
    )
    pre = preprocess(single, pcfg)
    ours = compute_angles(pre.filtered, list(ANGLES), side="near")
    res.side, res.facing = ours.side, int(ours.facing)
    res.frac_side_gated = round(_side_gate_fraction(single, ours.side, pcfg.score_threshold), 3)
    fps = float(track.fps)
    t_video = np.arange(track.n_frames) / fps

    # 2. the same angle definitions on the lab joint centres, same side. Compared over the
    # sit-to-stand window (first force seat-off − 1.5 s to last + 2.5 s)
    ref = trial.reference_track(fps)
    ref_angles = compute_angles(ref.track, list(ANGLES), side=ours.side)
    n = min(track.n_frames, ref.n_frames)
    force = trial.force_sts_events()
    lo_s, hi_s = 0.0, n / fps
    if force is not None and force.n_reps:
        lo_s = max(0.0, force.seat_off_s[0] - 1.5)
        hi_s = min(hi_s, force.seat_off_s[-1] + 2.5)
    win = (t_video[:n] >= lo_s) & (t_video[:n] <= hi_s)
    res.window_s = (round(lo_s, 2), round(hi_s, 2))
    frame = pd.DataFrame({"time_s": t_video[:n], "in_window": win})
    for name in ANGLES:
        a, b = ours[name][:n], ref_angles[name][:n]
        frame[f"{name}_video"] = a
        frame[f"{name}_mocap"] = b
        res.metrics.update(Agreement.of(a[win], b[win]).flat(f"{name}_mocap"))
    res.lag_knee_s = estimate_lag_s(
        ours["knee_flexion"][:n][win], ref_angles["knee_flexion"][:n][win], fps
    )
    # same statistics after removing lag (video shifted by whole frames), which
    # separates the dataset's video-to-mocap synchronisation from our angle error
    shift = -round((res.lag_knee_s or 0.0) * fps)  # a negative lag means ours is early
    for name in ANGLES:
        a = _shift(ours[name][:n], shift)
        res.metrics.update(Agreement.of(a[win], ref_angles[name][:n][win]).flat(f"{name}_synced"))
    seg_ours = segment_angles(pre.filtered, ours.side, ours.facing)
    seg_ref = segment_angles(ref.track, ours.side, ref_angles.facing)
    for name in SEGMENTS:
        a, b = _shift(seg_ours[name][:n], shift), seg_ref[name][:n]
        frame[f"{name}_video"] = seg_ours[name][:n]
        frame[f"{name}_mocap"] = b
        res.metrics.update(Agreement.of(a[win], b[win]).flat(f"{name}_synced"))

    # 2b. the same, after adjusting for height with off-side camera
    if np.isfinite(res.view_deg):
        res.cos_theta_nominal = round(float(np.cos(np.radians(90.0 - abs(res.view_deg)))), 3)
    for tag, est in (
        ("rect_thigh", estimate_view_thigh_length(pre.filtered, ours.side)),
        ("rect_shoulders", estimate_view_shoulders(pre.filtered)),
    ):
        setattr(res, "cos_theta_" + tag.removeprefix("rect_"), round(est.cos_theta, 3))
        res.metrics[f"{tag}_applied"] = est.applied
        if not est.applied:
            continue
        rect = rectify_sagittal(pre.filtered, est.cos_theta)
        ours_r = compute_angles(rect, list(ANGLES), side=ours.side)
        for name in ANGLES:
            a = _shift(ours_r[name][:n], shift)
            frame[f"{name}_{tag}"] = ours_r[name][:n]
            res.metrics.update(
                Agreement.of(a[win], ref_angles[name][:n][win]).flat(f"{name}_{tag}")
            )
        seg_r = segment_angles(rect, ours.side, ours.facing)
        for name in SEGMENTS:
            a = _shift(seg_r[name][:n], shift)
            res.metrics.update(Agreement.of(a[win], seg_ref[name][:n][win]).flat(f"{name}_{tag}"))

    # 3. the lab's own model coordinates, mapped to our conventions
    model = trial.reference_joint_angles(ours.side)
    if model is not None:
        tm = model["time"].to_numpy(dtype=np.float64)
        for name in ANGLES:
            b = resample_to(tm, model[name].to_numpy(dtype=np.float64), t_video[:n])
            frame[f"{name}_model"] = b
            res.metrics.update(Agreement.of(ours[name][:n][win], b[win]).flat(f"{name}_model"))

    # 4. sit-to-stand events against the force plates
    seg_params: dict[str, Any] = {
        **protocol.segmenter.params,
        "segment_signal_cutoff_hz": protocol.preprocess.segment_signal_cutoff_hz,
    }
    if trial.n_reps_expected < 0:  # count reps
        seg_params.update(n_reps_expected=-1, end_rule="window", window_s=n / fps + 1.0)
    else:
        seg_params["n_reps_expected"] = trial.n_reps_expected
    params = StsParams.from_dict(seg_params)
    seg = segment_sts(pre.filtered, fps, params, trunk_lean=ours["trunk_lean"])
    ours_seat_off = [r.seat_off / fps for r in seg.reps]
    res.reps_video = len(seg.reps)
    if seg.warnings:
        res.note = "; ".join(seg.warnings[:2])
    if len(ours_seat_off) >= 2:
        res.span_video_s = round(ours_seat_off[-1] - ours_seat_off[0], 3)
    if force is not None:
        res.reps_force = force.n_reps
        res.force_rule = force.rule
        res.span_force_s = round(force.span_s, 3) if force.span_s is not None else None
        if res.span_video_s is not None and res.span_force_s is not None:
            res.span_err_s = round(res.span_video_s - res.span_force_s, 3)
        pairs = match_reps(ours_seat_off, force.seat_off_s)
        res.reps_matched = len(pairs)
        errs = [d * 1000 for _, _, d in pairs]
        if errs:
            res.seat_off_err_ms_mean = round(float(np.mean(errs)), 1)
            res.seat_off_err_synced_ms_mean = round(
                float(np.mean(errs)) - 1000 * (res.lag_knee_s or 0.0), 1
            )
            res.seat_off_err_ms_sd = (
                round(float(np.std(errs, ddof=1)), 1) if len(errs) > 1 else None
            )
            res.seat_off_abs_err_ms_mean = round(float(np.mean(np.abs(errs))), 1)
        matched_ours = {i: (j, d) for i, j, d in pairs}
        matched_ref = {j for _, j, _ in pairs}
        for i, s in enumerate(ours_seat_off):
            j, d = matched_ours.get(i, (None, None))
            events.append(
                EventRow(
                    trial.dataset,
                    trial.subject,
                    trial.task,
                    camera,
                    i + 1,
                    round(s, 3),
                    round(force.seat_off_s[j], 3) if j is not None else None,
                    round(d * 1000, 1) if d is not None else None,
                )
            )
        for j, s in enumerate(force.seat_off_s):
            if j not in matched_ref:
                events.append(
                    EventRow(
                        trial.dataset, trial.subject, trial.task, camera, 0, None, round(s, 3), None
                    )
                )
    # per-rep range of motion against the mocap reference, over the matched video reps
    for name in ANGLES:
        rom_err: list[float] = []
        for rep in seg.reps:
            lo = max(0, int(rep.seat_off - 0.5 * fps))
            hi = min(n, int(rep.seated_return if rep.seated_return else rep.stand_reached + 1))
            if hi - lo < 3:
                continue
            a, b = ours[name][lo:hi], ref_angles[name][lo:hi]
            if np.isfinite(a).sum() < 3 or np.isfinite(b).sum() < 3:
                continue
            rom_err.append(float((np.nanmax(a) - np.nanmin(a)) - (np.nanmax(b) - np.nanmin(b))))
        res.metrics[f"{name}_rom_err_mean"] = round(float(np.mean(rom_err)), 2) if rom_err else None
        res.metrics[f"{name}_rom_err_abs_mean"] = (
            round(float(np.mean(np.abs(rom_err))), 2) if rom_err else None
        )

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(frames_dir / f"{trial.key}_cam{camera}.csv", index=False, float_format="%.3f")
    return res, events


# ---------------------------------------------------------------------------------------------
# Whole dataset
# ---------------------------------------------------------------------------------------------
def _trials_for(
    dataset: str, subjects: tuple[str, ...] | None, tasks: tuple[str, ...] | None
) -> list[LabTrial]:
    if dataset == "opencap":
        from ptvision.datasets import opencap

        return list(opencap.trials(tasks=tasks or opencap.STS_TASKS, subjects_wanted=subjects))
    raise ValueError(f"unknown dataset {dataset!r}; one of {DATASETS}")


def evaluate_video(
    dataset: str,
    *,
    cameras: tuple[int, ...] | None = None,
    subjects: tuple[str, ...] | None = None,
    tasks: tuple[str, ...] | None = None,
    limit: int | None = None,
    mode: str | None = None,
    reuse: bool = True,
    backend: Any = None,
    out_dir: Path | None = None,
    progress: Any = None,
) -> tuple[list[TrialCameraResult], Path]:
    """Score every (trial, camera) of `dataset`; writes trials.csv, events.csv, frames/ and
    summary.json under `<dataset dir>/derived/eval_video/` (or `out_dir`)."""
    from ptvision.clinical.protocol import load_protocol
    from ptvision.pipeline import AnalyzeOptions, _close_backend, _make_backend

    spec = load_manifest()[dataset]
    out_dir = out_dir or DatasetDirs.for_spec(spec).derived / "eval_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol("sts_5x")
    trials = _trials_for(dataset, subjects, tasks)
    if limit:
        trials = trials[:limit]
    jobs = [(t, c) for t in trials for c in sorted(t.videos) if cameras is None or c in cameras]
    planned = {(t.subject, t.task, c) for t, c in jobs}
    own_backend = backend is None
    if own_backend and jobs:
        backend = _make_backend(
            AnalyzeOptions(mode=mode), protocol.pose.mode, protocol.pose.det_frequency
        )
    results: list[TrialCameraResult] = []
    events: list[EventRow] = []
    try:
        for k, (trial, cam) in enumerate(jobs):
            if progress:
                progress(f"{trial.key} cam{cam}", k, len(jobs))
            trial_dir = out_dir / "trials" / trial.subject / trial.task / f"cam{cam}"
            try:
                r, ev = evaluate_trial_camera(
                    trial,
                    cam,
                    trial_dir=trial_dir,
                    backend=backend,
                    mode=mode,
                    reuse=reuse,
                    protocol=protocol,
                    frames_dir=out_dir / "frames",
                )
            except Exception as e:  # one bad trial must not end a multi-hour run
                r = TrialCameraResult(
                    trial.dataset,
                    trial.subject,
                    trial.task,
                    cam,
                    float(trial.view_deg.get(cam, np.nan)),
                )
                r.status, r.note = "error", f"{type(e).__name__}: {e}"
                ev = []
            results.append(r)
            events.extend(ev)
            _write_outputs(out_dir, dataset, results, events, backend, planned=planned)
    finally:
        if own_backend and backend is not None:
            _close_backend(backend)
    _write_outputs(out_dir, dataset, results, events, backend, planned=planned)
    return results, out_dir


def _prior_rows(path: Path, done: set[tuple[str, str, int]]) -> list[dict[str, Any]]:
    """Rows from an earlier run for trial-cameras this run does not touch, so partial runs
    (one camera, one subject) accumulate into one table instead of overwriting each other."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keep: list[dict[str, Any]] = []
    for r in rows:
        try:
            key = (str(r["subject"]), str(r["task"]), int(r["camera"]))
        except (KeyError, ValueError):
            continue
        if key not in done:
            keep.append({k: _parse(v) for k, v in r.items()})
    return keep


def _parse(v: Any) -> Any:
    if v is None or v == "":
        return None
    try:
        return int(v) if v.lstrip("-").isdigit() else float(v)
    except (ValueError, AttributeError):
        return v


def _write_outputs(
    out_dir: Path,
    dataset: str,
    results: list[TrialCameraResult],
    events: list[EventRow],
    backend: Any,
    *,
    planned: set[tuple[str, str, int]] | None = None,
) -> None:
    planned = planned or {(r.subject, r.task, r.camera) for r in results}
    prior = _prior_rows(out_dir / "trials.csv", planned)
    prior_events = _prior_rows(out_dir / "events.csv", planned)
    rows = prior + [r.row() for r in results]
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with (out_dir / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (out_dir / "events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(EventRow.__dataclass_fields__))
        w.writeheader()
        for row in prior_events:
            w.writerow(row)
        for e in events:
            w.writerow(asdict(e))
    merged = [_result_from_row(r) for r in prior] + results
    (out_dir / "summary.json").write_text(json.dumps(summarize(dataset, merged, backend), indent=2))


def _result_from_row(row: dict[str, Any]) -> TrialCameraResult:
    fields = TrialCameraResult.__dataclass_fields__
    base = {k: v for k, v in row.items() if k in fields and k != "metrics"}
    base["window_s"] = None
    res = TrialCameraResult(**base)
    res.metrics = {k: v for k, v in row.items() if k not in fields}
    return res


def _agg(values: list[float | None]) -> dict[str, float | None] | None:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 2),
        "median": round(float(np.median(v)), 2),
        "sd": round(float(v.std(ddof=1)), 2) if v.size > 1 else None,
        "min": round(float(v.min()), 2),
        "max": round(float(v.max()), 2),
    }


def _group_summary(rs: list[TrialCameraResult]) -> dict[str, Any]:
    ok = [r for r in rs if r.status == "ok"]
    out: dict[str, Any] = {"trials": len(rs), "ok": len(ok), "errors": len(rs) - len(ok)}
    for name in ANGLES:
        for ref in ("mocap", "synced", "model", "rect_thigh", "rect_shoulders"):
            for stat in ("rmse", "bias", "r", "slope"):
                out[f"{name}_{ref}_{stat}"] = _agg(
                    [r.metrics.get(f"{name}_{ref}_{stat}") for r in ok]
                )
        out[f"{name}_rom_err_abs_mean"] = _agg(
            [r.metrics.get(f"{name}_rom_err_abs_mean") for r in ok]
        )
    for name in SEGMENTS:
        for stat in ("rmse", "bias", "sd"):
            out[f"{name}_synced_{stat}"] = _agg(
                [r.metrics.get(f"{name}_synced_{stat}") for r in ok]
            )
    out["seat_off_err_ms_mean"] = _agg([r.seat_off_err_ms_mean for r in ok])
    out["seat_off_abs_err_ms_mean"] = _agg([r.seat_off_abs_err_ms_mean for r in ok])
    out["seat_off_err_synced_ms_mean"] = _agg([r.seat_off_err_synced_ms_mean for r in ok])
    out["span_err_s"] = _agg([r.span_err_s for r in ok])
    out["lag_knee_s"] = _agg([r.lag_knee_s for r in ok])
    for key in ("cos_theta_nominal", "cos_theta_thigh", "cos_theta_shoulders"):
        out[key] = _agg([getattr(r, key) for r in ok])
    out["rep_count_agrees"] = sum(
        1 for r in ok if r.reps_force is not None and r.reps_video == r.reps_force
    )
    out["rep_count_compared"] = sum(1 for r in ok if r.reps_force is not None)
    out["camera_view"] = {
        s: sum(1 for r in ok if r.camera_view_status == s) for s in ("pass", "warn", "fail")
    }
    return out


def summarize(
    dataset: str, results: list[TrialCameraResult], backend: Any = None
) -> dict[str, Any]:
    from ptvision import __version__

    by_cam: dict[str, Any] = {}
    for cam in sorted({r.camera for r in results}):
        rs = [r for r in results if r.camera == cam]
        by_cam[f"cam{cam}"] = {"view_deg": rs[0].view_deg, **_group_summary(rs)}
    by_subject = {
        s: _group_summary([r for r in results if r.subject == s])
        for s in sorted({r.subject for r in results})
    }
    info = getattr(backend, "info", None)
    dump = getattr(info, "model_dump", None)
    if callable(dump):
        info = {k: v for k, v in dump().items() if not k.endswith(("_url", "_sha256"))}
    elif is_dataclass(info) and not isinstance(info, type):
        info = {k: v for k, v in asdict(info).items() if not k.endswith(("_url", "_sha256"))}
    return {
        "dataset": dataset,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ptvision_version": __version__,
        "pose": info if isinstance(info, dict) else (str(info) if info is not None else None),
        "angles": list(ANGLES),
        "held_out": "no pipeline parameter was tuned on this dataset",
        "all": _group_summary(results),
        "by_camera": by_cam,
        "by_subject": by_subject,
    }
