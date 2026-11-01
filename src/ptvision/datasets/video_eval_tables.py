"""Render the video-vs-mocap results as tables"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from statistics import mean, median, stdev

from ptvision.datasets.registry import DatasetDirs, load_manifest

ANGLES = ("knee_flexion", "hip_flexion", "trunk_lean")


def _f(x: object, nd: int = 1, suffix: str = "") -> str:
    if x is None or x == "":
        return "–"
    try:
        return f"{float(str(x)):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return str(x)


def _med(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return median(vals) if vals else None


def _header(cols: list[str]) -> list[str]:
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


def _by_camera(rows: list[dict[str, str]]) -> Iterator[tuple[int, list[dict[str, str]]]]:
    """(camera, its scored rows) for every camera with at least one scored row."""
    for cam in sorted({int(r["camera"]) for r in rows}):
        rs = [r for r in rows if int(r["camera"]) == cam and r["status"] == "ok"]
        if rs:
            yield cam, rs


def camera_table(rows: list[dict[str, str]]) -> str:
    out = _header(
        [
            "camera",
            "view",
            "n",
            "knee RMSE as synced",
            "knee RMSE lag-corrected",
            "hip RMSE lag-corr.",
            "trunk RMSE lag-corr.",
            "knee bias",
            "knee ROM |err|",
            "seat-off err (lag-corr.)",
            "span err",
            "view check",
        ]
    )
    for cam, rs in _by_camera(rows):
        passes = sum(1 for r in rs if r["camera_view_status"] == "pass")
        cells = [
            f"cam{cam}",
            _f(rs[0]["view_deg"], 0, "°"),
            str(len(rs)),
            _f(_med(rs, "knee_flexion_mocap_rmse")),
            _f(_med(rs, "knee_flexion_synced_rmse")),
            _f(_med(rs, "hip_flexion_synced_rmse")),
            _f(_med(rs, "trunk_lean_synced_rmse")),
            _f(_med(rs, "knee_flexion_synced_bias"), 1, "°"),
            _f(_med(rs, "knee_flexion_rom_err_abs_mean"), 1, "°"),
            _f(_med(rs, "seat_off_err_synced_ms_mean"), 0, " ms"),
            _f(_med(rs, "span_err_s"), 2, " s"),
            f"{passes}/{len(rs)} pass",
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def side_table(rows: list[dict[str, str]]) -> str:
    """Per camera: how often the leg with the higher keypoint confidence is not the leg facing the
    camera, and how each leg's lag-corrected knee and hip angles agree with the lab's same leg"""
    out = _header(
        [
            "camera",
            "view",
            "n",
            "higher-confidence leg is the far one",
            "knee RMSE near leg",
            "knee RMSE far leg",
            "knee slope near leg",
            "knee slope far leg",
            "hip RMSE near leg",
            "hip RMSE far leg",
        ]
    )
    for cam, rs in _by_camera(rows):
        far = sum(1 for r in rs if r.get("side_by_confidence") not in (None, "", r["side"]))
        cells = [
            f"cam{cam}",
            _f(rs[0]["view_deg"], 0, "°"),
            str(len(rs)),
            f"{far}/{len(rs)}",
            _f(_med(rs, "knee_flexion_synced_rmse")),
            _f(_med(rs, "knee_flexion_far_rmse")),
            _f(_med(rs, "knee_flexion_synced_slope"), 2),
            _f(_med(rs, "knee_flexion_far_slope"), 2),
            _f(_med(rs, "hip_flexion_synced_rmse")),
            _f(_med(rs, "hip_flexion_far_rmse")),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def subject_table(rows: list[dict[str, str]], cameras: tuple[int, ...]) -> str:
    out = _header(
        [
            "subject",
            "trial-cameras",
            "knee RMSE (lag-corr.)",
            "hip RMSE",
            "trunk RMSE",
            "lag (s)",
            "reps video / force",
            "seat-off err",
            "span err",
        ]
    )
    for subj in sorted({r["subject"] for r in rows}):
        rs = [
            r
            for r in rows
            if r["subject"] == subj and r["status"] == "ok" and int(r["camera"]) in cameras
        ]
        if not rs:
            continue
        cells = [
            subj,
            str(len(rs)),
            _f(_med(rs, "knee_flexion_synced_rmse")),
            _f(_med(rs, "hip_flexion_synced_rmse")),
            _f(_med(rs, "trunk_lean_synced_rmse")),
            _f(_med(rs, "lag_knee_s"), 2),
            f"{_f(_med(rs, 'reps_video'), 0)} / {_f(_med(rs, 'reps_force'), 0)}",
            _f(_med(rs, "seat_off_err_synced_ms_mean"), 0, " ms"),
            _f(_med(rs, "span_err_s"), 2, " s"),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


SEGMENTS = ("shank_vertical", "thigh_vertical", "trunk_vertical", "trunk_shoulders_vertical")


def segment_table(rows: list[dict[str, str]], cameras: tuple[int, ...]) -> str:
    """Per-segment inclination error (video − lab, degrees from vertical) for the given cameras:
    where a joint-angle error comes from."""
    out = _header(["segment", "bias", "SD", "RMSE"])
    rs = [r for r in rows if r["status"] == "ok" and int(r["camera"]) in cameras]
    for seg in (*SEGMENTS, "knee_flexion", "hip_flexion"):
        if f"{seg}_synced_bias" not in rows[0]:
            continue
        cells = [
            seg.replace("_", " "),
            _f(_med(rs, f"{seg}_synced_bias"), 1, "°"),
            _f(_med(rs, f"{seg}_synced_sd"), 1, "°"),
            _f(_med(rs, f"{seg}_synced_rmse"), 1, "°"),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


CORRECTIONS = (
    ("synced", "as filmed"),
    ("rect_shoulders", "shoulder stretch"),
    ("rect_level", "perspective, level camera"),
    ("rect_calibrated", "perspective, lab calibration"),
)


def geometry_table(rows: list[dict[str, str]]) -> str:
    """Per camera: its angle off the side as placed (nominal), as calibrated by the lab, and as
    the shoulder estimate reads it from the video; its tilt; and whether the shoulders tell the
    camera's side (front or back of the person) the way the calibration does."""
    cols = ["camera", "view", "n", "cos θ nominal", "cos θ lab calibration", "cos θ shoulders"]
    out = _header([*cols, "tilt below level", "shoulders tell front from back"])
    for cam, rs in _by_camera(rows):
        known = [r for r in rs if r.get("front_shoulders") not in (None, "", "0")]
        agree = sum(1 for r in known if r["front_shoulders"] == r.get("front_calibrated"))
        cells = [
            f"cam{cam}",
            _f(rs[0]["view_deg"], 0, "°"),
            str(len(rs)),
            _f(_med(rs, "cos_theta_nominal"), 2),
            _f(_med(rs, "cos_theta_calibrated"), 2),
            _f(_med(rs, "cos_theta_shoulders"), 2),
            _f(_med(rs, "pitch_deg_calibrated"), 1, "°"),
            f"{agree}/{len(known)}",
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def correction_table(rows: list[dict[str, str]]) -> str:
    """Per camera and view correction: the lag-corrected knee / hip / trunk RMSE, over the
    trial-cameras the correction applies to."""
    out = _header(["camera", "view", "correction", "n", "knee RMSE", "hip RMSE", "trunk RMSE"])
    for cam, rs in _by_camera(rows):
        for tag, label in CORRECTIONS:
            done = [r for r in rs if r.get(f"knee_flexion_{tag}_rmse") not in (None, "")]
            if not done:
                continue
            cells = [f"cam{cam}", _f(rs[0]["view_deg"], 0, "°"), label, str(len(done))]
            cells += [_f(_med(done, f"{a}_{tag}_rmse")) for a in ANGLES]
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


EVENT_METRICS = ("knee_flexion_at_seat_off", "trunk_lean_at_seat_off", "peak_trunk_lean_rising")


def event_table(rows: list[dict[str, str]], cameras: tuple[int, ...]) -> str:
    """The reported angle metrics at the force-plate seat-offs, video − lab (lag-corrected),
    medians of the per-trial bias and RMSE over the given cameras. "as reported" is what the
    protocol outputs: the shoulder stretch where it applies, the angles as filmed elsewhere."""
    rs = [r for r in rows if r["status"] == "ok" and int(r["camera"]) in cameras]
    out = _header(["metric", "correction", "n", "bias", "RMSE"])
    for metric in EVENT_METRICS:
        for tag, label in (("shipped", "as reported"), *CORRECTIONS):
            done = [r for r in rs if r.get(f"{metric}_{tag}_rmse") not in (None, "")]
            if not done:
                continue
            cells = [metric.replace("_", " "), label, str(len(done))]
            cells += [
                _f(_med(done, f"{metric}_{tag}_bias"), 1, "°"),
                _f(_med(done, f"{metric}_{tag}_rmse"), 1, "°"),
            ]
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


KNEE_STEPS = (
    ("rect_shoulders", "shoulder stretch (as reported)"),
    ("rect_level", "perspective: camera angle from the shoulders, camera taken as level"),
    ("rect_true_yaw", "+ the lab's camera angle"),
    ("rect_true_tilt", "+ the lab's camera tilt"),
    ("rect_calibrated", "+ lens distortion removed (the lab calibration)"),
)


def knee_steps_table(rows: list[dict[str, str]], cameras: tuple[int, ...]) -> str:
    """The knee as the lab's calibration replaces the perspective model's inputs one at a time,
    medians over the given cameras: at the force-plate seat-offs, over the movement, and the
    slope against the lab. What each step removes is what that input costs."""
    rs = [r for r in rows if r["status"] == "ok" and int(r["camera"]) in cameras]
    out = _header(
        ["inputs", "n", "bias at seat-off", "RMSE at seat-off", "RMSE over the movement", "slope"]
    )
    for tag, label in KNEE_STEPS:
        done = [r for r in rs if r.get(f"knee_flexion_at_seat_off_{tag}_rmse") not in (None, "")]
        if not done:
            continue
        cells = [label, str(len(done))]
        cells += [
            _f(_med(done, f"knee_flexion_at_seat_off_{tag}_bias"), 1, "°"),
            _f(_med(done, f"knee_flexion_at_seat_off_{tag}_rmse"), 1, "°"),
            _f(_med(done, f"knee_flexion_{tag}_rmse"), 1, "°"),
            _f(_med(done, f"knee_flexion_{tag}_slope"), 2),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def offset_table(rows: list[dict[str, str]], cameras: tuple[int, ...]) -> str:
    """How steady the knee's error at seat-off is over the given cameras: its median, its spread
    over trial-cameras and over subjects' means, and the spread of the difference between a
    subject's first two trials from the same camera. A steady offset matters less for following
    one patient with one setup than its size suggests."""
    rs = [r for r in rows if r["status"] == "ok" and int(r["camera"]) in cameras]
    cols = ["correction", "n", "median bias", "SD over trial-cameras", "SD of subject means"]
    out = _header([*cols, "SD of a subject's trial-to-trial difference, same camera"])
    for tag, label in KNEE_STEPS[:2] + KNEE_STEPS[-1:]:
        key = f"knee_flexion_at_seat_off_{tag}_bias"
        done = [r for r in rs if r.get(key) not in (None, "")]
        if len(done) < 2:
            continue
        by_subject: dict[str, list[float]] = {}
        by_camera: dict[tuple[str, str], dict[str, float]] = {}
        for r in done:
            by_subject.setdefault(r["subject"], []).append(float(r[key]))
            by_camera.setdefault((r["subject"], r["camera"]), {})[r["task"]] = float(r[key])
        diffs = []
        for trials in by_camera.values():
            if len(trials) >= 2:
                first, second = sorted(trials)[:2]
                diffs.append(trials[first] - trials[second])
        means = [mean(v) for v in by_subject.values()]
        cells = [
            label,
            str(len(done)),
            _f(_med(done, key), 1, "°"),
            _f(stdev(float(r[key]) for r in done), 1, "°"),
            _f(stdev(means) if len(means) > 1 else None, 1, "°"),
            _f(stdev(diffs) if len(diffs) > 1 else None, 1, "°"),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def correction_subject_table(rows: list[dict[str, str]], cameras: tuple[int, ...]) -> str:
    """Per subject over the given cameras: hip and knee RMSE as filmed and after each correction,
    the view the decision rule looks at (no subject may get worse)."""
    cols = ["subject", "n"]
    for _tag, label in CORRECTIONS:
        cols += [f"knee {label}", f"hip {label}"]
    out = _header(cols)
    for subj in sorted({r["subject"] for r in rows}):
        rs = [
            r
            for r in rows
            if r["subject"] == subj and r["status"] == "ok" and int(r["camera"]) in cameras
        ]
        if not rs:
            continue
        cells = [subj, str(len(rs))]
        for tag, _label in CORRECTIONS:
            cells += [
                _f(_med(rs, f"knee_flexion_{tag}_rmse")),
                _f(_med(rs, f"hip_flexion_{tag}_rmse")),
            ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def slope_table(rows: list[dict[str, str]], cameras: tuple[int, ...]) -> str:
    """Per segment and angle over the given cameras: the slope of video against lab (1 = no
    scaling error) as filmed and after each correction."""
    cols = ["quantity"] + [f"slope {label}" for _tag, label in CORRECTIONS]
    out = _header(cols)
    rs = [r for r in rows if r["status"] == "ok" and int(r["camera"]) in cameras]
    for q in (*SEGMENTS, *ANGLES):
        cells = [q.replace("_", " ")]
        cells += [_f(_med(rs, f"{q}_{tag}_slope"), 2) for tag, _label in CORRECTIONS]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_rows(eval_dir: Path) -> list[dict[str, str]]:
    """The per-trial-camera rows of `trials.csv` under an `eval_video` output directory."""
    with (eval_dir / "trials.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def headline(rows: list[dict[str, str]], eval_dir: Path) -> str:
    """One line saying what was evaluated, when, with which pose model."""
    summary = json.loads((eval_dir / "summary.json").read_text())
    pose = summary.get("pose") or {}
    keys = ("pose_name", "mode", "device")
    model = ", ".join(str(pose[k]) for k in keys if isinstance(pose, dict) and k in pose)
    n_err = sum(1 for r in rows if r["status"] != "ok")
    return (
        f"{len(rows)} trial-cameras ({n_err} errors), evaluated {summary['evaluated_at'][:10]}, "
        f"ptvision {summary['ptvision_version']}, pose model {model or pose}. "
        "Medians over trials; RMSE in degrees over the sit-to-stand window."
    )


def main(names: list[str]) -> None:
    specs = load_manifest()
    for name in names:
        d = DatasetDirs.for_spec(specs[name]).derived / "eval_video"
        if not (d / "trials.csv").exists():
            print(f"## {name}\n\nno results under {d}\n")
            continue
        rows = load_rows(d)
        print(f"## {name}\n")
        print(headline(rows, d) + "\n")
        print(camera_table(rows) + "\n")
        print("Near leg against far leg:\n")
        print(side_table(rows) + "\n")
        side = tuple(
            int(c) for c in {r["camera"] for r in rows} if abs(float(_view(rows, c))) >= 60
        ) or tuple(int(c) for c in {r["camera"] for r in rows})
        cams = ", ".join(f"cam{c}" for c in sorted(side))
        print(f"Per subject over the most side-on cameras ({cams}):\n")
        print(subject_table(rows, side) + "\n")
        print(
            f"Segment inclinations over the same cameras ({cams}), medians of per-trial values:\n"
        )
        print(segment_table(rows, side) + "\n")
        print("Camera geometry:\n")
        print(geometry_table(rows) + "\n")
        print("View correction per camera:\n")
        print(correction_table(rows) + "\n")
        print(f"Reported angle metrics at the force-plate seat-offs ({cams}):\n")
        print(event_table(rows, side) + "\n")
        print(f"The knee, one calibrated input at a time ({cams}):\n")
        print(knee_steps_table(rows, side) + "\n")
        print(f"How steady the knee's offset at seat-off is ({cams}):\n")
        print(offset_table(rows, side) + "\n")
        print(f"View correction per subject ({cams}):\n")
        print(correction_subject_table(rows, side) + "\n")
        print(f"Slopes video vs lab ({cams}):\n")
        print(slope_table(rows, side) + "\n")


def _view(rows: list[dict[str, str]], cam: str) -> float:
    v = next(r["view_deg"] for r in rows if r["camera"] == cam)
    try:
        return float(v)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    main(sys.argv[1:] or ["opencap"])
