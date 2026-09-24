# [REVIEW] new file (BACKLOG B29)
"""UI-PRMD loader (Vicon marker positions, 100 Hz, 39 Plug-in-Gait markers, mm) and a sit-to-stand
check: every segmented episode of movement m05 is exactly one sit-to-stand, correct or incorrect.

Marker order from Table 2 of Vakanski et al., Data 2018;3(1):2 (standard Vicon full-body PiG order).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ptvision.datasets.c3d import C3DTrial
from ptvision.datasets.mocap_projection import ProjectedTrial, project_trial
from ptvision.datasets.registry import DatasetDirs, load_manifest

VICON_39 = (
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
    "C7",
    "T10",
    "CLAV",
    "STRN",
    "RBAK",
    "LSHO",
    "LUPA",
    "LELB",
    "LFRM",
    "LWRA",
    "LWRB",
    "LFIN",
    "RSHO",
    "RUPA",
    "RELB",
    "RFRM",
    "RWRA",
    "RWRB",
    "RFIN",
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LTHI",
    "LKNE",
    "LTIB",
    "LANK",
    "LHEE",
    "LTOE",
    "RTHI",
    "RKNE",
    "RTIB",
    "RANK",
    "RHEE",
    "RTOE",
)
VICON_RATE = 100.0
MOVEMENTS = {
    "m01": "deep squat",
    "m02": "hurdle step",
    "m03": "inline lunge",
    "m04": "side lunge",
    "m05": "sit to stand",
    "m06": "standing active straight leg raise",
    "m07": "standing shoulder abduction",
    "m08": "standing shoulder extension",
    "m09": "standing shoulder internal-external rotation",
    "m10": "standing shoulder scaption",
}


def read_vicon_positions(path: Path) -> C3DTrial:
    """A UI-PRMD Vicon positions .txt as a C3DTrial (metres, no events)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 3 * len(VICON_39):
        raise ValueError(f"{path.name}: expected {3 * len(VICON_39)} columns, got {data.shape[1]}")
    pts = data.reshape(data.shape[0], len(VICON_39), 3) / 1000.0
    pts[np.all(pts == 0, axis=2)] = np.nan
    return C3DTrial(path, VICON_RATE, list(VICON_39), pts, [], 0, "m")


@dataclass
class EpisodeResult:
    file: str
    movement: str
    subject: str
    episode: str
    correct: bool
    frames: int
    reps: int
    rise_time_s: float | None
    note: str


def episodes(kind: str = "m05") -> list[tuple[Path, bool]]:
    spec = load_manifest()["uiprmd"]
    raw = DatasetDirs.for_spec(spec).raw
    out: list[tuple[Path, bool]] = []
    for p in sorted(raw.rglob(f"{kind}_s*_e*_positions*.txt")):
        if "Vicon" not in str(p):
            continue
        out.append((p, "_inc" not in p.stem))
    return out


def evaluate_sts(
    *, fps: float = 30.0, limit: int | None = None
) -> tuple[list[EpisodeResult], Path]:
    """Run the sit-to-stand segmenter on every m05 episode; each should yield exactly one rise."""
    from ptvision.clinical.segmenters.sts import StsParams, segment_sts
    from ptvision.kinematics.preprocess import preprocess

    spec = load_manifest()["uiprmd"]
    out_dir = DatasetDirs.for_spec(spec).derived / "eval_sts"
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = episodes("m05")
    if limit:
        eps = eps[:limit]
    results: list[EpisodeResult] = []
    for path, correct in eps:
        parts = path.stem.split("_")
        try:
            trial = read_vicon_positions(path)
            pt: ProjectedTrial = project_trial(trial, fps=fps)
            pre = preprocess(pt.track)
            ev = segment_sts(
                pre.filtered, fps, StsParams(n_reps_expected=1, end_rule="fifth_stand")
            )
            rise = (ev.reps[0].stand_reached - ev.reps[0].seat_off) / fps if ev.reps else None
            results.append(
                EpisodeResult(
                    path.name,
                    parts[0],
                    parts[1],
                    parts[2],
                    correct,
                    pt.n_frames,
                    ev.n_reps,
                    rise,
                    "; ".join(ev.warnings[:2]),
                )
            )
        except Exception as e:
            results.append(
                EpisodeResult(
                    path.name, parts[0], parts[1], parts[2], correct, 0, -1, None, f"error: {e}"
                )
            )
    with (out_dir / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(EpisodeResult.__dataclass_fields__))
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)
    ok = [r for r in results if r.reps == 1]
    summary = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "episodes": len(results),
        "exactly_one_rep": len(ok),
        "zero_reps": sum(1 for r in results if r.reps == 0),
        "multi_reps": sum(1 for r in results if r.reps > 1),
        "errors": sum(1 for r in results if r.reps < 0),
        "correct_episodes_one_rep": sum(1 for r in results if r.correct and r.reps == 1),
        "incorrect_episodes_one_rep": sum(1 for r in results if not r.correct and r.reps == 1),
        "mean_rise_time_correct_s": round(
            float(np.mean([r.rise_time_s for r in ok if r.correct and r.rise_time_s])), 3
        )
        if any(r.correct and r.rise_time_s for r in ok)
        else None,
        "mean_rise_time_incorrect_s": round(
            float(np.mean([r.rise_time_s for r in ok if not r.correct and r.rise_time_s])), 3
        )
        if any((not r.correct) and r.rise_time_s for r in ok)
        else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results, out_dir
