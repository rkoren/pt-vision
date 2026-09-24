# [REVIEW] new file (BACKLOG B29)
"""Score the Zeni gait-event detector against labelled mocap datasets.

For each trial: c3d -> virtual side camera PoseTrack -> `segment_gait` -> match every labelled heel
strike / toe-off to the nearest detection of the same side and kind. Reports the fraction within
1 and 2 frames (Zeni 2008 reported 94 % within one frame for healthy treadmill walking), mean and
median absolute error, misses (no detection within 0.25 s) and extras (detections with no label).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ptvision.clinical.segmenters.gait_zeni import GaitParams, segment_gait_track
from ptvision.datasets.c3d import read_c3d
from ptvision.datasets.mocap_projection import ProjectedTrial, project_trial
from ptvision.datasets.registry import DatasetDirs, load_manifest

ProgressFn = Callable[[int, int | None], None]
MATCH_WINDOW_S = 0.25


@dataclass
class TrialScore:
    trial: str
    group: str
    n_frames: int
    n_truth: int
    n_detected: int
    matched: int
    missed: int
    extra: int
    boundary_extra: int = 0
    errors_frames: list[int] = field(default_factory=list)
    per_kind: dict[str, list[int]] = field(default_factory=dict)
    note: str = ""
    side_swapped: bool = False
    events_out: list[dict[str, object]] = field(default_factory=list)

    def row(self) -> dict[str, object]:
        e = np.abs(np.asarray(self.errors_frames)) if self.errors_frames else np.array([])
        return {
            "trial": self.trial,
            "group": self.group,
            "frames": self.n_frames,
            "truth_events": self.n_truth,
            "detected": self.n_detected,
            "matched": self.matched,
            "missed": self.missed,
            "extra": self.extra,
            "boundary_extra": self.boundary_extra,
            "side_swapped": self.side_swapped,
            "within_1": int((e <= 1).sum()),
            "within_2": int((e <= 2).sum()),
            "mae_frames": round(float(e.mean()), 2) if e.size else "",
            "note": self.note,
        }


def _match(
    truth: Sequence[tuple[str, str, int]], det: Sequence[tuple[str, str, int]], win: int
) -> tuple[list[tuple[int, int | None, int | None]], set[int]]:
    """Greedy nearest matching of truth events to detections of the same side and kind.

    Returns per-truth (truth_frame, detection_index|None, signed_error|None) and the used indices.
    """
    used: set[int] = set()
    out: list[tuple[int, int | None, int | None]] = []
    for side, kind, f in truth:
        best_i, best_d = None, None
        for i, (s, k, g) in enumerate(det):
            if i in used or s != side or k != kind:
                continue
            d = g - f
            if abs(d) <= win and (best_d is None or abs(d) < abs(best_d)):
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
        out.append((f, best_i, best_d))
    return out, used


def score_trial(pt: ProjectedTrial, params: GaitParams) -> TrialScore:
    ev = segment_gait_track(
        pt.track,
        pt.fps,
        params,
        image_width=pt.track.image_size[0] if pt.track.image_size else None,
    )
    det: list[tuple[str, str, int]] = [(e.side, e.kind, e.frame) for e in ev.events]
    win = round(MATCH_WINDOW_S * pt.fps)
    # Some datasets label a trial's sides the other way round (marker or context swap). If matching
    # with swapped sides recovers most events where the direct match recovers few, score the swapped
    # match and flag the trial.
    direct, _ = _match(pt.events, det, win)
    swapped_truth = [("right" if s == "left" else "left", k, f) for s, k, f in pt.events]
    swapped, _ = _match(swapped_truth, det, win)
    side_swapped = False
    if pt.events:
        hit_d = sum(1 for _, i, _ in direct if i is not None) / len(pt.events)
        hit_s = sum(1 for _, i, _ in swapped if i is not None) / len(pt.events)
        if hit_s >= 0.8 and hit_d <= 0.2:
            side_swapped = True
            pt = ProjectedTrial(pt.track, pt.scale, pt.fps, pt.lab, swapped_truth, pt.source)
    used: set[int] = set()
    errors: list[int] = []
    per_kind: dict[str, list[int]] = {"hs": [], "to": []}
    missed = 0
    events_out: list[dict[str, object]] = []
    for side, kind, f in pt.events:
        best_i, best_d = None, None
        for i, (s, k, g) in enumerate(det):
            if i in used or s != side or k != kind:
                continue
            d = g - f
            if abs(d) <= win and (best_d is None or abs(d) < abs(best_d)):
                best_i, best_d = i, d
        if best_i is None or best_d is None:
            missed += 1
            events_out.append(
                {
                    "side": side,
                    "kind": kind,
                    "truth": f,
                    "detected": "",
                    "error": "",
                    "pos": round(f / max(pt.n_frames - 1, 1), 2),
                }
            )
        else:
            used.add(best_i)
            errors.append(int(best_d))
            per_kind[kind].append(int(best_d))
            events_out.append(
                {
                    "side": side,
                    "kind": kind,
                    "truth": f,
                    "detected": det[best_i][2],
                    "error": int(best_d),
                    "pos": round(f / max(pt.n_frames - 1, 1), 2),
                }
            )
    # unmatched detections sitting within the match window of a clip boundary are events the dataset
    # could not label (the true strike lies just outside the recording); count them apart
    n = pt.n_frames
    unmatched = [i for i in range(len(det)) if i not in used]
    boundary = sum(1 for i in unmatched if det[i][2] <= win or det[i][2] >= n - 1 - win)
    extra = len(unmatched) - boundary
    return TrialScore(
        trial=pt.source.path.stem,
        group="",
        n_frames=pt.n_frames,
        n_truth=len(pt.events),
        n_detected=len(det),
        matched=len(errors),
        missed=missed,
        extra=extra,
        boundary_extra=boundary,
        errors_frames=errors,
        per_kind=per_kind,
        note=("side labels swapped in dataset; " if side_swapped else "")
        + "; ".join(ev.warnings[:2]),
        side_swapped=side_swapped,
        events_out=events_out,
    )


def _trials_for(name: str) -> list[tuple[Path, str]]:
    """(c3d path, group label) for a dataset; group = speed/condition or population."""
    spec = load_manifest()[name]
    raw = DatasetDirs.for_spec(spec).raw
    out: list[tuple[Path, str]] = []
    if name == "fukuchi2018":
        for p in sorted(raw.rglob("*.c3d")):
            stem = p.stem  # WBDS01walkO01C
            if "walk" not in stem or "static" in stem.lower():
                continue
            mode = "overground" if "walkO" in stem else "treadmill"
            out.append((p, f"{mode}/{stem[-1]}"))
    elif name == "schreiber2019":
        for p in sorted(raw.rglob("*.c3d")):
            if "_ST" in p.stem:
                continue
            cond = p.stem.split("_")[1] if "_" in p.stem else "?"
            out.append((p, cond))
    elif name == "vancriekinge2023":
        for p in sorted(raw.rglob("*.c3d")):
            pop = "stroke" if "Stroke" in str(p) else "healthy"
            out.append((p, pop))
    else:
        raise KeyError(f"no gait trials defined for dataset {name!r}")
    return out


@dataclass
class DatasetSummary:
    name: str
    out_dir: Path
    scores: list[TrialScore]
    params: GaitParams
    fps: float

    def aggregate(self, subset: list[TrialScore] | None = None) -> dict[str, object]:
        sc = subset if subset is not None else self.scores
        errs = np.abs(
            np.concatenate([np.asarray(s.errors_frames) for s in sc if s.errors_frames])
            if any(s.errors_frames for s in sc)
            else np.array([])
        )
        truth = sum(s.n_truth for s in sc)
        return {
            "trials": len(sc),
            "truth_events": truth,
            "matched": int(sum(s.matched for s in sc)),
            "missed": int(sum(s.missed for s in sc)),
            "extra": int(sum(s.extra for s in sc)),
            "boundary_extra": int(sum(s.boundary_extra for s in sc)),
            "side_swapped_trials": int(sum(1 for s in sc if s.side_swapped)),
            "within_1_frame": round(float((errs <= 1).mean()), 3) if errs.size else None,
            "within_2_frames": round(float((errs <= 2).mean()), 3) if errs.size else None,
            "mae_frames": round(float(errs.mean()), 2) if errs.size else None,
            "median_frames": float(np.median(errs)) if errs.size else None,
            "hs_bias_frames": round(
                float(np.mean([e for s in sc for e in s.per_kind.get("hs", [])])), 2
            )
            if any(s.per_kind.get("hs") for s in sc)
            else None,
            "to_bias_frames": round(
                float(np.mean([e for s in sc for e in s.per_kind.get("to", [])])), 2
            )
            if any(s.per_kind.get("to") for s in sc)
            else None,
        }

    def by_group(self) -> dict[str, dict[str, object]]:
        groups = sorted({s.group for s in self.scores})
        return {g: self.aggregate([s for s in self.scores if s.group == g]) for g in groups}

    def render(self) -> str:
        a = self.aggregate()
        lines = [
            f"{self.name}: {a['trials']} trials, {a['truth_events']} labelled events at "
            f"{self.fps:g} fps, method={self.params.method}",
            f"  matched {a['matched']}  missed {a['missed']}  extra {a['extra']} "
            f"(+{a['boundary_extra']} at clip edges, unlabelled); "
            f"side-swapped trials {a['side_swapped_trials']}",
            f"  within 1 frame {a['within_1_frame']}  within 2 frames {a['within_2_frames']}  "
            f"MAE {a['mae_frames']} frames",
            f"  bias: heel strike {a['hs_bias_frames']} frames, toe-off {a['to_bias_frames']} "
            "frames (+ = detected late)",
        ]
        for g, ag in self.by_group().items():
            lines.append(
                f"  [{g}] trials {ag['trials']}  within1 {ag['within_1_frame']}  "
                f"within2 {ag['within_2_frames']}  MAE {ag['mae_frames']}  "
                f"missed {ag['missed']}  extra {ag['extra']}"
            )
        return "\n".join(lines)


def evaluate_dataset(
    name: str,
    *,
    limit: int | None = None,
    fps: float = 30.0,
    method: str = "coordinate",
    progress: ProgressFn | None = None,
) -> DatasetSummary:
    spec = load_manifest()[name]
    dirs = DatasetDirs.for_spec(spec)
    trials = _trials_for(name)
    if limit:
        trials = trials[:limit]
    params = GaitParams(
        method=method, min_cycles=1, drop_edge_cycles=0, edge_margin_frac=0.0, bout_speed_frac=0.0
    )
    scores: list[TrialScore] = []
    for i, (path, group) in enumerate(trials):
        try:
            pt = project_trial(read_c3d(path), fps=fps)
            if not pt.events:
                continue
            s = score_trial(pt, params)
            s.group = group
            scores.append(s)
        except Exception as e:
            scores.append(TrialScore(path.stem, group, 0, 0, 0, 0, 0, 0, note=f"error: {e}"))
        if progress:
            progress(i + 1, len(trials))
    out_dir = dirs.derived / "eval_gait" / f"{method}_{int(fps)}fps"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = DatasetSummary(name, out_dir, scores, params, fps)
    with (out_dir / "trials.csv").open("w", newline="") as f:
        rows = [s.row() for s in scores]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["trial"])
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "events.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["trial", "group", "side", "kind", "truth", "detected", "error", "pos"]
        )
        w.writeheader()
        for sc in scores:
            for row in sc.events_out:
                w.writerow({"trial": sc.trial, "group": sc.group, **row})
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": name,
                "license": spec.license,
                "citation": spec.citation,
                "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "fps": fps,
                "params": params.to_dict(),
                "overall": summary.aggregate(),
                "by_group": summary.by_group(),
            },
            indent=2,
        )
    )
    return summary
