"""Loads a trial into the in-memory model the viewer widgets read"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ptvision.clinical.metrics.base import Metric, NormComparison
from ptvision.clinical.protocol import Protocol, load_protocol
from ptvision.files import read_json
from ptvision.kinematics.angles import AngleSeries, compute_angles, resolve_side
from ptvision.kinematics.preprocess import preprocess
from ptvision.pose.track import PoseTrack
from ptvision.pose.tracking import select_primary_person
from ptvision.quality.checks import QualityReport
from ptvision.trials.models import Capture, RunProvenance
from ptvision.trials.store import RunDir, TrialDir
from ptvision.viz.status import BoneStatus, Mode, Rule

PhaseKind = Literal["rise", "stand", "descent", "stance", "swing"]
DEFAULT_ANGLES = ["knee_flexion", "hip_flexion", "trunk_lean"]


@dataclass(frozen=True)
class EventMarker:
    frame: int
    label: str
    rep: int | None = None

    @property
    def text(self) -> str:
        rep = f"rep {self.rep + 1}  " if self.rep is not None else ""
        return f"{rep}{self.label}"


@dataclass(frozen=True)
class Phase:
    start: int
    end: int
    kind: PhaseKind


def events_from_dict(
    d: dict[str, Any],
) -> tuple[list[EventMarker], list[Phase], tuple[int, int] | None]:
    """Markers, phases, and the timed window from an `events.json` payload (STS today)."""
    markers: list[EventMarker] = []
    phases: list[Phase] = []
    window: tuple[int, int] | None = None
    if d.get("segmenter") == "sts_velocity_threshold":
        for r in d.get("reps", []):
            i = r.get("index")
            for key, label in (
                ("lean_onset", "lean onset"),
                ("seat_off", "seat-off"),
                ("stand_reached", "stand"),
                ("descent_start", "descent start"),
                ("seated_return", "seated"),
            ):
                if r.get(key) is not None:
                    markers.append(EventMarker(int(r[key]), label, i))
            phases.append(Phase(int(r["seat_off"]), int(r["stand_reached"]), "rise"))
            if r.get("descent_start") is not None and r.get("seated_return") is not None:
                phases.append(Phase(int(r["stand_reached"]), int(r["descent_start"]), "stand"))
                phases.append(Phase(int(r["descent_start"]), int(r["seated_return"]), "descent"))
        if d.get("test_start") is not None and d.get("test_end") is not None:
            window = (int(d["test_start"]), int(d["test_end"]))
    elif d.get("segmenter") == "gait_zeni":
        for e in d.get("events", []):
            label = ("heel strike" if e["kind"] == "hs" else "toe-off") + f" {e['side'][0].upper()}"
            markers.append(EventMarker(int(e["frame"]), label, None))
        # shade stance/swing of the side with the most steady cycles
        cycles = [c for c in d.get("cycles", []) if c.get("steady")]
        sides = {c["side"] for c in cycles}
        if sides:
            side = max(sides, key=lambda s: sum(1 for c in cycles if c["side"] == s))
            for c in cycles:
                if c["side"] == side:
                    phases.append(Phase(int(c["hs"]), int(c["to"]), "stance"))
                    phases.append(Phase(int(c["to"]), int(c["next_hs"]), "swing"))
        sw = d.get("steady_window")
        if sw:
            window = (int(sw[0]), int(sw[1]))
    markers.sort(key=lambda m: m.frame)
    return markers, phases, window


def _protocol_from_provenance(prov: RunProvenance | None) -> Protocol | None:
    if prov is None or not prov.protocol:
        return None
    src = str(prov.protocol.get("source", ""))
    pid = str(prov.protocol.get("id", ""))
    try:
        if src.startswith("builtin:"):
            return load_protocol(src.removeprefix("builtin:"))
        if src and Path(src).exists():
            return load_protocol(src)
        if pid:
            return load_protocol(pid)
    except (KeyError, FileNotFoundError):
        return None
    return None


@dataclass
class TrialSession:
    trial: TrialDir
    run: RunDir | None
    capture: Capture
    video_path: Path
    fps: float
    n_frames: int
    image_size: tuple[int, int]
    track: PoseTrack
    primary_slot: int
    primary_id: int
    angles: AngleSeries
    protocol: Protocol | None
    rules: list[Rule]
    status: BoneStatus
    events: list[EventMarker] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    test_window: tuple[int, int] | None = None
    metrics: list[Metric] = field(default_factory=list)
    norms: list[NormComparison] = field(default_factory=list)
    quality: QualityReport | None = None
    warnings: list[str] = field(default_factory=list)
    provenance: RunProvenance | None = None
    smoothed: np.ndarray | None = None  # (T, K, 2) primary, gap-filled + low-passed (viewer)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return self.status.edges

    @property
    def side(self) -> Literal["left", "right"]:
        return self.angles.side

    @property
    def title(self) -> str:
        name = self.protocol.protocol.name if self.protocol else "Pose only"
        return f"{name} · {self.trial.path.name}"

    def available_modes(self) -> list[Mode]:
        return self.status.available_modes()

    def default_mode(self) -> Mode:
        return "rules" if self.status.rules is not None else "confidence"

    @property
    def first_present_frame(self) -> int:
        """determines when clip actually starts"""
        if self.test_window is not None and 0 <= self.test_window[0] < self.n_frames:
            return int(self.test_window[0])
        sc = self.track.score[:, self.primary_slot]
        confident = (sc >= 0.3).mean(axis=1)
        whole = np.flatnonzero(confident >= 0.6)
        if whole.size:
            return int(whole[0])
        seen = np.flatnonzero(confident > 0)
        return int(seen[0]) if seen.size else 0

    def bone_colors(self, t: int, mode: Mode) -> tuple[np.ndarray, np.ndarray]:
        return self.status.colors(t, mode)

    def rule_values(self) -> dict[str, np.ndarray]:
        return self.angles.values()

    def angle_readout(self, t: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in self.angles.names:
            v = self.angles[name]
            if 0 <= t < len(v) and not np.isnan(v[t]):
                out[name] = float(v[t])
        return out

    def rep_at(self, t: int) -> tuple[int, int] | None:
        """current repetition for frame t, or None"""
        starts = sorted(m.frame for m in self.events if m.label == "seat-off" and m.rep is not None)
        if not starts:
            return None
        done = sum(1 for f in starts if f <= t)
        return (done, len(starts)) if done else None

    def default_angle_names(self) -> list[str]:
        wanted = self.protocol.report.angles if self.protocol else DEFAULT_ANGLES
        return [a for a in wanted if a in self.angles.names]

    def rules_for(self, angle: str) -> list[Rule]:
        return [r for r in self.rules if r.angle == angle]

    # ---- loading ----------------------------------------------------------------------
    @classmethod
    def load(cls, trial_dir: Path | str, run_id: str | None = None) -> TrialSession:
        trial = TrialDir(Path(trial_dir))
        if not trial.has_capture():
            raise FileNotFoundError(f"{trial.path} has no capture.json")
        capture = trial.read_capture()
        cam = capture.cameras[0]
        video = trial.path / cam.normalized_file
        if not trial.pose_path(cam.camera_id).exists():
            raise FileNotFoundError(f"{trial.path} has no pose data; run `ptv pose` first")
        track = PoseTrack.load(trial.pose_path(cam.camera_id))

        run = RunDir(trial.runs_dir / run_id) if run_id else trial.latest_run()
        prov: RunProvenance | None = None
        if run is not None and run.provenance_path.exists():
            prov = run.read_provenance()
        else:
            run = None

        if track.n_persons == 0:
            raise ValueError("no person was detected in this trial")
        pid: int | None = None
        if prov is not None:
            q = prov.quality or {}
            pp = q.get("primary_person")
            if isinstance(pp, int):
                pid = pp
        if pid is None or pid not in track.person_ids.tolist():
            pid = int(track.person_ids[select_primary_person(track.coords, track.score)])
        slot = int(np.flatnonzero(track.person_ids == pid)[0])
        single = track.select_person(pid)

        protocol = _protocol_from_provenance(prov)
        rules = protocol.rule_list() if protocol else []

        # [REVIEW]: the same gap-filled, low-passed track the angles use is
        # also what the viewer draws by default (raw keypoints jitter frame to frame and hidden
        # joints flicker); the panel's "smooth skeleton" box switches back to raw.
        pre = preprocess(single)
        smoothed = np.asarray(pre.filtered.coords[:, 0], dtype=np.float32)
        angles_path = run.path / "angles.parquet" if run is not None else None
        if angles_path is not None and angles_path.exists():
            angles = AngleSeries.load(
                angles_path, side_fallback=resolve_side(single, "near"), fps_fallback=cam.fps
            )
        else:
            angles = compute_angles(pre.filtered, DEFAULT_ANGLES, side="near")

        status = BoneStatus.build(
            single.score[:, 0],
            track.layout,
            values=angles.values(),
            rules=rules,
            side=angles.side,
        )

        events: list[EventMarker] = []
        phases: list[Phase] = []
        window: tuple[int, int] | None = None
        metrics: list[Metric] = []
        norms: list[NormComparison] = []
        quality: QualityReport | None = None
        warnings: list[str] = []
        if run is not None:
            ev_path = run.path / "events.json"
            if ev_path.exists():
                d = read_json(ev_path)
                events, phases, window = events_from_dict(d)
                warnings += list(d.get("warnings", []))
            m_path = run.path / "metrics.json"
            if m_path.exists():
                d = read_json(m_path)
                metrics = [Metric.model_validate(m) for m in d.get("metrics", [])]
                norms = [NormComparison.model_validate(n) for n in d.get("norms", [])]
            q_path = run.path / "quality.json"
            if q_path.exists():
                quality = QualityReport.model_validate(read_json(q_path))

        n_frames = min(track.n_frames, cam.n_frames or track.n_frames)
        return cls(
            trial=trial,
            run=run,
            capture=capture,
            video_path=video,
            fps=cam.fps,
            n_frames=n_frames,
            image_size=(cam.width, cam.height),
            track=track,
            smoothed=smoothed,
            primary_slot=slot,
            primary_id=pid,
            angles=angles,
            protocol=protocol,
            rules=rules,
            status=status,
            events=events,
            phases=phases,
            test_window=window,
            metrics=metrics,
            norms=norms,
            quality=quality,
            warnings=warnings,
            provenance=prov,
        )
