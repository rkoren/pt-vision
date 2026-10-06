"""Clinical joint and segment angles from 2D keypoints.

Conventions (documented in every report):
- knee_flexion: 0 deg with the shank aligned to the thigh (full extension), positive with flexion.
- hip_flexion: 0 deg with the thigh aligned to the trunk, positive with flexion.
- elbow_flexion: 0 deg with the forearm aligned to the arm.
- shoulder_flexion: angle between arm and trunk, 0 deg with the arm at the side, 180 deg overhead.
- trunk_lean: trunk (pelvis -> neck) angle from image vertical, positive when leaning toward the
  direction the person faces (forward). Facing is inferred from toe-heel direction.

Interior angles are unsigned, so they do not depend on which way the person faces. Side selection
("near") picks the side whose hip/knee/ankle have the higher mean confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ptvision.pose.track import PoseTrack

AngleKind = Literal["flexion3", "segment_vertical_signed"]
Side = Literal["left", "right", "near"]


@dataclass(frozen=True)
class AngleDef:
    name: str
    points: tuple[str, ...]  # keypoint names; "{S}" is replaced by "L"/"R"
    kind: AngleKind
    description: str
    segments: tuple[str, ...] = ()  # body segments this angle is about (see viz.skeleton.SEGMENTS)
    units: str = "deg"


ANGLE_DEFS: dict[str, AngleDef] = {
    d.name: d
    for d in (
        AngleDef(
            "knee_flexion",
            ("{S}Hip", "{S}Knee", "{S}Ankle"),
            "flexion3",
            "180 minus the interior hip-knee-ankle angle; 0 = full extension.",
            ("thigh", "shank"),
        ),
        AngleDef(
            "hip_flexion",
            ("Neck", "{S}Hip", "{S}Knee"),
            "flexion3",
            "180 minus the interior neck-hip-knee angle; 0 = thigh in line with trunk.",
            ("trunk", "thigh"),
        ),
        AngleDef(
            "elbow_flexion",
            ("{S}Shoulder", "{S}Elbow", "{S}Wrist"),
            "flexion3",
            "180 minus the interior shoulder-elbow-wrist angle; 0 = straight arm.",
            ("upper_arm", "forearm"),
        ),
        AngleDef(
            "shoulder_flexion",
            ("{S}Elbow", "{S}Shoulder", "{S}Hip"),
            "flexion3",
            "180 minus the interior elbow-shoulder-hip angle; 0 = arm at the side.",
            ("upper_arm", "trunk"),
        ),
        AngleDef(
            "trunk_lean",
            ("Hip", "Neck"),
            "segment_vertical_signed",
            "Pelvis-to-neck segment angle from vertical; positive = leaning forward "
            "(facing direction).",
            ("trunk",),
        ),
    )
}


def interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Unsigned angle at vertex b between rays b->a and b->c, degrees in [0, 180].

    Vectorised over the leading axis."""
    u = a - b
    v = c - b
    dot = np.sum(u * v, axis=-1)
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.clip(dot / (nu * nv), -1.0, 1.0)
    return np.asarray(np.degrees(np.arccos(cosang)), dtype=np.float64)


def segment_angle_from_vertical(base: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Signed angle of base->tip from image-up, positive toward +x (image right).

    Image y points down."""
    d = tip - base
    return np.asarray(np.degrees(np.arctan2(d[..., 0], -d[..., 1])), dtype=np.float64)


def facing_direction(track: PoseTrack, person: int = 0, min_score: float = 0.3) -> int:
    """+1 if the person faces image-right, -1 if image-left.

    Uses toe-heel direction first, nose-ear as a fallback, else +1."""
    lay = track.layout
    votes: list[float] = []
    for toe, heel in (("LBigToe", "LHeel"), ("RBigToe", "RHeel")):
        if lay.has(toe) and lay.has(heel):
            ok = (track.keypoint_score(toe, person) >= min_score) & (
                track.keypoint_score(heel, person) >= min_score
            )
            dx = track.keypoint(toe, person)[:, 0] - track.keypoint(heel, person)[:, 0]
            votes.extend(dx[ok & ~np.isnan(dx)].tolist())
    if not votes:
        for nose, ear in (("Nose", "LEar"), ("Nose", "REar")):
            if lay.has(nose) and lay.has(ear):
                ok = (track.keypoint_score(nose, person) >= min_score) & (
                    track.keypoint_score(ear, person) >= min_score
                )
                dx = track.keypoint(nose, person)[:, 0] - track.keypoint(ear, person)[:, 0]
                votes.extend(dx[ok & ~np.isnan(dx)].tolist())
    if not votes:
        return 1
    return 1 if float(np.median(votes)) >= 0 else -1


def resolve_side(track: PoseTrack, side: Side, person: int = 0) -> Literal["left", "right"]:
    if side in ("left", "right"):
        return side
    scores = {}
    for s in ("L", "R"):
        names = [f"{s}Hip", f"{s}Knee", f"{s}Ankle"]
        scores[s] = float(np.nanmean([track.keypoint_score(n, person).mean() for n in names]))
    return "left" if scores["L"] >= scores["R"] else "right"


@dataclass
class AngleSeries:
    frame: pd.DataFrame  # index = frame, one column per angle (degrees)
    side: Literal["left", "right"]
    facing: int
    definitions: dict[str, AngleDef]
    fps: float

    def __getitem__(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy(dtype=np.float64)

    @property
    def names(self) -> list[str]:
        return list(self.frame.columns)

    def values(self) -> dict[str, np.ndarray]:
        return {n: self[n] for n in self.names}

    def save(self, path: Path | str) -> Path:
        """Parquet with side/facing/fps in the file metadata (needed for rule coloring later)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(self.frame.reset_index(drop=True), preserve_index=False)
        md = {
            b"ptvision_angles_schema": b"1",
            b"side": self.side.encode(),
            b"facing": str(self.facing).encode(),
            b"fps": repr(float(self.fps)).encode(),
        }
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **md})
        pq.write_table(table, path, compression="zstd")
        return path

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        side_fallback: Literal["left", "right"] | None = None,
        fps_fallback: float | None = None,
    ) -> AngleSeries:
        """Load a saved series; files written before side metadata existed need the fallbacks."""
        table = pq.read_table(path)
        md = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
        df = table.to_pandas()
        df.index = pd.Index(np.arange(len(df)), name="frame")
        side = md.get("side") or side_fallback
        if side not in ("left", "right"):
            raise ValueError(f"{path}: no side metadata; pass side_fallback")
        fps = float(md["fps"]) if "fps" in md else fps_fallback
        if fps is None:
            raise ValueError(f"{path}: no fps metadata; pass fps_fallback")
        defs = {n: ANGLE_DEFS[n] for n in df.columns if n in ANGLE_DEFS}
        return cls(
            frame=df,
            side=side,
            facing=int(md.get("facing", "1")),
            definitions=defs,
            fps=fps,
        )


def compute_angles(
    track: PoseTrack,
    names: list[str] | None = None,
    *,
    side: Side = "near",
    person: int = 0,
) -> AngleSeries:
    """Compute the requested angles (default: all) for one person of a filtered track."""
    names = names or list(ANGLE_DEFS)
    resolved = resolve_side(track, side, person)
    prefix = "L" if resolved == "left" else "R"
    facing = facing_direction(track, person)
    cols: dict[str, np.ndarray] = {}
    used: dict[str, AngleDef] = {}
    for name in names:
        d = ANGLE_DEFS[name]
        pts = [track.keypoint(p.replace("{S}", prefix), person) for p in d.points]
        if d.kind == "flexion3":
            vals = 180.0 - interior_angle(pts[0], pts[1], pts[2])
        elif d.kind == "segment_vertical_signed":
            vals = facing * segment_angle_from_vertical(pts[0], pts[1])
        else:  # pragma: no cover
            raise ValueError(d.kind)
        cols[name] = vals.astype(np.float64)
        used[name] = d
    df = pd.DataFrame(cols, index=pd.Index(np.arange(track.n_frames), name="frame"))
    return AngleSeries(frame=df, side=resolved, facing=facing, definitions=used, fps=track.fps)
