"""Keypoint layouts and remapping between them.

Halpe-26 is the canonical internal layout: COCO-17 body plus head, neck, pelvis ("Hip") and
six foot points (big toe, small toe, heel per side). The heel and toe points are what gait
event detection needs, which is why COCO-17-only models are not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptvision._vendor.pose2sim.skeletons import HALPE_26_TREE


@dataclass(frozen=True)
class KeypointLayout:
    name: str
    names: tuple[str, ...]
    parents: tuple[int, ...]  # parent index per keypoint, -1 for root / unconnected
    left_right_pairs: tuple[tuple[int, int], ...]

    @property
    def n(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as e:
            raise KeyError(f"{name!r} not in layout {self.name}") from e

    def indices(self, *names: str) -> list[int]:
        return [self.index(n) for n in names]

    def has(self, name: str) -> bool:
        return name in self.names

    def edges(self) -> list[tuple[int, int]]:
        return [(p, i) for i, p in enumerate(self.parents) if p >= 0]

    def mirror(self, name: str) -> str:
        """Return the contralateral keypoint name (or the same name for midline points)."""
        for a, b in self.left_right_pairs:
            if self.names[a] == name:
                return self.names[b]
            if self.names[b] == name:
                return self.names[a]
        return name


def _halpe26() -> KeypointLayout:
    # Index order as emitted by RTMPose Halpe-26 checkpoints (MMPose halpe26 metainfo).
    names = (
        "Nose",
        "LEye",
        "REye",
        "LEar",
        "REar",
        "LShoulder",
        "RShoulder",
        "LElbow",
        "RElbow",
        "LWrist",
        "RWrist",
        "LHip",
        "RHip",
        "LKnee",
        "RKnee",
        "LAnkle",
        "RAnkle",
        "Head",
        "Neck",
        "Hip",
        "LBigToe",
        "RBigToe",
        "LSmallToe",
        "RSmallToe",
        "LHeel",
        "RHeel",
    )
    parents = [-1] * len(names)
    idx = {n: i for i, n in enumerate(names)}
    for name, kp_id, parent in HALPE_26_TREE:
        assert idx[name] == kp_id, f"Halpe-26 index mismatch for {name}: {idx[name]} vs {kp_id}"
        if parent is not None:
            parents[kp_id] = idx[parent]
    # Eyes and ears are not in Pose2Sim's kinematic tree; connect them COCO-style for drawing.
    parents[idx["LEye"]] = idx["Nose"]
    parents[idx["REye"]] = idx["Nose"]
    parents[idx["LEar"]] = idx["LEye"]
    parents[idx["REar"]] = idx["REye"]
    pairs = tuple(
        (idx[f"L{base}"], idx[f"R{base}"])
        for base in (
            "Eye",
            "Ear",
            "Shoulder",
            "Elbow",
            "Wrist",
            "Hip",
            "Knee",
            "Ankle",
            "BigToe",
            "SmallToe",
            "Heel",
        )
    )
    return KeypointLayout("halpe26", names, tuple(parents), pairs)


HALPE26 = _halpe26()

LAYOUTS: dict[str, KeypointLayout] = {HALPE26.name: HALPE26}

# Midline points that can be synthesised as the midpoint of two others when a source layout
# lacks them.
_DERIVED_MIDPOINTS: dict[str, tuple[str, str]] = {
    "Hip": ("LHip", "RHip"),
    "Neck": ("LShoulder", "RShoulder"),
}


def remap(
    coords: np.ndarray,
    score: np.ndarray,
    src: KeypointLayout,
    dst: KeypointLayout,
    *,
    derive_midpoints: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder keypoints from `src` layout to `dst` layout by name.

    Works on any leading shape: coords (..., K_src, D), score (..., K_src). Missing keypoints
    become NaN with score 0. Midline points absent from the source (pelvis, neck) are derived
    as midpoints when both parents exist.
    """
    if coords.shape[-2] != src.n or score.shape[-1] != src.n:
        raise ValueError(f"coords/score do not match layout {src.name} ({src.n} keypoints)")
    lead = coords.shape[:-2]
    out_c = np.full((*lead, dst.n, coords.shape[-1]), np.nan, dtype=coords.dtype)
    out_s = np.zeros((*lead, dst.n), dtype=score.dtype)
    for j, name in enumerate(dst.names):
        if src.has(name):
            i = src.index(name)
            out_c[..., j, :] = coords[..., i, :]
            out_s[..., j] = score[..., i]
        elif derive_midpoints and name in _DERIVED_MIDPOINTS:
            a, b = _DERIVED_MIDPOINTS[name]
            if src.has(a) and src.has(b):
                ia, ib = src.index(a), src.index(b)
                out_c[..., j, :] = (coords[..., ia, :] + coords[..., ib, :]) / 2
                out_s[..., j] = np.minimum(score[..., ia], score[..., ib])
    return out_c, out_s
