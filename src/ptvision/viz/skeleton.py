"""Named body segments mapped onto Halpe-26 edges, and the one drawing loop both renderers use."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ptvision.pose.layout import KeypointLayout

# "{S}" is replaced by "L"/"R". Midline segments have no placeholder.
SEGMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "trunk": (
        ("Hip", "Neck"),
        ("Hip", "LHip"),
        ("Hip", "RHip"),
        ("Neck", "LShoulder"),
        ("Neck", "RShoulder"),
    ),
    "head": (
        ("Neck", "Head"),
        ("Head", "Nose"),
        ("Nose", "LEye"),
        ("Nose", "REye"),
        ("LEye", "LEar"),
        ("REye", "REar"),
    ),
    "thigh": (("{S}Hip", "{S}Knee"),),
    "shank": (("{S}Knee", "{S}Ankle"),),
    "foot": (("{S}Ankle", "{S}Heel"), ("{S}Ankle", "{S}BigToe"), ("{S}BigToe", "{S}SmallToe")),
    "upper_arm": (("{S}Shoulder", "{S}Elbow"),),
    "forearm": (("{S}Elbow", "{S}Wrist"),),
}
SEGMENT_NAMES: tuple[str, ...] = tuple(SEGMENTS)


def is_sided(segment: str) -> bool:
    return any("{S}" in a or "{S}" in b for a, b in SEGMENTS[segment])


def edge_lookup(layout: KeypointLayout) -> dict[frozenset[int], int]:
    return {frozenset(e): i for i, e in enumerate(layout.edges())}


def segment_edges(
    layout: KeypointLayout, segment: str, side: Literal["left", "right"] | None
) -> list[int]:
    """Edge indices (into layout.edges()) for a segment.

    Sided segments need a side; with side=None both sides are returned."""
    if segment not in SEGMENTS:
        raise KeyError(f"unknown segment {segment!r}; known: {SEGMENT_NAMES}")
    lookup = edge_lookup(layout)
    prefixes = ["L", "R"] if side is None else ["L" if side == "left" else "R"]
    out: list[int] = []
    for a, b in SEGMENTS[segment]:
        sides = prefixes if ("{S}" in a or "{S}" in b) else [""]
        for p in sides:
            na, nb = a.replace("{S}", p), b.replace("{S}", p)
            key = frozenset((layout.index(na), layout.index(nb)))
            if key not in lookup:
                raise KeyError(f"segment {segment!r}: {na}-{nb} is not an edge of {layout.name}")
            out.append(lookup[key])
    return out


def bone_segments(
    coords: np.ndarray,
    edges: Sequence[tuple[int, int]],
    colors: np.ndarray,
    visible: np.ndarray,
) -> list[tuple[float, float, float, float, tuple[int, int, int]]]:
    """(K, 2) coords -> drawable (xa, ya, xb, yb, (r, g, b)); drops NaN endpoints and hidden
    bones."""
    out = []
    for i, (a, b) in enumerate(edges):
        if not visible[i]:
            continue
        pa, pb = coords[a], coords[b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        r, g, bl = (int(v) for v in colors[i])
        out.append((float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1]), (r, g, bl)))
    return out
