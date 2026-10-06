"""Per-frame, per-bone status in [0, 1] and the single red-amber-green gradient that renders it.

Two modes:
- "confidence": bone value = min confidence of its two joints, mapped CONF_LO..CONF_HI -> 0..1.
- "rules":      bone value = min over protocol angle-band rules touching the bone
                (NaN = not assessed); low confidence additionally dims the color toward grey.
Bones whose confidence is below CONF_HIDE are not drawn at all in either mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ptvision.pose.layout import KeypointLayout
from ptvision.viz.skeleton import segment_edges

Mode = Literal["confidence", "rules"]
MODES: tuple[Mode, ...] = ("confidence", "rules")

CONF_HIDE = 0.30  # below: bone hidden (matches the overlay kpt_thr and preprocess score_threshold)
CONF_FULL = 0.60  # at or above: no dimming in rules mode
CONF_LO, CONF_HI = 0.30, 0.80  # confidence-mode gradient endpoints

RGB = tuple[int, int, int]
STOPS: tuple[tuple[float, RGB], ...] = (
    (0.0, (0xD6, 0x28, 0x28)),  # outside range / low confidence
    (0.5, (0xF2, 0xA9, 0x00)),  # near limit
    (1.0, (0x2B, 0xA8, 0x4A)),  # in range / high confidence
)
NEUTRAL: RGB = (0x8C, 0x9B, 0xAB)  # not assessed
DIM_TARGET: RGB = (0x5A, 0x5F, 0x66)  # low confidence dimming target (rules mode)
OTHER_PERSON: RGB = (0x9E, 0x9E, 0x9E)
JOINT_RULES_MODE: RGB = (0xFF, 0xFF, 0xFF)


def hex_of(rgb: RGB | np.ndarray) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def status_rgb(s: np.ndarray | float) -> np.ndarray:
    """Map status values (any shape) to (..., 3) uint8 RGB. NaN -> NEUTRAL."""
    x = np.asarray(s, dtype=np.float64)
    out = np.empty((*x.shape, 3), dtype=np.float64)
    nan = np.isnan(x)
    xc = np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)
    pos = np.array([p for p, _ in STOPS])
    cols = np.array([c for _, c in STOPS], dtype=np.float64)
    for ch in range(3):
        out[..., ch] = np.interp(xc, pos, cols[:, ch])
    out[nan] = np.array(NEUTRAL, dtype=np.float64)
    return np.rint(out).astype(np.uint8)


def dim_rgb(rgb: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Lerp colors toward DIM_TARGET as confidence falls from CONF_FULL to CONF_HIDE."""
    c = np.asarray(conf, dtype=np.float64)
    w = np.clip((CONF_FULL - c) / (CONF_FULL - CONF_HIDE), 0.0, 1.0)[..., None]
    target = np.array(DIM_TARGET, dtype=np.float64)
    mixed = rgb.astype(np.float64) * (1 - w) + target * w
    return np.rint(mixed).astype(np.uint8)


def bone_confidence(score: np.ndarray, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    """(..., K) joint scores -> (..., B) bone confidence = min of the two joints (NaN -> 0)."""
    s = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0)
    a = np.array([e[0] for e in edges])
    b = np.array([e[1] for e in edges])
    return np.minimum(s[..., a], s[..., b])


def confidence_status(conf: np.ndarray | float) -> np.ndarray:
    c = np.asarray(conf, dtype=np.float64)
    return np.clip((c - CONF_LO) / (CONF_HI - CONF_LO), 0.0, 1.0)


@dataclass(frozen=True)
class Band:
    ok: tuple[float, float]
    warn: tuple[float, float]

    def __post_init__(self) -> None:
        a, b = self.ok
        c, d = self.warn
        if not a < b:
            raise ValueError(f"ok band must have lo < hi, got {self.ok}")
        if not (c <= a and d >= b):
            raise ValueError(f"warn band {self.warn} must contain ok band {self.ok}")


def band_status(x: np.ndarray | float, band: Band) -> np.ndarray:
    """1 inside ok; 1 -> 0.5 linearly from the ok edge to the warn edge; 0.5 -> 0 over one more
    warn-width beyond it; NaN passes through."""
    v = np.asarray(x, dtype=np.float64)
    a, b = band.ok
    c, d = band.warn
    w_lo = (a - c) if a > c else max((b - a) * 0.25, 1e-6)
    w_hi = (d - b) if d > b else max((b - a) * 0.25, 1e-6)
    out = np.ones_like(v)
    below = v < a
    above = v > b
    out[below] = 1.0 - 0.5 * (a - v[below]) / w_lo
    out[above] = 1.0 - 0.5 * (v[above] - b) / w_hi
    out = np.clip(out, 0.0, 1.0)
    out[np.isnan(v)] = np.nan
    return out


@dataclass(frozen=True)
class Rule:
    """Protocol-agnostic angle-band rule (built from `clinical.protocol.RuleSpec`)."""

    angle: str
    band: Band
    segments: tuple[str, ...]
    label: str = ""


def bone_rule_status(
    values: Mapping[str, np.ndarray],
    rules: Sequence[Rule],
    layout: KeypointLayout,
    side: Literal["left", "right"] | None,
) -> np.ndarray | None:
    """(T, B) status = min over rules touching each bone; NaN where no rule applies.
    Returns None when no rule can be evaluated (no matching angle series)."""
    edges = layout.edges()
    t = None
    for name in values:
        t = len(values[name])
        break
    if t is None:
        return None
    out = np.full((t, len(edges)), np.nan, dtype=np.float64)
    any_rule = False
    for rule in rules:
        if rule.angle not in values:
            continue
        s = band_status(values[rule.angle], rule.band)
        for seg in rule.segments:
            for e in segment_edges(layout, seg, side):
                col = out[:, e]
                out[:, e] = np.where(np.isnan(col), s, np.fmin(col, s))
                any_rule = True
    return out if any_rule else None


@dataclass
class BoneStatus:
    edges: list[tuple[int, int]]
    confidence: np.ndarray  # (T, B) raw min joint score
    rules: np.ndarray | None  # (T, B) rule status or None

    @classmethod
    def build(
        cls,
        score: np.ndarray,
        layout: KeypointLayout,
        *,
        values: Mapping[str, np.ndarray] | None = None,
        rules: Sequence[Rule] = (),
        side: Literal["left", "right"] | None = None,
    ) -> BoneStatus:
        edges = layout.edges()
        conf = bone_confidence(score, edges)
        rs = bone_rule_status(values, rules, layout, side) if values and rules else None
        return cls(edges, conf, rs)

    @property
    def n_frames(self) -> int:
        return int(self.confidence.shape[0])

    def available_modes(self) -> list[Mode]:
        return ["confidence", "rules"] if self.rules is not None else ["confidence"]

    def value(self, mode: Mode) -> np.ndarray:
        if mode == "confidence":
            return confidence_status(self.confidence)
        if self.rules is None:
            raise ValueError("rules mode not available")
        return self.rules

    def colors(self, t: int, mode: Mode) -> tuple[np.ndarray, np.ndarray]:
        """((B, 3) uint8 RGB, (B,) visible) for frame t."""
        conf = self.confidence[t]
        visible = conf >= CONF_HIDE
        if mode == "rules" and self.rules is not None:
            rgb = dim_rgb(status_rgb(self.rules[t]), conf)
        else:
            rgb = status_rgb(confidence_status(conf))
        return rgb, visible


def legend(mode: Mode) -> list[tuple[str, str]]:
    if mode == "rules":
        return [
            ("in range", hex_of(status_rgb(1.0))),
            ("near limit", hex_of(status_rgb(0.5))),
            ("outside range", hex_of(status_rgb(0.0))),
            ("not assessed", hex_of(NEUTRAL)),
            ("low confidence (dimmed)", hex_of(DIM_TARGET)),
        ]
    return [
        (f"confidence ≥ {CONF_HI:.2f}", hex_of(status_rgb(1.0))),
        (f"confidence {(CONF_LO + CONF_HI) / 2:.2f}", hex_of(status_rgb(0.5))),
        (f"confidence {CONF_LO:.2f} (hidden below)", hex_of(status_rgb(0.0))),
    ]
