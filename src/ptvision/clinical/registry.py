"""Decorator registries for segmenters and metrics, so protocols can reference them by name and a
test can assert every tier-1 metric carries a citation and a method version"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

SegmenterFn = Callable[..., Any]
MetricFn = Callable[..., Any]


@dataclass(frozen=True)
class SegmenterEntry:
    name: str
    version: str
    fn: SegmenterFn
    description: str


@dataclass(frozen=True)
class MetricEntry:
    name: str
    version: str
    tier: Literal[1, 2, 3]
    label: str
    units: str
    citation: str
    fn: MetricFn
    description: str


SEGMENTERS: dict[str, SegmenterEntry] = {}
METRICS: dict[str, MetricEntry] = {}


def register_segmenter(name: str, *, version: str) -> Callable[[SegmenterFn], SegmenterFn]:
    def deco(fn: SegmenterFn) -> SegmenterFn:
        if name in SEGMENTERS:
            raise ValueError(f"segmenter {name!r} already registered")
        SEGMENTERS[name] = SegmenterEntry(name, version, fn, (fn.__doc__ or "").strip())
        return fn

    return deco


def register_metric(
    name: str,
    *,
    version: str,
    tier: Literal[1, 2, 3],
    label: str,
    units: str,
    citation: str = "",
) -> Callable[[MetricFn], MetricFn]:
    if tier not in (1, 2, 3):
        raise ValueError("tier must be 1, 2 or 3")
    if tier == 1 and not citation:
        raise ValueError(f"tier-1 metric {name!r} must cite its validation source")

    def deco(fn: MetricFn) -> MetricFn:
        if name in METRICS:
            raise ValueError(f"metric {name!r} already registered")
        METRICS[name] = MetricEntry(
            name, version, tier, label, units, citation, fn, (fn.__doc__ or "").strip()
        )
        return fn

    return deco


def get_segmenter(name: str) -> SegmenterEntry:
    try:
        return SEGMENTERS[name]
    except KeyError as e:
        raise KeyError(f"unknown segmenter {name!r}; registered: {sorted(SEGMENTERS)}") from e


def get_metric(name: str) -> MetricEntry:
    try:
        return METRICS[name]
    except KeyError as e:
        raise KeyError(f"unknown metric {name!r}; registered: {sorted(METRICS)}") from e
