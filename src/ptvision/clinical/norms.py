"""Normative reference lookup"""

from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass
from importlib import resources
from typing import Literal

from ptvision.clinical.metrics.base import Metric, NormComparison
from ptvision.data.models import Subject


@dataclass(frozen=True)
class NormRow:
    age_lo: int
    age_hi: int
    sex: str
    statistic: str
    value: float
    units: str
    n: int | None


@dataclass(frozen=True)
class NormTable:
    norm_id: str
    metric: str
    population: str
    direction: str
    citation: str
    note: str
    rows: tuple[NormRow, ...]

    def lookup(self, age: int, sex: str | None, statistic: str = "mean") -> NormRow | None:
        sex = sex or "all"
        candidates = [
            r for r in self.rows if r.age_lo <= age <= r.age_hi and r.statistic == statistic
        ]
        exact = [r for r in candidates if r.sex == sex]
        if exact:
            return exact[0]
        general = [r for r in candidates if r.sex == "all"]
        return general[0] if general else None


def load_norms() -> dict[str, NormTable]:
    pkg = resources.files("ptvision.clinical.norm_tables")
    index = tomllib.loads(pkg.joinpath("index.toml").read_text(encoding="utf-8"))
    tables: dict[str, NormTable] = {}
    for norm_id, meta in index["norms"].items():
        text = pkg.joinpath(meta["file"]).read_text(encoding="utf-8")
        rows = []
        for r in csv.DictReader(text.splitlines()):
            rows.append(
                NormRow(
                    age_lo=int(r["age_lo"]),
                    age_hi=int(r["age_hi"]),
                    sex=r["sex"] or "all",
                    statistic=r["statistic"],
                    value=float(r["value"]),
                    units=r["units"],
                    n=int(r["n"]) if r.get("n") else None,
                )
            )
        tables[norm_id] = NormTable(
            norm_id=norm_id,
            metric=meta["metric"],
            population=meta["population"],
            direction=meta["direction"],
            citation=meta["citation"],
            note=meta.get("note", ""),
            rows=tuple(rows),
        )
    return tables


def compare(
    metric: Metric,
    norm_id: str,
    subject: Subject,
    tables: dict[str, NormTable] | None = None,
) -> NormComparison:
    tables = tables or load_norms()
    table = tables[norm_id]
    direction: Literal["higher_is_worse", "lower_is_worse"] = (
        "lower_is_worse" if table.direction == "lower_is_worse" else "higher_is_worse"
    )

    def nc(statement: str, *, row: NormRow | None, applicable: bool) -> NormComparison:
        return NormComparison(
            metric=metric.name,
            norm_id=norm_id,
            population=table.population,
            statistic=row.statistic if row else "mean",
            reference_value=row.value if row else None,
            units=metric.units,
            direction=direction,
            statement=statement,
            citation=table.citation,
            applicable=applicable,
        )

    if metric.value is None:
        return nc("Metric not available; no comparison made.", row=None, applicable=False)
    if subject.age_years is None:
        return nc("No reference band: subject age not recorded.", row=None, applicable=False)
    row = table.lookup(subject.age_years, subject.sex)
    if row is None:
        lo = min(r.age_lo for r in table.rows)
        hi = max(r.age_hi for r in table.rows)
        return nc(
            f"No reference band for age {subject.age_years} in this table (covers {lo}–{hi} y).",
            row=None,
            applicable=False,
        )
    diff = float(metric.value) - row.value
    worse = diff > 0 if direction == "higher_is_worse" else diff < 0
    if metric.units == "s":
        rel = "slower than" if worse else "faster than or equal to"
    else:
        rel = "above" if diff > 0 else "at or below"
    err = ""
    if metric.error is not None:
        err = f" (measurement resolution ±{metric.error:.2f} {metric.units})"
    sex_txt = "" if row.sex == "all" else f", {row.sex}"
    statement = (
        f"{metric.formatted()} is {rel} the reference mean of {row.value:g} {row.units} "
        f"for ages {row.age_lo}–{row.age_hi}{sex_txt}{err}. {table.note}"
    ).strip()
    return nc(statement, row=row, applicable=True)
