"""Readers for the OpenSim text formats that lab datasets ship: TRC marker files and
MOT/STO coordinate or force tables
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from ptvision.datasets.markers import MarkerTrial


def read_trc(path: Path | str) -> MarkerTrial:
    """A TRC marker file (OpenSim / Motion Analysis) as a MarkerTrial in metres.

    Layout: line 1 file type, line 2 header keys, line 3 header values, line 4 `Frame# Time` plus
    one marker name per three columns, line 5 `X1 Y1 Z1 ...`, then one row per frame. Missing
    values are empty cells."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 6:
        raise ValueError(f"{path.name}: not a TRC file (too short)")
    keys = lines[1].rstrip("\r").split("\t")
    vals = lines[2].rstrip("\r").split("\t")
    hdr = dict(zip(keys, vals, strict=False))
    rate = float(hdr.get("DataRate") or hdr.get("CameraRate") or 0)
    if rate <= 0:
        raise ValueError(f"{path.name}: missing DataRate")
    units = str(hdr.get("Units", "mm")).strip()
    first_frame = int(float(hdr.get("OrigDataStartFrame", 1) or 1))
    labels = [c.strip() for c in lines[3].rstrip("\r").split("\t")[2:] if c.strip()]
    rows: list[list[float]] = []
    width = 2 + 3 * len(labels)
    for line in lines[5:]:
        if not line.strip():
            continue
        cells = line.rstrip("\r").split("\t")
        row = [_float(c) for c in cells[:width]]
        row += [np.nan] * (width - len(row))
        rows.append(row)
    data = np.asarray(rows, dtype=np.float64)
    pts = data[:, 2:].reshape(data.shape[0], len(labels), 3)
    if units.lower().startswith("mm"):
        pts = pts / 1000.0
    return MarkerTrial(path, rate, labels, pts, first_frame, "m")


def read_mot(path: Path | str) -> pd.DataFrame:
    """A MOT/STO table (OpenSim coordinates, forces, marker errors) as a DataFrame.

    The header runs to `endheader`; `inDegrees=yes|no` is kept in `df.attrs["in_degrees"]`
    (None when the file does not say). Column names are the line after the header."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if end is None:
        raise ValueError(f"{path.name}: no endheader line")
    in_degrees: bool | None = None
    for line in lines[:end]:
        key, _, val = line.partition("=")
        if key.strip().lower() == "indegrees":
            in_degrees = val.strip().lower() == "yes"
    body = "\n".join(line for line in lines[end + 1 :] if line.strip())
    df = pd.read_csv(io.StringIO(body), sep=r"\s+", engine="python")
    df.attrs["in_degrees"] = in_degrees
    return df


def _float(cell: str) -> float:
    cell = cell.strip()
    if not cell:
        return np.nan
    try:
        return float(cell)
    except ValueError:
        return np.nan
