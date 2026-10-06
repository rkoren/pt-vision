"""OpenSim-compatible TRC (markers) and MOT (angles) writers, using Pose2Sim's header layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ptvision._vendor.pose2sim.trc_mot import write_mot as _write_mot
from ptvision._vendor.pose2sim.trc_mot import write_trc as _write_trc
from ptvision.kinematics.angles import AngleSeries
from ptvision.pose.track import PoseTrack


def write_trc(
    track: PoseTrack, path: Path | str, *, person: int = 0, units: str | None = None
) -> Path:
    """Write one person's keypoints as a TRC file. Image y (down) is flipped so Y points up; Z is 0
    for 2D data. Units default to 'px' for image coordinates and 'm' for world coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    units = units or ("m" if track.coord_space == "world_m" else "px")
    c = track.coords[:, person].astype(np.float64)  # (T, K, D)
    t, k, d = c.shape
    xyz = np.zeros((t, k, 3))
    xyz[..., 0] = c[..., 0]
    xyz[..., 1] = -c[..., 1] if track.coord_space == "image_px" else c[..., 1]
    if d == 3:
        xyz[..., 2] = c[..., 2]
    data = pd.DataFrame(xyz.reshape(t, k * 3))
    frames = pd.Series(np.arange(1, t + 1))
    time = pd.Series(track.time_s)
    names = track.layout.names
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        f"{track.fps}\t{track.fps}\t{t}\t{k}\t{units}\t{track.fps}\t1\t{t}\n",
        "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n",
        "\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(k)) + "\t\n",
    ]
    _write_trc(str(path), data, frames, time, header)  # type: ignore[no-untyped-call]
    return path


def write_mot(angles: AngleSeries, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = angles.frame.reset_index(drop=True)
    time = pd.Series(np.arange(len(df)) / angles.fps)
    header = [
        "Coordinates\n",
        "version=1\n",
        f"nRows={len(df)}\n",
        f"nColumns={len(df.columns) + 1}\n",
        "inDegrees=yes\n",
        "endheader\n",
        "time\t" + "\t".join(df.columns) + "\n",
    ]
    _write_mot(str(path), df, time, header)  # type: ignore[no-untyped-call]
    return path
