"""PoseTrack: keypoint time series with Parquet serialization.

Parquet schema v1 (one row per frame x person x keypoint, absent persons omitted):
  camera_id (str), frame (int32), time_s (float64), person_id (int16), keypoint (str),
  kp_index (int8), x, y, z (float32; z NaN for 2D), score (float32)
File metadata: ptvision_schema, layout, coord_space, fps, n_frames, image_width, image_height,
  camera_id, stage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ptvision.pose.layout import LAYOUTS, KeypointLayout

SCHEMA_VERSION = 1
CoordSpace = Literal["image_px", "world_m"]


@dataclass
class PoseTrack:
    layout: KeypointLayout
    fps: float
    coords: np.ndarray  # (T, P, K, D) float32, NaN = missing
    score: np.ndarray  # (T, P, K) float32
    person_ids: np.ndarray  # (P,) int16
    camera_id: str = "cam0"
    coord_space: CoordSpace = "image_px"
    image_size: tuple[int, int] | None = None  # (width, height)
    stage: str = "raw"
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float32)
        self.score = np.asarray(self.score, dtype=np.float32)
        self.person_ids = np.asarray(self.person_ids, dtype=np.int16)
        if self.coords.ndim != 4:
            raise ValueError(f"coords must be (T, P, K, D); got {self.coords.shape}")
        t, p, k, d = self.coords.shape
        if self.score.shape != (t, p, k):
            raise ValueError(f"score shape {self.score.shape} != {(t, p, k)}")
        if k != self.layout.n:
            raise ValueError(f"{k} keypoints but layout {self.layout.name} has {self.layout.n}")
        if d not in (2, 3):
            raise ValueError(f"coords last dim must be 2 or 3, got {d}")
        if self.person_ids.shape != (p,):
            raise ValueError(f"person_ids shape {self.person_ids.shape} != {(p,)}")

    # ---- shape helpers -------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_persons(self) -> int:
        return int(self.coords.shape[1])

    @property
    def n_dims(self) -> int:
        return int(self.coords.shape[3])

    @property
    def frame_idx(self) -> np.ndarray:
        return np.arange(self.n_frames, dtype=np.int32)

    @property
    def time_s(self) -> np.ndarray:
        return self.frame_idx / self.fps

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def present(self) -> np.ndarray:
        """(T, P) bool: person has at least one non-NaN keypoint in the frame."""
        return ~np.isnan(self.coords).all(axis=(2, 3))

    # ---- spurious tracks ------------------------------------------
    @property
    def spurious(self) -> np.ndarray:
        """(P,) bool: tracks marked as not a person (see `pose.tracking.spurious_slots`"""
        if "spurious_ids" not in self.extra_metadata:
            # older trials without the mask (or ones loaded by the app) get it computed once
            from ptvision.pose.identity import spurious_slots

            self.mark_spurious(spurious_slots(self.coords, self.score, fps=self.fps))
            self.extra_metadata.setdefault("spurious_ids", "")
        raw = self.extra_metadata.get("spurious_ids", "")
        ids = {int(x) for x in raw.split(",") if x.strip()}
        return np.array([int(i) in ids for i in self.person_ids], dtype=bool)

    def mark_spurious(self, mask: np.ndarray) -> None:
        ids = [str(int(i)) for i, m in zip(self.person_ids, mask, strict=True) if m]
        self.extra_metadata["spurious_ids"] = ",".join(ids)  # "" = computed, none spurious

    @property
    def n_persons_real(self) -> int:
        """Persons excluding spurious tracks."""
        return int((~self.spurious).sum())

    def presence_fraction(self) -> np.ndarray:
        """(P,) fraction of frames in which each person is present."""
        return self.present().mean(axis=0)

    def bboxes(self) -> np.ndarray:
        """(T, P, 4) xyxy bounding boxes from keypoints (NaN where absent)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = self.coords[..., 0]
            y = self.coords[..., 1]
            return np.stack(
                [
                    np.nanmin(x, axis=2),
                    np.nanmin(y, axis=2),
                    np.nanmax(x, axis=2),
                    np.nanmax(y, axis=2),
                ],
                axis=-1,
            )

    def select_person(self, person_id: int) -> PoseTrack:
        where = np.flatnonzero(self.person_ids == person_id)
        if where.size == 0:
            raise KeyError(f"person_id {person_id} not in track ({self.person_ids.tolist()})")
        p = int(where[0])
        return PoseTrack(
            layout=self.layout,
            fps=self.fps,
            coords=self.coords[:, p : p + 1],
            score=self.score[:, p : p + 1],
            person_ids=self.person_ids[p : p + 1],
            camera_id=self.camera_id,
            coord_space=self.coord_space,
            image_size=self.image_size,
            stage=self.stage,
            extra_metadata=dict(self.extra_metadata),
        )

    def keypoint(self, name: str, person: int = 0) -> np.ndarray:
        """(T, D) coordinates of one keypoint for one person slot."""
        return self.coords[:, person, self.layout.index(name), :]

    def keypoint_score(self, name: str, person: int = 0) -> np.ndarray:
        return self.score[:, person, self.layout.index(name)]

    # ---- serialization ---------------------------------------------------------------
    def _metadata(self) -> dict[str, str]:
        md = {
            "ptvision_schema": str(SCHEMA_VERSION),
            "layout": self.layout.name,
            "coord_space": self.coord_space,
            "fps": repr(float(self.fps)),
            "n_frames": str(self.n_frames),
            "camera_id": self.camera_id,
            "stage": self.stage,
        }
        if self.image_size is not None:
            md["image_width"] = str(self.image_size[0])
            md["image_height"] = str(self.image_size[1])
        md.update(self.extra_metadata)
        return md

    def to_frame(self) -> pd.DataFrame:
        k, d = self.coords.shape[2:]
        present = self.present()  # (T, P)
        ti, pi = np.nonzero(present)
        n = ti.size
        frame = np.repeat(ti.astype(np.int32), k)
        person = np.repeat(self.person_ids[pi], k)
        kp_index = np.tile(np.arange(k, dtype=np.int8), n)
        c = self.coords[ti, pi]  # (n, K, D)
        s = self.score[ti, pi]  # (n, K)
        z = c[..., 2].ravel() if d == 3 else np.full(n * k, np.nan, dtype=np.float32)
        df = pd.DataFrame(
            {
                "camera_id": pd.Categorical([self.camera_id] * (n * k)),
                "frame": frame,
                "time_s": frame.astype(np.float64) / self.fps,
                "person_id": person.astype(np.int16),
                "keypoint": pd.Categorical.from_codes(
                    kp_index, categories=pd.Index(list(self.layout.names))
                ),
                "kp_index": kp_index,
                "x": c[..., 0].ravel(),
                "y": c[..., 1].ravel(),
                "z": z,
                "score": s.ravel(),
            }
        )
        return df

    @classmethod
    def from_frame(cls, df: pd.DataFrame, metadata: dict[str, str]) -> PoseTrack:
        layout = LAYOUTS[metadata["layout"]]
        n_frames = int(metadata["n_frames"])
        fps = float(metadata["fps"])
        has_z = bool(len(df)) and not df["z"].isna().all()
        d = 3 if has_z else 2
        person_ids = np.array(sorted(pd.unique(df["person_id"]).tolist()), dtype=np.int16)
        p = len(person_ids)
        coords = np.full((n_frames, p, layout.n, d), np.nan, dtype=np.float32)
        score = np.zeros((n_frames, p, layout.n), dtype=np.float32)
        if len(df):
            pslot = np.searchsorted(person_ids, df["person_id"].to_numpy())
            fr = df["frame"].to_numpy()
            kp = df["kp_index"].to_numpy()
            coords[fr, pslot, kp, 0] = df["x"].to_numpy()
            coords[fr, pslot, kp, 1] = df["y"].to_numpy()
            if d == 3:
                coords[fr, pslot, kp, 2] = df["z"].to_numpy()
            score[fr, pslot, kp] = df["score"].to_numpy()
        image_size = None
        if "image_width" in metadata and "image_height" in metadata:
            image_size = (int(metadata["image_width"]), int(metadata["image_height"]))
        known = {
            "ptvision_schema",
            "layout",
            "coord_space",
            "fps",
            "n_frames",
            "camera_id",
            "stage",
            "image_width",
            "image_height",
        }
        extra = {k: v for k, v in metadata.items() if k not in known}
        return cls(
            layout=layout,
            fps=fps,
            coords=coords,
            score=score,
            person_ids=person_ids,
            camera_id=metadata.get("camera_id", "cam0"),
            coord_space=metadata.get("coord_space", "image_px"),  # type: ignore[arg-type]
            image_size=image_size,
            stage=metadata.get("stage", "raw"),
            extra_metadata=extra,
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(self.to_frame(), preserve_index=False)
        md = {k.encode(): v.encode() for k, v in self._metadata().items()}
        existing = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing, **md})
        pq.write_table(table, path, compression="zstd")
        return path

    @classmethod
    def load(cls, path: Path | str) -> PoseTrack:
        table = pq.read_table(path)
        raw_md = table.schema.metadata or {}
        metadata = {
            k.decode(): v.decode()
            for k, v in raw_md.items()
            if not k.startswith(b"pandas") and not k.startswith(b"ARROW")
        }
        if metadata.get("ptvision_schema") != str(SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported pose schema {metadata.get('ptvision_schema')!r} in {path}"
            )
        df = table.to_pandas()
        return cls.from_frame(df, metadata)
