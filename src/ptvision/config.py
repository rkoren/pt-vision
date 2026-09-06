"""Process-wide settings, resolved from environment variables with sensible defaults.

PTV_DATA_DIR       root of the patient/visit/trial store (default ~/ptvision-data)
PTV_MODEL_DIR      where ONNX weights are cached (default <user cache>/ptvision/models)
PTV_MODE           rtmlib mode: lightweight | balanced | performance (default balanced)
PTV_DEVICE         cpu | mps | cuda | cuda:N (default cpu; `mps` means the ONNX CoreML EP)
PTV_BACKEND        onnxruntime | opencv | openvino (default onnxruntime)
PTV_DET_FREQUENCY  run the person detector every N frames (default 4)
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs
from pydantic import BaseModel, Field


class Settings(BaseModel):
    data_dir: Path
    models_dir: Path
    mode: str = "balanced"
    device: str = "cpu"
    backend: str = "onnxruntime"
    det_frequency: int = Field(default=4, ge=1)


def settings() -> Settings:
    cache = Path(platformdirs.user_cache_dir("ptvision"))
    return Settings(
        data_dir=Path(os.environ.get("PTV_DATA_DIR", str(Path.home() / "ptvision-data"))),
        models_dir=Path(os.environ.get("PTV_MODEL_DIR", str(cache / "models"))),
        mode=os.environ.get("PTV_MODE", "balanced"),
        device=os.environ.get("PTV_DEVICE", "cpu"),
        backend=os.environ.get("PTV_BACKEND", "onnxruntime"),
        det_frequency=int(os.environ.get("PTV_DET_FREQUENCY", "4")),
    )
