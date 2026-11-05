"""[REVIEW] Put the pose models and demo clips the iOS app bundles into PTVision/Resources.

They stay out of git: the models come from the same pinned, checksummed archives and cache as
`ptv models pull` (through the package's model manager), and the clips are copies of files already
tracked in this repo.

    uv run python ios/scripts/fetch_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ptvision.pose.models import ModelManager

REPO = Path(__file__).resolve().parents[2]
RESOURCES = REPO / "ios" / "PTVision" / "Resources"
MODELS = ("yolox_tiny", "rtmpose_s_halpe26")
CLIPS = {
    "sts_side.mp4": REPO / "src" / "ptvision" / "samples" / "sts_side.mp4",
    "sports2d_demo_2s.mp4": REPO / "tests" / "fixtures" / "clip_2s_480p.mp4",
}


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    manager = ModelManager()
    for name in MODELS:
        onnx = manager.ensure(name).onnx_path
        shutil.copyfile(onnx, RESOURCES / f"{name}.onnx")
        print(f"{name}.onnx: copied from {onnx}")
    for filename, source in CLIPS.items():
        shutil.copyfile(source, RESOURCES / filename)
        print(f"{filename}: copied from {source.relative_to(REPO)}")


if __name__ == "__main__":
    main()
