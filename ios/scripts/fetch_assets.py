"""[REVIEW] Put the pose models and demo clips the iOS app bundles into PTVision/Resources.

They stay out of git: the models come from the same pinned archives as `ptv models pull`, and the
clips are copies of files already tracked in this repo. Standard library only.

    uv run --no-project python ios/scripts/fetch_assets.py
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESOURCES = REPO / "ios" / "PTVision" / "Resources"
MODELS = ("yolox_tiny", "rtmpose_s_halpe26")
CLIPS = {
    "sts_side.mp4": REPO / "src" / "ptvision" / "samples" / "sts_side.mp4",
    "sports2d_demo_2s.mp4": REPO / "tests" / "fixtures" / "clip_2s_480p.mp4",
}
OPENMMLAB = "https://download.openmmlab.com/mmpose/v1/projects/"
MIRROR = "https://huggingface.co/Tau-J/RTMPose/resolve/main/"


def download(url: str, dest: Path, sha256: str) -> None:
    for candidate in (url, url.replace(OPENMMLAB, MIRROR)):
        try:
            with urllib.request.urlopen(candidate, timeout=120) as response:
                dest.write_bytes(response.read())
        except OSError as error:
            print(f"  {candidate}: {error}")
            continue
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == sha256:
            return
        print(f"  {candidate}: sha256 {digest}, expected {sha256}")
    raise SystemExit(f"could not download a verified copy of {url}")


def fetch_model(name: str, spec: dict[str, str]) -> None:
    target = RESOURCES / f"{name}.onnx"
    if target.exists():
        print(f"{target.name}: already present")
        return
    print(f"{target.name}: downloading")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.zip"
        download(spec["url"], archive, spec["sha256"])
        with zipfile.ZipFile(archive) as zf:
            member = next(n for n in zf.namelist() if n.endswith("end2end.onnx"))
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    manifest = REPO / "src" / "ptvision" / "pose" / "models.toml"
    models = tomllib.loads(manifest.read_text(encoding="utf-8"))["models"]
    for name in MODELS:
        fetch_model(name, models[name])
    for filename, source in CLIPS.items():
        shutil.copyfile(source, RESOURCES / filename)
        print(f"{filename}: copied from {source.relative_to(REPO)}")


if __name__ == "__main__":
    main()
