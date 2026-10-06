"""Model weight manifest, download, cache, and checksum verification"""

from __future__ import annotations

import shutil
import ssl
import tempfile
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ptvision.config import settings
from ptvision.files import sha256_file

MODE_NAMES = ("lightweight", "balanced", "performance")

DownloadProgress = Callable[[str, int, int | None], None]  # (model name, done bytes, total or None)

_HF_MIRROR = {
    "https://download.openmmlab.com/mmpose/v1/projects/": "https://huggingface.co/Tau-J/RTMPose/resolve/main/",
}


def _ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle: python.org macOS builds otherwise fail certificate verification."""
    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def mirror_url(url: str) -> str | None:
    for prefix, mirror in _HF_MIRROR.items():
        if url.startswith(prefix):
            return mirror + url[len(prefix) :]
    return None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "det" | "pose"
    url: str
    input_size: tuple[int, int]
    sha256: str  # "" if not pinned
    license: str
    layout: str | None = None

    @property
    def archive_name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ResolvedModel:
    spec: ModelSpec
    onnx_path: Path
    archive_sha256: str
    onnx_sha256: str


@dataclass(frozen=True)
class ModePaths:
    mode: str
    det: ResolvedModel
    pose: ResolvedModel


def load_manifest() -> tuple[dict[str, ModelSpec], dict[str, dict[str, str]]]:
    text = resources.files("ptvision.pose").joinpath("models.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    models = {
        name: ModelSpec(
            name=name,
            kind=m["kind"],
            url=m["url"],
            input_size=(int(m["input_size"][0]), int(m["input_size"][1])),
            sha256=m.get("sha256", ""),
            license=m.get("license", "unknown"),
            layout=m.get("layout"),
        )
        for name, m in data["models"].items()
    }
    return models, data["modes"]


class ChecksumError(RuntimeError):
    pass


class ModelManager:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir) if models_dir is not None else settings().models_dir
        self.models, self.modes = load_manifest()

    # ---- paths -------------------------------------------------------------------------
    def archive_path(self, spec: ModelSpec) -> Path:
        return self.models_dir / spec.archive_name

    def extract_dir(self, spec: ModelSpec) -> Path:
        return self.models_dir / spec.archive_name.removesuffix(".zip")

    def _find_onnx(self, folder: Path) -> Path | None:
        candidates = sorted(folder.rglob("*.onnx"))
        if not candidates:
            return None
        preferred = [c for c in candidates if c.name == "end2end.onnx"]
        return preferred[0] if preferred else candidates[0]

    def is_available(self, name: str) -> bool:
        spec = self.models[name]
        return self._find_onnx(self.extract_dir(spec)) is not None

    # ---- download ----------------------------------------------------------------------
    def ensure(
        self,
        name: str,
        *,
        verify: bool = True,
        quiet: bool = False,
        progress: DownloadProgress | None = None,
    ) -> ResolvedModel:
        """Return the local .onnx path for a model, downloading and extracting if needed"""
        spec = self.models[name]
        self.models_dir.mkdir(parents=True, exist_ok=True)
        archive = self.archive_path(spec)
        folder = self.extract_dir(spec)

        if self._find_onnx(folder) is None:
            # handle archive already on disk
            have_archive = archive.exists() and (
                not verify or not spec.sha256 or sha256_file(archive) == spec.sha256
            )
            if not have_archive:
                self._download(spec.url, archive, quiet=quiet, progress=progress, label=name)
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(folder)

        archive_sha = sha256_file(archive)
        if verify and spec.sha256 and archive_sha != spec.sha256:
            raise ChecksumError(
                f"{name}: archive sha256 {archive_sha} does not match manifest {spec.sha256}. "
                f"Delete {archive} to re-download, or update models.toml if the upstream file "
                "changed."
            )
        onnx = self._find_onnx(folder)
        if onnx is None:
            raise FileNotFoundError(f"No .onnx file found in {folder}")
        return ResolvedModel(spec, onnx, archive_sha, sha256_file(onnx))

    def ensure_mode(
        self,
        mode: str,
        *,
        verify: bool = True,
        quiet: bool = False,
        progress: DownloadProgress | None = None,
    ) -> ModePaths:
        if mode not in self.modes:
            raise KeyError(f"Unknown mode {mode!r}; choose from {list(self.modes)}")
        m = self.modes[mode]
        return ModePaths(
            mode=mode,
            det=self.ensure(m["det"], verify=verify, quiet=quiet, progress=progress),
            pose=self.ensure(m["pose"], verify=verify, quiet=quiet, progress=progress),
        )

    def missing_for_mode(self, mode: str) -> list[str]:
        m = self.modes.get(mode, {})
        return [n for n in m.values() if not self.is_available(n)]

    def _download(
        self,
        url: str,
        dst: Path,
        *,
        quiet: bool,
        progress: DownloadProgress | None = None,
        label: str = "",
    ) -> None:
        """Download to `dst`, fall back to the Hugging Face mirror rtmlib uses"""
        urls = [url]
        mirror = mirror_url(url)
        if mirror:
            urls.append(mirror)
        last_error: Exception | None = None
        for candidate in urls:
            if not quiet:
                print(f"downloading {candidate}")
            with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False, suffix=".part") as tmp:
                tmp_path = Path(tmp.name)
            try:
                req = urllib.request.Request(candidate, headers={"User-Agent": "ptvision"})
                with (
                    urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp,
                    tmp_path.open("wb") as out,
                ):
                    total = int(resp.headers.get("Content-Length") or 0) or None
                    done = 0
                    while chunk := resp.read(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(label or dst.name, done, total)
                tmp_path.replace(dst)
                return
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                last_error = e
                if not quiet:
                    print(f"  failed: {e}")
        raise RuntimeError(f"Could not download {url}: {last_error}")

    # ---- listing -----------------------------------------------------------------------
    def status(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for name, spec in self.models.items():
            onnx = self._find_onnx(self.extract_dir(spec))
            archive = self.archive_path(spec)
            rows.append(
                {
                    "name": name,
                    "kind": spec.kind,
                    "available": onnx is not None,
                    "path": str(onnx) if onnx else "",
                    "size_mb": round(onnx.stat().st_size / 1e6, 1) if onnx else None,
                    "pinned": bool(spec.sha256),
                    "archive_sha256": sha256_file(archive) if archive.exists() else "",
                }
            )
        return rows
