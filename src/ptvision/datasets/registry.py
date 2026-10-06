"""Dataset manifest, download, verification, and provenance.

Datasets live under `$PTV_DATASETS_DIR/<name>/` (default `~/ptvision-data/datasets`):
  raw/        archives as downloaded, extracted in place
  derived/    anything we compute (never re-downloaded)
  PROVENANCE.json  license, source, citation, files with checksums, download date
"""

from __future__ import annotations

import hashlib
import os
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from ptvision.files import DownloadError, download, read_json, write_json

ProgressFn = Callable[[str, int, int | None], None]


@dataclass(frozen=True)
class DatasetFile:
    url: str
    name: str
    size: int | None = None
    md5: str | None = None
    extract: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    citation: str
    source: str
    license: str
    license_url: str
    kind: str
    notes: str
    files: tuple[DatasetFile, ...]
    target: str | None = None  # store under another dataset's directory

    @property
    def dir_name(self) -> str:
        return self.target or self.name

    @property
    def total_size(self) -> int:
        return sum(f.size or 0 for f in self.files)


__all__ = [
    "ChecksumError",
    "DatasetDirs",
    "DatasetFile",
    "DatasetSpec",
    "DownloadError",
    "PullResult",
    "datasets_root",
    "load_manifest",
    "pull",
    "status",
]


def datasets_root() -> Path:
    return Path(os.environ.get("PTV_DATASETS_DIR", str(Path.home() / "ptvision-data" / "datasets")))


def load_manifest() -> dict[str, DatasetSpec]:
    text = (
        resources.files("ptvision.datasets").joinpath("manifest.toml").read_text(encoding="utf-8")
    )
    data = tomllib.loads(text)
    out: dict[str, DatasetSpec] = {}
    for name, d in data["datasets"].items():
        files = tuple(
            DatasetFile(
                url=f["url"],
                name=f["name"],
                size=f.get("size"),
                md5=f.get("md5"),
                extract=bool(f.get("extract", False)),
            )
            for f in d["files"]
        )
        out[name] = DatasetSpec(
            name=name,
            title=d["title"],
            citation=d["citation"],
            source=d["source"],
            license=d["license"],
            license_url=d["license_url"],
            kind=d["kind"],
            notes=d.get("notes", ""),
            files=files,
            target=d.get("target"),
        )
    return out


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


class ChecksumError(RuntimeError):
    pass


@dataclass
class DatasetDirs:
    root: Path
    raw: Path
    derived: Path
    provenance: Path

    @classmethod
    def for_spec(cls, spec: DatasetSpec) -> DatasetDirs:
        root = datasets_root() / spec.dir_name
        return cls(root, root / "raw", root / "derived", root / "PROVENANCE.json")


@dataclass
class PullResult:
    spec: DatasetSpec
    dirs: DatasetDirs
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    extracted: list[str] = field(default_factory=list)


def _download(
    url: str,
    dst: Path,
    *,
    expected: int | None,
    label: str,
    progress: ProgressFn | None,
    attempts: int = 6,
) -> None:
    """Fetch one dataset file with the shared resumable downloader."""
    download(
        url,
        dst,
        expected_size=expected,
        progress=progress,
        label=label,
        attempts=attempts,
        user_agent="ptvision-datasets",
    )


def _extract(archive: Path, into: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(into)


def pull(
    spec: DatasetSpec,
    *,
    verify: bool = True,
    extract: bool = True,
    progress: ProgressFn | None = None,
) -> PullResult:
    dirs = DatasetDirs.for_spec(spec)
    dirs.raw.mkdir(parents=True, exist_ok=True)
    dirs.derived.mkdir(parents=True, exist_ok=True)
    res = PullResult(spec, dirs)
    prov = read_json(dirs.provenance) if dirs.provenance.exists() else {}
    files_prov: dict[str, dict[str, object]] = dict(prov.get("files", {}))

    def _save_provenance() -> None:
        write_json(
            dirs.provenance,
            {
                "name": spec.dir_name,
                "title": spec.title,
                "citation": spec.citation,
                "source": spec.source,
                "license": spec.license,
                "license_url": spec.license_url,
                "kind": spec.kind,
                "notes": spec.notes,
                "files": files_prov,
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )

    for f in spec.files:
        dst = dirs.raw / f.name
        # A file already on disk is kept when its checksum is known-good. The checksum is
        # recomputed when provenance has no record of it: a pull interrupted on a later file used
        # to lose the record and download finished archives again (10 GB wasted on COMFI video).
        have = dst.exists() and (
            not verify
            or not f.md5
            or files_prov.get(f.name, {}).get("md5") == f.md5
            or md5_file(dst) == f.md5
        )
        if have:
            dst.with_name(dst.name + ".part").unlink(missing_ok=True)
            res.skipped.append(f.name)
        else:
            _download(f.url, dst, expected=f.size, label=f.name, progress=progress)
            res.downloaded.append(f.name)
        md5 = (
            md5_file(dst)
            if (verify and (f.md5 or f.name not in files_prov))
            else files_prov.get(f.name, {}).get("md5")
        )
        if verify and f.md5 and md5 != f.md5:
            raise ChecksumError(f"{spec.name}/{f.name}: md5 {md5} != published {f.md5}")
        files_prov[f.name] = {
            "url": f.url,
            "md5": md5,
            "size": dst.stat().st_size,
            "downloaded_at": files_prov.get(f.name, {}).get("downloaded_at")
            or datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if extract and f.extract and dst.suffix.lower() == ".zip":
            marker = dirs.raw / f".{f.name}.extracted"
            if not marker.exists():
                _extract(dst, dirs.raw / dst.stem)
                marker.write_text(datetime.now(UTC).isoformat(timespec="seconds"))
                res.extracted.append(f.name)
        _save_provenance()  # after every file, so an interrupted pull keeps its progress

    _save_provenance()
    return res


def status(spec: DatasetSpec) -> dict[str, object]:
    dirs = DatasetDirs.for_spec(spec)
    present = [f.name for f in spec.files if (dirs.raw / f.name).exists()]
    return {
        "name": spec.name,
        "license": spec.license,
        "kind": spec.kind,
        "files": f"{len(present)}/{len(spec.files)}",
        "size_gb": round(spec.total_size / 1e9, 2),
        "dir": str(dirs.root),
        "provenance": dirs.provenance.exists(),
    }
