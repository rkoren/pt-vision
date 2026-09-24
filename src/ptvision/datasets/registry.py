# [REVIEW] new file (BACKLOG B29)
"""Dataset manifest, download, verification, and provenance.

Datasets live under `$PTV_DATASETS_DIR/<name>/` (default `~/ptvision-data/datasets`):
  raw/        archives as downloaded, extracted in place
  derived/    anything we compute (never re-downloaded)
  PROVENANCE.json  license, source, citation, files with checksums, download date
"""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import ssl
import time
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import certifi

from ptvision.io.jsonio import read_json, write_json

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


class DownloadError(RuntimeError):
    """The server closed the connection before the whole file arrived."""


def _download(
    url: str,
    dst: Path,
    *,
    expected: int | None,
    label: str,
    progress: ProgressFn | None,
    attempts: int = 6,
) -> None:
    """Stream ``url`` to ``dst`` through a resumable ``.part`` file.

    Zenodo drops multi-gigabyte transfers now and then; a short read is retried with an HTTP
    ``Range`` header from where the previous attempt stopped, and a download that still ends short
    of the published size raises ``DownloadError`` instead of surfacing later as a checksum
    mismatch.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    total = expected
    last_err: Exception | None = None
    for attempt in range(attempts):
        done = part.stat().st_size if part.exists() else 0
        if total is not None and done >= total:
            break
        headers = {"User-Agent": "ptvision-datasets"}
        if done:
            headers["Range"] = f"bytes={done}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
                if done and resp.status != 206:  # server ignored the range: start over
                    done = 0
                length = int(resp.headers.get("Content-Length") or 0)
                if total is None and length:
                    total = done + length
                with part.open("ab" if done else "wb") as out:
                    while blk := resp.read(1 << 20):
                        out.write(blk)
                        done += len(blk)
                        if progress:
                            progress(label, done, total)
        except (OSError, http.client.HTTPException) as e:  # includes IncompleteRead, timeouts
            last_err = e
            time.sleep(min(30, 2**attempt))
            continue
        if total is None or done >= total:
            break
        last_err = DownloadError(f"{label}: got {done} of {total} bytes")
        time.sleep(min(30, 2**attempt))
    done = part.stat().st_size if part.exists() else 0
    if total is not None and done != total:
        raise DownloadError(
            f"{label}: download stopped at {done} of {total} bytes after {attempts} attempts"
            + (f" ({last_err})" if last_err else "")
        )
    part.replace(dst)


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

    for f in spec.files:
        dst = dirs.raw / f.name
        if dst.exists() and (
            not verify or not f.md5 or files_prov.get(f.name, {}).get("md5") == f.md5
        ):
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


def remove_derived(spec: DatasetSpec) -> None:
    d = DatasetDirs.for_spec(spec).derived
    if d.exists():
        shutil.rmtree(d)
