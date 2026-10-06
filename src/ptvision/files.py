"""Small file helpers"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path | str, obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_default, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)
    return path


def read_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


# ---- downloads ----------------------------------------------------------------------------

DownloadProgress = Callable[[str, int, int | None], None]  # (label, done bytes, total or None)


class DownloadError(RuntimeError):
    """The transfer could not be completed (short read after retries, or every URL failed)."""


def download(
    url: str,
    dst: Path,
    *,
    expected_size: int | None = None,
    mirrors: Sequence[str] = (),
    progress: DownloadProgress | None = None,
    label: str = "",
    attempts: int = 6,
    timeout_s: int = 300,
    user_agent: str = "ptvision",
) -> None:
    """Stream a file to `dst` through a resumable `.part` file; the one downloader for model
    weights and datasets.

    A dropped connection is retried with an HTTP Range header from where it stopped, with
    exponential back-off; when the primary URL keeps failing the `mirrors` are tried in turn. A
    transfer that still ends short of `expected_size` (or of the server's Content-Length) raises
    `DownloadError` rather than surfacing later as a checksum mismatch. `progress(label, done,
    total)` is called about once per MB.
    """
    import http.client
    import ssl
    import time
    import urllib.request

    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    label = label or dst.name
    last_err: Exception | None = None
    for candidate in (url, *mirrors):
        total = expected_size
        for attempt in range(attempts):
            done = part.stat().st_size if part.exists() else 0
            if total is not None and done >= total:
                break
            headers = {"User-Agent": user_agent}
            if done:
                headers["Range"] = f"bytes={done}-"
            req = urllib.request.Request(candidate, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
                    if done and getattr(resp, "status", 206) != 206:  # range ignored: restart
                        done = 0
                    length = int(resp.headers.get("Content-Length") or 0)
                    if total is None and length:
                        total = done + length
                    with part.open("ab" if done else "wb") as out:
                        while blk := resp.read(1 << 20):
                            out.write(blk)
                            done += len(blk)
                            if progress is not None:
                                progress(label, done, total)
            except (OSError, http.client.HTTPException) as e:  # includes IncompleteRead, timeouts
                last_err = e
                if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                    break  # this URL will not work; try the next mirror
                time.sleep(min(30, 2**attempt))
                continue
            if total is None or done >= total:
                part.replace(dst)
                return
            last_err = DownloadError(f"{label}: got {done} of {total} bytes")
            time.sleep(min(30, 2**attempt))
        if part.exists() and total is not None and part.stat().st_size >= total:
            part.replace(dst)
            return
    done = part.stat().st_size if part.exists() else 0
    raise DownloadError(
        f"{label}: download failed after {attempts} attempts per URL"
        + (f", stopped at {done} bytes" if done else "")
        + (f" ({last_err})" if last_err else "")
    )
