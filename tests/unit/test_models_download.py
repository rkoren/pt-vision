"""[REVIEW] new file (demo MVP): byte progress while downloading model weights."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from ptvision.pose import models as M


class _FakeResp(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *a):  # type: ignore[no-untyped-def]
        self.close()


def test_download_reports_progress_and_extracts(tmp_path: Path, monkeypatch) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("end2end.onnx", b"x" * (3 << 20))  # 3 MB payload -> several progress calls
    payload = buf.getvalue()
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda req, **kw: _FakeResp(payload))

    mm = M.ModelManager(tmp_path)
    name = next(iter(mm.models))
    assert name in mm.missing_for_mode(next(m for m, v in mm.modes.items() if name in v.values()))
    calls: list[tuple[str, int, int | None]] = []
    res = mm.ensure(
        name, verify=False, quiet=True, progress=lambda n, d, t: calls.append((n, d, t))
    )
    assert res.onnx_path.exists()
    assert calls and calls[0][0] == name and calls[-1][1] == len(payload) == calls[-1][2]
    assert mm.is_available(name)


def test_existing_archive_is_extracted_not_redownloaded(tmp_path: Path, monkeypatch) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("end2end.onnx", b"y" * 1024)
    payload = buf.getvalue()
    mm = M.ModelManager(tmp_path)
    name = next(iter(mm.models))
    mm.models_dir.mkdir(parents=True, exist_ok=True)
    mm.archive_path(mm.models[name]).write_bytes(payload)  # archive present, nothing extracted
    calls = []
    monkeypatch.setattr(
        M.urllib.request, "urlopen", lambda req, **kw: calls.append(req) or _FakeResp(payload)
    )
    res = mm.ensure(name, verify=False, quiet=True)
    assert res.onnx_path.exists() and not calls, "must not download again"
