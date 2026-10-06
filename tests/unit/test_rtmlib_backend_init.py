"""[REVIEW] Non-onnxruntime backends used to run ORT-only lines and crash in __init__."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import rtmlib

from ptvision.pose.rtmlib_backend import RtmlibBackend


class _Tool:
    score_thr = 0.7

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None


class _StubModel:
    """Like rtmlib.BodyWithFeet built with backend="opencv": no ONNX `session` attribute."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.det_model = _Tool()
        self.pose_model = _Tool()


def _paths() -> SimpleNamespace:
    def part(name: str, size: tuple[int, int]) -> SimpleNamespace:
        spec = SimpleNamespace(name=name, url=f"https://example.invalid/{name}", input_size=size)
        return SimpleNamespace(spec=spec, onnx_path=f"/nowhere/{name}.onnx", onnx_sha256="0" * 64)

    return SimpleNamespace(det=part("det", (640, 640)), pose=part("pose", (192, 256)))


@pytest.mark.parametrize("backend", ["opencv", "openvino"])
def test_non_ort_backend_constructs(monkeypatch, backend: str) -> None:
    monkeypatch.setattr(rtmlib, "BodyWithFeet", _StubModel)
    models = SimpleNamespace(ensure_mode=lambda *a, **k: _paths())
    b = RtmlibBackend(backend=backend, models=models)  # type: ignore[arg-type]
    assert b._model.det_model.score_thr == b.det_score_thr  # threshold applies to every backend
    assert b.info.providers == []
    assert b.info.runtime is not None and b.info.runtime.startswith(backend)
