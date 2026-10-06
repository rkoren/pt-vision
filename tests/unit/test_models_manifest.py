from ptvision.pose.models import MODE_NAMES, load_manifest


def test_manifest_consistent() -> None:
    models, modes = load_manifest()
    assert set(modes) == set(MODE_NAMES)
    for mode, m in modes.items():
        assert models[m["det"]].kind == "det", mode
        assert models[m["pose"]].kind == "pose", mode
        assert models[m["pose"]].layout == "halpe26"
    for spec in models.values():
        assert spec.url.startswith("https://")
        assert spec.url.endswith(".zip")
        assert len(spec.input_size) == 2
        assert spec.sha256 == "" or len(spec.sha256) == 64
