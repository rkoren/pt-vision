from __future__ import annotations

from typer.testing import CliRunner

from ptvision.cli.main import app
from ptvision.samples import DEFAULT_SAMPLE, SAMPLES, sample


def test_sample_clip_is_shipped_and_small() -> None:
    smp = sample(DEFAULT_SAMPLE)
    assert smp.path.exists() and smp.path.suffix == ".mp4"
    assert smp.path.stat().st_size < 6_000_000, "keep the repo small"
    assert smp.protocol == "sts_5x" and smp.height_m and smp.age_years


def test_sample_clip_probes_as_720p_h264() -> None:
    import pytest

    from ptvision.video import probe

    try:
        info = probe(sample().path)
    except Exception as e:  # ffprobe missing
        pytest.skip(str(e))
    assert info.codec == "h264" and min(info.width, info.height) == 720
    assert 12 < info.duration_s < 15 and abs(info.fps - 30) < 0.1


def test_demo_help_and_unknown_sample() -> None:
    r = CliRunner().invoke(app, ["demo", "--help"])
    assert r.exit_code == 0 and "sample" in r.output
    import pytest

    with pytest.raises(KeyError):
        sample("nope")
    assert set(SAMPLES) == {"sit_to_stand"}
