from importlib import resources
from pathlib import Path

from typer.testing import CliRunner

from ptvision.cli.main import app
from ptvision.clinical.metrics.base import Metric
from ptvision.clinical.protocol import load_protocol
from ptvision.files import write_json
from ptvision.quality.checks import QualityCheck, QualityReport
from ptvision.report.bundle import ReportBundle, render_html


def _bundle(**extra: object) -> ReportBundle:
    m = Metric(
        name="sts_total_time",
        label="5xSTS time",
        value=12.34,
        units="s",
        error=0.07,
        error_kind="resolution",
        tier=1,
        method_version="0.1.0",
        citation="x",
    )
    b = ReportBundle(
        title="Five Times Sit-to-Stand",
        protocol={"id": "sts_5x", "version": "0.1.0", "name": "5xSTS", "summary": ""},
        trial_id="T-1",
        run_id="R-1",
        created_at="2026-09-01T00:00:00+00:00",
        ptvision_version="0.1.0",
        capture={},
        subject={},
        quality=QualityReport(
            status="warn",
            checks=[
                QualityCheck(
                    name="camera_motion",
                    status="warn",
                    value=3.1,
                    threshold=2.0,
                    message="camera moves",
                )
            ],
        ),
        primary=m,
        metrics=[m],
        events={"reps": [], "fps": 30.0},
        methods=["Pose: test"],
        **extra,  # type: ignore[arg-type]
    )
    return b


def test_render_minimal_bundle() -> None:
    html = render_html(_bundle(), "sts.html.j2")
    assert "12.34 s ± 0.07" in html
    assert "camera moves" in html
    assert "Not a medical device" in html
    assert 'class="banner warn"' in html


# [REVIEW] regression tests for the default report template and `ptv report`
def test_protocol_without_report_section_renders(tmp_path: Path) -> None:
    text = resources.files("ptvision.clinical.protocols").joinpath("sts_5x.toml").read_text()
    head, _, tail = text.partition("[report]")
    rest = tail.split("\n[", 1)
    toml = tmp_path / "custom.toml"
    toml.write_text(head + ("[" + rest[1] if len(rest) > 1 else ""), encoding="utf-8")
    proto = load_protocol(toml)
    assert "Not a medical device" in render_html(_bundle(), proto.report.template)


def test_ptv_report_uses_recorded_template_and_keeps_json(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "R-1"
    report_json = write_json(run / "report.json", _bundle(template="base.html.j2"))
    before = report_json.read_bytes()
    res = CliRunner().invoke(app, ["report", str(tmp_path), "--run", "R-1"])
    assert res.exit_code == 0, res.output
    html = (run / "report.html").read_text()
    assert "Not a medical device" in html and "<h2>Measurements</h2>" not in html
    assert report_json.read_bytes() == before
