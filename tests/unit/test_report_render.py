from ptvision.clinical.metrics.base import Metric
from ptvision.quality.checks import QualityCheck, QualityReport
from ptvision.report.bundle import ReportBundle, render_html


def test_render_minimal_bundle() -> None:
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
    )
    html = render_html(b, "sts.html.j2")
    assert "12.34 s ± 0.07" in html
    assert "camera moves" in html
    assert "Not a medical device" in html
    assert 'class="banner warn"' in html
