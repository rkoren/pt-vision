from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from ptvision.io.jsonio import write_json
from ptvision.report.schema import ReportBundle


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("ptvision.report", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(bundle: ReportBundle, template: str = "sts.html.j2") -> str:
    return _env().get_template(template).render(b=bundle)


def build_report(
    bundle: ReportBundle, run_dir: Path, template: str = "sts.html.j2"
) -> tuple[Path, Path]:
    """Write report.json (the primary artifact) and report.html rendered from it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(run_dir / "report.json", bundle)
    html_path = run_dir / "report.html"
    html_path.write_text(render_html(bundle, template), encoding="utf-8")
    return json_path, html_path
