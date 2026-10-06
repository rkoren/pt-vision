"""`ptv patient`, `ptv visit`, `ptv report`: the local data store and report re-rendering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from ptvision.cli.common import console, err, subject_from_flags
from ptvision.config import settings

if TYPE_CHECKING:
    from ptvision.trials.store import DataStore

patient_app = typer.Typer(help="Manage patients in the local data store.", no_args_is_help=True)
visit_app = typer.Typer(help="Manage visits.", no_args_is_help=True)


def _store() -> DataStore:
    from ptvision.trials.store import DataStore

    return DataStore(settings().data_dir)


@patient_app.command("new")
def patient_new(
    label: Annotated[str, typer.Option(help="Pseudonymous label (never a real name).")],
    patient_id: Annotated[str | None, typer.Option(help="Custom id (default generated).")] = None,
    age: Annotated[int | None, typer.Option()] = None,
    height_m: Annotated[float | None, typer.Option()] = None,
    sex: Annotated[str | None, typer.Option(help="f | m | other")] = None,
    notes: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create a patient record"""
    from ptvision.trials.models import Patient, Subject
    from ptvision.trials.store import new_id, now_iso

    st = _store()
    pid = patient_id or new_id("P")
    p = Patient(
        patient_id=pid,
        label=label,
        created_at=now_iso(),
        subject=subject_from_flags(age, height_m, sex) or Subject(),
        notes=notes,
    )
    st.create_patient(p)
    console.print(f"created patient [bold]{pid}[/] ({label}) in {st.root}")


@patient_app.command("list")
def patient_list() -> None:
    st = _store()
    t = Table("patient_id", "label", "age", "height m", "sex", "visits")
    for p in st.list_patients():
        t.add_row(
            p.patient_id,
            p.label,
            str(p.subject.age_years or ""),
            str(p.subject.height_m or ""),
            p.subject.sex or "",
            str(len(st.list_visits(p.patient_id))),
        )
    console.print(t)
    console.print(f"data_dir {st.root}")


@visit_app.command("new")
def visit_new(
    patient: Annotated[str, typer.Option(help="Patient id.")],
    date: Annotated[str | None, typer.Option(help="ISO date (default today).")] = None,
    episode: Annotated[
        str | None, typer.Option(help="Episode id/label this visit belongs to.")
    ] = None,
    notes: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create a visit for a patient."""
    from datetime import date as _date

    from ptvision.trials.models import Visit
    from ptvision.trials.store import new_id

    st = _store()
    vid = new_id("V")
    v = Visit(
        visit_id=vid,
        patient_id=patient,
        date=date or _date.today().isoformat(),
        episode_id=episode,
        notes=notes,
    )
    st.create_visit(v)
    console.print(f"created visit [bold]{vid}[/] for patient {patient} on {v.date}")


@visit_app.command("list")
def visit_list(patient: Annotated[str, typer.Option(help="Patient id.")]) -> None:
    st = _store()
    t = Table("visit_id", "date", "episode", "trials")
    for v in st.list_visits(patient):
        t.add_row(
            v.visit_id, v.date, v.episode_id or "", str(len(st.list_trials(patient, v.visit_id)))
        )
    console.print(t)


def report(
    trial_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Trial directory.")
    ],
    run: Annotated[str | None, typer.Option(help="Run id (default latest).")] = None,
) -> None:
    """Re-render report.html from an existing run's report.json."""
    from ptvision.files import read_json
    from ptvision.report.bundle import ReportBundle, build_report
    from ptvision.trials.store import RunDir, TrialDir

    trial = TrialDir(trial_dir)
    rd = RunDir(trial.runs_dir / run) if run else trial.latest_run()
    if rd is None or not (rd.path / "report.json").exists():
        err.print("[red]no report.json found for that run[/]")
        raise typer.Exit(1)
    bundle = ReportBundle.model_validate(read_json(rd.path / "report.json"))
    template = "sts.html.j2"
    _, html = build_report(bundle, rd.path, template)
    console.print(f"re-rendered {html}")
