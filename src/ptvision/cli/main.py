"""`ptv` command line"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from ptvision._version import __version__
from ptvision.config import settings

if TYPE_CHECKING:
    from ptvision.data.models import Subject
    from ptvision.data.store import DataStore

app = typer.Typer(
    name="ptv",
    help=(
        "ptvision: measurement tools for physical therapists from ordinary video. "
        "Not a medical device."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
models_app = typer.Typer(help="Manage ONNX model weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")
console = Console()
err = Console(stderr=True)


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


@app.callback()
def _root() -> None:
    pass


@app.command()
def version() -> None:
    """Print version and environment summary."""
    import onnxruntime as ort

    s = settings()
    console.print(f"ptvision {__version__}")
    console.print(
        f"python {sys.version.split()[0]}  onnxruntime {ort.__version__}  "
        f"providers {ort.get_available_providers()}"
    )
    console.print(f"models_dir {s.models_dir}")
    console.print(f"data_dir   {s.data_dir}")


@models_app.command("pull")
def models_pull(
    mode: Annotated[
        str, typer.Option(help="lightweight | balanced | performance | all")
    ] = "balanced",
) -> None:
    """Download and verify weights for a mode."""
    from ptvision.pose.models import MODE_NAMES, ModelManager

    mm = ModelManager()
    modes = list(MODE_NAMES) if mode == "all" else [mode]
    for m in modes:
        mp = mm.ensure_mode(m)
        for r in (mp.det, mp.pose):
            pinned = "pinned" if r.spec.sha256 else "unpinned"
            console.print(
                f"[green]ok[/] {m:12s} {r.spec.name:22s} "
                f"{r.onnx_path.stat().st_size / 1e6:6.1f} MB  {pinned}  "
                f"archive sha256 {r.archive_sha256}"
            )
    console.print(f"models_dir {mm.models_dir}")


@models_app.command("list")
def models_list() -> None:
    """Show which weights are cached."""
    from ptvision.pose.models import ModelManager

    mm = ModelManager()
    t = Table("name", "kind", "available", "size MB", "pinned", "path")
    for row in mm.status():
        t.add_row(
            str(row["name"]),
            str(row["kind"]),
            "[green]yes[/]" if row["available"] else "[red]no[/]",
            "" if row["size_mb"] is None else str(row["size_mb"]),
            "yes" if row["pinned"] else "no",
            str(row["path"]),
        )
    console.print(t)
    console.print(f"models_dir {mm.models_dir}")


@app.command()
def pose(
    video: Annotated[
        Path,
        typer.Argument(
            exists=True, help="Source video (any format ffmpeg reads) or trial directory."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output trial directory (default ./ptv_out/<clip>)."),
    ] = None,
    mode: Annotated[str | None, typer.Option(help="lightweight | balanced | performance")] = None,
    device: Annotated[str | None, typer.Option(help="cpu | mps (CoreML) | cuda")] = None,
    backend: Annotated[str | None, typer.Option(help="onnxruntime | opencv | openvino")] = None,
    det_frequency: Annotated[
        int | None, typer.Option(help="Run the detector every N frames.")
    ] = None,
    max_height: Annotated[
        int | None, typer.Option(help="Downscale to this height during ingest.")
    ] = None,
    overlay: Annotated[bool, typer.Option(help="Render an overlay video.")] = True,
    reuse: Annotated[
        bool, typer.Option(help="Reuse cached keypoints when the model matches.")
    ] = True,
) -> None:
    """Extract Halpe-26 keypoints from a video into Parquet (+ confidence-colored overlay video)."""
    from ptvision.io.video import require_ffmpeg
    from ptvision.pipeline import AnalyzeOptions, pose_only

    require_ffmpeg()
    opts = AnalyzeOptions(
        out_dir=out,
        mode=mode,
        device=device,
        backend=backend,
        det_frequency=det_frequency,
        max_height=max_height,
        overlay=overlay,
        reuse_pose=reuse,
    )
    with _progress() as prog:
        task = prog.add_task("starting", total=None)

        def on_progress(done: int, total: int | None) -> None:
            prog.update(task, completed=done, total=total)

        def on_stage(name: str) -> None:
            prog.update(task, description=name, completed=0, total=None)

        result = pose_only(video, opts, progress=on_progress, on_stage=on_stage)

    capture = result.run.path.parent.parent  # trial dir
    cap = (
        __import__("ptvision.data.store", fromlist=["TrialDir"])
        .TrialDir(capture)
        .read_capture()
        .cameras[0]
    )
    console.print(
        f"[bold]ingest[/] {cap.source_codec} {cap.source_width}x{cap.source_height} "
        f"@ {cap.source_fps_avg:.2f} fps{' VFR' if cap.source_is_vfr else ''}"
        f"{f', rotated {cap.rotation_applied_deg}°' if cap.rotation_applied_deg else ''} "
        f"-> {cap.width}x{cap.height} @ {cap.fps:.2f} fps, {cap.n_frames} frames"
    )
    tr = result.track
    fps_proc = tr.n_frames / result.seconds if result.seconds > 0 else float("inf")
    console.print(
        f"[bold]pose[/] {tr.n_frames} frames, {tr.n_persons} person(s), "
        f"primary id {result.primary_person}"
        + (
            " (reused cached keypoints)"
            if result.reused
            else f", {result.seconds:.1f} s ({fps_proc:.1f} fps)"
        )
    )
    if tr.n_persons and result.primary_person is not None:
        slot = int(list(tr.person_ids).index(result.primary_person))
        present = tr.present()[:, slot].mean()
        mean_score = float(tr.score[:, slot][tr.present()[:, slot]].mean()) if present > 0 else 0.0
        console.print(
            f"       primary present in {present:.0%} of frames, "
            f"mean keypoint score {mean_score:.2f}"
        )
    console.print(f"[bold]out[/]  {result.parquet_path}")
    if result.overlay_path:
        console.print(f"      {result.overlay_path}")
    console.print(f"      {result.run.provenance_path}")


@app.command()
def bench(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    frames: Annotated[int, typer.Option(help="Number of frames to time.")] = 150,
    modes: Annotated[str, typer.Option(help="Comma-separated modes.")] = "lightweight,balanced",
    devices: Annotated[str, typer.Option(help="Comma-separated devices.")] = "cpu",
    det_frequencies: Annotated[
        str, typer.Option(help="Comma-separated detector intervals.")
    ] = "1,4",
) -> None:
    """Time pose extraction per (mode, device, det_frequency) and print a Markdown table."""
    from ptvision.io.video import iter_frames, probe
    from ptvision.pose.rtmlib_backend import RtmlibBackend

    info = probe(video)
    buf = []
    for idx, frame in iter_frames(video):
        if idx >= frames:
            break
        buf.append((idx, frame))
    n = len(buf)
    size = (info.width, info.height)
    rows = []
    for mode in modes.split(","):
        for device in devices.split(","):
            for df in det_frequencies.split(","):
                try:
                    with console.status(f"{mode} / {device} / det every {df}"):
                        be = RtmlibBackend(
                            mode=mode.strip(), device=device.strip(), det_frequency=int(df)
                        )
                        be.estimate(iter(buf[:5]), fps=info.fps, image_size=size)  # warm-up
                        t0 = time.perf_counter()
                        tr = be.estimate(iter(buf), fps=info.fps, image_size=size)
                        dt = time.perf_counter() - t0
                    rows.append(
                        (mode, device, df, ",".join(be.info.providers), n / dt, tr.n_persons)
                    )
                    console.print(
                        f"{mode:12s} {device:5s} det/{df:<3s} {n / dt:6.1f} fps  "
                        f"providers {be.info.providers}"
                    )
                except Exception as e:
                    rows.append((mode, device, df, f"error: {e}", float("nan"), 0))
                    err.print(f"[red]{mode} {device} det/{df} failed:[/] {e}")
    console.print()
    console.print(
        f"Benchmark: `{video.name}` {size[0]}x{size[1]}, {n} frames, "
        f"python {sys.version.split()[0]}"
    )
    console.print()
    console.print("| mode | device | det every | providers | fps | persons |")
    console.print("|---|---|---|---|---|---|")
    for mode, device, df, prov, fps, persons in rows:
        console.print(f"| {mode} | {device} | {df} | {prov} | {fps:.1f} | {persons} |")


# ---------------------------------------------------------------------------------------------
# Protocol analysis
# ---------------------------------------------------------------------------------------------
def _subject_from_flags(age: int | None, height_m: float | None, sex: str | None) -> Subject | None:
    from ptvision.data.models import Subject

    if age is None and height_m is None and sex is None:
        return None
    return Subject.model_validate({"age_years": age, "height_m": height_m, "sex": sex})


@app.command()
def analyze(
    source: Annotated[
        Path, typer.Argument(exists=True, help="Source video, or an existing trial directory.")
    ],
    protocol: Annotated[
        str,
        typer.Option(
            "--protocol", "-p", help="Protocol id (see `ptv protocols list`) or a TOML path."
        ),
    ] = "sts_5x",
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Output trial directory for anonymous runs.")
    ] = None,
    patient: Annotated[str | None, typer.Option(help="Patient id in the data store.")] = None,
    visit: Annotated[str | None, typer.Option(help="Visit id in the data store.")] = None,
    person: Annotated[
        int | None, typer.Option(help="Track id of the subject (default: automatic).")
    ] = None,
    t0: Annotated[
        float | None, typer.Option(help="Manual test start time in seconds (overrides start_rule).")
    ] = None,
    age: Annotated[
        int | None, typer.Option(help="Subject age in years (for reference bands).")
    ] = None,
    height_m: Annotated[float | None, typer.Option(help="Subject height in metres.")] = None,
    sex: Annotated[str | None, typer.Option(help="f | m | other")] = None,
    mode: Annotated[str | None, typer.Option(help="lightweight | balanced | performance")] = None,
    device: Annotated[str | None, typer.Option(help="cpu | mps | cuda")] = None,
    det_frequency: Annotated[int | None, typer.Option()] = None,
    overlay: Annotated[bool, typer.Option(help="Render an overlay video.")] = True,
    reuse_pose: Annotated[
        bool, typer.Option(help="Reuse cached keypoints when the model matches.")
    ] = True,
    color_by: Annotated[
        str | None,
        typer.Option(
            help="Overlay coloring: rules | confidence (default rules if the protocol has any)."
        ),
    ] = None,
    max_height: Annotated[
        int | None, typer.Option(help="Downscale to this height during ingest (e.g. 1080 for 4K).")
    ] = None,
) -> None:
    """Run a clinical protocol on a video and write a report."""
    from ptvision.io.video import require_ffmpeg
    from ptvision.pipeline import AnalyzeOptions
    from ptvision.pipeline import analyze as _analyze

    require_ffmpeg()
    if (patient is None) != (visit is None):
        err.print("[red]--patient and --visit must be given together[/]")
        raise typer.Exit(2)
    opts = AnalyzeOptions(
        protocol=protocol,
        out_dir=out,
        patient_id=patient,
        visit_id=visit,
        person=person,
        mode=mode,
        device=device,
        det_frequency=det_frequency,
        overlay=overlay,
        reuse_pose=reuse_pose,
        manual_start_s=t0,
        subject=_subject_from_flags(age, height_m, sex),
        color_by=color_by,  # type: ignore[arg-type]
        max_height=max_height,
    )
    if color_by not in (None, "rules", "confidence"):
        err.print("[red]--color-by must be rules or confidence[/]")
        raise typer.Exit(2)
    with _progress() as prog:
        task = prog.add_task("starting", total=None)

        def on_progress(done: int, total: int | None) -> None:
            prog.update(task, completed=done, total=total)

        def on_stage(name: str) -> None:
            prog.update(task, description=name, completed=0, total=None)

        try:
            res = _analyze(source, opts, progress=on_progress, on_stage=on_stage)
        except RuntimeError as e:
            err.print(f"[red]analysis stopped:[/] {e}")
            raise typer.Exit(1) from e

    q = res.quality
    color = {"pass": "green", "warn": "yellow", "fail": "red"}[q.status]
    console.print(f"[bold]quality[/] [{color}]{q.status}[/]" + ("" if q.status == "pass" else ":"))
    for c in q.checks:
        if c.status != "pass":
            console.print(
                f"   [{'yellow' if c.status == 'warn' else 'red'}]{c.status}[/] "
                f"{c.name}: {c.message}"
            )
    console.print("[bold]metrics[/]")
    for m in res.metrics:
        side = f" ({m.side})" if m.side else ""
        console.print(f"   {m.label}{side}: [bold]{m.formatted()}[/]  tier {m.tier}")
    for w in res.warnings:
        console.print(f"   [yellow]note[/] {w}")
    console.print(f"[bold]report[/] {res.report_html}")
    console.print(f"       {res.report_json}")


protocols_app = typer.Typer(help="List and inspect clinical protocols.", no_args_is_help=True)
app.add_typer(protocols_app, name="protocols")


@protocols_app.command("list")
def protocols_list() -> None:
    from ptvision.clinical.protocol import builtin_protocol_ids, load_protocol

    t = Table("id", "version", "name", "view", "primary metric")
    for pid in builtin_protocol_ids():
        p = load_protocol(pid)
        prim = p.primary_metric()
        t.add_row(p.id, p.version, p.protocol.name, p.capture.view, prim.name if prim else "")
    console.print(t)


@protocols_app.command("show")
def protocols_show(
    protocol: Annotated[str, typer.Argument(help="Protocol id or TOML path.")],
) -> None:
    from ptvision.clinical import registry
    from ptvision.clinical.protocol import load_protocol

    p = load_protocol(protocol)
    console.print(f"[bold]{p.protocol.name}[/]  ({p.id} v{p.version})")
    console.print(p.protocol.summary)
    console.print(
        f"\n[bold]capture[/] view={p.capture.view} distance={p.capture.distance_m} m "
        f"camera height={p.capture.camera_height_m} m min fps={p.capture.min_fps}"
    )
    console.print(p.capture.instructions)
    console.print(f"\n[bold]segmenter[/] {p.segmenter.name} {p.segmenter.params}")
    t = Table("metric", "label", "tier", "units", "primary", "norm", "citation")
    for m in p.metrics:
        e = registry.get_metric(m.name)
        t.add_row(
            m.name,
            e.label,
            str(e.tier),
            e.units,
            "yes" if m.primary else "",
            m.norm or "",
            e.citation,
        )
    console.print(t)


# ---------------------------------------------------------------------------------------------
# Data store: patients, visits, trials
# ---------------------------------------------------------------------------------------------
patient_app = typer.Typer(help="Manage patients in the local data store.", no_args_is_help=True)
visit_app = typer.Typer(help="Manage visits.", no_args_is_help=True)
app.add_typer(patient_app, name="patient")
app.add_typer(visit_app, name="visit")


def _store() -> DataStore:
    from ptvision.data.store import DataStore

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
    """Create a patient record."""
    from ptvision.data.ids import new_id, now_iso
    from ptvision.data.models import Patient, Subject

    st = _store()
    pid = patient_id or new_id("P")
    p = Patient(
        patient_id=pid,
        label=label,
        created_at=now_iso(),
        subject=_subject_from_flags(age, height_m, sex) or Subject(),
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

    from ptvision.data.ids import new_id
    from ptvision.data.models import Visit

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


@app.command()
def report(
    trial_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Trial directory.")
    ],
    run: Annotated[str | None, typer.Option(help="Run id (default latest).")] = None,
) -> None:
    """Re-render report.html from an existing run's report.json."""
    from ptvision.data.store import RunDir, TrialDir
    from ptvision.io.jsonio import read_json
    from ptvision.report.build import build_report
    from ptvision.report.schema import ReportBundle

    trial = TrialDir(trial_dir)
    rd = RunDir(trial.runs_dir / run) if run else trial.latest_run()
    if rd is None or not (rd.path / "report.json").exists():
        err.print("[red]no report.json found for that run[/]")
        raise typer.Exit(1)
    bundle = ReportBundle.model_validate(read_json(rd.path / "report.json"))
    template = "sts.html.j2"
    _, html = build_report(bundle, rd.path, template)
    console.print(f"re-rendered {html}")


@app.command("app")
def app_cmd(
    source: Annotated[
        Path | None,
        typer.Argument(exists=True, help="Video to process, or a trial folder to open."),
    ] = None,
    protocol: Annotated[
        str, typer.Option("--protocol", "-p", help="pose | sts_5x | sts_30s | TOML path")
    ] = "pose",
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Output trial directory for a video.")
    ] = None,
    run: Annotated[str | None, typer.Option(help="Run id to open (default latest).")] = None,
) -> None:
    """Open the desktop viewer (drop a video in, watch progress, play with the skeleton overlay)."""
    try:
        from ptvision.app.main import main as app_main
    except ImportError as e:
        err.print(
            f"[red]desktop app dependencies missing:[/] {e}\ninstall with `uv sync --extra app`"
        )
        raise typer.Exit(1) from e
    try:
        from ptvision.io.video import require_ffmpeg

        require_ffmpeg()
    except Exception as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    raise typer.Exit(app_main(source, protocol=protocol, out_dir=out, run_id=run))


if __name__ == "__main__":
    app()
