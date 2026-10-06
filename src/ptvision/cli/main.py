"""`ptv` command line"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from ptvision._version import __version__
from ptvision.cli.common import console, err, progress_bar, subject_from_flags
from ptvision.config import settings

if TYPE_CHECKING:
    pass

app = typer.Typer(
    name="ptv",
    help=("ptvision: measurement tools for physical therapists"),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
models_app = typer.Typer(help="Manage ONNX model weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")


def _register_subcommands() -> None:
    """Sub-apps that live in their own modules (imported here to avoid import cycles)"""
    from ptvision.cli.datasets import datasets_app
    from ptvision.cli.records import patient_app, report, visit_app

    app.add_typer(datasets_app, name="datasets")
    app.add_typer(patient_app, name="patient")
    app.add_typer(visit_app, name="visit")
    app.command("report")(report)


def _finish_after_inference(code: int = 0) -> None:
    """leave the process without interpreter teardown once inference has run"""
    if os.environ.get("PTV_HARD_EXIT", "1") == "0":
        return
    console.file.flush()
    err.file.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


@app.callback()
def _root() -> None:
    pass


@app.command()
def version() -> None:
    """Print version and environment summary"""
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
    """Download and verify weights"""
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
                f"archive sha256 {r.archive_sha256}",
                soft_wrap=True,
            )
    console.print(f"models_dir {mm.models_dir}", soft_wrap=True)


@models_app.command("list")
def models_list() -> None:
    """Show which weights are cached"""
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
            exists=True, help="Source video (any format ffmpeg reads) or trial directory"
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
        int | None,
        typer.Option(
            help="Cap the shorter side of the frame during ingest (1080 turns 4K into 1920x1080 "
            "or 1080x1920)."
        ),
    ] = None,
    overlay: Annotated[bool, typer.Option(help="Render an overlay video.")] = True,
    reuse: Annotated[
        bool, typer.Option(help="Reuse cached keypoints when the model matches.")
    ] = True,
) -> None:
    """Extract Halpe-26 keypoints from a video into Parquet (+ confidence-colored overlay video)"""
    from ptvision.pipeline import AnalyzeOptions, TrialMismatchError, pose_only
    from ptvision.video import require_ffmpeg

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
    with progress_bar() as prog:
        task = prog.add_task("starting", total=None)

        def on_progress(done: int, total: int | None) -> None:
            prog.update(task, completed=done, total=total)

        def on_stage(name: str) -> None:
            prog.update(task, description=name, completed=0, total=None)

        try:
            result = pose_only(video, opts, progress=on_progress, on_stage=on_stage)
        except TrialMismatchError as e:
            err.print(f"[red]{e}[/]")
            raise typer.Exit(1) from e

    capture = result.run.path.parent.parent  # trial dir
    cap = (
        __import__("ptvision.trials.store", fromlist=["TrialDir"])
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
    _finish_after_inference()


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
    from ptvision.pose.rtmlib_backend import RtmlibBackend
    from ptvision.video import iter_frames, probe

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
        int | None,
        typer.Option(
            help="Cap the shorter side of the frame during ingest (1080 turns 4K into 1920x1080 "
            "or 1080x1920)."
        ),
    ] = None,
) -> None:
    """Run a clinical protocol on a video and write a report."""
    from ptvision.pipeline import AnalyzeOptions, TrialMismatchError
    from ptvision.pipeline import analyze as _analyze
    from ptvision.video import require_ffmpeg

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
        subject=subject_from_flags(age, height_m, sex),
        color_by=color_by,  # type: ignore[arg-type]
        max_height=max_height,
    )
    if color_by not in (None, "rules", "confidence"):
        err.print("[red]--color-by must be rules or confidence[/]")
        raise typer.Exit(2)
    with progress_bar() as prog:
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
        except TrialMismatchError as e:
            err.print(f"[red]{e}[/]")
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
    console.print(f"[bold]report[/] {res.report_html}", soft_wrap=True)
    console.print(f"       {res.report_json}", soft_wrap=True)
    _finish_after_inference()


protocols_app = typer.Typer(help="List and inspect clinical protocols.", no_args_is_help=True)
app.add_typer(protocols_app, name="protocols")


@protocols_app.command("list")
def protocols_list() -> None:
    from ptvision.clinical.protocol import builtin_protocol_ids, load_protocol

    t = Table("id", "version", "name", "view", "primary metric")
    t.columns[0].no_wrap = True
    t.columns[4].no_wrap = True
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
def _require_app() -> Callable[..., int]:
    """Import the desktop app and check ffmpeg, with one clear message per missing piece."""
    try:
        from ptvision.app.main import main as app_main
    except ImportError as e:
        err.print(
            f"[red]desktop app dependencies missing:[/] {e}\ninstall with `uv sync --extra app`"
        )
        raise typer.Exit(1) from e
    try:
        from ptvision.video import require_ffmpeg

        require_ffmpeg()
    except Exception as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    return app_main


@app.command()
def demo(
    pose_only: Annotated[
        bool, typer.Option("--pose-only", help="Skeleton and confidence colours only, no protocol.")
    ] = False,
) -> None:
    """Open the viewer on the bundled sample clip (a side-view sit-to-stand)"""
    from ptvision.samples import sample
    from ptvision.trials.models import Subject

    smp = sample()
    app_main = _require_app()
    subject = Subject(height_m=smp.height_m, age_years=smp.age_years, sex=smp.sex)  # type: ignore[arg-type]
    console.print(f"sample: {smp.description}")
    raise typer.Exit(
        app_main(
            smp.path,
            protocol="pose" if pose_only else smp.protocol,
            out_dir=Path("ptv_out") / "sample",
            subject=subject,
        )
    )


@app.command("app")
def app_cmd(
    source: Annotated[
        Path | None,
        typer.Argument(exists=True, help="Video to process, or a trial folder to open."),
    ] = None,
    protocol: Annotated[
        str,
        typer.Option(
            "--protocol",
            "-p",
            help="pose (skeleton only), a built-in protocol id (see `ptv protocols list`), "
            "or a TOML path",
        ),
    ] = "pose",
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Output trial directory for a video.")
    ] = None,
    run: Annotated[str | None, typer.Option(help="Run id to open (default latest).")] = None,
) -> None:
    """Open the desktop viewer (drop a video in, watch progress, play with the skeleton overlay).

    No video yet? `ptv demo` opens the bundled sample clip.
    Headless: PTV_APP_SCREENSHOT=<png> saves the window once the viewer shows, then exits.
    (Set QT_QPA_PLATFORM=offscreen when there is no display.)
    """
    app_main = _require_app()
    raise typer.Exit(app_main(source, protocol=protocol, out_dir=out, run_id=run))


_register_subcommands()


if __name__ == "__main__":
    app()
