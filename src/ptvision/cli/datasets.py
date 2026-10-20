"""`ptv datasets`: pull public validation datasets and run the evaluations"""

from __future__ import annotations

import time
from typing import Annotated, Any

import typer
from rich.table import Table

from ptvision.cli.common import console, err, progress_bar

datasets_app = typer.Typer(
    help="Public validation datasets: pull, list, evaluate.", no_args_is_help=True
)


@datasets_app.command("list")
def datasets_list() -> None:
    """Show known datasets, licenses, and what is on disk."""
    from ptvision.datasets.registry import datasets_root, load_manifest, status

    t = Table("name", "kind", "license", "files on disk", "size GB")
    t.columns[0].no_wrap = True
    for spec in load_manifest().values():
        st = status(spec)
        t.add_row(
            str(st["name"]),
            str(st["kind"]),
            str(st["license"]),
            str(st["files"]),
            str(st["size_gb"]),
        )
    console.print(t)
    console.print(
        f"root {datasets_root()}  (PTV_DATASETS_DIR); `ptv datasets pull <name>`", soft_wrap=True
    )


@datasets_app.command("pull")
def datasets_pull(
    names: Annotated[list[str], typer.Argument(help="Dataset names (see `ptv datasets list`).")],
    verify: Annotated[bool, typer.Option(help="Verify md5 checksums.")] = True,
) -> None:
    """Download, verify, extract, and record provenance (license, checksums, date)."""
    from ptvision.datasets.registry import load_manifest, pull

    specs = load_manifest()
    for name in names:
        if name not in specs:
            err.print(f"[red]unknown dataset {name!r}[/]; known: {', '.join(specs)}")
            raise typer.Exit(2)
    with progress_bar() as prog:
        task = prog.add_task("starting", total=None)
        plain: dict[str, Any] = {"label": "", "last_pct": -1, "last_t": 0.0, "t0": 0.0, "d0": 0}

        def on_progress(label: str, done: int, total: int | None) -> None:
            prog.update(task, description=label, completed=done, total=total)
            if console.is_terminal:
                return

            now = time.monotonic()
            if label != plain["label"]:
                plain.update(label=label, last_pct=-1, last_t=now, t0=now, d0=done)
            pct = int(100 * done / total) if total else -1
            due = pct // 5 != int(plain["last_pct"]) // 5 or now - float(plain["last_t"]) >= 30
            if not due:
                return
            rate = (done - int(plain["d0"])) / max(now - float(plain["t0"]), 1e-3) / 1e6
            size = (
                f"{done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)"
                if total
                else f"{done / 1e6:.1f} MB"
            )
            console.print(f"{label}: {size} {rate:.2f} MB/s", soft_wrap=True)
            plain.update(last_pct=pct, last_t=now)

        for name in names:
            spec = specs[name]
            if spec.manual:
                res = pull(spec)
                console.print(
                    f"[bold]{name}[/] is a manual download. {spec.notes}\n"
                    f"  source: {spec.source}\n  place it under: {res.dirs.raw}"
                )
                continue
            res = pull(spec, verify=verify, progress=on_progress)
            console.print(
                f"[bold]{name}[/] ({res.spec.license}): downloaded {len(res.downloaded)}, "
                f"already present {len(res.skipped)}, extracted {len(res.extracted)} "
                f"-> {res.dirs.root}"
            )


@datasets_app.command("eval-sts")
def datasets_eval_sts(
    limit: Annotated[int | None, typer.Option(help="Only the first N episodes.")] = None,
) -> None:
    """Run the sit-to-stand segmenter on UI-PRMD m05 episodes (each is exactly one repetition)."""
    from ptvision.datasets.uiprmd import evaluate_sts

    results, out_dir = evaluate_sts(limit=limit)
    n = len(results)
    one = sum(1 for r in results if r.reps == 1)
    console.print(
        f"uiprmd m05: {n} episodes, exactly one rise detected in {one} ({one / max(n, 1):.0%})"
    )
    n_zero = sum(1 for r in results if r.reps == 0)
    n_multi = sum(1 for r in results if r.reps > 1)
    n_err = sum(1 for r in results if r.reps < 0)
    console.print(f"  zero: {n_zero}  multiple: {n_multi}  errors: {n_err}")
    console.print(f"[bold]written[/] {out_dir}")


@datasets_app.command("eval-video")
def datasets_eval_video(
    dataset: Annotated[str, typer.Option(help="Dataset name (opencap).")] = "opencap",
    cameras: Annotated[
        str | None, typer.Option(help="Comma-separated camera indices; default all.")
    ] = None,
    subjects: Annotated[
        str | None, typer.Option(help="Comma-separated subject ids; default all.")
    ] = None,
    tasks: Annotated[
        str | None, typer.Option(help="Comma-separated task names; default the sit-to-stand ones.")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N trials.")] = None,
    mode: Annotated[
        str | None, typer.Option(help="Pose model mode (lightweight|balanced); default protocol.")
    ] = None,
    reuse: Annotated[bool, typer.Option(help="Reuse cached keypoints from a previous run.")] = True,
) -> None:
    """Run our video pipeline on lab-validation videos and score it against the lab's marker
    mocap (angle RMSE, bias, limits of agreement) and force plates (sit-to-stand timing)."""
    from ptvision.cli.main import _finish_after_inference
    from ptvision.datasets.video_eval import DATASETS, evaluate_video

    if dataset not in DATASETS:
        err.print(f"[red]unknown dataset {dataset!r}[/]; one of {', '.join(DATASETS)}")
        raise typer.Exit(2)

    def _split(s: str | None) -> tuple[str, ...] | None:
        return tuple(x.strip() for x in s.split(",") if x.strip()) if s else None

    cams = tuple(int(c) for c in _split(cameras) or ()) or None

    def on_progress(label: str, done: int, total: int) -> None:
        console.print(f"[{done + 1}/{total}] {label}", soft_wrap=True)

    results, out_dir = evaluate_video(
        dataset,
        cameras=cams,
        subjects=_split(subjects),
        tasks=_split(tasks),
        limit=limit,
        mode=mode,
        reuse=reuse,
        progress=on_progress,
    )
    t = Table(
        "camera", "view", "n", "knee RMSE", "hip RMSE", "trunk RMSE", "seat-off |err|", "span err"
    )
    for cam in sorted({r.camera for r in results}):
        rs = [r for r in results if r.camera == cam and r.status == "ok"]

        def med(key: str, rs: list = rs) -> str:  # type: ignore[type-arg]
            v = [r.metrics.get(key) for r in rs]
            v = [x for x in v if x is not None]
            return f"{sorted(v)[len(v) // 2]:.1f}" if v else "-"

        so = [r.seat_off_abs_err_ms_mean for r in rs if r.seat_off_abs_err_ms_mean is not None]
        sp = [r.span_err_s for r in rs if r.span_err_s is not None]
        view = next((r.view_deg for r in results if r.camera == cam), float("nan"))
        t.add_row(
            f"cam{cam}",
            f"{view:.0f}°",
            str(len(rs)),
            med("knee_flexion_mocap_rmse"),
            med("hip_flexion_mocap_rmse"),
            med("trunk_lean_mocap_rmse"),
            f"{sorted(so)[len(so) // 2]:.0f} ms" if so else "-",
            f"{sorted(sp)[len(sp) // 2]:+.2f} s" if sp else "-",
        )
    console.print(t)
    n_err = sum(1 for r in results if r.status != "ok")
    console.print(f"{len(results)} trial-cameras, {n_err} errors (medians over trials per camera)")
    console.print(f"[bold]written[/] {out_dir}")
    _finish_after_inference()
