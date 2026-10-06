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
            res = pull(specs[name], verify=verify, progress=on_progress)
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
