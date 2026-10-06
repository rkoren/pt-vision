"""Shared console objects and the progress bar used by every `ptv` command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from ptvision.trials.models import Subject

console = Console()
err = Console(stderr=True)


def progress_bar() -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def subject_from_flags(age: int | None, height_m: float | None, sex: str | None) -> Subject | None:
    from ptvision.trials.models import Subject

    if age is None and height_m is None and sex is None:
        return None
    return Subject.model_validate({"age_years": age, "height_m": height_m, "sex": sex})
