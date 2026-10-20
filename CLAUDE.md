# pt-vision

Computer-vision measurement tools to assist physical therapists (sit-to-stand tests today, gait and
range of motion later) from phone video

## Commands

```sh
uv sync --extra app --extra datasets            # the one install (dev + qt groups are default)
uv run ruff check . && uv run ruff format .     # lint + format (line length 100)
uv run mypy src                                 # strict on clinical/ and trials/
uv run pytest -m "not model"                    # fast suite (no weights needed)
uv run ptv models pull --mode lightweight && uv run pytest    # full suite
uv run ptv analyze clip.mov --protocol sts_5x   # clinical protocol -> report
uv run ptv app [clip.mov | trial_dir]           # desktop viewer
uv run ptv datasets eval-video --dataset opencap   # video vs lab mocap (manual download)
```

## Conventions

- Python 3.12 via `uv` (3.13 OK). Pin versions in `uv.lock`. `onnxruntime-gpu` clashes with
  `onnxruntime` on the module name, so CPU inference is the default.
- Halpe-26 for the keypoint layout (`ptvision.pose.layout.HALPE26`).
- Each metric registered with `@register_metric(name, version, tier, label, units, citation)`; tier-1
  metrics must cite validation. Protocols are TOML in `clinical/protocols/`. Norm tables are CSV with citations.
- Project license: Unlicense (public domain) for original code; vendored/adapted third-party code keeps
  its own license and is listed in `NOTICE`. Dependencies in `src/` must be permissive (Apache/MIT/BSD);
  `tests/test_license_firewall.py` enforces the research-asset ban.
- Vendored code goes in `src/ptvision/_vendor/<project>/` with its LICENSE and a `PROVENANCE.md`
  (upstream commit, files, local modifications).
- Capture facts live in `capture.json` (immutable per trial); every processing run writes its own
  `runs/<run_id>/provenance.json`. Never overwrite a previous run.
- Never commit patient video or identifiable data. Test fixtures come from BSD-licensed demo clips or
  synthetic generators (`tests/synthetic.py`).
- Validation evidence lives in `notebooks/validation.ipynb`, with tables computed from evaluation
  outputs, never typed. Evaluation datasets are held out: no pipeline parameter is tuned on them.
- Commit only when asked. Comment large code changes or new files with "[REVIEW]" and nudge a review
  with that tag present. Work items are GitHub issues; `CONTRIBUTING.md` has the workflow.

## Milestone checks

PTs should be able to run this on their laptop. At every milestone, do a fresh install and record
the results.
