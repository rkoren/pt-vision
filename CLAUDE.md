# pt-vision

Open computer-vision measurement tools to assist physical therapists (gait, functional tests, range of
motion) from ordinary phone video. **Not a medical device**: no diagnostic or disease language anywhere in
code, docs, reports, or UI. Measurements are reported against cited reference values with their known
error; interpretation is the clinician's. Video is processed locally and never leaves the machine.

## Commands

```sh
uv sync --extra app --group dev --group qt      # full dev install (Qt app + Qt tests)
uv run ruff check . && uv run ruff format .     # lint + format (line length 100)
uv run mypy src                                 # strict on clinical/ and trials/
uv run pytest -m "not model"                    # fast suite (no weights needed)
uv run ptv models pull --mode lightweight && uv run pytest    # full suite
uv run ptv pose clip.mov                        # keypoints -> Parquet + overlay
uv run ptv analyze clip.mov --protocol sts_5x   # clinical protocol -> report
uv run ptv app [clip.mov | trial_dir]           # desktop viewer
uv run ptv datasets pull uiprmd && uv run ptv datasets eval-sts   # public-data harness
```

## Conventions

- Python 3.12 via `uv` (3.13 OK; 3.14 cannot resolve `opensim`). Pin versions in `uv.lock`.
- Halpe-26 for the keypoint layout (`ptvision.pose.layout.HALPE26`)
- Each metric registered with `@register_metric(name, version, tier, label, units, citation)`; tier-1
  metrics must cite validation. Protocols are TOML in `clinical/protocols/`. Norm tables are CSV with citations.
- Project license: Unlicense (public domain) for original code; vendored/adapted third-party code keeps
  its own license and is listed in `NOTICE`. Dependencies in `src/` must be permissive (Apache/MIT/BSD);
  `tests/test_license_firewall.py` enforces the research-asset ban.
  Research-only assets (SMPL family, BEDLAM, GVHMR, HSMR/SKEL) and copyleft code should live only in `experiments/`.
- Vendored code goes in `src/ptvision/_vendor/<project>/` with its LICENSE and a `PROVENANCE.md`
  (upstream commit, files, local modifications).
- Capture facts live in `capture.json` (immutable per trial); every processing run writes its own
  `runs/<run_id>/provenance.json`. Never overwrite a previous run.
- Never commit patient video or identifiable data. Test fixtures come from BSD-licensed demo clips or synthetic generators (`tests/synthetic.py`).
- Commit only when asked. Comment large code changes or new files with "[REVIEW]" and avoid committing with that tag present

## Milestone checks

PTs should be able to run this on their laptop. At every milestone, do a fresh install
and record the results
