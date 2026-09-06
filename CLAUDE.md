# pt-vision

Open computer-vision measurement tools to assist physical therapists (gait, functional tests, range of
motion) from ordinary phone video. **Not a medical device**: no diagnostic or disease language anywhere in
code, docs, reports, or UI. Measurements are reported against cited reference values with their known
error; interpretation is the clinician's. Video is processed locally and never leaves the machine.

## Commands

```sh
uv sync --extra app --group dev --group qt      # full dev install (Qt app + Qt tests)
uv run ruff check . && uv run ruff format .     # lint + format (line length 100)
uv run mypy src                                 # strict on clinical/ and data/
uv run pytest -m "not model"                    # fast suite (no weights needed)
uv run ptv models pull --mode lightweight && uv run pytest    # full suite
uv run ptv pose clip.mov                        # keypoints -> Parquet + overlay
uv run ptv analyze clip.mov --protocol sts_5x   # clinical protocol -> report
uv run ptv app [clip.mov | trial_dir]           # desktop viewer
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
- Decisions that are not obvious from the code go in `docs/decisions/NNNN-*.md` (ADRs).
- Capture facts live in `capture.json` (immutable per trial); every processing run writes its own
  `runs/<run_id>/provenance.json`. Never overwrite a previous run.
- Never commit patient video or identifiable data. Test fixtures come from BSD-licensed demo clips or
  synthetic generators (`tests/synthetic.py`).
- Commit only when asked. Comment large code changes or new files with "[REVIEW]" and avoid committing with that tag present

## Branch workflow

Major changes happen on branches the user creates; Claude never creates branches, commits, or pushes.
1. Agree on the next major change (usually a backlog item, after its pick-up assessment).
2. The user creates the working branch and says so.
3. Claude drafts the change as the contents of a first commit and tests it (new files and large
   changes carry a `[REVIEW]` comment for the user to inspect).
4. The user reviews, removes `[REVIEW]` tags, commits, and opens a PR; follow-up fixes are drafted the
   same way on the same branch.
5. The user merges, `main` is pulled, and the loop repeats. Small doc-only edits may still go to `main`
   at the user's discretion.

## Backlog practice

`BACKLOG.md` at the repo root is the single list of things we are not doing right now.

- **Adding**: whenever an idea, follow-up, or reasonable next step comes up that we are not tackling in the
  current task, add it as a thorough entry: what, why, evidence or references, dependencies, rough effort,
  date added, and a pick-up check (what would make it stale). Be generous; good ideas are cheap to keep.
- **Picking up**: before starting an item, re-assess it critically. Is it still the right thing? Has the
  landscape (models, libraries, licenses) or our codebase moved? Do its dependencies exist? Then either do
  it, rewrite it, or move it to Dropped with a one-line reason. Never work backwards from a stale entry.
- **Closing a task**: add any next steps discovered during the task, and move finished items to Done.

## Milestone checks

PTs could run this on Windows or mac. At every milestone (a slice finished, the ffmpeg/distribution
decision implemented, before a PT hands-on session, and after any dependency bump or change to
`io/video.py`, `pose/models.py`, or `app/`), do a fresh install on a Windows machine following
`docs/portability/windows-log.md` and record the result in its log.

## Current Roadmap

Slice 0 (pose CLI), Slice 1 (Five-Times Sit-to-Stand) and the desktop viewer (`ptv app`) are done.
Next: confirm angle bands with the PT partner (B02), record real sit-to-stand clips (B03), the Windows
M1 baseline check (B49), then Slice 2 gait timing (Zeni 2008 events), Slice 3 range of motion, Slice 4
Timed Up and Go and longitudinal comparison. The approved plan lives at
`~/.claude/plans/new-project-here-we-re-bubbly-treasure.md`; evidence for the slice order is in
`docs/decisions/0005-metric-schema-wellness-framing.md`.
