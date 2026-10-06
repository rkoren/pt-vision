# Public datasets

`ptv datasets` pulls public motion-capture and video datasets with checksums and provenance, and
scores our algorithms against their ground truth. Nothing here is ever committed; files land under
`~/ptvision-data/datasets/<name>/` (`PTV_DATASETS_DIR` to move them) with `raw/` (as downloaded and
extracted), `derived/` (our evaluation outputs), and `PROVENANCE.json` (what was fetched, when, and
its hash). The manifest with every URL, checksum, and license is
`src/ptvision/datasets/manifest.toml`.

```sh
uv run ptv datasets list                       # what exists, sizes, licenses, what is on disk
uv run ptv datasets pull uiprmd                # download + verify + extract one dataset
uv run ptv datasets pull uiprmd
uv sync --extra datasets                       # ezc3d, needed only for the eval-* commands
uv run ptv datasets eval-sts                   # UI-PRMD sit-to-stand segmentation check
```

Evaluations use the mocap markers directly (a virtual side camera projects them into our keypoint
layout), so they need no model weights and run in minutes. What they measure and the current
numbers are in `docs/validation/uiprmd-sts.md`. The gait-event harness (`eval-gait`) was removed
with the gait protocol for the first release and returns with it.

## Automatic pulls

| name | what | size | license | used for |
|---|---|---|---|---|
| `uiprmd` | 10 subjects, 10 rehabilitation movements incl. sit-to-stand, Vicon 39 markers | 0.4 GB | PDDL | sit-to-stand segmentation |
| `comfi` | 18 participants (10 m, 8 f), 24 tasks incl. sit-to-stand, squatting, straight and circular walking; mocap + 3 force plates + camera parameters | 16 GB | CC BY 4.0 | sit-to-stand and gait against force-plate and mocap events |
| `comfi-video` | the six COMFI video archives: 4 synchronised cameras per task with per-frame timestamps, 1,696 videos | 56 GB | CC BY 4.0 | real-video validation of the pose model, held out |

Sizes are the download; extracted data roughly doubles them (COMFI with video: 147 GB on disk).

## Manual downloads

These need a login or an email and cannot be scripted. Put them where shown and add an `ACCESS.md`
next to the files saying who downloaded them, when, and under which terms.

- **OpenCap validation set** (Apache 2.0): walking, sit-to-stand, and more with synchronized phone
  video and force plates. Create a SimTK account, download `LabValidation_withVideos.zip` (19 GB)
  from the OpenCap project page, and unzip into `~/ptvision-data/datasets/opencap/raw/`.
- **KIMORE** (rehabilitation exercises with clinician scores): request by email from the authors;
  store under `~/ptvision-data/datasets/kimore/raw/`.

## Adding a dataset

Add an entry to `manifest.toml` with the host's exact byte size and md5/sha256 (Zenodo and Figshare
expose them in their APIs), the license you are relying on, and a `kind`. Write a loader under
`src/ptvision/datasets/` that yields a `PoseTrack` plus ground-truth events, and an evaluation that
writes `derived/<eval>/summary.json`. Only permissive or public-domain data belongs here; anything
research-only stays in `experiments/`.
