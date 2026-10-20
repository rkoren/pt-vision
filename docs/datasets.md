# Public datasets

`ptv datasets` pulls public motion-capture datasets with checksums and provenance, and scores our
algorithms against their ground truth. Nothing here is ever committed; files land under
`~/ptvision-data/datasets/<name>/` (`PTV_DATASETS_DIR` to move them) with `raw/` (as downloaded
and extracted), `derived/` (our evaluation outputs) and `PROVENANCE.json`. The manifest with every
URL, checksum and license is `src/ptvision/datasets/manifest.toml`. Results are read and plotted
by `notebooks/validation.ipynb`.

```sh
uv run ptv datasets list                           # what exists, sizes, licenses, what is on disk
uv run ptv datasets pull uiprmd                    # download + verify + extract (0.4 GB)
uv run ptv datasets eval-sts                       # UI-PRMD repetition counting (markers only, minutes)
uv run ptv datasets eval-video --dataset opencap   # our video pipeline vs lab mocap + force plates (~40 min)
```

| name | what | size | license | used for |
|---|---|---|---|---|
| `uiprmd` | 10 subjects, 10 rehabilitation movements incl. sit-to-stand, Vicon 39 markers, no video | 0.4 GB | PDDL | `eval-sts`: does every episode count as exactly one repetition |
| `opencap` (manual) | OpenCap LabValidation (Uhlrich et al. 2023): 10 subjects, 5 iPhones at 60 Hz synced with 100 Hz marker mocap, OpenSim inverse kinematics, 2 kHz force plates; two five-repetition sit-to-stand trials each | 23 GB | see the OpenCap data page (no license file in the archive; cite the paper) | `eval-video`: joint angles and sit-to-stand timing from real phone video |

**OpenCap is a manual download.** Get the `LabValidation_withVideos` folder from the OpenCap data
page linked in the manifest and unpack it so that `~/ptvision-data/datasets/opencap/raw/subject2/VideoData/`
exists. Add an `ACCESS.md` next to it saying who downloaded it, when, and under which terms, and
check the page's terms before redistributing anything derived from it. `ptv datasets list` shows
it as `manual: present` once the folder is there.

**Adding a dataset.** Add an entry to `manifest.toml` with the host's exact byte size and
md5/sha256, the license you are relying on, and a `kind`. Write a reader under
`src/ptvision/datasets/` that yields a `MarkerTrial` (markers) or a `LabTrial` (video plus
reference), and an evaluation that writes `derived/<eval>/summary.json`. Only permissive or
public-domain data belongs here, and evaluation data is held out: no pipeline parameter is tuned
on it.
