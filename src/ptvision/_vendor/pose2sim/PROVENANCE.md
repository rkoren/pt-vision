# Vendored Pose2Sim code

- Upstream: https://github.com/perfanalytics/pose2sim
- License: BSD 3-Clause (see LICENSE in this directory)
- Commit: `14d101c786e12abe24360534fb538d6ef589cfa0` (tag v0.10.49, released 2026-07-10)
- Vendored: 2026-09-01

| File here | Upstream source | Local modifications |
|---|---|---|
| skeletons.py | Pose2Sim/skeletons.py, HALPE_26 | anytree Node tree transcribed to (name, id, parent) tuples; verbatim tree kept in the docstring |
| angles.py | Pose2Sim/common.py: angle_dict, points_to_angles, fixed_angles | extracted verbatim; only `import numpy as np` added |
| trc_mot.py | Pose2Sim/common.py: read_mot, write_mot, read_trc, write_trc | extracted verbatim; imports added |
| filtering.py | Pose2Sim/filtering.py: hampel_filter | extracted verbatim; import added |
| tracking.py | Pose2Sim/common.py: pad_shape, sort_people_sports2d | extracted verbatim; imports added |

Why vendor instead of depend: Pose2Sim's core dependencies include opensim, PySide6, openvino, caliscope
and bvhsdk, and its modules import Qt and matplotlib at module load. The pieces above are small, pure
numpy/pandas functions.

To refresh: re-run the extraction against a new tag and update the commit above.
