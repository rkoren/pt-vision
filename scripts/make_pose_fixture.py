"""Turn a trial's pose Parquet into test fixture"""

from __future__ import annotations

import sys
from pathlib import Path

from ptvision.pose.track import PoseTrack
from ptvision.pose.tracking import select_primary_person


def main(src: str, dst: str) -> None:
    track = PoseTrack.load(src)
    slot = select_primary_person(track.coords, track.score)
    single = track.select_person(int(track.person_ids[slot]))
    single.extra_metadata["fixture_source"] = Path(src).name
    out = single.save(dst)
    print(
        f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, "
        f"{single.n_frames} frames @ {single.fps:.2f} fps)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
