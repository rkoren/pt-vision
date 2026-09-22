import numpy as np

from ptvision.app.session import TrialSession, events_from_dict
from ptvision.pose.layout import HALPE26
from ptvision.viz.skeleton import segment_edges


def test_load_analyzed_trial(synthetic_trial) -> None:
    s = TrialSession.load(synthetic_trial.trial_dir)
    assert s.run is not None and s.run.run_id == synthetic_trial.run_id
    assert s.n_frames == synthetic_trial.track.n_frames
    assert len(s.edges) == 25
    assert s.status.rules is not None and s.status.rules.shape == (s.n_frames, 25)
    assert s.side in ("left", "right")
    assert s.available_modes() == ["confidence", "rules"] and s.default_mode() == "rules"
    seat_offs = [e for e in s.events if e.label == "seat-off"]
    assert len(seat_offs) == 5
    assert len(s.phases) == 15 and s.test_window is not None
    assert {m.name for m in s.metrics} >= {"sts_total_time", "sts_rep_count"}
    assert s.quality is not None
    assert s.protocol is not None and s.protocol.id == "sts_5x" and len(s.rules) == 2
    assert s.default_angle_names() == ["knee_flexion", "hip_flexion", "trunk_lean"]
    # trunk turns red-ish during the lean with the tight bands, green when standing
    trunk = segment_edges(HALPE26, "trunk", None)
    t_lean = synthetic_trial.tl.seat_off[0]
    rgb_lean, _ = s.bone_colors(t_lean, "rules")
    rgb_stand, _ = s.bone_colors(synthetic_trial.tl.stand_reached[0] + 3, "rules")
    assert int(rgb_lean[trunk[0], 0]) > int(rgb_lean[trunk[0], 1])  # more red than green
    assert int(rgb_stand[trunk[0], 1]) > int(rgb_stand[trunk[0], 0])  # more green than red
    readout = s.angle_readout(t_lean)
    assert readout["trunk_lean"] > 20
    assert s.title.startswith("Five Times Sit-to-Stand")


def test_load_pose_only_trial(pose_only_trial) -> None:
    s = TrialSession.load(pose_only_trial.trial_dir)
    assert s.protocol is None and s.rules == []
    assert s.status.rules is None and s.available_modes() == ["confidence"]
    assert s.angles.names == ["knee_flexion", "hip_flexion", "trunk_lean"]  # computed on the fly
    assert s.events == [] and s.metrics == [] and s.quality is None
    assert np.isfinite(s.angles["knee_flexion"]).any()
    assert s.title.startswith("Pose only")


def test_events_from_dict_shapes() -> None:
    d = {
        "segmenter": "sts_velocity_threshold",
        "test_start": 10,
        "test_end": 90,
        "reps": [
            {
                "index": 0,
                "seat_off": 10,
                "peak": 20,
                "stand_reached": 25,
                "descent_start": 40,
                "seated_return": 50,
                "lean_onset": 5,
            },
            {
                "index": 1,
                "seat_off": 60,
                "peak": 70,
                "stand_reached": 75,
                "descent_start": None,
                "seated_return": None,
                "lean_onset": None,
            },
        ],
    }
    markers, phases, window = events_from_dict(d)
    assert [m.frame for m in markers] == [5, 10, 25, 40, 50, 60, 75]
    assert [p.kind for p in phases] == ["rise", "stand", "descent", "rise"]
    assert window == (10, 90)
    assert events_from_dict({"segmenter": "other"}) == ([], [], None)


def test_events_from_dict_gait() -> None:
    d = {
        "segmenter": "gait_zeni",
        "steady_window": [30, 120],
        "events": [
            {"frame": 30, "side": "left", "kind": "hs", "bout": 0},
            {"frame": 50, "side": "left", "kind": "to", "bout": 0},
        ],
        "cycles": [
            {
                "side": "left",
                "bout": 0,
                "hs": 30,
                "to": 50,
                "next_hs": 63,
                "contra_hs": 46,
                "contra_to": 34,
                "steady": True,
            },
            {
                "side": "right",
                "bout": 0,
                "hs": 46,
                "to": 66,
                "next_hs": 79,
                "contra_hs": 63,
                "contra_to": 50,
                "steady": False,
            },
        ],
    }
    markers, phases, window = events_from_dict(d)
    assert [m.label for m in markers] == ["heel strike L", "toe-off L"]
    assert [(p.kind, p.start, p.end) for p in phases] == [("stance", 30, 50), ("swing", 50, 63)]
    assert window == (30, 120)
