# Vendored from Pose2Sim (https://github.com/perfanalytics/pose2sim), BSD 3-Clause License.
# Copyright (c) 2022, perfanalytics / David Pagnon. See LICENSE in this directory.
# Upstream: Pose2Sim/skeletons.py (HALPE_26) at commit 14d101c786e12abe24360534fb538d6ef589cfa0 (tag v0.10.49). See PROVENANCE.md for local modifications.
# ruff: noqa
# mypy: ignore-errors

"""HALPE_26 kinematic tree, transcribed from Pose2Sim's anytree definition as
(name, model_index, parent_name) tuples so the core package does not need anytree.

Upstream source (verbatim, for parity checks):

    HALPE_26 = Node("Hip", id=19, children=[
        Node("RHip", id=12, children=[
            Node("RKnee", id=14, children=[
                Node("RAnkle", id=16, children=[
                    Node("RBigToe", id=21, children=[Node("RSmallToe", id=23)]),
                    Node("RHeel", id=25),
                ]),
            ]),
        ]),
        Node("LHip", id=11, children=[
            Node("LKnee", id=13, children=[
                Node("LAnkle", id=15, children=[
                    Node("LBigToe", id=20, children=[Node("LSmallToe", id=22)]),
                    Node("LHeel", id=24),
                ]),
            ]),
        ]),
        Node("Neck", id=18, children=[
            Node("Head", id=17, children=[Node("Nose", id=0)]),
            Node("RShoulder", id=6, children=[Node("RElbow", id=8, children=[Node("RWrist", id=10)])]),
            Node("LShoulder", id=5, children=[Node("LElbow", id=7, children=[Node("LWrist", id=9)])]),
        ]),
    ])
"""

HALPE_26_TREE: list[tuple[str, int, str | None]] = [
    ("Hip", 19, None),
    ("RHip", 12, "Hip"),
    ("RKnee", 14, "RHip"),
    ("RAnkle", 16, "RKnee"),
    ("RBigToe", 21, "RAnkle"),
    ("RSmallToe", 23, "RBigToe"),
    ("RHeel", 25, "RAnkle"),
    ("LHip", 11, "Hip"),
    ("LKnee", 13, "LHip"),
    ("LAnkle", 15, "LKnee"),
    ("LBigToe", 20, "LAnkle"),
    ("LSmallToe", 22, "LBigToe"),
    ("LHeel", 24, "LAnkle"),
    ("Neck", 18, "Hip"),
    ("Head", 17, "Neck"),
    ("Nose", 0, "Head"),
    ("RShoulder", 6, "Neck"),
    ("RElbow", 8, "RShoulder"),
    ("RWrist", 10, "RElbow"),
    ("LShoulder", 5, "Neck"),
    ("LElbow", 7, "LShoulder"),
    ("LWrist", 9, "LElbow"),
]
