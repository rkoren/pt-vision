"""ptvision: open computer-vision measurement tools to assist physical therapists.

Data flows left to right; each package depends only on the ones before it:

    video.py    video probe and normalisation (ffmpeg), frame access; files.py: hashing, JSON
    pose        keypoints per frame (RTMPose via rtmlib), tracking, PoseTrack storage, overlays
    kinematics  cleaning, joint angles, pixel-to-metre scale, TRC/MOT export
    clinical    protocols (TOML), segmenters, metrics, reference values
    quality     capture checks that gate whether a result can be trusted
    report      the JSON report bundle and its HTML rendering
    pipeline    the stages above in order, writing one run folder per analysis
    trials      the trial store on disk (capture.json, runs/<id>/provenance.json)
    viz         colour semantics and skeleton segments shared by overlays and the app
    app         the PySide6 viewer (optional extra); cli is the `ptv` command line
    datasets    public validation datasets: pull with checksums, evaluate
    samples     the bundled demo clip

Not a medical device: measurements are reported against cited reference values with their known
error; interpretation is the clinician's.
"""

from ptvision._version import __version__

__all__ = ["__version__"]
