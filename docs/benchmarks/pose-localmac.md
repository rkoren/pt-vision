# Pose extraction benchmark

Command: `ptv bench sports2d_demo.mp4 --frames 150 --modes lightweight,balanced --devices cpu,mps --det-frequencies 1,4`
Input: Sports2D demo clip, 1768x994, two people in frame. onnxruntime 1.29.0, rtmlib 0.0.16. 2026-09-01.

| mode | device | det every | providers | fps | persons tracked |
|---|---|---|---|---|---|
| lightweight (RTMPose-s + YOLOX-tiny) | cpu | 1 | CPUExecutionProvider | 45.3 | 3 |
| lightweight | cpu | 4 | CPUExecutionProvider | 86.3 | 2 |
| lightweight | mps (CoreML EP) | 1 | — | error | — |
| lightweight | mps (CoreML EP) | 4 | — | error | — |
| balanced (RTMPose-m + YOLOX-m) | cpu | 1 | CPUExecutionProvider | 11.6 | 4 |
| balanced | cpu | 4 | CPUExecutionProvider | 28.0 | 4 |
| balanced | mps (CoreML EP) | 1 | — | error | — |
| balanced | mps (CoreML EP) | 4 | — | error | — |

Observations
- The person detector dominates cost. Running YOLOX every 4th frame and propagating boxes from the
  previous frame's keypoints (like rtmlib's PoseTracker) is 2–2.5x faster with no visible tracking loss.
- The ONNX Runtime CoreML execution provider fails at inference (`Non-zero status`) for these models on
  ORT 1.29. Pose2Sim gates CoreML to ORT <= 1.26 for the same reason. `--device mps` now raises a clear
  error; CPU is the default. Revisit when ORT or the exported models change.
- Defaults: `balanced`, `cpu`, `det_frequency=4`. A 60 s clinic clip at 30 fps takes about 65 s.
- "persons tracked" > 2 at `det every 1` reflects brief spurious detections creating short-lived ids; the
  primary-subject selection is unaffected, and the quality report flags other people in frame.
