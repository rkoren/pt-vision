"""Stage orchestration: ingest, pose, quality, clean, angles, segment, metrics, report"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ptvision._version import __version__
from ptvision.config import settings
from ptvision.files import read_json, sha256_file, write_json
from ptvision.pose.base import PoseBackend, PreviewFn, ProgressFn
from ptvision.pose.identity import select_primary_person, spurious_slots
from ptvision.pose.overlay import render_overlay
from ptvision.pose.track import PoseTrack
from ptvision.trials.models import CameraCapture, Capture, RunProvenance, Subject, View
from ptvision.trials.store import DataStore, RunDir, TrialDir, base_provenance, new_id, now_iso
from ptvision.video import iter_frames, normalize, probe
from ptvision.viz import status as vs

StageHook = Callable[[str], None]


class TrialMismatchError(ValueError):
    """The trial directory already holds a different source video."""


# ---------------------------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------------------------
@dataclass
class IngestOptions:
    copy_source: bool = True
    max_height: int | None = None
    fps: float | None = None
    view: View = "unknown"
    subject: Subject | None = None
    protocol_id: str | None = None
    protocol_version: str | None = None
    notes: str | None = None
    patient_id: str | None = None
    visit_id: str | None = None
    episode_id: str | None = None


def ingest(
    source: Path, trial: TrialDir, opts: IngestOptions | None = None, *, camera_id: str = "cam0"
) -> Capture:
    """Probe, hash, optionally copy, and normalize a source video into a trial directory."""
    opts = opts or IngestOptions()
    source = Path(source)
    if trial.has_capture():
        if holds_other_video(trial, source):
            raise TrialMismatchError(
                f"{trial.path} was ingested from a different video than {source}"
            )
        return trial.read_capture()
    src_info = probe(source)
    src_sha = sha256_file(source)
    if opts.copy_source:
        trial.source_dir.mkdir(parents=True, exist_ok=True)
        dst_src = trial.source_dir / source.name
        if not dst_src.exists():
            shutil.copy2(source, dst_src)
        source_ref = str(dst_src.relative_to(trial.path))
    else:
        source_ref = str(source.resolve())

    norm_path = trial.video_path(camera_id)
    norm_info = normalize(source, norm_path, fps=opts.fps, max_height=opts.max_height)

    cam = CameraCapture(
        camera_id=camera_id,
        source_file=source_ref,
        source_sha256=src_sha,
        source_codec=src_info.codec,
        source_pix_fmt=src_info.pix_fmt,
        source_width=src_info.width,
        source_height=src_info.height,
        source_fps_avg=src_info.fps,
        source_fps_nominal=src_info.nominal_fps,
        source_is_vfr=src_info.is_vfr,
        rotation_applied_deg=src_info.rotation_deg,
        normalized_file=str(norm_path.relative_to(trial.path)),
        width=norm_info.width,
        height=norm_info.height,
        fps=norm_info.fps,
        n_frames=norm_info.n_frames,
        duration_s=norm_info.duration_s,
        view=opts.view,
    )
    capture = Capture(
        trial_id=trial.trial_id or new_id("T"),
        patient_id=opts.patient_id,
        visit_id=opts.visit_id,
        episode_id=opts.episode_id,
        protocol_id=opts.protocol_id,
        protocol_version=opts.protocol_version,
        recorded_at=src_info.creation_time,
        ingested_at=now_iso(),
        subject=opts.subject or Subject(),
        cameras=[cam],
        notes=opts.notes,
    )
    trial.write_capture(capture)
    return capture


# ---------------------------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------------------------
def pose_matches(prov_path: Path, backend: PoseBackend) -> bool:
    """True if an existing run's pose provenance matches this backend (so Parquet can be reused)."""
    if not prov_path.exists():
        return False
    try:
        prev = RunProvenance.model_validate(read_json(prov_path))
    except Exception:
        return False
    if prev.pose is None:
        return False
    keys = {
        "backend",
        "model_class",
        "mode",
        "det_sha256",
        "pose_sha256",
        "det_frequency",
        "tracker",
    }
    return prev.pose.model_dump(include=keys) == backend.info.model_dump(include=keys)


def extract_pose(
    trial: TrialDir,
    backend: PoseBackend,
    prov: RunProvenance,
    *,
    camera_id: str = "cam0",
    reuse: bool = True,
    progress: ProgressFn | None = None,
    on_stage: StageHook | None = None,
    preview: PreviewFn | None = None,
) -> tuple[PoseTrack, bool]:
    """Run (or reuse) pose extraction for one camera; fills `prov.pose*` and timing."""
    capture = trial.read_capture()
    cam = next(c for c in capture.cameras if c.camera_id == camera_id)
    video = trial.path / cam.normalized_file
    parquet = trial.pose_path(camera_id)
    prov.pose = backend.info

    latest = trial.latest_run()
    t0 = time.perf_counter()
    reused = (
        reuse
        and parquet.exists()
        and latest is not None
        and pose_matches(latest.provenance_path, backend)
    )
    if reused:
        if on_stage:
            on_stage("pose (reusing cached keypoints)")
        track = PoseTrack.load(parquet)
    else:
        if on_stage:
            on_stage("pose")
        track = backend.estimate(
            iter_frames(video),
            fps=cam.fps,
            image_size=(cam.width, cam.height),
            n_frames=cam.n_frames,
            progress=progress,
            preview=preview,
        )
        track.camera_id = camera_id
        track.mark_spurious(spurious_slots(track.coords, track.score, fps=track.fps))
        track.save(parquet)
    prov.timing_s["pose"] = round(time.perf_counter() - t0, 3)
    prov.pose_reused = bool(reused)
    prov.pose_parquet = str(parquet.relative_to(trial.path))
    prov.pose_parquet_sha256 = sha256_file(parquet)
    return track, reused


@dataclass
class PoseResult:
    track: PoseTrack
    parquet_path: Path
    run: RunDir
    primary_person: int | None
    overlay_path: Path | None
    seconds: float
    reused: bool


def run_pose(
    trial: TrialDir,
    backend: PoseBackend,
    *,
    camera_id: str = "cam0",
    overlay: bool = True,
    reuse: bool = True,
    progress: ProgressFn | None = None,
    on_stage: StageHook | None = None,
    preview: PreviewFn | None = None,
) -> PoseResult:
    """`ptv pose`: extract keypoints, pick the primary person, render overlay, record provenance.

    The overlay is colored by tracking confidence (no protocol, so no rules)."""
    run = trial.new_run()
    prov = base_provenance(run.run_id)
    track, reused = extract_pose(
        trial,
        backend,
        prov,
        camera_id=camera_id,
        reuse=reuse,
        progress=progress,
        on_stage=on_stage,
        preview=preview,
    )
    primary: int | None = None
    if track.n_persons:
        primary = int(track.person_ids[select_primary_person(track.coords, track.score)])
    prov.quality = {
        "primary_person": primary,
        "n_persons": track.n_persons_real,
        "n_tracks": track.n_persons,
    }

    overlay_path: Path | None = None
    if overlay:
        if on_stage:
            on_stage("overlay")
        t1 = time.perf_counter()
        video = trial.path / trial.read_capture().cameras[0].normalized_file
        status = None
        if primary is not None:
            slot = int(np.flatnonzero(track.person_ids == primary)[0])
            status = vs.BoneStatus.build(track.score[:, slot], track.layout)
        overlay_path = render_overlay(
            video,
            track,
            run.path / "overlay.mp4",
            primary=primary,
            status=status,
            mode="confidence",
            progress=progress,
        )
        prov.timing_s["overlay"] = round(time.perf_counter() - t1, 3)

    run.write_provenance(prov)
    trial.set_latest(run)
    return PoseResult(
        track, trial.pose_path(camera_id), run, primary, overlay_path, prov.timing_s["pose"], reused
    )


# ---------------------------------------------------------------------------------------------
# Analyze (protocol-driven)
# ---------------------------------------------------------------------------------------------
@dataclass
class AnalyzeOptions:
    protocol: str = "sts_5x"
    out_dir: Path | None = None
    patient_id: str | None = None
    visit_id: str | None = None
    person: int | None = None
    mode: str | None = None
    device: str | None = None
    backend: str | None = None
    det_frequency: int | None = None
    overlay: bool = True
    reuse_pose: bool = True
    manual_start_s: float | None = None
    subject: Subject | None = None
    max_height: int | None = None
    color_by: vs.Mode | None = None  # overlay coloring; None = rules when available else confidence
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzeResult:
    trial: TrialDir
    run: RunDir
    report_json: Path
    report_html: Path
    metrics: list[Any]
    quality: Any
    events: Any
    primary_person: int | None
    warnings: list[str]


def _resolve_trial(source: Path, opts: AnalyzeOptions, store: DataStore | None) -> TrialDir:
    if source.is_dir():
        trial = TrialDir(source)
        if not trial.has_capture():
            raise FileNotFoundError(f"{source} is a directory without capture.json")
        return trial
    if opts.patient_id and opts.visit_id:
        store = store or DataStore(settings().data_dir)
        return store.new_trial(opts.patient_id, opts.visit_id)
    if opts.out_dir is None:
        return DataStore.anonymous_trial(default_trial_dir(source))
    trial = DataStore.anonymous_trial(opts.out_dir)
    if holds_other_video(trial, source):
        raise TrialMismatchError(
            f"{opts.out_dir} already holds a different video; choose another output directory"
        )
    return trial


# handle same name clip
def holds_other_video(trial: TrialDir, source: Path, sha: str | None = None) -> bool:
    """True when the trial was ingested from a file whose content differs from `source`."""
    if not trial.has_capture():
        return False
    cams = trial.read_capture().cameras
    return bool(cams) and cams[0].source_sha256 != (sha or sha256_file(source))


def default_trial_dir(source: Path) -> Path:
    """`ptv_out/<stem>`, or `<stem>-2`, `<stem>-3`, ... when that directory already holds a
    different video with the same file name."""
    base = Path("ptv_out") / source.stem
    candidate, n, sha = base, 1, None
    while TrialDir(candidate).has_capture():
        sha = sha or sha256_file(source)
        if not holds_other_video(TrialDir(candidate), source, sha):
            break
        n += 1
        candidate = base.with_name(f"{base.name}-{n}")
    return candidate


def _prepare_trial(
    source: Path, opts: AnalyzeOptions, store: DataStore | None, **ingest_extra: Any
) -> TrialDir:
    """Resolve the trial, ingest the video once, and keep the capture's subject in sync with the
    options (the CLI flags or the app's drop-page form). Shared by `pose_only` and `analyze`."""
    trial = _resolve_trial(source, opts, store)
    if not trial.has_capture():
        ingest(
            source,
            trial,
            IngestOptions(
                max_height=opts.max_height,
                subject=opts.subject,
                patient_id=opts.patient_id,
                visit_id=opts.visit_id,
                **ingest_extra,
            ),
        )
    if opts.subject is not None:
        capture = trial.read_capture()
        if capture.subject != opts.subject:
            capture.subject = opts.subject
            trial.write_capture(capture)
    return trial


@contextmanager
def _backend_for(
    opts: AnalyzeOptions,
    backend: PoseBackend | None,
    mode_default: str | None,
    det_default: int | None,
    *,
    progress: ProgressFn | None,
    on_stage: StageHook | None,
) -> Iterator[PoseBackend]:
    """Yield the caller's backend, or build one and release its model sessions afterwards ."""
    if backend is not None:
        yield backend
        return
    if on_stage:
        on_stage("loading model")
    own = _make_backend(opts, mode_default, det_default, progress=progress, on_stage=on_stage)
    try:
        yield own
    finally:
        _close_backend(own)


def _close_backend(backend: PoseBackend) -> None:
    """Release model sessions created by the pipeline itself ."""
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _make_backend(
    opts: AnalyzeOptions,
    mode_default: str | None,
    det_default: int | None,
    *,
    progress: ProgressFn | None = None,
    on_stage: StageHook | None = None,
) -> PoseBackend:
    """Build the pose backend; a first-time weight download is announced as its own stage with
    byte progress so the app's bar shows it (demo MVP)."""
    from ptvision.pose.models import ModelManager
    from ptvision.pose.rtmlib_backend import RtmlibBackend

    s = settings()
    mode = opts.mode or mode_default or s.mode
    mm = ModelManager()
    download_progress = None
    missing = mm.missing_for_mode(mode)
    if missing and on_stage:
        on_stage(f"downloading model weights ({mode}, one time, about 150 MB)")
    if missing and progress:

        def download_progress(name: str, done: int, total: int | None) -> None:
            progress(done, total)

    return RtmlibBackend(
        mode=mode,
        device=opts.device or s.device,
        backend=opts.backend or s.backend,
        det_frequency=opts.det_frequency or det_default or s.det_frequency,
        models=mm,
        download_progress=download_progress,
    )


def pose_only(
    source: Path,
    opts: AnalyzeOptions | None = None,
    *,
    store: DataStore | None = None,
    backend: PoseBackend | None = None,
    progress: ProgressFn | None = None,
    on_stage: StageHook | None = None,
    preview: PreviewFn | None = None,
) -> PoseResult:
    """Ingest (if needed) + pose extraction + confidence-colored overlay, no protocol.
    Shared by `ptv pose` and the desktop app's "Pose only" mode."""
    opts = opts or AnalyzeOptions()
    source = Path(source)
    if on_stage:
        on_stage("ingest")
    trial = _prepare_trial(source, opts, store)
    with _backend_for(opts, backend, None, None, progress=progress, on_stage=on_stage) as be:
        return run_pose(
            trial,
            be,
            overlay=opts.overlay,
            reuse=opts.reuse_pose,
            progress=progress,
            on_stage=on_stage,
            preview=preview,
        )


def analyze(
    source: Path,
    opts: AnalyzeOptions | None = None,
    *,
    store: DataStore | None = None,
    backend: PoseBackend | None = None,
    progress: ProgressFn | None = None,
    on_stage: StageHook | None = None,
    preview: PreviewFn | None = None,
) -> AnalyzeResult:
    """Full protocol run: ingest -> pose -> quality -> preprocess -> angles -> segment -> metrics ->
    norms -> export -> figures -> report -> provenance."""
    # Imported here so `ptv pose` stays light.
    from ptvision.clinical import registry
    from ptvision.clinical.metrics.base import Metric, MetricContext, NormComparison
    from ptvision.clinical.norms import compare, load_norms
    from ptvision.clinical.protocol import load_protocol
    from ptvision.kinematics.angles import ANGLE_DEFS, compute_angles
    from ptvision.kinematics.export import write_mot, write_trc
    from ptvision.kinematics.preprocess import PreprocessConfig, preprocess
    from ptvision.pose.overlay import thumbnail
    from ptvision.quality import checks as Q
    from ptvision.report.bundle import Figure, ReportBundle, build_report
    from ptvision.report.figures import (
        fig_angles,
        fig_per_rep,
        fig_sts_trajectory,
    )

    opts = opts or AnalyzeOptions()
    protocol = load_protocol(opts.protocol)
    source = Path(source)

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    # 1. trial + ingest
    stage("ingest")
    trial = _prepare_trial(
        source,
        opts,
        store,
        view=protocol.capture.view,
        protocol_id=protocol.id,
        protocol_version=protocol.version,
    )
    capture = trial.read_capture()
    cam = capture.cameras[0]
    video = trial.path / cam.normalized_file

    # 2. pose
    backend_cm = _backend_for(
        opts,
        backend,
        protocol.pose.mode,
        protocol.pose.det_frequency,
        progress=progress,
        on_stage=on_stage,
    )
    run = trial.new_run()
    prov = base_provenance(run.run_id)
    prov.protocol = {
        "id": protocol.id,
        "version": protocol.version,
        "sha256": protocol.source_sha256,
        "source": protocol.source_path,
    }
    with backend_cm as be:  # sessions are released as soon as extraction is done
        pose_info = be.info  # kept for the methods text after the sessions are closed
        track, _reused = extract_pose(
            trial,
            be,
            prov,
            reuse=opts.reuse_pose,
            progress=progress,
            on_stage=on_stage,
            preview=preview,
        )

    warnings: list[str] = []
    checks: list[Q.QualityCheck] = Q.check_capture(cam, protocol)
    if track.n_persons == 0:
        checks.append(
            Q.QualityCheck(
                name="subject_present",
                status="fail",
                value=0.0,
                threshold=0.95,
                message="No person detected in the video.",
            )
        )
        quality = Q.combine(checks)
        _write_minimal(run, prov, quality)
        raise RuntimeError("No person detected; see quality.json")

    if opts.person is not None:
        pid = opts.person
    else:
        pid = int(track.person_ids[select_primary_person(track.coords, track.score)])
    single = track.select_person(pid)

    # 3. quality (raw)
    stage("quality checks")
    checks += Q.check_track(single, protocol)
    checks += Q.check_multi_person(track)
    if video.exists():
        checks.append(Q.check_camera_motion(video))

    # 4. preprocess + angles
    stage("filtering")
    pcfg = PreprocessConfig(
        score_threshold=protocol.preprocess.score_threshold,
        interp_max_gap_s=protocol.preprocess.interp_max_gap_s,
        filter_type=protocol.preprocess.filter_type,
        filter_order=protocol.preprocess.filter_order,
        filter_cutoff_hz=protocol.preprocess.filter_cutoff_hz,
    )
    pre = preprocess(single, pcfg)
    prov.preprocess = {
        **pcfg.to_dict(),
        **pre.stats,
        "segment_signal_cutoff_hz": protocol.preprocess.segment_signal_cutoff_hz,
    }
    angle_names = list(
        dict.fromkeys([*protocol.report.angles, "trunk_lean", "knee_flexion", "hip_flexion"])
    )
    angle_names = [a for a in angle_names if a in ANGLE_DEFS]
    angles = compute_angles(pre.filtered, angle_names, side="near")
    angles_raw = compute_angles(pre.raw, angle_names, side=angles.side)

    # 5. segment
    stage("segmenting")
    seg_entry = registry.get_segmenter(protocol.segmenter.name)
    seg_params = {
        **protocol.segmenter.params,
        "segment_signal_cutoff_hz": protocol.preprocess.segment_signal_cutoff_hz,
        **opts.extra_params,
    }
    if opts.manual_start_s is not None:
        seg_params["manual_start_s"] = opts.manual_start_s
        seg_params["start_rule"] = "manual"
    events = seg_entry.fn(
        pre.filtered,
        cam.fps,
        seg_params,
        trunk_lean=angles["trunk_lean"] if "trunk_lean" in angles.names else None,
        image_size=(cam.width, cam.height),
    )
    prov.segmenter = {"name": seg_entry.name, "version": seg_entry.version, "params": seg_params}
    warnings += list(getattr(events, "warnings", []))
    expected = int(seg_params.get("n_reps_expected", -1))
    if expected > 0:
        n = getattr(events, "n_reps", 0)
        checks.append(
            Q.QualityCheck(
                name="repetition_count",
                status="pass" if n == expected else "fail",
                value=float(n),
                threshold=float(expected),
                message=f"{n} repetitions detected"
                if n == expected
                else (
                    f"{n} repetitions detected but the protocol expects {expected}; "
                    "check the overlay and the trajectory figure."
                ),
            )
        )
    if getattr(events, "truncated", False):
        checks.append(
            Q.QualityCheck(
                name="test_truncated",
                status="fail",
                value=None,
                threshold=None,
                message="the test starts or ends at the clip boundary, so the primary timing is "
                "not trustworthy; re-record with two seconds of stillness before and after.",
            )
        )
    quality = Q.combine(checks)

    # 6. metrics + norms
    stage("metrics")
    ctx = MetricContext(
        raw=pre.raw,
        track=pre.filtered,
        angles=angles,
        events=events,
        capture=capture,
        protocol=protocol,
        fps=cam.fps,
    )
    metrics: list[Metric] = []
    norms: list[NormComparison] = []
    primary: Metric | None = None
    primary_norm: NormComparison | None = None
    norm_tables = load_norms()
    for spec in protocol.metrics:
        entry = registry.get_metric(spec.name)
        m = entry.fn(ctx)
        metrics.append(m)
        prov.metrics[m.name] = m.method_version
        if spec.norm:
            nc = compare(m, spec.norm, capture.subject, norm_tables)
            norms.append(nc)
        if spec.primary:
            primary = m
            primary_norm = norms[-1] if spec.norm else None

    # 7. artifacts
    stage("writing artifacts")
    write_json(run.path / "quality.json", quality)
    write_json(run.path / "events.json", events.to_dict() if hasattr(events, "to_dict") else events)
    write_json(run.path / "metrics.json", {"metrics": metrics, "norms": norms})
    angles.save(run.path / "angles.parquet")
    write_trc(pre.filtered, run.path / "keypoints_filtered_px.trc")
    write_trc(pre.raw, run.path / "keypoints_raw_px.trc")
    write_mot(angles, run.path / "angles.mot")

    figures: list[Figure] = []
    thumbs: list[Figure] = []
    fig_dir = run.path / "figures"
    if hasattr(events, "height_norm"):
        fig_sts_trajectory(events, fig_dir / "sts_trajectory.png")
        figures.append(
            Figure(
                name="sts_trajectory",
                path="figures/sts_trajectory.png",
                caption=(
                    "Normalised pelvis height with detected rising (green), standing (blue) and "
                    "sitting (orange) phases, and the timed interval (dashed)."
                ),
            )
        )
        if getattr(events, "reps", None):
            fig_per_rep(events, fig_dir / "per_rep.png")
            figures.append(
                Figure(
                    name="per_rep",
                    path="figures/per_rep.png",
                    caption="Rise and sit durations per repetition.",
                )
            )
    fig_angles(
        angles_raw.frame,
        angles.frame,
        cam.fps,
        events if hasattr(events, "height_norm") else None,
        fig_dir / "angles.png",
        names=[a for a in protocol.report.angles if a in angles.names],
    )
    figures.append(
        Figure(
            name="angles",
            path="figures/angles.png",
            caption=(
                f"Joint angles ({angles.side} side; facing "
                f"{'right' if angles.facing > 0 else 'left'}), raw in grey and filtered in blue."
            ),
        )
    )
    if video.exists() and getattr(events, "reps", None):
        for r in events.reps:
            p = fig_dir / f"seat_off_{r.index + 1}.jpg"
            try:
                thumbnail(video, track, r.seat_off, p)
                thumbs.append(
                    Figure(
                        name=f"seat_off_{r.index + 1}",
                        path=f"figures/{p.name}",
                        caption=f"rep {r.index + 1} seat-off, frame {r.seat_off}",
                    )
                )
            except Exception as e:
                warnings.append(f"thumbnail for rep {r.index + 1} failed: {e}")

    rules = protocol.rule_list()
    status = vs.BoneStatus.build(
        single.score[:, 0], track.layout, values=angles.values(), rules=rules, side=angles.side
    )
    color_mode: vs.Mode = opts.color_by or ("rules" if status.rules is not None else "confidence")
    if opts.overlay and video.exists():
        stage("overlay")
        render_overlay(
            video,
            track,
            run.path / "overlay.mp4",
            primary=pid,
            status=status,
            mode=color_mode,
            progress=progress,
        )

    # 8. report + provenance
    stage("report")
    prov.preprocess["overlay_color_by"] = color_mode
    prov.quality = {
        "status": quality.status,
        "failed": [c.name for c in quality.failures],
        "warned": [c.name for c in quality.warnings],
        "primary_person": pid,
        "n_persons": track.n_persons_real,
        "n_tracks": track.n_persons,
    }
    methods = [
        (
            f"Pose: {pose_info.pose_name} ({pose_info.layout}) with {pose_info.det_name} "
            f"detector every {pose_info.det_frequency} frames, {pose_info.runtime}."
        ),
        (
            f"Preprocessing: keypoints below confidence {pcfg.score_threshold} removed; gaps up to "
            f"{pcfg.interp_max_gap_s} s linearly interpolated; Hampel outlier rejection (window "
            f"{pcfg.hampel_window}); zero-phase Butterworth low-pass {pcfg.filter_cutoff_hz:g} Hz "
            f"(order {pcfg.filter_order})."
        ),
        (
            f"Segmentation: {seg_entry.name} v{seg_entry.version}: "
            f"{seg_entry.description.splitlines()[0] if seg_entry.description else ''}"
        ),
        (
            f"Angles computed on the {angles.side} side (higher keypoint confidence). Tier 1 = "
            "timing/count (validated ICC > 0.9 vs clinicians); tier 2 = sagittal angles (typical "
            "RMSE 4–6°); ankle and frontal-plane measures are not reported."
        ),
    ]
    bundle = ReportBundle(
        title=protocol.protocol.name,
        protocol={
            "id": protocol.id,
            "version": protocol.version,
            "name": protocol.protocol.name,
            "summary": protocol.protocol.summary,
        },
        trial_id=trial.trial_id,
        run_id=run.run_id,
        created_at=prov.created_at,
        ptvision_version=__version__,
        capture=cam.model_dump(),
        subject=capture.subject.model_dump(),
        quality=quality,
        primary=primary,
        primary_norm=primary_norm,
        metrics=metrics,
        norms=norms,
        events=events.to_dict() if hasattr(events, "to_dict") else {},
        figures=figures,
        thumbnails=thumbs,
        methods=methods,
        angle_definitions=[
            {"name": n, "description": d.description} for n, d in angles.definitions.items()
        ],
        provenance=prov.model_dump(mode="json"),
        warnings=warnings,
        template=protocol.report.template,
    )
    report_json, report_html = build_report(bundle, run.path, bundle.template)
    run.write_provenance(prov)
    trial.set_latest(run)
    return AnalyzeResult(
        trial, run, report_json, report_html, metrics, quality, events, pid, warnings
    )


def _write_minimal(run: RunDir, prov: RunProvenance, quality: Any) -> None:
    write_json(run.path / "quality.json", quality)
    prov.quality = {"status": quality.status, "failed": [c.name for c in quality.failures]}
    run.write_provenance(prov)
