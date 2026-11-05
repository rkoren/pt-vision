// [REVIEW] Live frames → pose model → filming checks.
// Source is the back camera on a device, or the bundled sts_side.mp4 looped at 30 fps
// (Simulator, or "Use sample clip" on a device), so the same model path runs everywhere.

import AVFoundation
import CoreImage
import CoreMotion
import Foundation
import Observation

enum FrameSource: String, CaseIterable, Identifiable, Sendable {
    case camera, sample
    var id: String { rawValue }
    var label: String { self == .camera ? "Camera" : "Sample clip" }

    static var defaultSource: FrameSource {
        #if targetEnvironment(simulator)
        .sample
        #else
        .camera
        #endif
    }
}

enum CheckState: Sendable, Equatable { case pass, warn, fail, pending }

struct LiveCheck: Identifiable, Sendable, Equatable {
    let id: String
    let label: String
    var state: CheckState
    var fix: String
}

@MainActor
@Observable
final class LiveCapture {
    // published to the UI
    private(set) var image: CGImage?
    private(set) var pose: PoseFrame?
    private(set) var modelFPS: Double = 0
    private(set) var tilt: PhoneTilt?
    private(set) var steady: Bool?
    private(set) var cameraDenied = false
    private(set) var stands = 0
    private(set) var recordedKnee: [Double?] = []

    var backend: PoseBackend = .rtmpose { didSet { engine.setBackend(backend) } }
    var source: FrameSource = .defaultSource { didSet { if started { restart() } } }
    private var started = false
    var recording = false {
        didSet {
            if recording { counter = StandCounter(); stands = 0; recordedKnee = [] }
        }
    }

    private let engine = CaptureEngine()
    private let motion = CMMotionManager()
    private var counter = StandCounter()
    private var smoothedRotation = 0.0
    private var viewRatios: [Double] = []   // last ~2 s; pt-vision rates the view over the clip, not per frame

    init() {
        engine.onFrame = { [weak self] image, pose, fps in
            Task { @MainActor in self?.receive(image: image, pose: pose, fps: fps) }
        }
    }

    func start() {
        started = true
        engine.setBackend(backend)
        restart()
        startMotion()
    }

    func stop() {
        engine.stop()
        motion.stopDeviceMotionUpdates()
    }

    private func restart() {
        engine.stop()
        switch source {
        case .camera:
            Task {
                let ok = await AVCaptureDevice.requestAccess(for: .video)
                cameraDenied = !ok
                if ok { engine.startCamera() } else { source = .sample }
            }
        case .sample:
            if let url = Bundle.main.url(forResource: "sts_side", withExtension: "mp4") {
                engine.startSample(url)
            }
        }
    }

    private func startMotion() {
        guard motion.isDeviceMotionAvailable else { tilt = nil; steady = nil; return }
        motion.deviceMotionUpdateInterval = 1 / 30
        motion.startDeviceMotionUpdates(to: .main) { [weak self] m, _ in
            guard let self, let m else { return }
            // Roll and pitch from upright portrait: both 0° when the phone's long edge is vertical.
            let g = m.gravity
            self.tilt = PhoneTilt(gravityX: g.x, y: g.y, z: g.z)
            let r = m.rotationRate
            self.smoothedRotation = 0.85 * self.smoothedRotation + 0.15 * sqrt(r.x * r.x + r.y * r.y + r.z * r.z)
            self.steady = self.smoothedRotation < 0.08
        }
    }

    private func receive(image: CGImage, pose: PoseFrame?, fps: Double) {
        self.image = image
        self.pose = pose
        self.modelFPS = fps
        if let r = pose?.viewRatio { viewRatios.append(r); if viewRatios.count > 60 { viewRatios.removeFirst() } }
        if recording {
            counter.update(knee: pose?.kneeFlexion)
            stands = counter.count
            recordedKnee.append(pose?.kneeFlexion)
        }
    }

    // MARK: checks (same rules as pt-vision quality/checks.py, live)

    var checks: [LiveCheck] {
        var out: [LiveCheck] = []
        if let tilt {
            let a = tilt.worstDegrees
            out.append(.init(id: "level", label: "Phone is level", state: a < 3 ? .pass : a < 8 ? .warn : .fail,
                             fix: tilt.fix))
        } else {
            out.append(.init(id: "level", label: "Phone is level", state: .pass, fix: "No motion sensor here; not checked"))
        }

        if let r = medianViewRatio {
            let state: CheckState = r < 0.35 ? .pass : r < 0.50 ? .warn : .fail
            out.append(.init(id: "view", label: "Filming from the side", state: state,
                             fix: state == .warn ? "Slightly angled · timing is fine, side-on is best for angles" : "Looks frontal · turn to face their side"))
        } else {
            out.append(.init(id: "view", label: "Filming from the side", state: .pending, fix: pose == nil ? "Looking for the person…" : "Checking the camera angle…"))
        }

        let inFrame = pose?.wholeBodyInFrame ?? false
        out.append(.init(id: "frame", label: "Whole body, feet and chair in frame", state: pose == nil ? .pending : inFrame ? .pass : .fail,
                         fix: pose == nil ? "Looking for the person…" : "Move back so the head and feet fit"))

        if let steady {
            out.append(.init(id: "steady", label: "Phone is steady", state: steady ? .pass : .warn,
                             fix: "Use a tripod or lean it on something"))
        } else {
            out.append(.init(id: "steady", label: "Phone is steady", state: .pass, fix: "No motion sensor here; not checked"))
        }
        return out
    }

    private var medianViewRatio: Double? {
        guard viewRatios.count >= 5 else { return nil }
        let s = viewRatios.sorted()
        return s[s.count / 2]
    }

    /// Nothing failing or pending. A slightly angled view (amber) still counts: the laptop corrects
    /// the joint angles for it and the timing doesn't depend on it. An amber level or steady check
    /// doesn't count, because nothing corrects for those.
    var ready: Bool {
        checks.allSatisfy { $0.state == .pass || ($0.id == "view" && $0.state == .warn) }
    }
}

/// Runs on its own serial queue: frames in, pose out. Late frames are dropped.
final class CaptureEngine: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    var onFrame: (@Sendable (CGImage, PoseFrame?, Double) -> Void)?

    private let queue = DispatchQueue(label: "ptvision.capture", qos: .userInitiated)
    private let ci = CIContext()
    private var estimator: PoseEstimator?
    private var session: AVCaptureSession?
    private var sampleTask: Task<Void, Never>?
    private var lastStamp = CFAbsoluteTimeGetCurrent()
    private var fps = 0.0
    private var busy = false

    func setBackend(_ b: PoseBackend) {
        queue.async { self.estimator = b.make() }
    }

    func stop() {
        sampleTask?.cancel(); sampleTask = nil
        let s = session; session = nil
        queue.async { s?.stopRunning() }
    }

    // MARK: camera

    func startCamera() {
        queue.async {
            let s = AVCaptureSession()
            s.sessionPreset = .hd1280x720
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device), s.canAddInput(input) else { return }
            s.addInput(input)
            let out = AVCaptureVideoDataOutput()
            out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            out.alwaysDiscardsLateVideoFrames = true
            out.setSampleBufferDelegate(self, queue: self.queue)
            guard s.canAddOutput(out) else { return }
            s.addOutput(out)
            if let c = out.connection(with: .video), c.isVideoRotationAngleSupported(90) {
                c.videoRotationAngle = 90   // portrait buffers, so keypoints match what the PT sees
            }
            self.session = s
            s.startRunning()
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        process(pb)
    }

    // MARK: sample clip

    func startSample(_ url: URL) {
        sampleTask = Task.detached(priority: .userInitiated) { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.playOnce(url)
            }
        }
    }

    private func playOnce(_ url: URL) async {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let reader = try? AVAssetReader(asset: asset) else { return }
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        reader.startReading()
        let start = CFAbsoluteTimeGetCurrent()
        while !Task.isCancelled, let sb = out.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) {
            // pace to real time like a camera; drop frames if the model is slower
            let pts = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            let wait = pts - (CFAbsoluteTimeGetCurrent() - start)
            if wait > 0 { try? await Task.sleep(for: .seconds(wait)) }
            else if wait < -0.05 { continue }   // behind real time: drop, like a camera would
            let pixel = UncheckedBuffer(pb)
            queue.sync { self.process(pixel.buffer) }
        }
        reader.cancelReading()
    }

    // MARK: shared

    private func process(_ pb: CVPixelBuffer) {
        if busy { return }
        busy = true
        defer { busy = false }
        let pose = estimator?.estimate(pb)
        let now = CFAbsoluteTimeGetCurrent()
        let dt = now - lastStamp
        lastStamp = now
        if dt > 0 { fps = 0.9 * fps + 0.1 * (1 / dt) }
        guard let cg = ci.createCGImage(CIImage(cvPixelBuffer: pb), from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))) else { return }
        onFrame?(cg, pose, fps)
    }
}

private struct UncheckedBuffer: @unchecked Sendable {
    let buffer: CVPixelBuffer
    init(_ b: CVPixelBuffer) { buffer = b }
}
