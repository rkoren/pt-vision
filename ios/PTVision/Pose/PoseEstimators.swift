// [REVIEW] Two on-device pose backends behind one protocol:
// - RTMPoseEstimator: pt-vision's own models (YOLOX-tiny person detector + RTMPose-s Halpe-26,
//   MMPose, Apache-2.0) run with ONNX Runtime. Pre/post-processing mirrors rtmlib 0.0.16
//   (BGR input, letterbox 114 padding, 1.25 bbox padding, affine crop, SimCC argmax / 2).
// - VisionPoseEstimator: Apple's built-in body pose (19 joints, no feet), for comparison.

import CoreVideo
import Foundation
import OnnxRuntimeBindings
import Vision

protocol PoseEstimator: AnyObject, Sendable {
    var name: String { get }
    /// Called on the capture queue, one frame at a time. BGRA pixel buffer, upright.
    func estimate(_ pixelBuffer: CVPixelBuffer) -> PoseFrame?
}

enum PoseBackend: String, CaseIterable, Identifiable, Sendable {
    case rtmpose, vision
    var id: String { rawValue }
    var label: String {
        switch self {
        case .rtmpose: "RTMPose (pt-vision)"
        case .vision: "Apple Vision"
        }
    }

    func make() -> PoseEstimator? {
        switch self {
        case .rtmpose: try? RTMPoseEstimator()
        case .vision: VisionPoseEstimator()
        }
    }
}

// MARK: - BGRA access

private struct BGRAImage {
    let base: UnsafePointer<UInt8>
    let width: Int, height: Int, rowBytes: Int

    @inline(__always) func pixel(_ x: Int, _ y: Int) -> (b: Float, g: Float, r: Float) {
        let p = base + y * rowBytes + x * 4
        return (Float(p[0]), Float(p[1]), Float(p[2]))
    }

    /// Bilinear sample; outside the image is 0 (cv2.warpAffine BORDER_CONSTANT).
    @inline(__always) func bilinear(_ x: Float, _ y: Float) -> (b: Float, g: Float, r: Float) {
        let x0 = Int(floor(x)), y0 = Int(floor(y))
        guard x0 >= 0, y0 >= 0, x0 + 1 < width, y0 + 1 < height else {
            if x0 >= 0, y0 >= 0, x0 < width, y0 < height { return pixel(x0, y0) }
            return (0, 0, 0)
        }
        let fx = x - Float(x0), fy = y - Float(y0)
        let a = pixel(x0, y0), b = pixel(x0 + 1, y0), c = pixel(x0, y0 + 1), d = pixel(x0 + 1, y0 + 1)
        func mix(_ p: Float, _ q: Float, _ r: Float, _ s: Float) -> Float {
            (p * (1 - fx) + q * fx) * (1 - fy) + (r * (1 - fx) + s * fx) * fy
        }
        return (mix(a.b, b.b, c.b, d.b), mix(a.g, b.g, c.g, d.g), mix(a.r, b.r, c.r, d.r))
    }
}

private func withBGRA<T>(_ pb: CVPixelBuffer, _ body: (BGRAImage) throws -> T) rethrows -> T? {
    guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA else { return nil }
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
    let img = BGRAImage(base: base.assumingMemoryBound(to: UInt8.self),
                        width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                        rowBytes: CVPixelBufferGetBytesPerRow(pb))
    return try body(img)
}

// MARK: - RTMPose via ONNX Runtime

final class RTMPoseEstimator: PoseEstimator, @unchecked Sendable {
    let name = "RTMPose-s · Halpe-26"

    private let env: ORTEnv
    private let det: ORTSession
    private let pose: ORTSession
    private let detInput: String, poseInput: String
    private let detOutputs: [String], poseOutputs: [String]

    private let detSize = 416                     // YOLOX-tiny input (square)
    private let poseW = 192, poseH = 256          // RTMPose input (w, h)
    private let mean: (Float, Float, Float) = (123.675, 116.28, 103.53)   // applied to B, G, R as rtmlib does
    private let std: (Float, Float, Float) = (58.395, 57.12, 57.375)

    private var trackBox: CGRect?
    private var framesSinceDetect = 0

    init() throws {
        guard let detPath = Bundle.main.path(forResource: "yolox_tiny", ofType: "onnx"),
              let posePath = Bundle.main.path(forResource: "rtmpose_s_halpe26", ofType: "onnx")
        else { throw CocoaError(.fileNoSuchFile) }
        let env = try ORTEnv(loggingLevel: .warning)
        self.env = env
        det = try Self.session(env, detPath)
        pose = try Self.session(env, posePath)
        detInput = try det.inputNames()[0]
        poseInput = try pose.inputNames()[0]
        detOutputs = try det.outputNames()
        poseOutputs = try pose.outputNames()
    }

    private static func session(_ env: ORTEnv, _ path: String) throws -> ORTSession {
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(2)
        try opts.setGraphOptimizationLevel(.all)
        #if !targetEnvironment(simulator)
        let coreml = ORTCoreMLExecutionProviderOptions()
        coreml.createMLProgram = true
        try? opts.appendCoreMLExecutionProvider(with: coreml)
        #endif
        return try ORTSession(env: env, modelPath: path, sessionOptions: opts)
    }

    func estimate(_ pixelBuffer: CVPixelBuffer) -> PoseFrame? {
        let result = try? withBGRA(pixelBuffer) { img -> PoseFrame? in
            if trackBox == nil || framesSinceDetect >= 6 {
                if let box = try detect(img) { trackBox = box }
                framesSinceDetect = 0
            }
            framesSinceDetect += 1
            guard let box = trackBox else { return nil }
            let frame = try estimatePose(img, box: box)
            // Track: next frame's box from this frame's confident keypoints; re-detect if lost.
            if frame.meanRequiredScore < 0.3 {
                trackBox = nil
            } else {
                let good = frame.points.indices.filter { frame.scores[$0] >= 0.3 }.map { frame.points[$0] }
                if let minX = good.map(\.x).min(), let maxX = good.map(\.x).max(),
                   let minY = good.map(\.y).min(), let maxY = good.map(\.y).max() {
                    let padX = (maxX - minX) * 0.1, padY = (maxY - minY) * 0.1
                    trackBox = CGRect(x: minX - padX, y: minY - padY, width: maxX - minX + 2 * padX, height: maxY - minY + 2 * padY)
                }
            }
            return frame
        }
        return result ?? nil ?? nil
    }

    /// YOLOX: top-left letterbox to 416², pad 114, raw BGR 0–255, NCHW. Returns the largest person box.
    private func detect(_ img: BGRAImage) throws -> CGRect? {
        let s = detSize
        let ratio = min(Float(s) / Float(img.height), Float(s) / Float(img.width))
        let nw = Int(Float(img.width) * ratio), nh = Int(Float(img.height) * ratio)
        var input = [Float](repeating: 114, count: 3 * s * s)
        input.withUnsafeMutableBufferPointer { buf in
            let plane = s * s
            for y in 0..<nh {
                let sy = min(img.height - 1, Int((Float(y) + 0.5) / ratio))
                for x in 0..<nw {
                    let sx = min(img.width - 1, Int((Float(x) + 0.5) / ratio))
                    let px = img.pixel(sx, sy)
                    let i = y * s + x
                    buf[i] = px.b; buf[plane + i] = px.g; buf[2 * plane + i] = px.r
                }
            }
        }
        let out = try run(det, input: detInput, outputs: detOutputs, data: input, shape: [1, 3, s, s])
        guard let (dets, shape) = out.first(where: { $0.shape.last.map { $0 >= 5 } ?? false }).map({ ($0.data, $0.shape) }) else { return nil }
        let stride = shape.last!
        let n = dets.count / stride
        var best: CGRect?, bestArea: CGFloat = 0
        for i in 0..<n {
            let o = i * stride
            let score: Float = stride == 5 ? dets[o + 4] : dets[o + 4] * dets[o + 5]   // end2end vs raw head
            guard score > 0.3 else { continue }
            var r: CGRect
            if stride == 5 {
                r = CGRect(x: CGFloat(dets[o] / ratio), y: CGFloat(dets[o + 1] / ratio),
                           width: CGFloat((dets[o + 2] - dets[o]) / ratio), height: CGFloat((dets[o + 3] - dets[o + 1]) / ratio))
            } else {
                continue   // the bundled end2end model includes NMS; raw heads are not expected
            }
            r = r.intersection(CGRect(x: 0, y: 0, width: img.width, height: img.height))
            let area = r.width * r.height
            if area > bestArea { best = r; bestArea = area }
        }
        return best
    }

    /// RTMPose: bbox → centre/scale ×1.25 at 192:256, affine crop, (v − mean) / std on BGR, SimCC decode.
    private func estimatePose(_ img: BGRAImage, box: CGRect) throws -> PoseFrame {
        let cx = Float(box.midX), cy = Float(box.midY)
        var sw = Float(box.width) * 1.25, sh = Float(box.height) * 1.25
        let aspect = Float(poseW) / Float(poseH)
        if sw > sh * aspect { sh = sw / aspect } else { sw = sh * aspect }
        let kx = sw / Float(poseW), ky = sh / Float(poseH)

        var input = [Float](repeating: 0, count: 3 * poseW * poseH)
        input.withUnsafeMutableBufferPointer { buf in
            let plane = poseW * poseH
            for v in 0..<poseH {
                let sy = cy + (Float(v) - Float(poseH) / 2) * ky
                for u in 0..<poseW {
                    let sx = cx + (Float(u) - Float(poseW) / 2) * kx
                    let px = img.bilinear(sx, sy)
                    let i = v * poseW + u
                    buf[i] = (px.b - mean.0) / std.0
                    buf[plane + i] = (px.g - mean.1) / std.1
                    buf[2 * plane + i] = (px.r - mean.2) / std.2
                }
            }
        }
        let outs = try run(pose, input: poseInput, outputs: poseOutputs, data: input, shape: [1, 3, poseH, poseW])
        // simcc_x is [1, K, W*2], simcc_y is [1, K, H*2]
        guard let sx = outs.first(where: { $0.shape.last == poseW * 2 }),
              let sy = outs.first(where: { $0.shape.last == poseH * 2 }) else { throw CocoaError(.coderInvalidValue) }
        let k = Halpe26.count, wx = poseW * 2, wy = poseH * 2
        var points = [CGPoint](repeating: .zero, count: k), scores = [Float](repeating: 0, count: k)
        for j in 0..<k {
            let (ix, mx) = argmax(sx.data, from: j * wx, count: wx)
            let (iy, my) = argmax(sy.data, from: j * wy, count: wy)
            let score = 0.5 * (mx + my)
            let lx = Float(ix) / 2, ly = Float(iy) / 2
            points[j] = CGPoint(x: CGFloat(lx / Float(poseW) * sw + cx - sw / 2),
                                y: CGFloat(ly / Float(poseH) * sh + cy - sh / 2))
            scores[j] = score
        }
        return PoseFrame(points: points, scores: scores, imageSize: CGSize(width: img.width, height: img.height), backend: name)
    }

    private func argmax(_ a: [Float], from: Int, count: Int) -> (Int, Float) {
        var bi = 0, bv = -Float.infinity
        for i in 0..<count where a[from + i] > bv { bv = a[from + i]; bi = i }
        return (bi, bv)
    }

    private struct Tensor { let data: [Float]; let shape: [Int] }

    private func run(_ session: ORTSession, input: String, outputs: [String], data: [Float], shape: [Int]) throws -> [Tensor] {
        let bytes = NSMutableData(bytes: data, length: data.count * MemoryLayout<Float>.stride)
        let value = try ORTValue(tensorData: bytes, elementType: .float, shape: shape.map { NSNumber(value: $0) })
        let res = try session.run(withInputs: [input: value], outputNames: Set(outputs), runOptions: nil)
        return try outputs.compactMap { name in
            guard let v = res[name] else { return nil }
            let info = try v.tensorTypeAndShapeInfo()
            guard info.elementType == .float else { return nil }
            let raw = try v.tensorData() as Data
            let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return Tensor(data: floats, shape: info.shape.map(\.intValue))
        }
    }
}

// MARK: - Apple Vision

final class VisionPoseEstimator: PoseEstimator, @unchecked Sendable {
    let name = "Apple Vision · 19 joints"
    private let request = VNDetectHumanBodyPoseRequest()

    private static let map: [(VNHumanBodyPoseObservation.JointName, Int)] = [
        (.nose, Halpe26.nose), (.leftEye, Halpe26.lEye), (.rightEye, Halpe26.rEye),
        (.leftEar, Halpe26.lEar), (.rightEar, Halpe26.rEar),
        (.leftShoulder, Halpe26.lShoulder), (.rightShoulder, Halpe26.rShoulder),
        (.leftElbow, Halpe26.lElbow), (.rightElbow, Halpe26.rElbow),
        (.leftWrist, Halpe26.lWrist), (.rightWrist, Halpe26.rWrist),
        (.leftHip, Halpe26.lHip), (.rightHip, Halpe26.rHip),
        (.leftKnee, Halpe26.lKnee), (.rightKnee, Halpe26.rKnee),
        (.leftAnkle, Halpe26.lAnkle), (.rightAnkle, Halpe26.rAnkle),
        (.neck, Halpe26.neck), (.root, Halpe26.hip),
    ]

    func estimate(_ pixelBuffer: CVPixelBuffer) -> PoseFrame? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        guard (try? handler.perform([request])) != nil,
              let obs = request.results?.first,
              let joints = try? obs.recognizedPoints(.all) else { return nil }
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer)), h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        var points = [CGPoint](repeating: .zero, count: Halpe26.count)
        var scores = [Float](repeating: 0, count: Halpe26.count)
        for (name, i) in Self.map {
            guard let jp = joints[name] else { continue }
            points[i] = CGPoint(x: jp.location.x * w, y: (1 - jp.location.y) * h)
            scores[i] = jp.confidence
        }
        // Head: no Vision joint; use the nose so the skeleton has a head bone.
        points[Halpe26.head] = points[Halpe26.nose]
        scores[Halpe26.head] = scores[Halpe26.nose]
        return PoseFrame(points: points, scores: scores, imageSize: CGSize(width: w, height: h), backend: name)
    }
}
