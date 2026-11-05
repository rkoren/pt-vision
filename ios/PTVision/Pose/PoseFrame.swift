// [REVIEW] Halpe-26 layout and the angle maths, ported from pt-vision
// (pose/layout.py HALPE26, kinematics/angles.py, kinematics/view.py thresholds).

import CoreGraphics
import Foundation

enum Halpe26 {
    static let count = 26
    static let nose = 0, lEye = 1, rEye = 2, lEar = 3, rEar = 4
    static let lShoulder = 5, rShoulder = 6, lElbow = 7, rElbow = 8, lWrist = 9, rWrist = 10
    static let lHip = 11, rHip = 12, lKnee = 13, rKnee = 14, lAnkle = 15, rAnkle = 16
    static let head = 17, neck = 18, hip = 19
    static let lBigToe = 20, rBigToe = 21, lSmallToe = 22, rSmallToe = 23, lHeel = 24, rHeel = 25

    static let bones: [(Int, Int)] = [
        (lAnkle, lKnee), (lKnee, lHip), (rAnkle, rKnee), (rKnee, rHip),
        (lHip, hip), (rHip, hip), (hip, neck), (neck, head),
        (neck, lShoulder), (neck, rShoulder), (lShoulder, lElbow), (lElbow, lWrist),
        (rShoulder, rElbow), (rElbow, rWrist),
        (lAnkle, lHeel), (lAnkle, lBigToe), (rAnkle, rHeel), (rAnkle, rBigToe),
    ]

    /// The protocol's required keypoints for sit-to-stand (sts_5x.toml [capture]).
    static let required = [lHip, rHip, lKnee, rKnee, lAnkle, rAnkle, lShoulder, rShoulder, neck, hip]
}

/// One person's keypoints in image pixel coordinates (y down), with per-joint confidence.
struct PoseFrame: Sendable {
    var points: [CGPoint]      // Halpe26.count entries
    var scores: [Float]
    var imageSize: CGSize
    var backend: String

    static let minScore: Float = 0.3   // pt-vision preprocess score_threshold

    func ok(_ i: Int) -> Bool { scores[i] >= Self.minScore }
    func p(_ i: Int) -> CGPoint { points[i] }

    // MARK: side and facing (kinematics/angles.py resolve_side / facing_direction)

    enum Side: String, Sendable { case left, right }

    /// The leg nearer the camera: the right leg of a person facing image-right, the left leg of one
    /// facing image-left. The leg with the higher keypoint confidence only when the facing is
    /// unknown; on oblique views that is usually the far leg.
    var side: Side {
        if facingVote > 0 { return .right }
        if facingVote < 0 { return .left }
        let l = (scores[Halpe26.lHip] + scores[Halpe26.lKnee] + scores[Halpe26.lAnkle]) / 3
        let r = (scores[Halpe26.rHip] + scores[Halpe26.rKnee] + scores[Halpe26.rAnkle]) / 3
        return l >= r ? .left : .right
    }

    /// +1 facing image-right, -1 facing image-left, 0 if neither the feet nor the face show it:
    /// toe ahead of heel on both feet, else the nose ahead of both ears.
    var facingVote: Int {
        var votes: [CGFloat] = []
        for (toe, heel) in [(Halpe26.lBigToe, Halpe26.lHeel), (Halpe26.rBigToe, Halpe26.rHeel)] where ok(toe) && ok(heel) {
            votes.append(p(toe).x - p(heel).x)
        }
        if votes.isEmpty {
            for ear in [Halpe26.lEar, Halpe26.rEar] where ok(Halpe26.nose) && ok(ear) {
                votes.append(p(Halpe26.nose).x - p(ear).x)
            }
        }
        guard !votes.isEmpty else { return 0 }
        let s = votes.sorted(), m = s.count / 2
        let median = s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
        return median >= 0 ? 1 : -1
    }

    /// The facing as a sign for the trunk lean, image-right when unknown.
    var facing: CGFloat { facingVote < 0 ? -1 : 1 }

    // MARK: angles (degrees)

    private static func interior(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Double {
        let u = CGVector(dx: a.x - b.x, dy: a.y - b.y), v = CGVector(dx: c.x - b.x, dy: c.y - b.y)
        let nu = hypot(u.dx, u.dy), nv = hypot(v.dx, v.dy)
        guard nu > 0, nv > 0 else { return .nan }
        let cosang = max(-1, min(1, (u.dx * v.dx + u.dy * v.dy) / (nu * nv)))
        return acos(Double(cosang)) * 180 / .pi
    }

    /// 0° with the shank aligned to the thigh, positive with flexion.
    var kneeFlexion: Double? {
        let (h, k, a) = side == .left ? (Halpe26.lHip, Halpe26.lKnee, Halpe26.lAnkle) : (Halpe26.rHip, Halpe26.rKnee, Halpe26.rAnkle)
        guard ok(h), ok(k), ok(a) else { return nil }
        return 180 - Self.interior(p(h), p(k), p(a))
    }

    /// 0° with the thigh aligned to the trunk (neck–hip–knee), positive with flexion.
    var hipFlexion: Double? {
        let (h, k) = side == .left ? (Halpe26.lHip, Halpe26.lKnee) : (Halpe26.rHip, Halpe26.rKnee)
        guard ok(Halpe26.neck), ok(h), ok(k) else { return nil }
        return 180 - Self.interior(p(Halpe26.neck), p(h), p(k))
    }

    /// Pelvis→neck from image vertical, positive leaning toward the facing direction.
    var trunkLean: Double? {
        guard ok(Halpe26.hip), ok(Halpe26.neck) else { return nil }
        let d = CGVector(dx: p(Halpe26.neck).x - p(Halpe26.hip).x, dy: p(Halpe26.neck).y - p(Halpe26.hip).y)
        return Double(facing) * atan2(Double(d.dx), Double(-d.dy)) * 180 / .pi
    }

    func angle(_ a: Angle) -> Double? {
        switch a {
        case .knee: kneeFlexion
        case .hip: hipFlexion
        case .trunk: trunkLean
        }
    }

    // MARK: capture checks (quality/checks.py)

    /// Shoulder separation / trunk length. < 0.35 side view, < 0.50 oblique, else frontal.
    var viewRatio: Double? {
        guard ok(Halpe26.lShoulder), ok(Halpe26.rShoulder), ok(Halpe26.neck), ok(Halpe26.hip) else { return nil }
        let sep = abs(p(Halpe26.lShoulder).x - p(Halpe26.rShoulder).x)
        let trunk = hypot(p(Halpe26.neck).x - p(Halpe26.hip).x, p(Halpe26.neck).y - p(Halpe26.hip).y)
        guard trunk > 1 else { return nil }
        return Double(sep / trunk)
    }

    /// Every required keypoint confident and at least 2% inside the frame edge.
    var wholeBodyInFrame: Bool {
        let mx = imageSize.width * 0.02, my = imageSize.height * 0.02
        return Halpe26.required.allSatisfy { i in
            ok(i) && p(i).x > mx && p(i).x < imageSize.width - mx && p(i).y > my && p(i).y < imageSize.height - my
        }
    }

    var meanRequiredScore: Float {
        Halpe26.required.map { scores[$0] }.reduce(0, +) / Float(Halpe26.required.count)
    }
}

/// The phone's tilt from upright portrait, from gravity in the device's axes (x right, y up, z out
/// of the screen). Roll turns the picture, which adds to the trunk lean degree for degree; pitch
/// tips the camera down or up, which the angles take as level.
struct PhoneTilt: Sendable, Equatable {
    var rollDegrees: Double
    var pitchDownDegrees: Double

    init(gravityX x: Double, y: Double, z: Double) {
        rollDegrees = atan2(x, -y) * 180 / .pi
        pitchDownDegrees = asin(max(-1, min(1, -z))) * 180 / .pi
    }

    var worstDegrees: Double { max(abs(rollDegrees), abs(pitchDownDegrees)) }

    var fix: String {
        if abs(pitchDownDegrees) >= abs(rollDegrees) {
            let way = pitchDownDegrees > 0 ? "down" : "up"
            return "Tipped \(Int(abs(pitchDownDegrees).rounded()))° \(way) · aim it level at their side"
        }
        return "Rotated \(Int(abs(rollDegrees).rounded()))° · straighten the phone"
    }
}

/// Counts full stands live from knee flexion with hysteresis (seated > 70°, standing < 25°).
struct StandCounter: Sendable {
    private(set) var count = 0
    private var seatedSeen = false

    mutating func update(knee: Double?) {
        guard let knee else { return }
        if knee > 70 { seatedSeen = true }
        if seatedSeen, knee < 25 { count += 1; seatedSeen = false }
    }
}
