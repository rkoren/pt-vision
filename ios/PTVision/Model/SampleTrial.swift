// [REVIEW] Prototype data model. The sample trial reproduces the summary numbers from
// pt-vision's bundled sts_side demo run; per-rep times and angle curves are synthetic.

import Foundation

enum TestKind: String, CaseIterable, Identifiable, Sendable {
    case sts5x, sts30s, pose
    var id: String { rawValue }

    var name: String {
        switch self {
        case .sts5x: "Five Times Sit-to-Stand"
        case .sts30s: "30-Second Chair Stand"
        case .pose: "Pose only"
        }
    }

    var shortName: String {
        switch self {
        case .sts5x: "5xSTS"
        case .sts30s: "30s chair stand"
        case .pose: "Pose"
        }
    }

    var subtitle: String {
        switch self {
        case .sts5x: "Time for five stands, arms crossed"
        case .sts30s: "Stands completed in 30 seconds"
        case .pose: "Skeleton tracking, no measurements"
        }
    }
}

enum Sex: String, CaseIterable, Identifiable, Sendable {
    case unspecified = "Not given", female = "Female", male = "Male"
    var id: String { rawValue }
}

struct Patient: Sendable {
    var age: Int = 71
    var heightInches: Int = 66
    var sex: Sex = .female

    var heightText: String { "\(heightInches / 12)′ \(heightInches % 12)″" }
}

enum Angle: String, CaseIterable, Identifiable, Sendable {
    case knee, hip, trunk
    var id: String { rawValue }

    var label: String {
        switch self {
        case .knee: "Knee"
        case .hip: "Hip"
        case .trunk: "Trunk lean"
        }
    }

    /// Provisional "expected range" bands from sts_5x.toml (marked "confirm with PT").
    var okRange: ClosedRange<Double> {
        switch self {
        case .knee: 0...110
        case .hip: 0...120
        case .trunk: 0...40
        }
    }

    var warnRange: ClosedRange<Double> {
        switch self {
        case .knee: 0...130
        case .hip: 0...140
        case .trunk: -10...55
        }
    }

    var axis: ClosedRange<Double> {
        switch self {
        case .knee, .hip: 0...130
        case .trunk: -10...50
        }
    }
}

enum MovementPhase: String, Sendable {
    case seated, rising, standing, sitting

    var word: String {
        switch self {
        case .seated: "seated"
        case .rising: "rising"
        case .standing: "standing"
        case .sitting: "sitting"
        }
    }
}

struct Metric: Identifiable, Sendable {
    let id = UUID()
    let label: String
    let detail: String
    let value: String
    let error: String
    let validated: Bool
}

struct ReferenceBand: Identifiable, Sendable {
    let id = UUID()
    let ages: ClosedRange<Int>
    let meanSeconds: Double
}

/// The bundled sample: 13.3 s clip, five reps. Summary values match pt-vision's demo report.
enum SampleTrial {
    static let duration = 13.3
    static let fps = 30.0
    static let seatOffs: [Double] = [2.25, 4.68, 7.11, 9.54, 11.97]
    static let rise = 0.65, stand = 0.40, descent = 0.72
    static let riseTimes: [Double] = [0.62, 0.69, 0.66, 0.60, 0.68]

    static let totalTime = 10.43
    static let totalError = 0.07

    static let references: [ReferenceBand] = [
        .init(ages: 60...69, meanSeconds: 11.4),
        .init(ages: 70...79, meanSeconds: 12.6),
        .init(ages: 80...89, meanSeconds: 14.8),
    ]
    static let referenceCitation = "Bohannon 2006"

    static func reference(for age: Int) -> ReferenceBand? {
        references.first { $0.ages.contains(age) }
    }

    static let metrics: [Metric] = [
        .init(label: "Repetitions", detail: "Full stands found", value: "5", error: "", validated: true),
        // tier 2 in clinical/metrics/sts.py: the timings' spread is the SD over reps, not an error;
        // the trunk lean's error is its RMSE against lab motion capture (ANGLE_RMSE_DEG)
        .init(label: "Rise time", detail: "Seat-off to standing", value: "0.65 s", error: "SD 0.04", validated: false),
        .init(label: "Sit time", detail: "Start of descent to seated", value: "0.72 s", error: "SD 0.03", validated: false),
        .init(label: "Cycle time", detail: "Seat-off to next seat-off", value: "2.43 s", error: "SD 0.12", validated: false),
        .init(label: "Trunk lean at seat-off", detail: "Forward from vertical", value: "12°", error: "RMSE 2.6°", validated: false),
    ]

    // MARK: synthetic kinematics

    static func phase(at t: Double) -> (rep: Int?, phase: MovementPhase, dt: Double) {
        guard let i = seatOffs.lastIndex(where: { t >= $0 }) else { return (nil, .seated, 0) }
        let dt = t - seatOffs[i]
        if dt < rise { return (i, .rising, dt) }
        if i == seatOffs.count - 1 || dt < rise + stand { return (i, .standing, dt) }
        if dt < rise + stand + descent { return (i, .sitting, dt) }
        return (i, .seated, dt)
    }

    private static func ease(_ x: Double) -> Double {
        0.5 - 0.5 * cos(.pi * min(1, max(0, x)))
    }

    static func knee(_ t: Double) -> Double {
        let p = phase(at: t)
        switch p.phase {
        case .rising: return 95 - 93 * ease(p.dt / rise)
        case .standing: return 2
        case .sitting: return 2 + 93 * ease((p.dt - rise - stand) / descent)
        case .seated: return 95
        }
    }

    private static func bump(_ t: Double, amp: Double, width: Double) -> Double {
        seatOffs.reduce(0) { $0 + amp * exp(-pow((t - $1 - 0.08) / width, 2)) }
    }

    static func hip(_ t: Double) -> Double { 5 + (knee(t) - 2) / 93 * 92 + bump(t, amp: 14, width: 0.28) }
    static func trunk(_ t: Double) -> Double { (knee(t) - 2) / 93 * 5 + bump(t, amp: 8, width: 0.3) }

    static func value(_ angle: Angle, at t: Double) -> Double {
        switch angle {
        case .knee: knee(t)
        case .hip: hip(t)
        case .trunk: trunk(t)
        }
    }

    struct Sample: Identifiable, Sendable {
        var id: Double { t }
        let t: Double
        let value: Double
    }

    static func series(_ angle: Angle, step: Double = 0.05) -> [Sample] {
        stride(from: 0, through: duration, by: step).map { Sample(t: $0, value: value(angle, at: $0)) }
    }

    struct PhaseSpan: Identifiable, Sendable {
        let id = UUID()
        let start: Double
        let end: Double
        let phase: MovementPhase
    }

    static let phaseSpans: [PhaseSpan] = seatOffs.enumerated().flatMap { i, s -> [PhaseSpan] in
        if i == seatOffs.count - 1 {
            return [.init(start: s, end: s + rise, phase: .rising), .init(start: s + rise, end: duration, phase: .standing)]
        }
        return [
            .init(start: s, end: s + rise, phase: .rising),
            .init(start: s + rise, end: s + rise + stand, phase: .standing),
            .init(start: s + rise + stand, end: s + rise + stand + descent, phase: .sitting),
        ]
    }

    static func status(_ angle: Angle, _ v: Double) -> BoneStatus {
        if angle.okRange.contains(v) { return .inRange }
        if angle.warnRange.contains(v) { return .nearLimit }
        return .outside
    }
}

enum BoneStatus: Sendable { case inRange, nearLimit, outside, notAssessed }
