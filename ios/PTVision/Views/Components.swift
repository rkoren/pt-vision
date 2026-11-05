// [REVIEW] Shared drawing: the sit-to-stand scene, skeleton, and small UI pieces.

import SwiftUI

/// 2D side-view stick figure computed from knee flexion and trunk lean (degrees).
struct SkeletonPose {
    var ankle, toe, knee, hip, shoulder, elbow, head: CGPoint

    /// Units: metres, origin at the ankle, y up. Facing +x.
    init(knee kDeg: Double, trunkLean: Double) {
        let r = { (d: Double) in d * .pi / 180 }
        let a = r(kDeg * 0.15), k = r(kDeg), tl = r(trunkLean)
        ankle = .zero
        toe = CGPoint(x: 0.18, y: 0)
        knee = CGPoint(x: 0.45 * sin(a), y: 0.45 * cos(a))
        hip = CGPoint(x: knee.x + 0.45 * sin(a - k), y: knee.y + 0.45 * cos(a - k))
        shoulder = CGPoint(x: hip.x + 0.52 * sin(tl), y: hip.y + 0.52 * cos(tl))
        elbow = CGPoint(x: shoulder.x + 0.12, y: shoulder.y - 0.14)
        head = CGPoint(x: shoulder.x + 0.15 * sin(tl), y: shoulder.y + 0.15 * cos(tl))
    }
}

/// The "video": a dark room, a chair, and the tracked skeleton. Stands in for real footage.
struct TrialScene: View {
    var t: Double
    var colored = true
    var showChairBack = true
    /// Draw a plain grey person (camera view) instead of the tracked skeleton.
    var personOnly = false
    /// Floor line as a fraction of height, and metres→points scale as a fraction of width
    /// (nil = fit to height, for the short video panels).
    var floor: CGFloat = 0.86
    var widthScale: CGFloat?

    static func geometry(_ size: CGSize, floor: CGFloat, widthScale: CGFloat?) -> (floorY: CGFloat, scale: CGFloat, ankleX: CGFloat) {
        (size.height * floor, widthScale.map { size.width * $0 } ?? size.height * 0.5, size.width * 0.56)
    }

    var body: some View {
        Canvas { ctx, size in
            let (floorY, scale, ankleX) = Self.geometry(size, floor: floor, widthScale: widthScale)
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(white: 0.14)))
            ctx.fill(Path(CGRect(x: 0, y: floorY, width: size.width, height: size.height - floorY)), with: .color(Color(white: 0.19)))
            func p(_ m: CGPoint) -> CGPoint { CGPoint(x: ankleX + m.x * scale, y: floorY - m.y * scale) }

            // chair
            let chair = Color(white: 0.3)
            let seatY = floorY - 0.47 * scale
            let seatX0 = ankleX - 0.62 * scale, seatW = 0.42 * scale
            ctx.fill(Path(roundedRect: CGRect(x: seatX0, y: seatY, width: seatW, height: 0.05 * scale), cornerRadius: 3), with: .color(chair))
            ctx.fill(Path(CGRect(x: seatX0 + 4, y: seatY, width: 0.035 * scale, height: floorY - seatY)), with: .color(chair))
            ctx.fill(Path(CGRect(x: seatX0 + seatW - 0.035 * scale - 4, y: seatY, width: 0.035 * scale, height: floorY - seatY)), with: .color(chair))
            if showChairBack {
                ctx.fill(Path(roundedRect: CGRect(x: seatX0, y: seatY - 0.45 * scale, width: 0.035 * scale, height: 0.47 * scale), cornerRadius: 3), with: .color(chair))
            }

            let knee = SampleTrial.knee(t), lean = SampleTrial.trunk(t)
            let pose = SkeletonPose(knee: knee, trunkLean: lean)
            if personOnly {
                let skin = Color(red: 0.72, green: 0.70, blue: 0.67)
                let w = 0.11 * scale
                func limb(_ a: CGPoint, _ b: CGPoint) {
                    var path = Path(); path.move(to: p(a)); path.addLine(to: p(b))
                    ctx.stroke(path, with: .color(skin), style: StrokeStyle(lineWidth: w, lineCap: .round))
                }
                limb(pose.ankle, pose.toe); limb(pose.ankle, pose.knee); limb(pose.knee, pose.hip)
                limb(pose.hip, pose.shoulder); limb(pose.shoulder, pose.elbow)
                let h = p(pose.head), hr = 0.1 * scale
                ctx.fill(Path(ellipseIn: CGRect(x: h.x - hr, y: h.y - hr, width: hr * 2, height: hr * 2)), with: .color(skin))
                return
            }

            let leg = colored ? SampleTrial.status(.knee, knee).color : .statusGreen
            let trunk = colored ? SampleTrial.status(.trunk, lean).color : .statusGreen
            let lw = max(4, min(size.height, scale * 2) * 0.022)

            func bone(_ a: CGPoint, _ b: CGPoint, _ c: Color) {
                var path = Path(); path.move(to: p(a)); path.addLine(to: p(b))
                ctx.stroke(path, with: .color(c), style: StrokeStyle(lineWidth: lw, lineCap: .round))
            }
            bone(pose.ankle, pose.toe, leg)
            bone(pose.ankle, pose.knee, leg)
            bone(pose.knee, pose.hip, leg)
            bone(pose.hip, pose.shoulder, trunk)
            bone(pose.shoulder, pose.elbow, .neutralBone)
            for j in [pose.ankle, pose.knee, pose.hip, pose.shoulder] {
                let c = p(j)
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - lw * 0.8, y: c.y - lw * 0.8, width: lw * 1.6, height: lw * 1.6)), with: .color(.white))
            }
            let h = p(pose.head), hr = 0.075 * scale
            ctx.stroke(Path(ellipseIn: CGRect(x: h.x - hr, y: h.y - hr, width: hr * 2, height: hr * 2)), with: .color(trunk), lineWidth: lw * 0.8)
        }
        .accessibilityLabel("Video frame with the tracked skeleton")
    }
}

struct SectionHeader: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.footnote)
            .textCase(.uppercase)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// On the screens that still show the bundled sample, whatever was filmed.
struct SampleDataBanner: View {
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.statusAmber)
            Text("**Sample data, not this recording.** The measurement comes from the laptop.")
                .font(.footnote)
                .foregroundStyle(.primary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.statusAmber.opacity(0.14), in: .rect(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }
}

struct ValidatedTag: View {
    let validated: Bool
    var body: some View {
        Text(validated ? "Validated" : "Estimate")
            .font(.caption2.weight(.semibold))
            .foregroundStyle(validated ? Color.accent : .secondary)
            .padding(.horizontal, 8).padding(.vertical, 2)
            .background(validated ? Color.accent.opacity(0.16) : Color(.tertiarySystemFill), in: .capsule)
    }
}

/// Where the patient's time sits against the age-band reference means.
struct ReferenceBar: View {
    let value: Double
    let highlightAges: ClosedRange<Int>?
    private let lo = 8.0, hi = 16.0

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let x = { (v: Double) in (v - lo) / (hi - lo) * w }
            ZStack(alignment: .leading) {
                Capsule().fill(Color(.tertiarySystemFill)).frame(height: 6)
                ForEach(SampleTrial.references) { band in
                    let on = band.ages == highlightAges
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(on ? Color.primary : Color.secondary.opacity(0.5))
                        .frame(width: 3, height: on ? 22 : 16)
                        .offset(x: x(band.meanSeconds) - 1.5)
                }
                Circle().fill(Color.accent)
                    .frame(width: 16, height: 16)
                    .overlay(Circle().stroke(.background, lineWidth: 3))
                    .shadow(color: .black.opacity(0.25), radius: 1.5, y: 1)
                    .offset(x: x(value) - 8)
            }
            .frame(height: 24)
        }
        .frame(height: 24)
        .accessibilityElement()
        .accessibilityLabel("Patient time \(value, specifier: "%.2f") seconds against age-band reference means")
    }
}

struct FilmingNote: View {
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.noteInk)
            VStack(alignment: .leading, spacing: 4) {
                Text("Joint angles: read with care").font(.subheadline.weight(.semibold)).foregroundStyle(Color.noteInk)
                Text("The phone was at an angle to their side, so angles are less exact. **Timings are not affected.** Next time, face the phone straight at their side.")
                    .font(.subheadline)
                    .foregroundStyle(.primary.opacity(0.9))
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.noteFill, in: .rect(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.noteInk.opacity(0.18)))
    }
}
