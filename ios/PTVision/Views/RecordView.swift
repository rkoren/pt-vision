// [REVIEW] Screen 2: live camera (or the sample clip) with pt-vision's pose model running per
// frame. The skeleton, joint angles, filming checks and the stand counter are all live.

import SwiftUI

struct RecordView: View {
    @Environment(Session.self) private var session
    @Environment(\.dismiss) private var dismiss
    @Namespace private var glass
    @State private var capture = LiveCapture()
    @State private var recordStart: Date?

    var body: some View {
        ZStack {
            viewfinder.ignoresSafeArea()
            VStack(spacing: 0) {
                topBar
                liveAngles
                    .padding(.top, 10)
                Spacer()
                statusPanel
                    .padding(.horizontal, 12)
                    .padding(.bottom, 18)
                shutter
                Text(hint)
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.85))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .padding(.top, 10)
                    .padding(.bottom, 6)
            }
        }
        .toolbarVisibility(.hidden, for: .navigationBar)
        .preferredColorScheme(.dark)
        .onAppear {
            if session.useSample { capture.source = .sample }
            capture.start()
        }
        .onDisappear { capture.stop() }
        .animation(.spring(duration: 0.35, bounce: 0), value: capture.ready)
        .animation(.spring(duration: 0.35, bounce: 0), value: capture.recording)
    }

    // MARK: viewfinder + overlay

    private var viewfinder: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                if let img = capture.image {
                    Image(decorative: img, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: geo.size.width, height: geo.size.height)
                        .clipped()
                } else {
                    ProgressView().tint(.white)
                }
                if let pose = capture.pose {
                    SkeletonOverlay(pose: pose).frame(width: geo.size.width, height: geo.size.height)
                }
                let inFrame = capture.pose?.wholeBodyInFrame ?? false
                RoundedRectangle(cornerRadius: 22)
                    .stroke(inFrame ? Color.statusGreen : Color.statusAmber, style: StrokeStyle(lineWidth: 2.5, dash: [10, 8]))
                    .padding(.horizontal, geo.size.width * 0.06)
                    .padding(.top, geo.size.height * 0.12)
                    .padding(.bottom, geo.size.height * 0.08)
                    .opacity(capture.recording ? 0 : 0.9)
                    .animation(.easeOut(duration: 0.25), value: inFrame)
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Camera view with the tracked skeleton")
    }

    private var topBar: some View {
        HStack(alignment: .top) {
            Button { dismiss() } label: {
                Image(systemName: "xmark").font(.body.weight(.semibold)).frame(width: 30, height: 30).foregroundStyle(.white)
            }
            .buttonStyle(.glass)
            .buttonBorderShape(.circle)
            .accessibilityLabel("Close camera")

            Spacer()

            Group {
                if capture.recording {
                    Label(clock, systemImage: "circle.fill")
                        .font(.headline.monospacedDigit())
                        .padding(.horizontal, 14).padding(.vertical, 7)
                        .glassEffect(.regular.tint(.statusRed), in: .capsule)
                        .glassEffectID("status", in: glass)
                } else {
                    VStack(spacing: 2) {
                        Text(capture.source == .sample ? "Demo clip · \(session.test.shortName)" : "\(session.test.shortName) · side view").font(.subheadline.weight(.semibold))
                        Text("\(capture.pose?.backend ?? capture.backend.label) · \(Int(capture.modelFPS.rounded())) fps")
                            .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 14).padding(.vertical, 6)
                    .glassEffect(.regular, in: .capsule)
                    .glassEffectID("status", in: glass)
                }
            }
            Spacer()

            Menu {
                Picker("Model", selection: $capture.backend) {
                    ForEach(PoseBackend.allCases) { Text($0.label).tag($0) }
                }
                Picker("Source", selection: $capture.source) {
                    ForEach(FrameSource.allCases) { Text($0.label).tag($0) }
                }
            } label: {
                Image(systemName: "slider.horizontal.3").font(.body.weight(.semibold)).frame(width: 30, height: 30).foregroundStyle(.white)
            }
            .buttonStyle(.glass)
            .buttonBorderShape(.circle)
            .accessibilityLabel("Model and source")
            .disabled(capture.recording)
        }
        .padding(.horizontal, 14)
        .padding(.top, 6)
    }

    /// Not the measurement: the phone runs the smaller RTMPose-s, unfiltered, with no view
    /// correction, so these can differ from what the laptop reports for the same recording.
    private var liveAngles: some View {
        VStack(spacing: 6) {
            liveAngleChips
            Text("Live preview · the laptop gives the measurement")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.75))
                .padding(.horizontal, 10).padding(.vertical, 3)
                .background(.black.opacity(0.35), in: .capsule)
        }
    }

    private var liveAngleChips: some View {
        GlassEffectContainer(spacing: 8) {
            HStack(spacing: 8) {
                ForEach(Angle.allCases) { a in
                    let v = capture.pose?.angle(a)
                    VStack(spacing: 0) {
                        Text(a == .trunk ? "Trunk" : a.label).font(.caption2).foregroundStyle(.secondary)
                        Text(v.map { "\(Int($0.rounded()))°" } ?? "–")
                            .font(.title3.weight(.semibold).monospacedDigit())
                            .contentTransition(.numericText())
                    }
                    .frame(width: 74, height: 46)
                    .glassEffect(.regular, in: .rect(cornerRadius: 14))
                }
            }
        }
    }

    private var statusPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            if capture.recording {
                VStack(alignment: .leading, spacing: 4) {
                    Text(capture.stands == 0 ? "Say “Ready, go” when they are seated and still" : "Stands seen: \(capture.stands) of 5")
                        .font(.headline)
                        .contentTransition(.numericText())
                    Text("Keep filming 2 seconds after the 5th stand, then tap stop.")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                .padding(16)
            } else {
                let checks = capture.checks
                let open = checks.filter { $0.state != .pass }
                HStack(spacing: 12) {
                    Image(systemName: capture.ready ? "checkmark.seal.fill" : "viewfinder")
                        .font(.title3)
                        .foregroundStyle(capture.ready ? Color.accent : .white)
                        .contentTransition(.symbolEffect(.replace))
                    Text(capture.ready
                         ? (open.isEmpty ? "Ready to film" : "Ready to film · \(open.count) note\(open.count == 1 ? "" : "s")")
                         : "\(checks.count - open.count) of \(checks.count) filming checks passed")
                        .font(.subheadline.weight(.semibold))
                        .contentTransition(.numericText())
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 16).padding(.vertical, 8)
                ForEach(open) { check in
                    HStack(spacing: 12) {
                        Image(systemName: icon(check.state))
                            .font(.title3)
                            .foregroundStyle(color(check.state))
                            .contentTransition(.symbolEffect(.replace))
                        VStack(alignment: .leading, spacing: 1) {
                            Text(check.label).font(.subheadline.weight(.medium))
                            Text(check.fix).font(.footnote).foregroundStyle(Color(red: 1, green: 0.82, blue: 0.48))
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 16).padding(.vertical, 6)
                    .accessibilityElement(children: .combine)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
        }
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 24))
    }

    private func icon(_ s: CheckState) -> String {
        switch s {
        case .pass: "checkmark.circle.fill"
        case .warn: "exclamationmark.circle.fill"
        case .fail: "xmark.circle.fill"
        case .pending: "circle.dashed"
        }
    }

    private func color(_ s: CheckState) -> Color {
        switch s {
        case .pass: .statusGreen
        case .warn: .statusAmber
        case .fail: .statusRed
        case .pending: .white.opacity(0.6)
        }
    }

    private var shutter: some View {
        Button {
            if capture.recording {
                capture.recording = false
                session.path.append(.processing)
            } else {
                recordStart = .now
                capture.recording = true
            }
        } label: {
            ZStack {
                Circle().strokeBorder(.white, lineWidth: 5).frame(width: 78, height: 78)
                RoundedRectangle(cornerRadius: capture.recording ? 8 : 31)
                    .fill(Color.statusRed)
                    .frame(width: capture.recording ? 30 : 62, height: capture.recording ? 30 : 62)
            }
            .contentShape(.circle)
        }
        .buttonStyle(ShutterPressStyle())
        .accessibilityLabel(capture.recording ? "Stop recording" : "Start recording")
        .sensoryFeedback(.impact(weight: .medium), trigger: capture.recording)
        .sensoryFeedback(.increase, trigger: capture.stands)
    }

    private var clock: String {
        let s = recordStart.map { Int(Date.now.timeIntervalSince($0)) } ?? 0
        return String(format: "%02d:%02d", s / 60, s % 60)
    }

    private var hint: String {
        if capture.cameraDenied { return "Camera access is off. Showing the sample clip instead." }
        if capture.recording { return "Tap stop when they have finished." }
        return capture.ready ? "All set. Tap to start recording." : "You can record now, but problems will be noted on the result."
    }
}

/// Halpe-26 skeleton drawn over an aspect-filled frame. Bones use the protocol's angle bands
/// (knee for the legs, trunk lean for the trunk); low-confidence joints are hidden.
struct SkeletonOverlay: View {
    let pose: PoseFrame

    var body: some View {
        Canvas { ctx, size in
            let W = pose.imageSize.width, H = pose.imageSize.height
            guard W > 0, H > 0 else { return }
            let s = max(size.width / W, size.height / H)
            let ox = (size.width - W * s) / 2, oy = (size.height - H * s) / 2
            func v(_ i: Int) -> CGPoint { CGPoint(x: pose.points[i].x * s + ox, y: pose.points[i].y * s + oy) }

            let legColor = pose.kneeFlexion.map { SampleTrial.status(.knee, $0).color } ?? .neutralBone
            let trunkColor = pose.trunkLean.map { SampleTrial.status(.trunk, $0).color } ?? .neutralBone
            let legs: Set<Int> = [Halpe26.lAnkle, Halpe26.rAnkle, Halpe26.lKnee, Halpe26.rKnee, Halpe26.lHeel, Halpe26.rHeel, Halpe26.lBigToe, Halpe26.rBigToe]
            let lw = max(3, size.width * 0.012)

            for (a, b) in Halpe26.bones where pose.ok(a) && pose.ok(b) {
                let color: Color
                if legs.contains(a) || legs.contains(b) { color = legColor }
                else if (a == Halpe26.hip && b == Halpe26.neck) { color = trunkColor }
                else { color = .neutralBone }
                var p = Path(); p.move(to: v(a)); p.addLine(to: v(b))
                ctx.stroke(p, with: .color(color.opacity(Double(min(1, pose.scores[a] + 0.3)))), style: StrokeStyle(lineWidth: lw, lineCap: .round))
            }
            for i in 0..<Halpe26.count where pose.ok(i) {
                let c = v(i), r = lw * 0.75
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(.white))
            }
        }
        .allowsHitTesting(false)
    }
}

private struct ShutterPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1)
            .animation(.spring(duration: 0.2, bounce: 0), value: configuration.isPressed)
    }
}
