// [REVIEW] Screen 4: video with overlays, glass transport controls, headline result,
// filming note, interactive angle chart, reps, measurements, notes and share sheet.

import Charts
import SwiftUI

struct ResultView: View {
    @Environment(Session.self) private var session

    @State private var t = 8.52
    @State private var playing = false
    @State private var speed = 1.0
    @State private var angle: Angle = .knee
    @State private var showShare = false

    private var phase: (rep: Int?, phase: MovementPhase, dt: Double) { SampleTrial.phase(at: t) }

    var body: some View {
        @Bindable var session = session
        VStack(spacing: 0) {
            video
            transport
            ScrollView {
                VStack(spacing: 14) {
                    SampleDataBanner()
                    headline
                    FilmingNote()
                    chartCard
                    reps
                    measurements
                    notes
                    NavigationLink(value: Route.report) {
                        Text("View report").font(.headline).foregroundStyle(.black).frame(maxWidth: .infinity, minHeight: 36)
                    }
                    .buttonStyle(.glassProminent)
                    .padding(.top, 4)
                }
                .padding(16)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(session.test.shortName)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden()
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button("New test", systemImage: "chevron.backward") { session.startOver() }
                    .labelStyle(.titleAndIcon)
            }
            ToolbarItem(placement: .principal) {
                VStack(spacing: 0) {
                    Text(session.test.shortName).font(.headline)
                    Text("Today, 10:42 · Age \(session.patient.age)").font(.caption2).foregroundStyle(.secondary)
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button("Share", systemImage: "square.and.arrow.up") { showShare = true }
            }
        }
        .sheet(isPresented: $showShare) {
            ShareSheet()
                .presentationDetents([.medium])
        }
        .task(id: playing) {
            guard playing else { return }
            if t >= SampleTrial.duration - 0.01 { t = 0 }
            while playing, !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(33))
                let next = t + 0.033 * speed
                if next >= SampleTrial.duration { t = SampleTrial.duration; playing = false; break }
                t = next
            }
        }
    }

    // MARK: video + transport

    private var video: some View {
        TrialScene(t: t)
            .frame(height: 240)
            .overlay(alignment: .topLeading) {
                Text(repBadge)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .glassEffect(.regular, in: .capsule)
                    .padding(10)
                    .contentTransition(.numericText())
            }
            .overlay(alignment: .topTrailing) {
                GlassEffectContainer(spacing: 6) {
                    VStack(alignment: .trailing, spacing: 6) {
                        ForEach(Angle.allCases) { a in
                            HStack(spacing: 8) {
                                Text(a == .trunk ? "Trunk" : a.label).font(.caption2).foregroundStyle(.secondary)
                                Text("\(Int(SampleTrial.value(a, at: t).rounded()))°")
                                    .font(.headline.monospacedDigit())
                                    .contentTransition(.numericText())
                            }
                            .padding(.horizontal, 10).padding(.vertical, 4)
                            .frame(minWidth: 96, alignment: .trailing)
                            .glassEffect(.regular, in: .rect(cornerRadius: 10))
                        }
                    }
                }
                .padding(10)
            }
            .environment(\.colorScheme, .dark)
    }

    private var transport: some View {
        VStack(spacing: 2) {
            GeometryReader { geo in
                ForEach(Array(SampleTrial.seatOffs.enumerated()), id: \.offset) { i, s in
                    Button {
                        playing = false; t = s
                    } label: {
                        Text("\(i + 1)").font(.caption2.bold())
                            .frame(width: 22, height: 22)
                            .foregroundStyle(phase.rep == i ? .black : .primary)
                            .background(phase.rep == i ? Color.accent : Color(.tertiarySystemFill), in: .circle)
                    }
                    .buttonStyle(.plain)
                    .position(x: 11 + (geo.size.width - 22) * s / SampleTrial.duration, y: 11)
                    .accessibilityLabel("Jump to rep \(i + 1)")
                }
            }
            .frame(height: 22)

            Slider(value: Binding(get: { t }, set: { playing = false; t = $0 }), in: 0...SampleTrial.duration)
                .accessibilityLabel("Playback position")

            HStack {
                GlassEffectContainer(spacing: 8) {
                    HStack(spacing: 8) {
                        Button { step(-1) } label: { Image(systemName: "backward.frame.fill").frame(width: 26, height: 26) }
                            .accessibilityLabel("Back one frame")
                        Button { playing.toggle() } label: {
                            Image(systemName: playing ? "pause.fill" : "play.fill")
                                .font(.title3)
                                .foregroundStyle(.black)
                                .frame(width: 34, height: 34)
                                .contentTransition(.symbolEffect(.replace))
                        }
                        .buttonStyle(.glassProminent)
                        .accessibilityLabel(playing ? "Pause" : "Play")
                        Button { step(1) } label: { Image(systemName: "forward.frame.fill").frame(width: 26, height: 26) }
                            .accessibilityLabel("Forward one frame")
                    }
                    .buttonStyle(.glass)
                    .buttonBorderShape(.circle)
                }
                Text("\(Text("\(t, specifier: "%.2f")").fontWeight(.semibold)) \(Text("/ 13.3 s").foregroundStyle(.secondary))")
                    .font(.subheadline.monospacedDigit())
                Spacer()
                Button(speedLabel) {
                    speed = speed == 1 ? 0.5 : speed == 0.5 ? 0.25 : 1
                }
                .font(.subheadline.weight(.semibold).monospacedDigit())
                .buttonStyle(.glass)
                .accessibilityLabel("Playback speed \(speedLabel)")
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.bar)
    }

    // MARK: cards

    private var headline: some View {
        let band = SampleTrial.reference(for: session.patient.age)
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Five-times sit-to-stand time").font(.subheadline.weight(.medium)).foregroundStyle(.secondary)
                Spacer()
                ValidatedTag(validated: true)
            }
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("10.43").font(.system(size: 50, weight: .bold)).tracking(-1.2).monospacedDigit()
                Text("s").font(.title3.weight(.semibold))
                Text("± 0.07").font(.subheadline).foregroundStyle(.secondary)
            }
            ReferenceBar(value: SampleTrial.totalTime, highlightAges: band?.ages)
            if let band {
                let diff = band.meanSeconds - SampleTrial.totalTime
                Text("10.43 s is **\(abs(diff), specifier: "%.1f") s \(diff >= 0 ? "faster" : "slower")** than the reference mean of \(band.meanSeconds, specifier: "%.1f") s for ages \(band.ages.lowerBound)–\(band.ages.upperBound).")
                    .font(.subheadline)
            } else {
                Text("No reference values for age \(session.patient.age). The table covers ages 60–89.")
                    .font(.subheadline).foregroundStyle(.secondary)
            }
            Text("\(SampleTrial.referenceCitation) · shorter is faster").font(.caption).foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 20))
    }

    private var chartCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Angle", selection: $angle) {
                ForEach(Angle.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)

            AngleChart(angle: angle, playhead: t) { newT in
                playing = false; t = newT
            }
            .frame(height: 150)
            .animation(.spring(duration: 0.35, bounce: 0), value: angle)

            HStack(spacing: 12) {
                legend(Color.riseFill, "Rising")
                legend(Color(.systemGray5), "Standing")
                legend(Color.sitFill, "Sitting")
                HStack(spacing: 4) {
                    Rectangle().stroke(style: StrokeStyle(lineWidth: 1.5, dash: [3, 2])).frame(width: 12, height: 0.1)
                    Text("Expected range")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 20))
    }

    private func legend(_ c: Color, _ s: String) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 2).fill(c).frame(width: 10, height: 10)
            Text(s)
        }
    }

    private var reps: some View {
        VStack(spacing: 6) {
            SectionHeader(text: "Repetitions · tap to jump")
            VStack(spacing: 0) {
                ForEach(Array(SampleTrial.seatOffs.enumerated()), id: \.offset) { i, s in
                    Button {
                        playing = false; t = s
                    } label: {
                        HStack {
                            Text("Rep \(i + 1)").fontWeight(.semibold).frame(width: 56, alignment: .leading)
                            Text("at \(s, specifier: "%.2f") s").foregroundStyle(.secondary)
                            Spacer()
                            Text("rise \(SampleTrial.riseTimes[i], specifier: "%.2f") s")
                        }
                        .monospacedDigit()
                        .padding(.horizontal, 16)
                        .frame(minHeight: 46)
                        .background(phase.rep == i ? Color.accent.opacity(0.14) : .clear)
                        .contentShape(.rect)
                    }
                    .buttonStyle(.plain)
                    if i < 4 { Divider().padding(.leading, 16) }
                }
            }
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 14))
        }
    }

    private var measurements: some View {
        VStack(spacing: 6) {
            SectionHeader(text: "Measurements")
            VStack(spacing: 0) {
                ForEach(Array(SampleTrial.metrics.enumerated()), id: \.element.id) { i, m in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.label)
                            Text(m.detail).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 3) {
                            HStack(alignment: .firstTextBaseline, spacing: 5) {
                                Text(m.value).fontWeight(.semibold).monospacedDigit()
                                Text(m.error).font(.caption).foregroundStyle(.secondary)
                            }
                            ValidatedTag(validated: m.validated)
                        }
                    }
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    if i < SampleTrial.metrics.count - 1 { Divider().padding(.leading, 16) }
                }
            }
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 14))
        }
    }

    private var notes: some View {
        @Bindable var session = session
        return VStack(spacing: 6) {
            SectionHeader(text: "Notes")
            TextField("Arm use, balance, pain, footwear… Printed on the report.", text: $session.notes, axis: .vertical)
                .lineLimit(3...6)
                .padding(14)
                .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 14))
        }
    }

    // MARK: helpers

    private var repBadge: String {
        guard let rep = phase.rep else { return "Before the test" }
        return "Rep \(rep + 1) of 5 · \(phase.phase.word)"
    }

    private var speedLabel: String { speed == 1 ? "1×" : speed == 0.5 ? "½×" : "¼×" }

    private func step(_ frames: Double) {
        playing = false
        t = min(SampleTrial.duration, max(0, t + frames / SampleTrial.fps))
    }
}

struct AngleChart: View {
    let angle: Angle
    let playhead: Double
    var lineColor: Color = .accent
    var onSeek: ((Double) -> Void)?

    var body: some View {
        Chart {
            ForEach(SampleTrial.phaseSpans) { span in
                RectangleMark(xStart: .value("Start", span.start), xEnd: .value("End", span.end))
                    .foregroundStyle(fill(span.phase))
            }
            RuleMark(y: .value("Upper", angle.okRange.upperBound))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3])).foregroundStyle(.gray)
            RuleMark(y: .value("Lower", angle.okRange.lowerBound))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3])).foregroundStyle(.gray)
            ForEach(SampleTrial.series(angle)) { s in
                LineMark(x: .value("Time", s.t), y: .value(angle.label, s.value))
                    .foregroundStyle(lineColor)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .interpolationMethod(.catmullRom)
            }
            RuleMark(x: .value("Now", playhead))
                .foregroundStyle(.primary)
                .lineStyle(StrokeStyle(lineWidth: 2))
        }
        .chartXScale(domain: 0...SampleTrial.duration)
        .chartYScale(domain: angle.axis)
        .chartXAxis { AxisMarks(values: .stride(by: 2)) { AxisValueLabel() ; AxisGridLine().foregroundStyle(.clear) } }
        .chartYAxis { AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { v in
            AxisGridLine()
            AxisValueLabel { if let d = v.as(Double.self) { Text("\(Int(d))°") } }
        } }
        .chartOverlay { proxy in
            GeometryReader { geo in
                Rectangle().fill(.clear).contentShape(.rect)
                    .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                        guard let onSeek, let frame = proxy.plotFrame else { return }
                        let origin = geo[frame].origin
                        if let x: Double = proxy.value(atX: g.location.x - origin.x) {
                            onSeek(min(SampleTrial.duration, max(0, x)))
                        }
                    })
            }
        }
        .accessibilityLabel("\(angle.label) over the clip, five rise-and-sit cycles")
    }

    private func fill(_ p: MovementPhase) -> Color {
        switch p {
        case .rising: .riseFill
        case .standing: Color(.systemGray5)
        case .sitting: .sitFill
        case .seated: .clear
        }
    }
}

struct ShareSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(Session.self) private var session

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ShareLink(item: Exports.reportPDF(session: session), preview: SharePreview("ptvision report")) {
                        row("doc.richtext.fill", .accent, "Report (PDF)", "Result, measurements and notes")
                    }
                    ShareLink(item: Exports.anglesCSV(), preview: SharePreview("Angles CSV")) {
                        row("tablecells.fill", .gray, "Angles", "Knee, hip and trunk per frame (CSV)")
                    }
                    ShareLink(item: Exports.eventsCSV(), preview: SharePreview("Events CSV")) {
                        row("list.bullet.rectangle.fill", .gray, "Events", "Seat-off, stand and seated times (CSV)")
                    }
                } footer: {
                    Text("The video itself is never included. Share only through your clinic’s approved apps.")
                }
            }
            .navigationTitle("Share")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done", systemImage: "xmark") { dismiss() }
                }
            }
        }
    }

    private func row(_ icon: String, _ tint: Color, _ title: String, _ sub: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(tint == .accent ? .black : .white)
                .frame(width: 32, height: 32)
                .background(tint, in: .rect(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).foregroundStyle(.primary)
                Text(sub).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
