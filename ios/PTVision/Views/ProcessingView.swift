// [REVIEW] Screen 3: named stages and a skeleton preview. Progress is simulated for now.

import SwiftUI

struct ProcessingView: View {
    @Environment(Session.self) private var session
    @State private var progress = 0.0

    private let stages: [(label: String, end: Double)] = [
        ("Tracking the skeleton", 0.70), ("Checking the filming", 0.76), ("Smoothing", 0.80),
        ("Finding the five stands", 0.88), ("Measuring", 0.94), ("Writing the report", 1.0),
    ]

    private var finished: Bool { progress >= 1 }
    private var currentIndex: Int? { stages.firstIndex { progress < $0.end } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                TrialScene(t: (progress * SampleTrial.duration).truncatingRemainder(dividingBy: SampleTrial.duration))
                    .frame(height: 280)
                    .clipShape(.rect(cornerRadius: 22))
                    .overlay(alignment: .topLeading) {
                        Text("Live preview")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 10).padding(.vertical, 5)
                            .glassEffect(.regular, in: .capsule)
                            .padding(12)
                    }

                VStack(alignment: .leading, spacing: 8) {
                    Text(finished ? "Result ready" : (currentIndex.map { stages[$0].label + "…" } ?? "Starting…"))
                        .font(.title2.bold())
                        .contentTransition(.opacity)
                    ProgressView(value: progress)
                        .tint(finished ? .statusGreen : .accent)
                    Text(finished ? "Done · 5 stands measured" : "\(Int(progress * 100))% · about \(Int(((1 - progress) * 15).rounded())) s left")
                        .font(.footnote.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                VStack(spacing: 0) {
                    ForEach(Array(stages.enumerated()), id: \.offset) { i, stage in
                        stageRow(i, stage.label)
                        if i < stages.count - 1 { Divider().padding(.leading, 46) }
                    }
                }
                .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))

                Text("Simulated: the measurement runs on the laptop the video is sent to.")
                    .font(.footnote).foregroundStyle(.secondary)
                    .padding(.horizontal, 4)
            }
            .padding(16)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Analysing")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden()
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button("Cancel") { session.startOver() }
            }
        }
        .safeAreaInset(edge: .bottom) {
            Button {
                if finished { session.path.append(.result) } else { progress = 1 }
            } label: {
                Text(finished ? "View result" : "Skip ahead (prototype)")
                    .font(.headline)
                    .foregroundStyle(finished ? Color.black : Color.accent)
                    .frame(maxWidth: .infinity, minHeight: 36)
            }
            .buttonStyle(finished ? AnyPrimitiveButtonStyle(.glassProminent) : AnyPrimitiveButtonStyle(.glass))
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .task {
            while progress < 1, !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(60))
                progress = min(1, progress + 0.009)
            }
        }
        .sensoryFeedback(.success, trigger: finished)
    }

    @ViewBuilder
    private func stageRow(_ i: Int, _ label: String) -> some View {
        let start = i == 0 ? 0 : stages[i - 1].end
        let done = progress >= stages[i].end
        let current = !done && progress >= start
        HStack(spacing: 12) {
            Group {
                if done {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.accent)
                } else if current {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "circle").foregroundStyle(.tertiary)
                }
            }
            .frame(width: 22)
            Text(label)
                .fontWeight(current ? .semibold : .regular)
                .foregroundStyle(done || current ? .primary : .secondary)
            Spacer()
            Text(detail(i, current: current, done: done))
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .font(.subheadline)
        .padding(.horizontal, 14)
        .frame(minHeight: 46)
    }

    private func detail(_ i: Int, current: Bool, done: Bool) -> String {
        if i == 0, current { return "\(Int((progress / stages[0].end) * 399)) / 399" }
        if i == 1, done { return "1 note" }
        if i == 3, done { return "5 found" }
        return ""
    }
}

/// Lets one button switch between the glass and glassProminent styles.
struct AnyPrimitiveButtonStyle: PrimitiveButtonStyle {
    private let make: (Configuration) -> AnyView
    init<S: PrimitiveButtonStyle>(_ style: S) {
        make = { AnyView(style.makeBody(configuration: $0)) }
    }
    func makeBody(configuration: Configuration) -> some View { make(configuration) }
}
