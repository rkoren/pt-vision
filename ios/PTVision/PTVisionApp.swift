// [REVIEW] App entry and navigation for the ptvision iPhone prototype (issue #20).

import SwiftUI

enum Route: Hashable {
    case record, processing, result, report
}

@MainActor
@Observable
final class Session {
    var test: TestKind = .sts5x
    var patient = Patient()
    var notes = ""
    var path: [Route] = []
    /// Run the live screen on the bundled demo clip instead of the camera.
    var useSample = false

    func startOver() { path.removeAll() }
}

@main
struct PTVisionApp: App {
    @State private var session = Session()

    var body: some Scene {
        WindowGroup {
            @Bindable var session = session
            NavigationStack(path: $session.path) {
                StartView()
                    .navigationDestination(for: Route.self) { route in
                        switch route {
                        case .record: RecordView()
                        case .processing: ProcessingView()
                        case .result: ResultView()
                        case .report: ReportView()
                        }
                    }
            }
            .environment(session)
            .tint(.accent)
            .preferredColorScheme(.dark)
            .onAppear {
                applyLaunchArguments()
                DemoClips.copyToDocuments()
            }
        }
    }

    /// `-PTVRoute result` etc. jumps straight to a screen (for screenshots).
    private func applyLaunchArguments() {
        guard let raw = UserDefaults.standard.string(forKey: "PTVRoute") else { return }
        let map: [String: [Route]] = [
            "record": [.record], "processing": [.processing],
            "result": [.result], "report": [.result, .report],
        ]
        session.path = map[raw] ?? []
    }
}

/// Puts pt-vision's demo clips in the app's Documents folder so they show in the Files app
/// (On My iPhone › ptvision), where they can be saved to Photos or shared.
enum DemoClips {
    static let files = [
        ("sts_side", "pt-vision sit-to-stand demo.mp4"),
        ("sports2d_demo_2s", "Sports2D demo (BSD-3, 2 s).mp4"),
    ]

    static func copyToDocuments() {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        for (resource, name) in files {
            let dest = docs.appending(path: name)
            guard !fm.fileExists(atPath: dest.path),
                  let src = Bundle.main.url(forResource: resource, withExtension: "mp4") else { continue }
            try? fm.copyItem(at: src, to: dest)
        }
    }
}

extension Color {
    /// Apple Fitness green. Use on dark surfaces or as a fill under black text; never as text on white.
    static let accent = Color(red: 0xA6 / 255, green: 1, blue: 0)
    /// Darker green for text and lines on the light, printable report.
    static let accentInk = Color(red: 0.24, green: 0.42, blue: 0)
    static let statusGreen = Color(red: 0x2B / 255, green: 0xA8 / 255, blue: 0x4A / 255)
    static let statusAmber = Color(red: 0xF2 / 255, green: 0xA9 / 255, blue: 0)
    static let statusRed = Color(red: 0xD6 / 255, green: 0x28 / 255, blue: 0x28 / 255)
    static let neutralBone = Color(red: 0x8C / 255, green: 0x9B / 255, blue: 0xAB / 255)
    static let noteInk = Color(red: 1, green: 0.8, blue: 0.45)
    static let noteFill = Color(red: 1, green: 0.62, blue: 0.1).opacity(0.14)
    static let reportNoteInk = Color(red: 0x6B / 255, green: 0x3F / 255, blue: 0)
    static let reportNoteFill = Color(red: 1, green: 0.97, blue: 0.925)
    static let riseFill = Color(red: 0.25, green: 0.55, blue: 1).opacity(0.28)
    static let sitFill = Color(red: 1, green: 0.6, blue: 0.2).opacity(0.28)
}

extension BoneStatus {
    var color: Color {
        switch self {
        case .inRange: .statusGreen
        case .nearLimit: .statusAmber
        case .outside: .statusRed
        case .notAssessed: .neutralBone
        }
    }
}
