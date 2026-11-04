// [REVIEW] Screen 5: the report as a document, plus PDF/CSV export helpers.

import Charts
import SwiftUI

struct ReportView: View {
    @Environment(Session.self) private var session

    var body: some View {
        ScrollView {
            ReportDocument(test: session.test, patient: session.patient, notes: session.notes)
                .background(.white, in: .rect(cornerRadius: 8))
                .shadow(color: .black.opacity(0.08), radius: 12, y: 4)
                .padding(16)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Report")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                ShareLink(item: Exports.reportPDF(session: session), preview: SharePreview("ptvision report")) {
                    Label("Share PDF", systemImage: "square.and.arrow.up")
                }
            }
        }
    }
}

/// The printable report. Rendered on screen and into the PDF.
struct ReportDocument: View {
    let test: TestKind
    let patient: Patient
    let notes: String

    var body: some View {
        let band = SampleTrial.reference(for: patient.age)
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 4) {
                Text("MEASUREMENT REPORT").font(.caption.weight(.semibold)).tracking(0.6).foregroundStyle(Color.accentInk)
                Text(test.name).font(.title2.bold()).tracking(-0.4)
                Text("Trial sts_side · 3 November 2026, 10:42").font(.footnote).foregroundStyle(.secondary)
                HStack(spacing: 14) {
                    Text("Age \(patient.age)")
                    Text(patient.heightText)
                    Text(patient.sex.rawValue)
                    Text("Right side")
                }
                .font(.footnote)
                .padding(.top, 4)
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Text("Five-times sit-to-stand time").font(.footnote).foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("10.43").font(.system(size: 44, weight: .bold)).tracking(-1).monospacedDigit()
                    Text("s").font(.title3.weight(.semibold))
                    Text("± 0.07").font(.footnote).foregroundStyle(.secondary)
                }
                if let band {
                    Text("Faster than the reference mean of **\(band.meanSeconds, specifier: "%.1f") s** for ages \(band.ages.lowerBound)–\(band.ages.upperBound).")
                        .font(.subheadline)
                }
                Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 3) {
                    ForEach(SampleTrial.references) { r in
                        GridRow {
                            Text("Ages \(r.ages.lowerBound)–\(r.ages.upperBound)")
                            Text("\(r.meanSeconds, specifier: "%.1f") s").monospacedDigit()
                        }
                        .fontWeight(r.ages == band?.ages ? .semibold : .regular)
                    }
                }
                .font(.footnote)
                .padding(.top, 4)
                Text("Reference: \(SampleTrial.referenceCitation). Timed from the first seat-off to the fifth stand.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Text("**Filming: 1 note, 6 checks passed.** The phone was at an angle to the person’s side; joint angles were corrected and are less exact. Timings are not affected.")
                .font(.footnote)
                .foregroundStyle(Color.reportNoteInk)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.reportNoteFill, in: .rect(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 8) {
                Text("Measurements").font(.headline)
                ForEach(SampleTrial.metrics) { m in
                    HStack {
                        Text(m.label)
                        Spacer()
                        Text(m.value).fontWeight(.semibold).monospacedDigit()
                        Text(m.error).foregroundStyle(.secondary).frame(width: 50, alignment: .leading)
                        Text(m.validated ? "Validated" : "Estimate").font(.caption).foregroundStyle(.secondary).frame(width: 60, alignment: .trailing)
                    }
                    .font(.footnote)
                    Divider()
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Knee flexion over the test").font(.headline)
                AngleChart(angle: .knee, playhead: -1, lineColor: .accentInk)
                    .frame(height: 120)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Repetition times (s from start of video)").font(.headline)
                Grid(alignment: .trailing, horizontalSpacing: 16, verticalSpacing: 4) {
                    GridRow {
                        Text("Rep").gridColumnAlignment(.leading)
                        Text("Seat-off"); Text("Standing"); Text("Starts to sit"); Text("Seated")
                    }
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    ForEach(0..<5, id: \.self) { i in
                        let s = SampleTrial.seatOffs[i]
                        GridRow {
                            Text("\(i + 1)").fontWeight(.semibold)
                            Text(String(format: "%.2f", s))
                            Text(String(format: "%.2f", s + SampleTrial.riseTimes[i]))
                            Text(i < 4 ? String(format: "%.2f", s + SampleTrial.rise + SampleTrial.stand) : "–")
                            Text(i < 4 ? String(format: "%.2f", s + SampleTrial.rise + SampleTrial.stand + SampleTrial.descent) : "–")
                        }
                        .font(.footnote.monospacedDigit())
                    }
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Clinician notes").font(.headline)
                Text(notes.isEmpty ? "No notes." : notes)
                    .font(.footnote)
                    .foregroundStyle(notes.isEmpty ? .secondary : .primary)
            }

            Text("How this was computed: skeleton tracked on this device (26 body and foot points); low-confidence joints dropped, short gaps filled, 6 Hz low-pass filter; repetitions found from hip-height velocity; joint angles corrected for the camera angle. Protocol sts_5x v0.1.0.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(28)
        .foregroundStyle(.black)
        .environment(\.colorScheme, .light)
    }
}

@MainActor
enum Exports {
    private static var dir: URL { FileManager.default.temporaryDirectory }

    static func anglesCSV() -> URL {
        var csv = "frame,time_s,knee_flexion_deg,hip_flexion_deg,trunk_lean_deg\n"
        let n = Int(SampleTrial.duration * SampleTrial.fps)
        for f in 0...n {
            let t = Double(f) / SampleTrial.fps
            csv += String(format: "%d,%.3f,%.1f,%.1f,%.1f\n", f, t, SampleTrial.knee(t), SampleTrial.hip(t), SampleTrial.trunk(t))
        }
        return write(csv, "ptvision-angles.csv")
    }

    static func eventsCSV() -> URL {
        var csv = "rep,seat_off_s,stand_s,descent_start_s,seated_s\n"
        for (i, s) in SampleTrial.seatOffs.enumerated() {
            let last = i == SampleTrial.seatOffs.count - 1
            csv += String(format: "%d,%.2f,%.2f,", i + 1, s, s + SampleTrial.riseTimes[i])
            csv += last ? ",\n" : String(format: "%.2f,%.2f\n", s + SampleTrial.rise + SampleTrial.stand, s + SampleTrial.rise + SampleTrial.stand + SampleTrial.descent)
        }
        return write(csv, "ptvision-events.csv")
    }

    static func reportPDF(session: Session) -> URL {
        let url = dir.appending(path: "ptvision-report.pdf")
        let renderer = ImageRenderer(content:
            ReportDocument(test: session.test, patient: session.patient, notes: session.notes)
                .frame(width: 612)
                .background(.white)
        )
        renderer.render { size, draw in
            var box = CGRect(origin: .zero, size: size)
            guard let pdf = CGContext(url as CFURL, mediaBox: &box, nil) else { return }
            pdf.beginPDFPage(nil)
            draw(pdf)
            pdf.endPDFPage()
            pdf.closePDF()
        }
        return url
    }

    private static func write(_ text: String, _ name: String) -> URL {
        let url = dir.appending(path: name)
        try? text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
