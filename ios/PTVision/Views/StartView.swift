// [REVIEW] Screen 1: choose the test, optional patient details, then film.

import SwiftUI

struct StartView: View {
    @Environment(Session.self) private var session

    var body: some View {
        @Bindable var session = session
        List {
            Section {
                Text("Film the test, and the measurements are ready in about 15 seconds. The video stays on this iPhone.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .listRowBackground(Color.clear)
                    .listRowInsets(EdgeInsets(top: 0, leading: 4, bottom: 4, trailing: 4))
            }

            Section("Test") {
                ForEach(TestKind.allCases) { test in
                    Button {
                        session.test = test
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(test.name).foregroundStyle(.primary)
                                Text(test.subtitle).font(.footnote).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if session.test == test {
                                Image(systemName: "checkmark")
                                    .font(.body.weight(.semibold))
                                    .foregroundStyle(Color.accent)
                                    .transition(.scale.combined(with: .opacity))
                            }
                        }
                        .contentShape(.rect)
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(session.test == test ? .isSelected : [])
                }
            }

            Section {
                Stepper(value: $session.patient.age, in: 18...110) {
                    LabeledContent("Age") {
                        Text("\(session.patient.age)").monospacedDigit().foregroundStyle(.primary)
                    }
                }
                Picker("Height", selection: $session.patient.heightInches) {
                    ForEach(54...78, id: \.self) { inches in
                        Text("\(inches / 12)′ \(inches % 12)″").tag(inches)
                    }
                }
                Picker("Sex", selection: $session.patient.sex) {
                    ForEach(Sex.allCases) { Text($0.rawValue).tag($0) }
                }
            } header: {
                Text("Patient · optional")
            } footer: {
                Text("Age picks the reference values the result is compared with. No names are stored.")
            }
        }
        .animation(.spring(duration: 0.3, bounce: 0), value: session.test)
        .navigationTitle("New test")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("History", systemImage: "clock.arrow.circlepath") {
                    session.path = [.result]
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            GlassEffectContainer(spacing: 12) {
                VStack(spacing: 10) {
                    Button {
                        session.useSample = false
                        session.path.append(.record)
                    } label: {
                        Label("Film the test", systemImage: "camera.fill")
                            .font(.headline)
                            .foregroundStyle(.black)
                            .frame(maxWidth: .infinity, minHeight: 36)
                    }
                    .buttonStyle(.glassProminent)

                    HStack(spacing: 10) {
                        Button {
                            session.path.append(.processing)
                        } label: {
                            Text("Choose from Photos").frame(maxWidth: .infinity, minHeight: 30)
                        }
                        Button {
                            session.useSample = true
                            session.path.append(.record)
                        } label: {
                            Text("Try the demo clip").frame(maxWidth: .infinity, minHeight: 30)
                        }
                    }
                    .buttonStyle(.glass)
                    .font(.subheadline.weight(.semibold))
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
    }
}
