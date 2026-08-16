// swift-tools-version: 6.0
import PackageDescription

// A single helper binary for the two jobs that need native macOS APIs:
//
//   overheard-helper capture   Core Audio process taps, replacing BlackHole
//   overheard-helper diarize   FluidAudio speaker diarization on the ANE
//
// Deliberately one binary rather than two: one code-signing target, one TCC
// identity for the audio-capture permission, one artifact to rebuild and ship.
//
// Language mode v5: the Core Audio IOProc is a C callback that mutates shared
// buffers, which Swift 6 strict concurrency cannot reason about. Safety there
// comes from an explicit lock in CaptureSession.
let package = Package(
    name: "overheard-helper",
    // 14.4 rather than 14.0: the process-tap API arrived in 14.2, and the
    // aggregate-device plumbing it depends on is only dependable from 14.4.
    platforms: [.macOS("14.4")],
    products: [
        .executable(name: "overheard-helper", targets: ["OverheardHelper"])
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.15.5")
    ],
    targets: [
        .executableTarget(
            name: "OverheardHelper",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            path: "Sources/OverheardHelper",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
