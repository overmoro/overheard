import Foundation

let version = "0.1.0"

let arguments = Array(CommandLine.arguments.dropFirst())

guard let command = arguments.first else {
    print(
        """
        overheard-helper \(version)

        Native macOS support for Overheard.

          capture    Stream system audio (and the mic) captured via Core Audio taps
          diarize    Speaker diarization on the Apple Neural Engine
          download   Fetch the diarization models ahead of first use

        Run a subcommand with --help for its options.
        """)
    exit(0)
}

let rest = Array(arguments.dropFirst())

switch command {
case "capture":
    runCapture(rest)
case "diarize":
    runDiarize(rest)
case "download":
    runDownload(rest)
case "--version", "version":
    print(version)
    exit(0)
default:
    FileHandle.standardError.write("Unknown command: \(command)\n".data(using: .utf8)!)
    exit(64)
}
