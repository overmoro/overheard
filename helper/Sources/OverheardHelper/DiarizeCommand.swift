import FluidAudio
import Foundation

/// Speaker diarization via FluidAudio, running on the Apple Neural Engine.
///
/// Replaces pyannote, which needed a Hugging Face account, an accepted model
/// licence and a token on disk. FluidAudio's models download on first use with
/// no credentials.
///
/// Emits JSON on stdout so the Python side can parse it directly. Embeddings are
/// omitted unless asked for: they are 256 floats per segment and dwarf the rest
/// of the payload, but they are what makes cross-meeting speaker identity
/// possible, so they are available on request.
func runDiarize(_ arguments: [String]) -> Never {
    var audioPath: String?
    var outputPath: String?
    var speakers: Int?
    var minSpeakers: Int?
    var maxSpeakers: Int?
    var includeEmbeddings = false

    var index = 0
    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "--speakers":
            index += 1
            if index < arguments.count { speakers = Int(arguments[index]) }
        case "--min-speakers":
            index += 1
            if index < arguments.count { minSpeakers = Int(arguments[index]) }
        case "--max-speakers":
            index += 1
            if index < arguments.count { maxSpeakers = Int(arguments[index]) }
        case "--output":
            index += 1
            if index < arguments.count { outputPath = arguments[index] }
        case "--embeddings":
            includeEmbeddings = true
        case "--help":
            print(
                """
                overheard-helper diarize <audio-file> [options]

                Writes JSON {segments: [{speaker, start, end}], speakerCount} to stdout,
                or to --output. Models download automatically on first run.

                  --speakers <n>       Exact expected speaker count
                  --min-speakers <n>   Lower bound when the count is not known
                  --max-speakers <n>   Upper bound when the count is not known
                  --embeddings         Include the 256-dim embedding per segment
                  --output <file>      Write JSON to a file instead of stdout
                """)
            exit(0)
        default:
            if !argument.hasPrefix("--") && audioPath == nil { audioPath = argument }
        }
        index += 1
    }

    guard let audioPath else {
        Status.fatal("No audio file given", code: "no_input")
    }
    guard FileManager.default.fileExists(atPath: audioPath) else {
        Status.fatal("Audio file not found: \(audioPath)", code: "no_input")
    }

    let semaphore = DispatchSemaphore(value: 0)
    var exitCode: Int32 = 0

    Task {
        defer { semaphore.signal() }
        do {
            var config = OfflineDiarizerConfig.default
            if let speakers {
                config = config.withSpeakers(exactly: speakers)
            } else if minSpeakers != nil || maxSpeakers != nil {
                config = config.withSpeakers(min: minSpeakers, max: maxSpeakers)
            }

            let manager = OfflineDiarizerManager(config: config)
            let started = Date()
            let result = try await manager.process(URL(fileURLWithPath: audioPath))
            let elapsed = Date().timeIntervalSince(started)

            let segments: [[String: Any]] = result.segments.map { segment in
                var entry: [String: Any] = [
                    "speaker": segment.speakerId,
                    "start": Double(segment.startTimeSeconds),
                    "end": Double(segment.endTimeSeconds),
                    "quality": Double(segment.qualityScore),
                ]
                if includeEmbeddings {
                    entry["embedding"] = segment.embedding.map { Double($0) }
                }
                return entry
            }

            let speakerIds = Set(result.segments.map { $0.speakerId })
            let payload: [String: Any] = [
                "segments": segments,
                "speakerCount": speakerIds.count,
                "processingSeconds": elapsed,
            ]

            let data = try JSONSerialization.data(withJSONObject: payload)
            if let outputPath {
                try data.write(to: URL(fileURLWithPath: outputPath))
                Status.emit("done", ["output": outputPath, "speakerCount": speakerIds.count])
            } else {
                FileHandle.standardOutput.write(data)
                FileHandle.standardOutput.write("\n".data(using: .utf8)!)
            }
        } catch {
            Status.emit("fatal", ["code": "diarize_failed", "message": "\(error)"])
            exitCode = 1
        }
    }

    semaphore.wait()
    exit(exitCode)
}
