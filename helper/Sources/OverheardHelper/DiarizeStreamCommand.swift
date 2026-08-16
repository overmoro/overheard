import FluidAudio
import Foundation

/// Streaming speaker diarization for the live transcript.
///
/// Reads raw 32-bit float mono PCM at 16 kHz on stdin and emits newline
/// delimited JSON on stdout as speakers resolve:
///
///     {"type":"segment","speaker":1,"start":3.2,"end":5.6,"final":true}
///
/// Sortformer is end-to-end: it assigns speakers as audio arrives rather than
/// clustering the whole recording afterwards, which is what makes live
/// attribution possible at all. The trade is a hard ceiling of four speakers,
/// and labels that are positional rather than identities.
///
/// Segments are emitted twice when they start tentative and later confirm, with
/// `final` distinguishing the two. Callers should replace on matching speaker
/// and start time rather than append blindly.
func runDiarizeStream(_ arguments: [String]) -> Never {
    var chunkSeconds = 1.0

    var index = 0
    while index < arguments.count {
        switch arguments[index] {
        case "--chunk-seconds":
            index += 1
            if index < arguments.count, let value = Double(arguments[index]) {
                chunkSeconds = max(0.25, min(5.0, value))
            }
        case "--help":
            print(
                """
                overheard-helper diarize-stream [options]

                Reads f32le mono 16 kHz PCM on stdin, writes newline delimited
                JSON segments on stdout. Ends on EOF or SIGTERM.

                  --chunk-seconds <n>   Audio accumulated per inference (default 1.0)

                Sortformer supports at most four concurrent speakers.
                """)
            exit(0)
        default:
            break
        }
        index += 1
    }

    let sampleRate = 16000
    let chunkSamples = Int(Double(sampleRate) * chunkSeconds)

    let semaphore = DispatchSemaphore(value: 0)
    var exitCode: Int32 = 0

    Task {
        defer { semaphore.signal() }

        let diarizer: SortformerDiarizer
        do {
            // Default config is NVIDIA's low-latency preset, roughly 1s of lag.
            let config = SortformerConfig.default
            diarizer = SortformerDiarizer(config: config)
            let models = try await SortformerModels.loadFromHuggingFace(config: config)
            diarizer.initialize(models: models)
        } catch {
            Status.emit("fatal", ["code": "model_load_failed", "message": "\(error)"])
            exitCode = 1
            return
        }

        Status.emit("ready", ["sampleRate": sampleRate, "chunkSeconds": chunkSeconds])

        let input = FileHandle.standardInput
        let output = FileHandle.standardOutput
        var pending = Data()
        var emittedFinal = Set<String>()

        func emit(_ segments: [DiarizerSegment], frameDuration: Float, isFinal: Bool) {
            for segment in segments {
                let start = Double(Float(segment.startFrame) * frameDuration)
                let end = Double(Float(segment.endFrame) * frameDuration)
                guard end > start else { continue }

                // Confirmed segments are emitted once; tentative ones may be
                // revised, so they are re-sent until they settle.
                let key = "\(segment.speakerIndex)-\(segment.startFrame)-\(segment.endFrame)"
                if isFinal {
                    if emittedFinal.contains(key) { continue }
                    emittedFinal.insert(key)
                }

                let payload: [String: Any] = [
                    "type": "segment",
                    "speaker": segment.speakerIndex,
                    "start": start,
                    "end": end,
                    "final": isFinal,
                ]
                if let data = try? JSONSerialization.data(withJSONObject: payload) {
                    output.write(data)
                    output.write("\n".data(using: .utf8)!)
                }
            }
        }

        let frameDuration = diarizer.config.frameDurationSeconds

        func drain(_ update: DiarizerTimelineUpdate?) {
            guard let update else { return }
            emit(update.finalizedSegments, frameDuration: frameDuration, isFinal: true)
            emit(update.tentativeSegments, frameDuration: frameDuration, isFinal: false)
        }

        while true {
            let data = input.availableData
            if data.isEmpty { break }  // EOF
            pending.append(data)

            let frameBytes = MemoryLayout<Float>.size
            while pending.count >= chunkSamples * frameBytes {
                let slice = pending.prefix(chunkSamples * frameBytes)
                pending.removeFirst(chunkSamples * frameBytes)

                let samples = slice.withUnsafeBytes { raw -> [Float] in
                    Array(raw.bindMemory(to: Float.self))
                }
                diarizer.addAudio(samples)
                do {
                    drain(try diarizer.process())
                } catch {
                    Status.emit("warning", ["message": "streaming update failed: \(error)"])
                }
            }
        }

        do {
            drain(try diarizer.finalizeSession())
        } catch {
            Status.emit("warning", ["message": "finalize failed: \(error)"])
        }
        Status.emit("done", [:])
    }

    semaphore.wait()
    exit(exitCode)
}
