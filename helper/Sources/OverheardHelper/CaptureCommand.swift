import AudioToolbox
import CoreAudio
import Foundation

/// Captures system audio via a Core Audio process tap, optionally alongside the
/// microphone, and streams both to stdout as interleaved 32-bit float PCM.
///
/// Both sources live in one private aggregate device read by a single IOProc,
/// so the tracks are sample-aligned by construction rather than by correlating
/// two independently clocked streams afterwards.
///
/// Channel layout on stdout is announced in the `ready` message and is either
/// `["mic", "system"]` or `["system"]`. System audio is folded to mono here:
/// everything downstream (ASR, diarization) wants mono, and folding at the
/// source halves the bytes crossing the pipe.
final class CaptureSession {

    struct Options {
        var includePIDs: [pid_t] = []
        var excludePIDs: [pid_t] = []
        var captureMic = true
    }

    private let options: Options
    private var tapID = AudioObjectID(0)
    private var aggregateID = AudioObjectID(0)
    private var procID: AudioDeviceIOProcID?

    /// Guards `micFirst` / `expectedBuffers`, which the IOProc reads.
    private let lock = NSLock()

    /// Audio handed from the IOProc to the writer thread.
    ///
    /// The IOProc runs on a Core Audio realtime thread. Writing to stdout there
    /// would block whenever the reader falls behind and the pipe fills, and a
    /// stalled realtime thread is what trips Core Audio's watchdog. Buffering
    /// under a lock and draining on a normal thread keeps the callback bounded.
    private let bufferLock = NSLock()
    private var outgoing = [Float]()
    private var writerThread: Thread?
    private var draining = true

    /// Roughly ten seconds of stereo audio at 48 kHz. Past this the consumer
    /// has stopped for good, and dropping is better than growing without end.
    private let maxBufferedSamples = 48_000 * 2 * 10
    private var reportedOverflow = false
    private var micFirst = false
    private var expectedBuffers = 0
    private var reportedMismatch = false

    private let out = FileHandle.standardOutput

    init(options: Options) {
        self.options = options
    }

    // MARK: - Setup

    func start() {
        let outputDevice = CA.defaultDevice(kAudioHardwarePropertyDefaultOutputDevice)
        guard let outputUID = CA.deviceUID(outputDevice) else {
            Status.fatal("No default output device", code: "no_output_device")
        }

        // A tap follows the output device, so it captures whatever the user
        // actually hears without them rerouting anything.
        let tapDescription: CATapDescription
        if !options.includePIDs.isEmpty {
            tapDescription = CATapDescription(
                stereoMixdownOfProcesses: processObjects(for: options.includePIDs))
        } else {
            tapDescription = CATapDescription(
                stereoGlobalTapButExcludeProcesses: processObjects(for: options.excludePIDs))
        }
        tapDescription.name = "Overheard"
        tapDescription.isPrivate = true
        tapDescription.muteBehavior = .unmuted

        let tapStatus = AudioHardwareCreateProcessTap(tapDescription, &tapID)
        guard tapStatus == noErr else {
            Status.fatal(
                "Could not create audio tap (status \(tapStatus)). System audio recording "
                    + "permission may have been denied.", code: "tap_failed")
        }

        // The input device is deliberately the main (clock) sub-device.
        //
        // A process tap only produces callbacks while the tapped output device
        // is actually playing. Clocking the aggregate off the output device
        // therefore stops the IOProc dead during silence, which loses the
        // microphone track as well and collapses gaps so timestamps no longer
        // correspond to wall-clock time. The input device runs continuously, so
        // clocking off it keeps frames flowing whether or not anything is
        // playing, and the tap simply contributes silence when idle.
        let inputDevice = CA.defaultDevice(kAudioHardwarePropertyDefaultInputDevice)
        let inputUID = CA.deviceUID(inputDevice)
        let micChannels = inputUID == nil ? 0 : CA.inputChannelLayout(inputDevice).reduce(0, +)

        var subDevices: [[String: Any]] = []
        var micUID: String?

        if let inputUID {
            // Included even when the mic track isn't wanted, purely as the clock.
            subDevices.append([
                kAudioSubDeviceUIDKey: inputUID,
                kAudioSubDeviceDriftCompensationKey: 1,
            ])
            if options.captureMic {
                micUID = inputUID
            }
        } else {
            Status.emit(
                "warning",
                [
                    "message":
                        "No input device to clock the aggregate; capture will pause during silence"
                ])
            subDevices.append([kAudioSubDeviceUIDKey: outputUID])
        }

        let clockUID = inputUID ?? outputUID

        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Overheard Capture",
            kAudioAggregateDeviceUIDKey: "au.com.overmoro.overheard.capture.\(UUID().uuidString)",
            kAudioAggregateDeviceMainSubDeviceKey: clockUID,
            // Private keeps it out of Sound preferences and other apps' device lists
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            // Must be false. With auto-start enabled the aggregate's IO only
            // runs while the tapped output device is playing, so the whole
            // device (microphone included) stalls during silence and the
            // recorded timeline no longer matches wall-clock time. Disabled,
            // the aggregate clocks off the input device and runs continuously,
            // with the tap contributing silence while nothing is playing.
            kAudioAggregateDeviceTapAutoStartKey: false,
            kAudioAggregateDeviceSubDeviceListKey: subDevices,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapDriftCompensationKey: true,
                    kAudioSubTapUIDKey: tapDescription.uuid.uuidString,
                ]
            ],
        ]

        let aggregateStatus = AudioHardwareCreateAggregateDevice(
            aggregateDescription as CFDictionary, &aggregateID)
        guard aggregateStatus == noErr else {
            Status.fatal(
                "Could not create aggregate device (status \(aggregateStatus))",
                code: "aggregate_failed")
        }

        let layout = CA.inputChannelLayout(aggregateID)
        let rate = CA.sampleRate(aggregateID, scope: kAudioObjectPropertyScopeInput)
        let haveMic = micUID != nil && layout.count > 1

        lock.lock()
        micFirst = haveMic
        expectedBuffers = layout.count
        lock.unlock()

        // Sanity-check the documented ordering rather than trusting it blindly:
        // sub-device inputs come first, taps after.
        if haveMic, let first = layout.first, micChannels > 0, first != micChannels {
            Status.emit(
                "warning",
                [
                    "message":
                        "Unexpected input layout \(layout); mic reports \(micChannels) channels"
                ])
        }

        Status.emit(
            "ready",
            [
                "sampleRate": rate,
                "channels": haveMic ? 2 : 1,
                "layout": haveMic ? ["mic", "system"] : ["system"],
                "format": "f32le",
                "inputLayout": layout,
                "micDevice": micUID as Any,
                "outputDevice": outputUID,
            ])

        let status = AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) {
            [weak self] _, inInputData, _, _, _ in
            self?.handle(inInputData)
        }
        guard status == noErr, procID != nil else {
            Status.fatal("Could not create IOProc (status \(status))", code: "ioproc_failed")
        }
        let writer = Thread { [weak self] in self?.writeLoop() }
        writer.name = "overheard.capture.writer"
        writer.qualityOfService = .userInitiated
        writer.start()
        writerThread = writer

        guard AudioDeviceStart(aggregateID, procID!) == noErr else {
            Status.fatal("Could not start capture", code: "start_failed")
        }
    }

    // MARK: - Audio callback

    /// Interleaves the mic and system tracks into the output frame stream.
    /// Runs on a Core Audio realtime thread: no allocation beyond the scratch
    /// buffer, no logging, no locking beyond the two flags it reads.
    private func handle(_ inInputData: UnsafePointer<AudioBufferList>) {
        let buffers = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer(mutating: inInputData))

        lock.lock()
        let haveMic = micFirst
        let expected = expectedBuffers
        lock.unlock()

        guard buffers.count == expected, buffers.count > 0 else {
            reportMismatchOnce(got: buffers.count, expected: expected)
            return
        }

        // The tap is the last input buffer; the mic, when present, is the first.
        let systemBuffer = buffers[buffers.count - 1]
        let systemChannels = Int(systemBuffer.mNumberChannels)
        guard systemChannels > 0, let systemData = systemBuffer.mData else { return }
        let systemSamples = Int(systemBuffer.mDataByteSize) / MemoryLayout<Float>.size
        let frames = systemSamples / systemChannels
        guard frames > 0 else { return }
        let system = systemData.assumingMemoryBound(to: Float.self)

        var micPointer: UnsafeMutablePointer<Float>?
        var micChannels = 0
        if haveMic {
            let micBuffer = buffers[0]
            if let data = micBuffer.mData, micBuffer.mNumberChannels > 0 {
                micPointer = data.assumingMemoryBound(to: Float.self)
                micChannels = Int(micBuffer.mNumberChannels)
            }
        }

        let outChannels = haveMic ? 2 : 1
        var interleaved = [Float](repeating: 0, count: frames * outChannels)

        for frame in 0..<frames {
            // Fold system audio to mono
            var sum: Float = 0
            for channel in 0..<systemChannels {
                sum += system[frame * systemChannels + channel]
            }
            let systemMono = sum / Float(systemChannels)

            if haveMic {
                var micMono: Float = 0
                if let mic = micPointer, micChannels > 0 {
                    var micSum: Float = 0
                    for channel in 0..<micChannels {
                        micSum += mic[frame * micChannels + channel]
                    }
                    micMono = micSum / Float(micChannels)
                }
                interleaved[frame * 2] = micMono
                interleaved[frame * 2 + 1] = systemMono
            } else {
                interleaved[frame] = systemMono
            }
        }

        bufferLock.lock()
        if outgoing.count + interleaved.count > maxBufferedSamples {
            let overflowed = !reportedOverflow
            reportedOverflow = true
            bufferLock.unlock()
            if overflowed {
                Status.emit("warning", ["message": "Output backlog full, dropping audio"])
            }
            return
        }
        outgoing.append(contentsOf: interleaved)
        bufferLock.unlock()
    }

    /// Drains buffered audio to stdout, off the realtime thread.
    private func writeLoop() {
        while true {
            bufferLock.lock()
            let batch = outgoing
            outgoing.removeAll(keepingCapacity: true)
            let keepGoing = draining
            bufferLock.unlock()

            if !batch.isEmpty {
                batch.withUnsafeBufferPointer { out.write(Data(buffer: $0)) }
            } else if !keepGoing {
                return
            } else {
                // Nothing ready; yield rather than spin.
                Thread.sleep(forTimeInterval: 0.005)
            }
        }
    }

    private func reportMismatchOnce(got: Int, expected: Int) {
        lock.lock()
        let already = reportedMismatch
        reportedMismatch = true
        lock.unlock()
        if !already {
            Status.emit(
                "warning",
                ["message": "Input buffer count \(got) does not match expected \(expected)"])
        }
    }

    // MARK: - Teardown

    func stop() {
        if let procID {
            AudioDeviceStop(aggregateID, procID)
            AudioDeviceDestroyIOProcID(aggregateID, procID)
            self.procID = nil
        }
        if aggregateID != 0 {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = 0
        }
        if tapID != 0 {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = 0
        }

        // Let the writer flush what the IOProc already handed over.
        bufferLock.lock()
        draining = false
        bufferLock.unlock()
        for _ in 0..<200 {
            bufferLock.lock()
            let remaining = outgoing.count
            bufferLock.unlock()
            if remaining == 0 { break }
            Thread.sleep(forTimeInterval: 0.005)
        }
        writerThread = nil
        try? out.synchronize()
    }

    private func processObjects(for pids: [pid_t]) -> [AudioObjectID] {
        pids.compactMap { pid in
            var object = AudioObjectID(0)
            var size = UInt32(MemoryLayout<AudioObjectID>.size)
            var pidValue = pid
            var address = AudioObjectPropertyAddress(
                mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain)
            let status = AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address,
                UInt32(MemoryLayout<pid_t>.size), &pidValue, &size, &object)
            guard status == noErr, object != 0 else {
                Status.emit("warning", ["message": "No audio process for pid \(pid)"])
                return nil
            }
            return object
        }
    }
}

func runCapture(_ arguments: [String]) -> Never {
    var options = CaptureSession.Options()
    var index = 0
    while index < arguments.count {
        switch arguments[index] {
        case "--include-pid":
            index += 1
            if index < arguments.count, let pid = pid_t(arguments[index]) {
                options.includePIDs.append(pid)
            }
        case "--exclude-pid":
            index += 1
            if index < arguments.count, let pid = pid_t(arguments[index]) {
                options.excludePIDs.append(pid)
            }
        case "--no-mic":
            options.captureMic = false
        case "--help":
            print(
                """
                overheard-helper capture [options]

                Streams interleaved 32-bit float PCM to stdout and newline-delimited
                JSON status to stderr. Recording stops on SIGINT or SIGTERM.

                  --include-pid <pid>   Capture only this process (repeatable)
                  --exclude-pid <pid>   Capture everything except this process (repeatable)
                  --no-mic              System audio only, no microphone track
                """)
            exit(0)
        default:
            break
        }
        index += 1
    }

    let session = CaptureSession(options: options)

    // Tear the aggregate device and tap down cleanly, otherwise they can linger
    // in Core Audio after the process exits.
    let stopOnce = DispatchWorkItem { session.stop(); exit(0) }
    for signalNumber in [SIGINT, SIGTERM] {
        signal(signalNumber, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .main)
        source.setEventHandler { stopOnce.perform() }
        source.resume()
        signalSources.append(source)
    }

    session.start()
    dispatchMain()
}

/// Signal sources must outlive the function that creates them.
private var signalSources: [DispatchSourceSignal] = []
