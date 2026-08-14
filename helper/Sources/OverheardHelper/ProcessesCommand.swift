import AudioToolbox
import CoreAudio
import Foundation

/// Report which processes are currently running audio.
///
/// Meeting detection previously asked whether an app was running at all, so a
/// backgrounded Chrome with no call in it reported a Google Meet. Core Audio
/// knows which processes actually have live audio, which is the question worth
/// asking, and needs no additional permission beyond what capture already uses.
func runProcesses(_ arguments: [String]) -> Never {
    if arguments.contains("--help") {
        print(
            """
            overheard-helper processes

            Lists processes with live audio as JSON on stdout:
              [{"pid": 123, "bundleID": "us.zoom.xos", "input": true, "output": true}]

            input is true when the process is capturing (microphone in use),
            output when it is playing. A meeting is typically both.
            """)
        exit(0)
    }

    var listAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyProcessObjectList,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)

    var size: UInt32 = 0
    guard
        AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &listAddress, 0, nil, &size) == noErr,
        size > 0
    else {
        FileHandle.standardOutput.write("[]\n".data(using: .utf8)!)
        exit(0)
    }

    let count = Int(size) / MemoryLayout<AudioObjectID>.size
    var objects = [AudioObjectID](repeating: 0, count: count)
    guard
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &listAddress, 0, nil, &size, &objects) == noErr
    else {
        Status.fatal("Could not read the audio process list", code: "process_list_failed")
    }

    func uint32Property(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: UInt32 = 0
        var valueSize = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &valueSize, &value) == noErr
        else { return nil }
        return value
    }

    func stringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var valueSize = UInt32(MemoryLayout<CFTypeRef?>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &valueSize, &value) == noErr
        else { return nil }
        return value?.takeRetainedValue() as String?
    }

    var report: [[String: Any]] = []
    for object in objects where object != 0 {
        let input = (uint32Property(object, kAudioProcessPropertyIsRunningInput) ?? 0) != 0
        let output = (uint32Property(object, kAudioProcessPropertyIsRunningOutput) ?? 0) != 0
        guard input || output else { continue }

        var entry: [String: Any] = ["input": input, "output": output]
        if let pid = uint32Property(object, kAudioProcessPropertyPID) {
            entry["pid"] = Int(Int32(bitPattern: pid))
        }
        if let bundleID = stringProperty(object, kAudioProcessPropertyBundleID), !bundleID.isEmpty {
            entry["bundleID"] = bundleID
        }
        report.append(entry)
    }

    guard let data = try? JSONSerialization.data(withJSONObject: report) else {
        Status.fatal("Could not encode the process list", code: "encode_failed")
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    exit(0)
}
