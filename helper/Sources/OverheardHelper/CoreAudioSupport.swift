import AudioToolbox
import CoreAudio
import Foundation

/// Thin wrappers over the Core Audio property API, which is otherwise four
/// lines of ceremony per read.
enum CA {

    static func defaultDevice(_ selector: AudioObjectPropertySelector) -> AudioObjectID {
        var id = AudioObjectID(0)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id)
        return status == noErr ? id : AudioObjectID(0)
    }

    static func deviceUID(_ id: AudioObjectID) -> String? {
        guard id != 0 else { return nil }
        var uid: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<CFTypeRef?>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &uid) == noErr else {
            return nil
        }
        return uid?.takeRetainedValue() as String?
    }

    static func deviceName(_ id: AudioObjectID) -> String? {
        guard id != 0 else { return nil }
        var name: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<CFTypeRef?>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceNameCFString,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &name) == noErr else {
            return nil
        }
        return name?.takeRetainedValue() as String?
    }

    /// Channel counts of each input buffer the device presents, in order.
    static func inputChannelLayout(_ id: AudioObjectID) -> [Int] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr, size > 0 else {
            return []
        }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, raw) == noErr else {
            return []
        }
        let list = UnsafeMutableAudioBufferListPointer(
            raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.map { Int($0.mNumberChannels) }
    }

    static func sampleRate(_ id: AudioObjectID, scope: AudioObjectPropertyScope) -> Double {
        var format = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamFormat,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, &format) == noErr else {
            return 0
        }
        return format.mSampleRate
    }
}

/// Newline-delimited JSON status messages on stderr, keeping stdout pure PCM.
enum Status {
    private static let lock = NSLock()

    static func emit(_ type: String, _ fields: [String: Any] = [:]) {
        var payload: [String: Any] = ["type": type]
        payload.merge(fields) { _, new in new }
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
            var line = String(data: data, encoding: .utf8)
        else { return }
        line += "\n"
        lock.lock()
        FileHandle.standardError.write(line.data(using: .utf8)!)
        lock.unlock()
    }

    static func fatal(_ message: String, code: String = "error") -> Never {
        emit("fatal", ["code": code, "message": message])
        exit(1)
    }
}
