"""Audio device discovery and recording."""

import os
import subprocess
import sys
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_DEVICE_NAME = "Meeting Capture"
FALLBACK_DEVICES = ["BlackHole", "MacBook Pro Microphone"]
SAMPLE_RATE = 16000
CHANNELS = 1


def find_device(name: str) -> int | None:
    """Find an audio input device by name (case-insensitive partial match)."""
    for d in sd.query_devices():
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return d["index"]
    return None


def find_recording_device(preferred: str = DEFAULT_DEVICE_NAME) -> int | None:
    """Find the best available recording device with fallback chain."""
    device_id = find_device(preferred)
    if device_id is not None:
        return device_id
    for name in FALLBACK_DEVICES:
        device_id = find_device(name)
        if device_id is not None:
            return device_id
    return None


def list_input_devices() -> list[dict]:
    """List all available input devices."""
    return [
        {"index": d["index"], "name": d["name"], "channels": d["max_input_channels"]}
        for d in sd.query_devices()
        if d["max_input_channels"] > 0
    ]


class Recorder:
    """Simple audio recorder using sounddevice."""

    def __init__(self, device_id: int, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._paused: bool = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def start(self) -> None:
        self._chunks = []
        self._paused = False

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio: {status}", file=sys.stderr)
            if not self._paused:
                self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray | None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks, axis=0)
        self._chunks = []
        return audio

    def save(self, audio: np.ndarray, path: str) -> str:
        sf.write(path, audio, self.sample_rate)
        return path


# ---------------------------------------------------------------------------
# CoreAudio device creation
# ---------------------------------------------------------------------------

# Swift snippet that creates an Aggregate Device named "Meeting Capture"
# combining BlackHole 2ch (loopback) + MacBook Pro Microphone.
_CREATE_AGGREGATE_SWIFT = """\
import CoreAudio
import Foundation

var propSize: UInt32 = 0
var prop = AudioObjectPropertyAddress(
    mSelector: kAudioHardwarePropertyDevices,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain
)
AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &prop, 0, nil, &propSize)
let deviceCount = Int(propSize) / MemoryLayout<AudioDeviceID>.size
var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &prop, 0, nil, &propSize, &deviceIDs)

func deviceUID(_ id: AudioDeviceID) -> String {
    var uidProp = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var uidRef: CFString = "" as CFString
    var sz = UInt32(MemoryLayout<CFString>.size)
    AudioObjectGetPropertyData(id, &uidProp, 0, nil, &sz, &uidRef)
    return uidRef as String
}

func deviceName(_ id: AudioDeviceID) -> String {
    var nameProp = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceNameCFString,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var nameRef: CFString = "" as CFString
    var sz = UInt32(MemoryLayout<CFString>.size)
    AudioObjectGetPropertyData(id, &nameProp, 0, nil, &sz, &nameRef)
    return nameRef as String
}

var bhUID = ""; var micUID = ""
for id in deviceIDs {
    let name = deviceName(id)
    if name.contains("BlackHole 2ch") { bhUID = deviceUID(id) }
    if name.contains("MacBook Pro Microphone") { micUID = deviceUID(id) }
}
// Check already exists
for id in deviceIDs {
    if deviceName(id) == "Meeting Capture" { print("ALREADY_EXISTS"); exit(0) }
}
guard !bhUID.isEmpty && !micUID.isEmpty else { print("MISSING_DEVICES"); exit(1) }

let desc: NSDictionary = [
    kAudioAggregateDeviceNameKey: "Meeting Capture",
    kAudioAggregateDeviceUIDKey: "com.overheard.MeetingCapture",
    kAudioAggregateDeviceSubDeviceListKey: [
        [kAudioSubDeviceUIDKey: bhUID, kAudioSubDeviceDriftCompensationKey: 1],
        [kAudioSubDeviceUIDKey: micUID, kAudioSubDeviceDriftCompensationKey: 1],
    ],
    kAudioAggregateDeviceMasterSubDeviceKey: micUID,
    kAudioAggregateDeviceIsPrivateKey: 0
]
var newID: AudioDeviceID = 0
let st = AudioHardwareCreateAggregateDevice(desc, &newID)
print(st == noErr ? "CREATED" : "FAILED:\\(st)")
exit(st == noErr ? 0 : 1)
"""

# Swift snippet that creates a Multi-Output Device named "Meeting Monitor"
# combining MacBook Pro Speakers + BlackHole 2ch.
_CREATE_MULTIOUT_SWIFT = """\
import CoreAudio
import Foundation

var propSize: UInt32 = 0
var prop = AudioObjectPropertyAddress(
    mSelector: kAudioHardwarePropertyDevices,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain
)
AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &prop, 0, nil, &propSize)
let deviceCount = Int(propSize) / MemoryLayout<AudioDeviceID>.size
var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &prop, 0, nil, &propSize, &deviceIDs)

func deviceUID(_ id: AudioDeviceID) -> String {
    var uidProp = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var uidRef: CFString = "" as CFString
    var sz = UInt32(MemoryLayout<CFString>.size)
    AudioObjectGetPropertyData(id, &uidProp, 0, nil, &sz, &uidRef)
    return uidRef as String
}

func deviceName(_ id: AudioDeviceID) -> String {
    var nameProp = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceNameCFString,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var nameRef: CFString = "" as CFString
    var sz = UInt32(MemoryLayout<CFString>.size)
    AudioObjectGetPropertyData(id, &nameProp, 0, nil, &sz, &nameRef)
    return nameRef as String
}

var bhUID = ""; var speakerUID = ""
for id in deviceIDs {
    let name = deviceName(id)
    if name.contains("BlackHole 2ch") { bhUID = deviceUID(id) }
    if name.contains("MacBook Pro Speakers") { speakerUID = deviceUID(id) }
}
for id in deviceIDs {
    if deviceName(id) == "Meeting Monitor" { print("ALREADY_EXISTS"); exit(0) }
}
guard !bhUID.isEmpty && !speakerUID.isEmpty else { print("MISSING_DEVICES"); exit(1) }

let desc: NSDictionary = [
    kAudioAggregateDeviceNameKey: "Meeting Monitor",
    kAudioAggregateDeviceUIDKey: "com.overheard.MeetingMonitor",
    kAudioAggregateDeviceSubDeviceListKey: [
        [kAudioSubDeviceUIDKey: speakerUID, kAudioSubDeviceDriftCompensationKey: 0],
        [kAudioSubDeviceUIDKey: bhUID, kAudioSubDeviceDriftCompensationKey: 1],
    ],
    kAudioAggregateDeviceMasterSubDeviceKey: speakerUID,
    kAudioAggregateDeviceIsPrivateKey: 0,
    kAudioAggregateDeviceIsStackedKey: 0
]
var newID: AudioDeviceID = 0
let st = AudioHardwareCreateAggregateDevice(desc, &newID)
print(st == noErr ? "CREATED" : "FAILED:\\(st)")
exit(st == noErr ? 0 : 1)
"""


def _run_swift_snippet(code: str) -> tuple[bool, str]:
    """Write Swift code to a temp file, compile+run it, parse output."""
    with tempfile.NamedTemporaryFile(suffix=".swift", mode="w", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            ["swift", path],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        if output == "ALREADY_EXISTS":
            return True, "Already exists"
        if output == "CREATED":
            return True, "Created successfully"
        if output == "MISSING_DEVICES":
            return False, "Required devices not found — is BlackHole 2ch installed?"
        if result.returncode != 0:
            err = (result.stderr or output)[:200]
            return False, f"Error: {err}"
        return True, output or "Done"
    except subprocess.TimeoutExpired:
        return False, "Timed out"
    except FileNotFoundError:
        return False, "swift not found — create device manually in Audio MIDI Setup"
    finally:
        os.unlink(path)


def create_aggregate_device() -> tuple[bool, str]:
    """Create 'Meeting Capture' aggregate device (BlackHole 2ch + MacBook Pro Microphone).

    Returns (success, message).
    """
    return _run_swift_snippet(_CREATE_AGGREGATE_SWIFT)


def create_multi_output_device() -> tuple[bool, str]:
    """Create 'Meeting Monitor' multi-output device (MacBook Pro Speakers + BlackHole 2ch).

    Returns (success, message).
    """
    return _run_swift_snippet(_CREATE_MULTIOUT_SWIFT)
