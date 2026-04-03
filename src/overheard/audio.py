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

# Number of audio frames used for live level metering (~100ms at 16 kHz)
_LEVEL_WINDOW_FRAMES = 1600


def find_device(name: str) -> int | None:
    """Find an audio input device by name (case-insensitive partial match)."""
    for d in sd.query_devices():
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return d["index"]
    return None


def _device_has_signal(device_id: int) -> bool:
    """Quick half-second test to check a device actually captures audio."""
    try:
        info = sd.query_devices(device_id)
        rate = int(info["default_samplerate"])
        channels = min(info["max_input_channels"], 2)
        rec = sd.rec(int(0.5 * rate), samplerate=rate, channels=channels,
                     device=device_id, dtype="float32")
        sd.wait()
        return float(np.max(np.abs(rec))) > 0.0
    except Exception:
        return False


def find_recording_device(preferred: str = DEFAULT_DEVICE_NAME) -> int | None:
    """Find the best available recording device with fallback chain.

    Tests each candidate for actual signal — skips devices that return silence.
    """
    candidates = [preferred] + FALLBACK_DEVICES
    for name in candidates:
        device_id = find_device(name)
        if device_id is not None and _device_has_signal(device_id):
            print(f"Audio: using device '{name}' [{device_id}]", file=sys.stderr)
            return device_id
        elif device_id is not None:
            print(f"Audio: skipping '{name}' [{device_id}] — no signal", file=sys.stderr)
    return None


def list_input_devices() -> list[dict]:
    """List all available input devices."""
    return [
        {"index": d["index"], "name": d["name"], "channels": d["max_input_channels"]}
        for d in sd.query_devices()
        if d["max_input_channels"] > 0
    ]


class Recorder:
    """Audio recorder using sounddevice.

    Attempts multi-channel recording when the device has >=3 channels
    (e.g. Meeting Capture aggregate: BlackHole 2ch + mic). Falls back to
    mono if the device reports fewer channels.

    Channel layout convention for 3-channel aggregate:
        ch 0, 1 — system audio (BlackHole)
        ch 2    — microphone
    """

    # Minimum channel count to attempt multi-channel recording
    _MIN_MULTICHANNEL = 3

    def __init__(self, device_id: int, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.device_id = device_id
        self.sample_rate = sample_rate

        # Detect actual channel count from device
        try:
            info = sd.query_devices(device_id)
            avail = info["max_input_channels"]
        except Exception:
            avail = 1

        if avail >= self._MIN_MULTICHANNEL:
            self.channels = avail
            self._is_multichannel = True
            self._channels_info: dict | None = {
                "mic_channel": avail - 1,
                "system_channels": list(range(avail - 1)),
            }
        else:
            self.channels = channels
            self._is_multichannel = False
            self._channels_info = None

        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._paused: bool = False
        self._level_buf: np.ndarray | None = None  # last N frames for metering

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def start(self) -> None:
        self._chunks = []
        self._paused = False
        self._level_buf = None

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio: {status}", file=sys.stderr)
            if not self._paused:
                data = indata.copy()
                self._chunks.append(data)
                # Keep a rolling window for level metering
                self._level_buf = data

        self._stream = sd.InputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> tuple[np.ndarray | None, dict | None]:
        """Stop recording.

        Returns:
            (audio_array, channels_info) where channels_info is either
            {"mic_channel": int, "system_channels": [int, ...]} for multichannel,
            or None for mono.
        """
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return None, None
        audio = np.concatenate(self._chunks, axis=0)
        self._chunks = []
        self._level_buf = None
        return audio, self._channels_info

    def get_levels(self) -> tuple[float, float]:
        """Return (mic_rms, system_rms) from the last captured audio buffer.

        For mono recordings, both values are the same.
        Returns (0.0, 0.0) when not recording.
        """
        buf = self._level_buf
        if buf is None or len(buf) == 0:
            return 0.0, 0.0

        if self._is_multichannel and self._channels_info is not None:
            mic_ch = self._channels_info["mic_channel"]
            sys_chs = self._channels_info["system_channels"]
            mic_rms = float(np.sqrt(np.mean(buf[:, mic_ch] ** 2)))
            if sys_chs:
                sys_data = buf[:, sys_chs]
                sys_rms = float(np.sqrt(np.mean(sys_data ** 2)))
            else:
                sys_rms = mic_rms
        else:
            # Mono — collapse to single channel
            mono = buf[:, 0] if buf.ndim > 1 else buf
            rms = float(np.sqrt(np.mean(mono ** 2)))
            mic_rms = sys_rms = rms

        return mic_rms, sys_rms

    def save(self, audio: np.ndarray, path: str) -> str:
        """Write audio array to WAV. Handles both mono and multi-channel."""
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
