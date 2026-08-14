"""Capture reader tests.

The frame reassembly tests here cover a real bug: the reader used to discard the
trailing bytes of a short read, which left the next read starting mid-frame and
permanently swapped the microphone and system channels for the rest of the
recording. Ragged read sizes are the whole point, so they are the default here
rather than an edge case.
"""

import threading

import numpy as np
import pytest

from overheard.capture import TapRecorder


class RaggedPipe:
    """A stdout stand-in returning awkward, non-frame-aligned byte counts.

    Real pipes return whatever bytes happen to be available, which is frequently
    not a whole number of audio frames.
    """

    def __init__(self, data: bytes, sizes):
        self.data = data
        self.sizes = sizes
        self.offset = 0
        self.call = 0

    def read(self, _want):
        if self.offset >= len(self.data):
            return b""
        size = self.sizes[self.call % len(self.sizes)]
        self.call += 1
        chunk = self.data[self.offset:self.offset + size]
        self.offset += size
        return chunk


def interleaved_ramps(frames: int):
    """Two channels that are trivial to check for misalignment.

    Channel 0 counts up, channel 1 counts down, so any shift shows up as a
    mismatch at the first affected sample rather than as plausible-looking audio.
    """
    mic = np.arange(frames, dtype=np.float32)
    system = -np.arange(frames, dtype=np.float32)
    data = np.empty(frames * 2, dtype=np.float32)
    data[0::2] = mic
    data[1::2] = system
    return data, mic, system


def drain(recorder: TapRecorder, pipe: RaggedPipe):
    recorder._stop_event = threading.Event()
    recorder._proc = type("Proc", (), {"stdout": pipe})()
    recorder._read_audio()
    return np.concatenate(recorder._chunks, axis=0)


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param([1023, 5, 4097, 7, 31], id="ragged"),
        pytest.param([1], id="one-byte-at-a-time"),
        pytest.param([3], id="never-a-whole-frame"),
        pytest.param([8], id="exactly-one-frame"),
        pytest.param([32768], id="large-aligned"),
        pytest.param([8191], id="large-misaligned"),
    ],
)
def test_frames_survive_ragged_reads(sizes):
    """Every frame arrives intact and in order whatever the read boundaries."""
    frames = 4000
    data, mic, system = interleaved_ramps(frames)

    recorder = TapRecorder()
    recorder.channels = 2
    out = drain(recorder, RaggedPipe(data.tobytes(), sizes))

    assert len(out) == frames, "frames were lost"
    assert np.array_equal(out[:, 0], mic), "microphone channel corrupted"
    assert np.array_equal(out[:, 1], system), "system channel corrupted"


def test_channels_are_not_transposed():
    """Guard the specific symptom: mic and system swapping places.

    Equal-length ramps would still compare equal under some off-by-one errors,
    so assert the channels are distinguishable and land the right way round.
    """
    data, mic, system = interleaved_ramps(500)
    recorder = TapRecorder()
    recorder.channels = 2
    out = drain(recorder, RaggedPipe(data.tobytes(), [37]))

    assert out[1, 0] == 1.0 and out[1, 1] == -1.0
    assert not np.array_equal(out[:, 0], out[:, 1])


def test_mono_stream():
    recorder = TapRecorder(capture_mic=False)
    recorder.channels = 1
    samples = np.arange(1000, dtype=np.float32)
    out = drain(recorder, RaggedPipe(samples.tobytes(), [13, 7, 4096]))

    assert out.shape == (1000, 1)
    assert np.array_equal(out[:, 0], samples)


def test_paused_capture_keeps_nothing():
    """While paused, audio is read from the pipe but not retained."""
    data, _, _ = interleaved_ramps(1000)
    recorder = TapRecorder()
    recorder.channels = 2
    recorder._paused = True
    recorder._stop_event = threading.Event()
    recorder._proc = type("Proc", (), {"stdout": RaggedPipe(data.tobytes(), [4096])})()
    recorder._read_audio()
    assert recorder._chunks == []


def test_empty_stream_terminates():
    recorder = TapRecorder()
    recorder.channels = 2
    recorder._stop_event = threading.Event()
    recorder._proc = type("Proc", (), {"stdout": RaggedPipe(b"", [8])})()
    recorder._read_audio()
    assert recorder._chunks == []


def test_tap_receives_blocks_and_errors_are_contained():
    """A failing tap must never interrupt capture."""
    data, _, _ = interleaved_ramps(600)

    seen = []
    recorder = TapRecorder()
    recorder.channels = 2
    recorder.set_tap(seen.append)
    out = drain(recorder, RaggedPipe(data.tobytes(), [4096]))
    assert seen and sum(len(b) for b in seen) == len(out)

    exploding = TapRecorder()
    exploding.channels = 2

    def boom(_block):
        raise RuntimeError("tap failure")

    exploding.set_tap(boom)
    survived = drain(exploding, RaggedPipe(data.tobytes(), [4096]))
    assert len(survived) == 600, "a tap error must not cost us audio"


def test_channels_info_contract():
    """The layout the transcription pipeline relies on.

    Written against ``_channels_info`` / ``_is_multichannel`` when app.py read
    those private names directly. Phase 2 promoted them, so the names here moved
    with them; every value asserted is unchanged, which is what shows the rename
    was a rename and not a rewrite.
    """
    stereo = TapRecorder()
    assert stereo.channels_info == {"mic_channel": 0, "system_channels": [1]}
    assert stereo.is_multichannel is True

    mono = TapRecorder(capture_mic=False)
    assert mono.channels_info is None
    assert mono.is_multichannel is False


def test_both_backends_satisfy_the_audio_source_protocol():
    """The Protocol only earns its place if it actually matches both backends.

    isinstance on a runtime_checkable Protocol is a hasattr check, so this
    catches a member being renamed on one backend and not the other. It cannot
    catch a signature drifting; that is what the type checker is for.
    """
    from overheard.audio import Recorder
    from overheard.protocols import AudioSource

    assert isinstance(TapRecorder(), AudioSource)

    # Device -1 never resolves, so Recorder falls through to its mono defaults
    # without opening a stream. Building one is the point: half its interface is
    # assigned in __init__ rather than declared on the class.
    assert isinstance(Recorder(-1), AudioSource)
