"""End-to-end pipeline test.

One run through transcribe_audio with real speech, asserting the thing that
actually matters: the right words attributed to the right people, in a file the
vault can read. The unit tests cover the parts; this covers them working
together.
"""

import subprocess

import numpy as np
import pytest
import soundfile as sf

from ..conftest import requires_helper, requires_say

pytestmark = [pytest.mark.integration, requires_helper, requires_say]

DUAL = {"mic_channel": 2, "system_channels": [0, 1]}


@pytest.fixture(scope="module")
def dual_speech_wav(tmp_path_factory):
    """A 3-channel recording: the local speaker on the mic, a remote on system.

    Includes 15 percent bleed of the remote voice into the microphone, which is
    what happens on laptop speakers and what echo suppression has to survive.
    """
    directory = tmp_path_factory.mktemp("dual_speech")

    def speak(voice: str, line: str, name: str):
        aiff, wav = directory / f"{name}.aiff", directory / f"{name}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), line], check=True, timeout=120)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
            check=True, timeout=120,
        )
        return sf.read(wav, dtype="float32")[0]

    local = speak("Daniel", "Right, let us kick off with the budget review.", "local")
    remote = speak("Karen", "Thanks. The numbers look better than we forecast.", "remote")

    rate = 16000
    gap = int(rate * 1.0)
    length = len(local) + gap + len(remote)
    mic = np.zeros(length, dtype=np.float32)
    system = np.zeros(length, dtype=np.float32)

    mic[: len(local)] = local
    offset = len(local) + gap
    system[offset:offset + len(remote)] = remote
    mic[offset:offset + len(remote)] = remote * 0.15   # speaker bleed

    path = directory / "dual.wav"
    sf.write(path, np.stack([system, system, mic], axis=1), rate)
    return path


def speaker_lines(markdown_path):
    return [line for line in open(markdown_path).read().splitlines() if line.startswith("**")]


def test_dual_track_transcript_names_both_speakers(dual_speech_wav, tmp_path, isolated_config):
    from overheard.details_panel import MeetingDetails
    from overheard.pipeline import transcribe_audio

    details = MeetingDetails(
        name="Budget Sync", source="zoom", location="Zoom",
        attendees=["Sarah Kelly", "Don Reddin"],
    )
    output = tmp_path / "transcript.md"
    transcribe_audio(
        str(dual_speech_wav), str(output),
        meeting_details=details, mic_speaker="Don Reddin", channels_info=DUAL,
    )

    content = output.read_text()
    lines = speaker_lines(output)

    assert "[[Don Reddin]]" in content, "the local speaker should be named from the mic track"
    assert "[[Sarah Kelly]]" in content, "the remote speaker should take the attendee name"

    # Frontmatter the vault relies on
    assert content.startswith("---")
    assert "type: transcript" in content
    assert "source: zoom" in content

    # The local speaker talks first in the fixture
    assert "Don Reddin" in lines[0]

    # Echo suppression: the remote line must not be transcribed twice
    assert content.lower().count("forecast") == 1, "remote speech was duplicated by echo"


def test_speaker_is_recognised_in_a_later_meeting(dual_speech_wav, tmp_path, isolated_config):
    """The unambiguous case: one remote speaker, one unclaimed attendee.

    A second meeting with no attendee list and no mic_speaker should still name
    both people, from voice alone.
    """
    from overheard.details_panel import MeetingDetails
    from overheard.speakers import SpeakerLibrary
    from overheard.pipeline import transcribe_audio

    first = MeetingDetails(
        name="First", source="zoom", location="Zoom",
        attendees=["Sarah Kelly", "Don Reddin"],
    )
    transcribe_audio(
        str(dual_speech_wav), str(tmp_path / "first.md"),
        meeting_details=first, mic_speaker="Don Reddin", channels_info=DUAL,
    )

    learned = SpeakerLibrary().names()
    assert "Don Reddin" in learned
    assert "Sarah Kelly" in learned, "an unambiguous remote assignment should be learned"

    second = MeetingDetails(name="Second", source="zoom", location="Zoom", attendees=[])
    output = tmp_path / "second.md"
    transcribe_audio(
        str(dual_speech_wav), str(output),
        meeting_details=second, mic_speaker=None, channels_info=DUAL,
    )

    content = output.read_text()
    assert "[[Don Reddin]]" in content
    assert "[[Sarah Kelly]]" in content


def test_mono_recording_still_produces_a_transcript(dual_speech_wav, tmp_path, isolated_config):
    """No channel layout: one folded track, speakers separated by clustering."""
    from overheard.details_panel import MeetingDetails
    from overheard.pipeline import transcribe_audio

    data, rate = sf.read(dual_speech_wav, dtype="float32")
    mono_path = tmp_path / "mono.wav"
    sf.write(mono_path, data.mean(axis=1), rate)

    details = MeetingDetails(
        name="Mono", source="in-person", location="Office", attendees=["Sarah Kelly"],
    )
    output = tmp_path / "mono.md"
    transcribe_audio(
        str(mono_path), str(output), meeting_details=details, channels_info=None,
    )

    assert speaker_lines(output), "a mono recording should still yield attributed lines"
    assert "budget" in output.read_text().lower()
