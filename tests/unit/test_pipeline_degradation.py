"""Speaker memory failing must cost the names, not the meeting.

The voice-matching block runs after every track has been transcribed, and
app._on_details_confirmed deletes the temp WAV in a finally whenever
transcribe_audio raises. So anything escaping that block destroys a fully
transcribed meeting AND its audio, to save nothing: speaker memory only decides
what the speakers are called.

That is not hypothetical. The diarizer's embedding guard and this consumer are
two places that have to agree about shape, and they have disagreed once
already, in a way that raised inside numpy rather than at the parse. The guard
was corrected; this exists so the next disagreement cannot cost a transcript.
"""

import pytest

from overheard import pipeline


@pytest.fixture
def one_track_meeting(monkeypatch, tmp_path):
    """Everything up to the speaker-memory block, stubbed at the seams.

    ASR and diarization are the slow parts and are not what this file is about,
    so they are replaced. Everything downstream of them is real: canonicalize,
    assign, merge, name and render all run, which is what makes "the transcript
    survived" an assertion about the actual output file.
    """
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"")

    monkeypatch.setattr(pipeline, "check_audio_signal", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "warn_silent_tracks", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "split_tracks",
        lambda *a, **k: [{"label": "mixed", "path": str(audio), "temp": False}],
    )
    monkeypatch.setattr(
        pipeline, "transcribe_parakeet",
        lambda *a, **k: [{"start": 0.0, "end": 1.0, "text": "hello there"}],
    )
    monkeypatch.setattr(
        pipeline, "diarize",
        lambda *a, **k: [
            {"start": 0.0, "end": 1.0, "speaker": "A", "embedding": [0.1, 0.2]},
        ],
    )
    monkeypatch.setattr(pipeline.cfg, "get", lambda key, *a: {
        "speaker_memory": True,
        "speaker_match_threshold": 0.5,
    }.get(key))

    return str(audio), str(tmp_path / "out.md")


def test_the_transcript_survives_speaker_memory_raising(
    one_track_meeting, monkeypatch, capsys
):
    """The whole point of the guard around the voice-matching block.

    Asserting transcribe_audio returned is not enough on its own: the file has
    to be on disk with the words in it, because the recording it came from is
    deleted the moment this raises.
    """
    import overheard.speakers as speakers

    def explode(*args, **kwargs):
        raise ValueError("operands could not be broadcast together")

    monkeypatch.setattr(speakers, "mean_embeddings", explode)

    audio, output = one_track_meeting
    result = pipeline.transcribe_audio(audio, output)

    assert result == output
    written = open(output).read()
    assert "hello there" in written, (
        "the transcript was lost to a speaker-memory failure, and the caller "
        "deletes the WAV when this raises"
    )
    assert "speaker memory failed" in capsys.readouterr().err, (
        "the failure must be reported, or it degrades silently and the missing "
        "names look like a diarization problem"
    )


def test_the_transcript_survives_the_library_raising(
    one_track_meeting, monkeypatch, capsys
):
    """mean_embeddings is not the only thing in that block that can raise.

    SpeakerLibrary reads a JSON file off disk and matches against it. Guarding
    only the call that failed once would be pinning the instance and leaving
    the class, which is the mistake this unit has made repeatedly.
    """
    import overheard.speakers as speakers

    class Exploding:
        def __init__(self, *a, **k):
            raise OSError("speakers.json is unreadable")

    monkeypatch.setattr(speakers, "SpeakerLibrary", Exploding)

    audio, output = one_track_meeting
    result = pipeline.transcribe_audio(audio, output)

    assert result == output
    assert "hello there" in open(output).read()
    assert "speaker memory failed" in capsys.readouterr().err
