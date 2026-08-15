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
    # 256 floats, the width FluidAudio actually emits and the width
    # diarization._usable_embedding requires. A short stand-in here would be
    # dropped by that guard, so speaker memory would find nothing to match and
    # the success test below would fail for a reason that has nothing to do
    # with what it is testing.
    monkeypatch.setattr(
        pipeline, "diarize",
        lambda *a, **k: [
            {"start": 0.0, "end": 1.0, "speaker": "A", "embedding": [0.1] * 256},
        ],
    )
    monkeypatch.setattr(pipeline.cfg, "get", lambda key, *a: {
        "speaker_memory": True,
        "speaker_match_threshold": 0.5,
    }.get(key))

    return str(audio), str(tmp_path / "out.md")


@pytest.fixture
def two_track_meeting(monkeypatch, tmp_path):
    """A mic track and a system track, which is what learning requires.

    _learn_confident_voices only trusts the local speaker, and "local" means the
    mic track: with a single mixed track local_labels is empty and nothing is
    ever confident, so a mixed-track fixture cannot reach library.remember at
    all. The stubs key on the track path so the two sides differ; identical
    segments would be treated as echo and drop_echo would empty the mic side,
    which empties local_labels again by a different route.
    """
    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    for path in (mic, system):
        path.write_bytes(b"")

    monkeypatch.setattr(pipeline, "check_audio_signal", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "warn_silent_tracks", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "split_tracks", lambda *a, **k: [
        {"label": "mic", "path": str(mic), "temp": False},
        {"label": "system", "path": str(system), "temp": False},
    ])

    def segments_for(path, **kwargs):
        if path == str(mic):
            return [{"start": 0.0, "end": 1.0, "text": "hello there"}]
        return [{"start": 5.0, "end": 6.0, "text": "good to see you"}]

    def turns_for(path, **kwargs):
        base = 0.0 if path == str(mic) else 5.0
        return [{
            "start": base, "end": base + 1.0, "speaker": "A",
            "embedding": [0.1 if path == str(mic) else 0.9] * 256,
        }]

    monkeypatch.setattr(pipeline, "transcribe_parakeet", segments_for)
    monkeypatch.setattr(pipeline, "diarize", turns_for)
    monkeypatch.setattr(pipeline.cfg, "get", lambda key, *a: {
        "speaker_memory": True,
        "speaker_match_threshold": 0.5,
    }.get(key))

    return str(tmp_path / "in.wav"), str(tmp_path / "out.md")


def test_a_remembered_voice_reaches_the_transcript(
    one_track_meeting, isolated_config
):
    """Speaker memory working, which nothing asserted.

    Every other test in this file asserts the block FAILS gracefully, and they
    are all satisfied by "transcript on disk" plus a warning on stderr. Raising
    on the first line of the try produced exactly that, so cross-meeting voice
    identity, the feature the whole embedding and library stack exists for,
    could be completely dead with all tests green. Testing only the fallback is
    the standing defect class with the branches swapped.

    This drives the real SpeakerLibrary against an isolated path: seed a name,
    hand the pipeline an embedding that matches it, and require the name to
    come out the far end in the rendered markdown.
    """
    from overheard.speakers import SpeakerLibrary

    library = SpeakerLibrary()
    library.remember("Don Reddin", [0.1] * 256)
    library.save()

    audio, output = one_track_meeting
    pipeline.transcribe_audio(audio, output)

    written = open(output).read()
    assert "Don Reddin" in written, (
        "a voice already in the library was not recognised, so speaker memory "
        "is doing nothing and every failure test here would still pass"
    )


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


def test_a_failed_match_does_not_disable_learning(
    two_track_meeting, isolated_config, monkeypatch, capsys
):
    """The one state where the except handler's reset actually does anything.

    Both other failure tests raise before `library` is ever bound, so the reset
    line was unreachable under test and its only reachable effect was wrong: it
    set library to None, which makes _learn_confident_voices return early. A
    user with one bad entry in speakers.json would stop accumulating any voice
    profile at all, including the local speaker, whose name comes off the mic
    track and has nothing to do with matching.
    """
    from overheard.speakers import SpeakerLibrary

    def explode(*args, **kwargs):
        raise ValueError("speakers.json holds a corrupt entry")

    monkeypatch.setattr(SpeakerLibrary, "match_labels", explode)

    audio, output = two_track_meeting
    pipeline.transcribe_audio(audio, output, mic_speaker="Don Reddin")

    assert "speaker memory failed" in capsys.readouterr().err
    assert "hello there" in open(output).read()

    stored = SpeakerLibrary()._entries
    assert stored, (
        "matching failed and learning was disabled with it, so the local "
        "speaker's voice was never stored despite being known from the mic track"
    )


def test_the_library_write_cannot_cost_the_meeting(
    two_track_meeting, isolated_config, monkeypatch, capsys
):
    """_learn_confident_voices runs after write_markdown, and was unguarded.

    Anything escaping it costs the WAV, because the caller unlinks tmp_path in a
    finally, and shows the user an error for a meeting whose transcript is
    already safely on disk. It is also the half that writes persistent state,
    which is exactly where the round-ten library clobber landed.
    """
    from overheard.speakers import SpeakerLibrary

    def explode(*args, **kwargs):
        raise OSError("disk full while saving speakers.json")

    monkeypatch.setattr(SpeakerLibrary, "remember", explode)

    audio, output = two_track_meeting
    result = pipeline.transcribe_audio(audio, output, mic_speaker="Don Reddin")

    assert result == output
    assert "hello there" in open(output).read()
    assert "could not update the voice library" in capsys.readouterr().err


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
