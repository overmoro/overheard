"""Diarization failing must cost the speaker labels, not the transcript.

The pipeline runs ASR first and diarizes afterwards, so by the time this is
called the meeting has already been transcribed. Returning None instead of an
empty list would blow up canonicalize_turns at pipeline.py:83 and lose the
whole transcript at the last step, which is the outcome the module docstring
calls unacceptable.

That degradation used to be a two-branch body with a pyannote fallback under
it. The single-engine cut rewrote it into one line and it had no test at all,
so nothing pinned the behaviour the cut promised to preserve.
"""

import json
import subprocess

import pytest

from overheard import diarization


@pytest.fixture(autouse=True)
def no_config_reads(monkeypatch):
    """Keep these hermetic: no helper binary, no subprocess, unless asked."""
    monkeypatch.setattr(
        diarization, "_diarize_fluidaudio", lambda *a, **k: None, raising=True
    )


class TestUnavailableIsNotFatal:
    def test_no_helper_binary_gives_an_empty_list(self):
        """None means "this backend could not run", and must not escape."""
        assert diarization.diarize("meeting.wav") == []

    def test_the_empty_list_is_a_list_not_none(self):
        """canonicalize_turns iterates it, so None would raise on the caller.

        Pinned as a type rather than as equality, because `== []` also passes
        for anything falsy while `turns[0]` further down the pipeline would
        still raise.
        """
        result = diarization.diarize("meeting.wav")
        assert isinstance(result, list)

    def test_a_backend_that_found_no_speech_is_passed_through(self, monkeypatch):
        """[] and None mean different things and must not be conflated.

        [] is "it ran and heard nobody", None is "it could not run". Both come
        out as [] here, but only because the backend distinguishes them first.
        """
        monkeypatch.setattr(diarization, "_diarize_fluidaudio", lambda *a, **k: [])
        assert diarization.diarize("meeting.wav") == []

    def test_real_turns_are_returned_unchanged(self, monkeypatch):
        turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        monkeypatch.setattr(diarization, "_diarize_fluidaudio", lambda *a, **k: turns)
        assert diarization.diarize("meeting.wav") == turns


class TestTheBackendReportsUnavailableRatherThanRaising:
    """_diarize_fluidaudio is what has to return None, so pin that too."""

    @pytest.fixture(autouse=True)
    def real_backend(self, monkeypatch):
        monkeypatch.undo()

    def test_a_missing_helper_reports_none(self, monkeypatch):
        monkeypatch.setattr("overheard.helper.helper_path", lambda: None)
        assert diarization._diarize_fluidaudio("meeting.wav") is None

    def test_a_failing_subprocess_does_not_raise(self, monkeypatch, tmp_path):
        """A non-zero exit is "unavailable", not an exception up the pipeline."""
        binary = tmp_path / "overheard-helper"
        binary.write_text("")
        monkeypatch.setattr("overheard.helper.helper_path", lambda: binary)

        class Result:
            returncode = 1
            stdout = b""
            stderr = b"models missing"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        assert diarization._diarize_fluidaudio("meeting.wav") is None

    def test_unparseable_output_does_not_raise(self, monkeypatch, tmp_path):
        """Whatever the helper prints, the transcript still has to survive."""
        binary = tmp_path / "overheard-helper"
        binary.write_text("")
        monkeypatch.setattr("overheard.helper.helper_path", lambda: binary)

        class Result:
            returncode = 0
            stdout = b"not json at all"
            stderr = b""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        try:
            result = diarization._diarize_fluidaudio("meeting.wav")
        except json.JSONDecodeError:
            pytest.fail(
                "unparseable helper output escaped as an exception, which would "
                "lose an already-transcribed meeting at the last step"
            )
        assert result is None, "unparseable output must read as unavailable"

    def test_a_timeout_does_not_raise(self, monkeypatch, tmp_path):
        """A 30 minute cap exists, and hitting it must not lose the transcript."""
        binary = tmp_path / "overheard-helper"
        binary.write_text("")
        monkeypatch.setattr("overheard.helper.helper_path", lambda: binary)

        def timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="overheard-helper", timeout=1800)

        monkeypatch.setattr(subprocess, "run", timeout)
        assert diarization._diarize_fluidaudio("meeting.wav") is None

    def test_the_helper_being_unrunnable_does_not_raise(self, monkeypatch, tmp_path):
        """A bundle whose helper lost its execute bit still has to transcribe."""
        binary = tmp_path / "overheard-helper"
        binary.write_text("")
        monkeypatch.setattr("overheard.helper.helper_path", lambda: binary)

        def oserror(*a, **k):
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(subprocess, "run", oserror)
        assert diarization._diarize_fluidaudio("meeting.wav") is None


class TestMalformedSegmentsDoNotCostTheTranscript:
    """Well-formed JSON is not well-formed output.

    The parse is guarded; the loop that reads each segment was not. It runs
    after transcription has completed, and app._on_details_confirmed deletes
    the temp WAV when transcribe_audio raises, so a single bad segment
    destroyed a fully transcribed meeting and left an error notification.
    """

    @pytest.fixture(autouse=True)
    def real_backend(self, monkeypatch):
        monkeypatch.undo()

    @pytest.fixture
    def helper_returning(self, monkeypatch, tmp_path):
        def install(payload):
            binary = tmp_path / "overheard-helper"
            binary.write_text("")
            monkeypatch.setattr("overheard.helper.helper_path", lambda: binary)

            class Result:
                returncode = 0
                stdout = json.dumps(payload).encode()
                stderr = b""

            monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        return install

    @pytest.mark.parametrize("segment", [
        pytest.param({"start": 0.0, "end": 1.0}, id="missing-speaker"),
        pytest.param({"end": 1.0, "speaker": "A"}, id="missing-start"),
        pytest.param({"start": None, "end": 1.0, "speaker": "A"}, id="null-start"),
        pytest.param({"start": "0.0s", "end": 1.0, "speaker": "A"}, id="non-numeric"),
        pytest.param({"start": 0.0, "end": "later", "speaker": "A"}, id="non-numeric-end"),
    ])
    def test_a_malformed_segment_is_skipped_not_fatal(self, helper_returning, segment):
        helper_returning({"segments": [segment]})
        result = diarization._diarize_fluidaudio("meeting.wav")
        assert result == [], f"{segment} should be skipped, got {result}"

    def test_good_segments_survive_a_bad_neighbour(self, helper_returning):
        """One bad segment costs that segment, not the other speakers."""
        helper_returning({"segments": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": None, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_02"},
        ]})
        result = diarization._diarize_fluidaudio("meeting.wav")
        assert [t["speaker"] for t in result] == ["SPEAKER_00", "SPEAKER_02"]

    def test_a_segments_key_that_is_not_a_list_does_not_raise(self, helper_returning):
        helper_returning({"segments": "unexpected"})
        assert diarization._diarize_fluidaudio("meeting.wav") in ([], None)
