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
        pytest.param({"start": 0.0, "end": 1.0, "speaker": {"n": 1}}, id="dict-speaker"),
        pytest.param({"start": 0.0, "end": 1.0, "speaker": ["A"]}, id="list-speaker"),
        pytest.param({"start": 0.0, "end": 1.0, "speaker": 7}, id="int-speaker"),
    ])
    def test_a_malformed_segment_is_skipped_not_fatal(self, helper_returning, segment):
        helper_returning({"segments": [segment]})
        result = diarization._diarize_fluidaudio("meeting.wav")
        assert result == [], f"{segment} should be skipped, got {result}"

    @pytest.mark.parametrize("embedding", [
        pytest.param("oops", id="string"),
        pytest.param({"v": 1}, id="dict"),
        pytest.param([0.1, "x", 0.3], id="list-with-a-string"),
        pytest.param([[0.1], [0.2]], id="nested"),
        pytest.param([0.1, 0.2], id="too-narrow"),
        pytest.param([0.1] * 512, id="too-wide"),
    ])
    def test_a_malformed_embedding_costs_the_embedding_not_the_turn(
        self, helper_returning, embedding
    ):
        """An embedding is optional metadata and must not cost the diarization.

        This asserted the whole segment was dropped until round ten, which is
        what made a bad segment so expensive: start, end and speaker went with
        it. With the width taken from the first arrival, one malformed leading
        segment discarded every good one behind it, assign_speakers fell to its
        `if not turns` branch, and the meeting lost all speaker attribution.

        Nothing downstream re-checks the vector: speakers.mean_embeddings hands
        it to np.asarray(dtype="float64"), and pipeline calls that after
        transcription. So it has to be dropped here, and only it.
        """
        helper_returning({"segments": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "embedding": embedding},
        ]})
        result = diarization._diarize_fluidaudio("meeting.wav", with_embeddings=True)

        assert len(result) == 1, f"the turn was discarded with its embedding: {result}"
        assert "embedding" not in result[0], (
            f"{embedding!r} reached the caller, which hands it to numpy"
        )
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_one_bad_leading_embedding_cannot_discard_the_whole_meeting(
        self, helper_returning
    ):
        """The critical from round ten, as its own case.

        The width used to be whatever the first accepted embedding happened to
        be, so a 2-wide segment at the front made every correct 256-wide segment
        behind it the outlier. Four of five turns were thrown away and the only
        surviving vector was the malformed one, which then went into the
        permanent speaker library and replaced a real profile.
        """
        helper_returning({"segments": [
            {"start": 0.0, "end": 1.0, "speaker": "A", "embedding": [0.1, 0.2]},
            {"start": 1.0, "end": 2.0, "speaker": "B", "embedding": [0.1] * 256},
            {"start": 2.0, "end": 3.0, "speaker": "A", "embedding": [0.2] * 256},
            {"start": 3.0, "end": 4.0, "speaker": "B", "embedding": [0.3] * 256},
        ]})
        result = diarization._diarize_fluidaudio("meeting.wav", with_embeddings=True)

        assert len(result) == 4, (
            f"a bad leading segment cost {4 - len(result)} good turns, and "
            "losing every turn labels the whole meeting SPEAKER_00"
        )
        widths = {len(t["embedding"]) for t in result if t.get("embedding")}
        assert widths == {256}, f"the wrong width became canonical: {widths}"

    def test_whatever_survives_the_guard_survives_the_code_that_reads_it(
        self, helper_returning
    ):
        """The guard is only worth what its consumers can actually swallow.

        Asserting the guard skipped something tests the guard against itself.
        This runs the two functions that receive its output on the meeting-
        destroying path, so the assertion is the property that matters: nothing
        the diarizer can emit reaches them in a shape they raise on.
        """
        from overheard import speakers

        helper_returning({"segments": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "embedding": "oops"},
            {"start": 1.0, "end": 2.0, "speaker": {"n": 1}},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01", "embedding": [0.3, "x"]},
            # Two ACCEPTED embeddings for one speaker, so mean_embeddings takes
            # its accumulate branch. Without a second survivor on the same label
            # that branch never runs, and it is the only line in mean_embeddings
            # that can raise: an earlier version of this test left exactly one
            # survivor and therefore asserted nothing. The assertion below now
            # fails rather than letting that recur, which is how the round-ten
            # width change was caught here rather than in review.
            {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01", "embedding": [0.1] * 256},
            {"start": 4.0, "end": 5.0, "speaker": "SPEAKER_01", "embedding": [0.5] * 256},
            # Ragged against those two. Same speaker, wrong width.
            {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_01", "embedding": [0.2] * 128},
        ]})
        turns = diarization._diarize_fluidaudio("meeting.wav", with_embeddings=True)

        widths = {len(t["embedding"]) for t in turns if t.get("embedding")}
        assert len(widths) <= 1, f"the guard emitted ragged embeddings: {widths}"
        assert sum(1 for t in turns if t.get("embedding")) >= 2, (
            "fewer than two embeddings survived, so mean_embeddings never "
            "reaches the accumulate branch and this test proves nothing"
        )

        canonical, _order = diarization.canonicalize_turns(turns)
        speakers.mean_embeddings(canonical)

    def test_good_segments_survive_a_bad_neighbour(self, helper_returning):
        """One bad segment costs that segment, not the other speakers."""
        helper_returning({"segments": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": None, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_02"},
        ]})
        result = diarization._diarize_fluidaudio("meeting.wav")
        assert [t["speaker"] for t in result] == ["SPEAKER_00", "SPEAKER_02"]

    @pytest.mark.parametrize("payload, why", [
        pytest.param({"segments": None}, "a Swift Encodable optional array emits null",
                     id="segments-null"),
        pytest.param([], "the helper emitted a bare array", id="payload-list"),
        pytest.param(None, "the helper emitted null", id="payload-null"),
        pytest.param("ok", "the helper emitted a bare string", id="payload-string"),
        pytest.param({"segments": 5}, "segments is not a sequence", id="segments-int"),
        pytest.param({"segments": [None, "x", 3]}, "the segments are not objects",
                     id="segments-not-objects"),
    ])
    def test_malformed_payloads_read_as_unavailable(self, helper_returning, payload, why):
        """Well-formed JSON is not well-formed output.

        `.get`'s default fires only when a key is ABSENT, not when it is null,
        and a bare list, null or string payload has no `.get` at all. Each of
        these raised past diarize() and out of transcribe_audio, whose caller
        deletes the temp WAV, so the diarizer destroyed a meeting it had
        already transcribed.
        """
        helper_returning(payload)
        result = diarization._diarize_fluidaudio("meeting.wav")
        assert result is None or result == [], f"{why}: got {result!r}"

    def test_a_string_payload_is_unavailable_not_empty(self, helper_returning):
        """Discriminates the guard from the pre-fix accident.

        Before the guard, iterating a string yielded characters that the inner
        handler skipped one by one, so the function returned [] and an
        `in ([], None)` assertion passed either way. None is the honest answer:
        the backend could not be read.
        """
        helper_returning("unexpected")
        assert diarization._diarize_fluidaudio("meeting.wav") is None
