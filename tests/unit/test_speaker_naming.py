"""Speaker naming tests.

Naming draws on three signals of decreasing reliability: a voice matched against
the library (evidence), which track carried the audio (measurement), and the
attendee list order (convention). The precedence between them is the part that
went wrong before, so it is what these tests pin down.
"""

import pytest

from overheard.transcribe import (
    _build_speaker_map,
    _canonicalize_turns,
    _fallback_speaker_names,
)

from ..conftest import turn


class TestBuildSpeakerMap:
    def test_mic_speaker_takes_the_first_local_label(self):
        mapping = _build_speaker_map(
            ["MIC_00"], ["SYS_00"], ["Sarah Kelly", "Don Reddin"], mic_speaker="Don Reddin"
        )
        assert mapping["MIC_00"] == "Don Reddin"
        assert mapping["SYS_00"] == "Sarah Kelly"

    def test_local_speaker_name_is_not_handed_out_twice(self):
        """The local speaker being in the attendee list must not double-assign."""
        mapping = _build_speaker_map(
            ["MIC_00"], ["SYS_00"], ["Don Reddin", "Sarah Kelly"], mic_speaker="Don Reddin"
        )
        assert sorted(mapping.values()) == ["Don Reddin", "Sarah Kelly"]

    def test_name_matching_is_case_and_space_insensitive(self):
        mapping = _build_speaker_map(
            ["MIC_00"], ["SYS_00"], ["  don reddin ", "Sarah Kelly"], mic_speaker="Don Reddin"
        )
        assert mapping["SYS_00"] == "Sarah Kelly"

    def test_remote_speakers_are_named_before_extra_local_ones(self):
        """The attendee list describes the people on the call.

        Extra microphone speakers are others in the room, so remote voices have
        the stronger claim on a listed name.
        """
        mapping = _build_speaker_map(
            ["MIC_00", "MIC_01"], ["SYS_00"], ["Sarah Kelly"], mic_speaker="Don Reddin"
        )
        assert mapping["SYS_00"] == "Sarah Kelly"
        assert "MIC_01" not in mapping

    def test_library_match_outranks_attendee_order(self):
        """A matched embedding is evidence and must win."""
        mapping = _build_speaker_map(
            ["MIC_00"], ["SYS_00", "SYS_01"],
            ["Alice", "Bob"],
            mic_speaker="Don Reddin",
            known={"SYS_01": "Bob"},
        )
        assert mapping["SYS_01"] == "Bob", "library match was overwritten"
        assert mapping["SYS_00"] == "Alice", "Bob must not be handed out again"

    def test_library_match_on_the_mic_label_beats_mic_speaker(self):
        mapping = _build_speaker_map(
            ["MIC_00"], [], ["Sarah Kelly"],
            mic_speaker="Don Reddin",
            known={"MIC_00": "Someone Else"},
        )
        assert mapping["MIC_00"] == "Someone Else"

    def test_mic_speaker_is_skipped_when_the_library_already_claimed_that_name(self):
        """Guards one person appearing as two speakers."""
        mapping = _build_speaker_map(
            ["MIC_00"], ["SYS_00"], [],
            mic_speaker="Don Reddin",
            known={"SYS_00": "Don Reddin"},
        )
        assert list(mapping.values()).count("Don Reddin") == 1
        assert "MIC_00" not in mapping

    def test_in_person_meeting_with_several_mic_speakers(self):
        """No system track: everyone is in the room, on one microphone."""
        mapping = _build_speaker_map(
            ["MIC_00", "MIC_01", "MIC_02"], [],
            ["Sarah Kelly", "Raj Patel"],
            mic_speaker="Don Reddin",
        )
        assert mapping == {
            "MIC_00": "Don Reddin",
            "MIC_01": "Sarah Kelly",
            "MIC_02": "Raj Patel",
        }

    def test_no_attendees_leaves_remote_speakers_unnamed(self):
        mapping = _build_speaker_map(["MIC_00"], ["SYS_00"], [], mic_speaker="Don")
        assert mapping == {"MIC_00": "Don"}

    def test_more_speakers_than_attendees_runs_out_quietly(self):
        mapping = _build_speaker_map(
            ["MIC_00"], ["SYS_00", "SYS_01", "SYS_02"], ["Alice"], mic_speaker="Don"
        )
        assert mapping["SYS_00"] == "Alice"
        assert "SYS_01" not in mapping and "SYS_02" not in mapping

    def test_blank_attendee_entries_are_ignored(self):
        mapping = _build_speaker_map(
            ["MIC_00"], ["SYS_00"], ["", "  ", "Alice"], mic_speaker="Don"
        )
        assert mapping["SYS_00"] == "Alice"

    def test_no_speakers_at_all(self):
        assert _build_speaker_map([], [], ["Alice"], mic_speaker="Don") == {}


class TestCanonicalizeTurns:
    def test_speakers_are_renumbered_by_first_speech(self):
        """The diarizer numbers by cluster order, which is arbitrary."""
        turns = [
            turn(5.0, 9.0, "S7"),
            turn(0.0, 4.0, "S3"),
            turn(10.0, 12.0, "S7"),
        ]
        relabelled, order = _canonicalize_turns(turns)
        assert order == ["SPEAKER_00", "SPEAKER_01"]
        by_start = {t["start"]: t["speaker"] for t in relabelled}
        assert by_start[0.0] == "SPEAKER_00", "first voice heard must be 00"
        assert by_start[5.0] == "SPEAKER_01"
        assert by_start[10.0] == "SPEAKER_01"

    def test_prefix_namespaces_each_track(self):
        _, order = _canonicalize_turns([turn(0.0, 1.0, "S1")], prefix="MIC")
        assert order == ["MIC_00"]

    def test_ordering_is_stable_regardless_of_input_order(self):
        turns = [turn(3.0, 4.0, "b"), turn(1.0, 2.0, "a")]
        _, first = _canonicalize_turns(turns)
        _, second = _canonicalize_turns(list(reversed(turns)))
        assert first == second

    def test_empty_input(self):
        assert _canonicalize_turns([]) == ([], [])

    def test_other_keys_survive_relabelling(self):
        """Embeddings must not be dropped: speaker identity depends on them."""
        relabelled, _ = _canonicalize_turns([turn(0.0, 1.0, "S1", embedding=[0.5, 0.5])])
        assert relabelled[0]["embedding"] == [0.5, 0.5]


class TestFallbackSpeakerNames:
    def test_unnamed_speakers_are_numbered_by_first_appearance(self):
        segments = [{"speaker": "SYS_01"}, {"speaker": "MIC_00"}, {"speaker": "SYS_01"}]
        assert _fallback_speaker_names(segments, {}) == {
            "SYS_01": "Speaker 1",
            "MIC_00": "Speaker 2",
        }

    def test_already_named_speakers_are_skipped(self):
        segments = [{"speaker": "MIC_00"}, {"speaker": "SYS_00"}]
        assert _fallback_speaker_names(segments, {"MIC_00": "Don"}) == {"SYS_00": "Speaker 1"}

    def test_numbering_is_continuous_across_tracks(self):
        """One sequence for the reader, not one per track."""
        segments = [{"speaker": "MIC_00"}, {"speaker": "SYS_00"}, {"speaker": "MIC_01"}]
        names = _fallback_speaker_names(segments, {})
        assert sorted(names.values()) == ["Speaker 1", "Speaker 2", "Speaker 3"]

    def test_segments_without_a_speaker_are_ignored(self):
        assert _fallback_speaker_names([{"text": "hi"}], {}) == {}
