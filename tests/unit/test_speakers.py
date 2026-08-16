"""Voice library tests.

A library match outranks every other naming signal, so a wrong entry propagates
to every future transcript with nothing to correct it. These tests cover both
directions: recognising a returning voice, and refusing to guess.
"""

import json

import numpy as np
import pytest

from overheard.speakers import DEFAULT_THRESHOLD, SpeakerLibrary, _cosine, mean_embeddings

from ..conftest import turn


def vector(seed: int, dims: int = 256):
    """A deterministic unit-ish embedding, distinct per seed."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=dims).tolist()


@pytest.fixture
def library(tmp_path):
    return SpeakerLibrary(tmp_path / "speakers.json")


class TestCosine:
    def test_identical_vectors(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_scale_does_not_matter(self):
        assert _cosine([1.0, 2.0], [3.0, 6.0]) == pytest.approx(1.0)

    def test_degenerate_inputs_return_zero_rather_than_raising(self):
        assert _cosine([], []) == 0.0
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


class TestMatching:
    def test_a_known_voice_is_recognised(self, library):
        embedding = vector(1)
        library.remember("Sarah Kelly", embedding)
        name, score = library.match(embedding)
        assert name == "Sarah Kelly" and score == pytest.approx(1.0)

    def test_an_unknown_voice_is_refused(self, library):
        library.remember("Sarah Kelly", vector(1))
        name, score = library.match(vector(99))
        assert name is None, "an unfamiliar voice must not be given a name"
        assert score < DEFAULT_THRESHOLD

    def test_threshold_is_respected(self, library):
        library.remember("Sarah Kelly", vector(1))
        # A permissive threshold accepts what the default rejects
        assert library.match(vector(99), threshold=-1.0)[0] == "Sarah Kelly"

    def test_empty_library_matches_nothing(self, library):
        assert library.match(vector(1)) == (None, 0.0)

    def test_one_name_per_label_within_a_recording(self, library):
        """Two voices in one meeting cannot both be the same person."""
        base = vector(1)
        library.remember("Sarah Kelly", base)
        matched = library.match_labels(
            {"SYS_00": base, "SYS_01": base}, threshold=-1.0
        )
        assert list(matched.values()).count("Sarah Kelly") == 1

    def test_best_scoring_label_wins_the_name(self, library):
        exact = vector(1)
        library.remember("Sarah Kelly", exact)
        matched = library.match_labels(
            {"SYS_00": vector(50), "SYS_01": exact}, threshold=-1.0
        )
        assert matched["SYS_01"] == "Sarah Kelly"

    def test_labels_below_threshold_are_left_unmatched(self, library):
        library.remember("Sarah Kelly", vector(1))
        assert library.match_labels({"SYS_00": vector(99)}) == {}


class TestRemember:
    def test_first_sighting_stores_one_sample(self, library):
        library.remember("Sarah Kelly", vector(1))
        described = library.describe()
        assert len(described) == 1
        name, samples, updated = described[0]
        assert (name, samples) == ("Sarah Kelly", 1)
        assert updated, "an updated date should be recorded"

    def test_repeated_sightings_average_rather_than_replace(self, library):
        """One noisy meeting must not dominate a well-known voice."""
        library.remember("Sarah Kelly", [1.0, 0.0])
        library.remember("Sarah Kelly", [0.0, 1.0])
        name, samples, _ = library.describe()[0]
        assert samples == 2
        stored = library._entries["Sarah Kelly"]["embedding"]
        assert stored == pytest.approx([0.5, 0.5])

    def test_running_mean_weights_by_sample_count(self, library):
        for _ in range(3):
            library.remember("Sarah Kelly", [1.0, 0.0])
        library.remember("Sarah Kelly", [0.0, 1.0])
        stored = library._entries["Sarah Kelly"]["embedding"]
        assert stored[0] > stored[1], "the established voice should dominate"

    def test_a_changed_embedding_size_resets_rather_than_corrupts(self, library):
        library.remember("Sarah Kelly", [1.0, 0.0])
        library.remember("Sarah Kelly", [1.0, 2.0, 3.0])
        assert len(library._entries["Sarah Kelly"]["embedding"]) == 3

    def test_empty_embedding_is_ignored(self, library):
        library.remember("Sarah Kelly", [])
        assert library.names() == []


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "speakers.json"
        first = SpeakerLibrary(path)
        first.remember("Sarah Kelly", vector(1))
        first.save()

        assert SpeakerLibrary(path).names() == ["Sarah Kelly"]

    def test_forget_removes_and_persists(self, tmp_path):
        path = tmp_path / "speakers.json"
        lib = SpeakerLibrary(path)
        lib.remember("Sarah Kelly", vector(1))
        lib.remember("Raj Patel", vector(2))
        lib.save()

        assert lib.forget("Sarah Kelly") is True
        assert lib.forget("Nobody") is False
        assert SpeakerLibrary(path).names() == ["Raj Patel"]

    def test_corrupt_file_degrades_to_empty(self, tmp_path):
        """A damaged library must not stop a meeting being transcribed."""
        path = tmp_path / "speakers.json"
        path.write_text("{not json")
        assert SpeakerLibrary(path).names() == []

    def test_unexpected_json_shape_degrades_to_empty(self, tmp_path):
        path = tmp_path / "speakers.json"
        path.write_text(json.dumps(["not", "a", "dict"]))
        assert SpeakerLibrary(path).names() == []

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert SpeakerLibrary(tmp_path / "absent.json").names() == []


class TestMeanEmbeddings:
    def test_longer_turns_carry_more_weight(self):
        """A long turn is more of the voice than a two-word interjection."""
        turns = [
            turn(0.0, 10.0, "S1", embedding=[1.0, 0.0]),
            turn(10.0, 11.0, "S1", embedding=[0.0, 1.0]),
        ]
        result = mean_embeddings(turns)["S1"]
        assert result[0] > result[1] * 5

    def test_speakers_are_kept_separate(self):
        turns = [
            turn(0.0, 1.0, "S1", embedding=[1.0, 0.0]),
            turn(1.0, 2.0, "S2", embedding=[0.0, 1.0]),
        ]
        result = mean_embeddings(turns)
        assert result["S1"] == pytest.approx([1.0, 0.0])
        assert result["S2"] == pytest.approx([0.0, 1.0])

    def test_turns_without_embeddings_are_skipped(self):
        assert mean_embeddings([turn(0.0, 1.0, "S1")]) == {}

    def test_zero_length_turns_are_skipped(self):
        assert mean_embeddings([turn(1.0, 1.0, "S1", embedding=[1.0])]) == {}

    def test_no_turns(self):
        assert mean_embeddings([]) == {}
