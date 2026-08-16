"""A voice library cannot be reconstructed, so losing it has to be hard.

Every other kind of state this app holds can be rebuilt: settings have defaults,
transcripts can be re-run from a WAV. A voice profile is accumulated across many
meetings from audio that is deleted afterwards unless keep_recordings is on, so
once it is gone it is gone.

It was not hard to lose. save() truncated the file before writing it, load()
reset to an empty dict on any read failure and printed to a stderr a bundled app
has no terminal for, and remember() replaced a profile outright when a vector
disagreed about shape. A width bug in the diarizer guard reached that last one
and turned twelve meetings of accumulated data into a two-float entry, which is
what these tests exist to make impossible rather than merely fixed.
"""

import json

import pytest

from overheard import speakers


@pytest.fixture
def library_path(tmp_path, monkeypatch):
    path = tmp_path / "speakers.json"
    monkeypatch.setattr(speakers, "LIBRARY_PATH", path)
    return path


def _seeded(library_path, name="Don Reddin", meetings=12):
    """A profile with real history behind it, saved the way the app saves it."""
    library = speakers.SpeakerLibrary()
    for _ in range(meetings):
        library.remember(name, [0.5] * 256)
    library.save()
    return library


class TestAnAccumulatedProfileIsNotDiscarded:
    def test_a_disagreeing_vector_cannot_replace_a_real_profile(
        self, library_path, capsys
    ):
        """The exact loss a diarizer width bug caused, now refused.

        remember() reset samples to 1 and wrote the new vector, so one malformed
        embedding cost every meeting that came before it.
        """
        _seeded(library_path)
        library = speakers.SpeakerLibrary()
        library.remember("Don Reddin", [0.1, 0.2])
        library.save()

        entry = json.load(open(library_path))["Don Reddin"]
        assert len(entry["embedding"]) == 256, "the profile was overwritten"
        assert entry["samples"] == 12, "the accumulated history was reset"
        assert "refusing to replace" in capsys.readouterr().err, (
            "refusing silently is its own problem: the user would see voice "
            "matching quietly stop working with nothing to act on"
        )

    def test_the_refusal_names_a_way_out(self, library_path, capsys):
        """A legitimate upstream width change lands here too, and looks the same.

        It cannot be distinguished from corruption, so it is not auto-accepted.
        That is only defensible if the message says what to do about it.
        """
        _seeded(library_path)
        speakers.SpeakerLibrary().remember("Don Reddin", [0.1] * 128)

        err = capsys.readouterr().err
        assert "Forget this speaker" in err
        assert "256" in err and "128" in err, (
            f"the message must name both shapes to be actionable, got {err!r}"
        )

    def test_a_single_sample_entry_is_still_replaceable(self, library_path):
        """The refusal is about protecting evidence, not about being immovable.

        One sample is a guess that has been stored once. Refusing to update it
        would leave a bad first impression of a voice permanently in place.
        """
        library = speakers.SpeakerLibrary()
        library.remember("Sarah", [0.5] * 256)
        library.save()
        assert json.load(open(library_path))["Sarah"]["samples"] == 1

        library = speakers.SpeakerLibrary()
        library.remember("Sarah", [0.1] * 128)
        library.save()
        assert len(json.load(open(library_path))["Sarah"]["embedding"]) == 128

    def test_forget_really_is_the_way_out(self, library_path):
        """The escape the refusal message promises has to actually work."""
        _seeded(library_path)
        library = speakers.SpeakerLibrary()
        assert library.forget("Don Reddin") is True

        library = speakers.SpeakerLibrary()
        library.remember("Don Reddin", [0.1] * 128)
        library.save()
        assert len(json.load(open(library_path))["Don Reddin"]["embedding"]) == 128


class TestTheFileSurvivesABadWrite:
    def test_a_corrupt_library_is_recovered_from_the_backup(
        self, library_path, capsys
    ):
        """The backup is only worth keeping if load() actually reaches for it.

        Writing one and never reading it is the shape of a safety net that
        exists in the commit message and nowhere else.
        """
        _seeded(library_path)
        # Save once more so a backup of the good content exists.
        speakers.SpeakerLibrary().save()
        library_path.write_text("{ truncated half-writ")

        library = speakers.SpeakerLibrary()

        assert "Don Reddin" in library.names(), "the library was lost"
        assert library._entries["Don Reddin"]["samples"] == 12
        assert "recovered" in capsys.readouterr().err

    def test_an_interrupted_write_leaves_the_old_file_intact(
        self, library_path, monkeypatch
    ):
        """Atomicity, as an observable rather than as a claim about os.replace.

        The previous implementation opened the real path with "w", which
        truncates before anything is written, so a failure partway through left
        a file that parsed as nothing. Here the failure happens after the temp
        file is written and before it is renamed.
        """
        _seeded(library_path)
        before = library_path.read_text()

        def fail_rename(*args, **kwargs):
            raise OSError("no space left on device")

        library = speakers.SpeakerLibrary()
        library.remember("Sarah", [0.9] * 256)
        monkeypatch.setattr(speakers.os, "replace", fail_rename)
        library.save()

        assert library_path.read_text() == before, (
            "an interrupted write damaged the library it was replacing"
        )
        assert speakers.SpeakerLibrary().names() == ["Don Reddin"]

    def test_a_failed_write_leaves_no_temp_files_behind(
        self, library_path, monkeypatch
    ):
        """Otherwise the config directory silently fills with half-written copies."""
        _seeded(library_path)

        def fail_rename(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(speakers.os, "replace", fail_rename)
        speakers.SpeakerLibrary().save()

        strays = [p.name for p in library_path.parent.glob("*.tmp")]
        assert not strays, f"left behind {strays}"

    def test_an_empty_library_does_not_overwrite_a_good_backup(self, library_path):
        """The failure mode that makes a backup useless.

        If load() ever resets to {} and the next save() writes that empty dict
        through, the backup is overwritten by the emptiness one save later and
        the data is gone for good. load() recovering rather than resetting is
        what prevents it, and this asserts the end state rather than the branch.
        """
        _seeded(library_path)
        speakers.SpeakerLibrary().save()
        library_path.write_text("not json at all")

        recovered = speakers.SpeakerLibrary()
        recovered.save()

        assert json.load(open(library_path))["Don Reddin"]["samples"] == 12
