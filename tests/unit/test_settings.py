"""Settings schema tests.

The point of settings.py is that a default exists in exactly one place. These
tests hold that line: they assert the schema is complete, that no caller can
restate a default, and that the two keys which previously had two different
"defaults" now have one.
"""

import json

import pytest

from overheard import config as cfg
from overheard import settings


class TestSchema:
    def test_every_key_read_by_the_app_is_declared(self):
        """A key the code reads but the schema omits would raise at runtime.

        Kept as an explicit list rather than derived from a grep: the grep would
        pass by construction, which is not the same as being right.
        """
        for key in (
            "output_dir", "obsidian_enabled", "obsidian_vault", "obsidian_inbox",
            "local_speaker_name", "engine", "live_preview", "capture_backend",
            "diarizer", "live_speakers", "speaker_memory",
            "speaker_match_threshold", "keep_recordings", "hf_token",
        ):
            assert key in settings.DEFAULTS, f"{key} is read by the app but not declared"

    def test_unknown_keys_are_an_error_not_a_shrug(self, isolated_config):
        """A mistyped key used to read as 'absent, use the fallback'."""
        with pytest.raises(KeyError):
            cfg.get("outupt_dir")
        with pytest.raises(KeyError):
            cfg.set_value("nonexistent_key", 1)

    def test_get_takes_no_default_argument(self):
        """The signature is the enforcement. A default cannot be passed in."""
        with pytest.raises(TypeError):
            cfg.get("engine", "whisper")

    def test_settings_are_frozen(self):
        with pytest.raises(Exception):
            settings.load().engine = "whisper"


class TestFixedBugs:
    def test_the_local_speaker_name_has_one_default(self, isolated_config):
        """live_panel defaulted to "You" while config.py said "Don".

        Both were defensible readings, which is exactly why nothing caught it:
        the transcript said Don and the live panel said You, for the same voice
        in the same meeting.
        """
        assert settings.DEFAULTS["local_speaker_name"] == "Don"
        assert cfg.get("local_speaker_name") == "Don"

    def test_the_output_directory_has_one_default(self, isolated_config):
        """transcribe.OUTPUT_DIR fell back to ~/meeting-transcripts.

        Nothing read it, so the disagreement was invisible; it is deleted now,
        and this pins the one remaining answer.
        """
        assert cfg.get("output_dir").endswith("/overheard/transcripts")


class TestReadingTheFile:
    def test_stored_values_win_over_defaults(self, isolated_config):
        cfg.set_value("engine", "whisper")
        assert cfg.get("engine") == "whisper"

    def test_absent_keys_fall_back_to_the_declared_default(self, isolated_config):
        (isolated_config / "config.json").write_text(json.dumps({"engine": "whisper"}))
        loaded = settings.load()
        assert loaded.engine == "whisper"
        assert loaded.diarizer == "auto"

    def test_a_corrupt_file_reads_as_all_defaults(self, isolated_config):
        (isolated_config / "config.json").write_text("{not json")
        assert settings.load() == settings.Settings()

    def test_a_missing_file_reads_as_all_defaults(self, isolated_config):
        assert settings.load() == settings.Settings()

    def test_unknown_stored_keys_are_ignored_not_fatal(self, isolated_config):
        (isolated_config / "config.json").write_text(
            json.dumps({"engine": "whisper", "from_a_future_version": True})
        )
        assert settings.load().engine == "whisper"

    def test_unknown_stored_keys_survive_a_write(self, isolated_config):
        """Dropping a key a newer version wrote would silently lose settings."""
        path = isolated_config / "config.json"
        path.write_text(json.dumps({"from_a_future_version": True}))
        cfg.set_value("engine", "whisper")
        assert json.loads(path.read_text())["from_a_future_version"] is True


class TestCoercion:
    def test_a_number_written_as_a_string_is_coerced(self, isolated_config):
        (isolated_config / "config.json").write_text(
            json.dumps({"speaker_match_threshold": "0.85"})
        )
        assert settings.load().speaker_match_threshold == 0.85

    def test_a_bool_written_as_a_string_is_coerced(self, isolated_config):
        (isolated_config / "config.json").write_text(
            json.dumps({"live_preview": "false"})
        )
        assert settings.load().live_preview is False

    def test_an_unusable_value_falls_back_to_the_default(self, isolated_config, capsys):
        """Better a default than an exception midway through a transcription."""
        (isolated_config / "config.json").write_text(
            json.dumps({"speaker_match_threshold": "very high"})
        )
        assert settings.load().speaker_match_threshold == 0.70
        assert "speaker_match_threshold" in capsys.readouterr().err

    def test_a_wrongly_typed_string_falls_back(self, isolated_config):
        (isolated_config / "config.json").write_text(json.dumps({"engine": 3}))
        assert settings.load().engine == "parakeet"

    def test_an_uncoercible_value_is_repaired_by_an_unrelated_write(self, isolated_config):
        """A write to one key also repairs a different key that was already broken.

        This is deliberate, not incidental: repairing once and warning once
        beats carrying the bad value forward and warning on every future read.
        """
        (isolated_config / "config.json").write_text(
            json.dumps({"speaker_match_threshold": "very high"})
        )
        cfg.set_value("live_preview", False)
        reloaded = json.loads((isolated_config / "config.json").read_text())
        assert reloaded["speaker_match_threshold"] == 0.70
        assert reloaded["live_preview"] is False
