"""Meeting source detection tests.

Detection used to ask whether an application was running, so an open Chrome
reported a Google Meet on every recording and that wrong value went into the
transcript's frontmatter. It now asks which process is capturing the microphone,
because that is what distinguishes a call from a browser playing a video.
"""

import pytest

from overheard import meeting


def process(bundle_id=None, input=False, output=False, pid=1234):
    entry = {"input": input, "output": output, "pid": pid}
    if bundle_id is not None:
        entry["bundleID"] = bundle_id
    return entry


class TestDetectSource:
    def patch_processes(self, monkeypatch, processes):
        monkeypatch.setattr(meeting, "_audio_processes", lambda: processes)

    def test_a_process_capturing_input_identifies_the_platform(self, monkeypatch):
        self.patch_processes(monkeypatch, [process("us.zoom.xos", input=True, output=True)])
        assert meeting.detect_source() == "zoom"

    @pytest.mark.parametrize(
        "bundle_id,expected",
        [
            ("us.zoom.xos", "zoom"),
            ("com.microsoft.teams", "teams"),
            ("com.microsoft.teams2", "teams"),
            ("com.google.Chrome", "meet"),
            ("com.apple.Safari", "meet"),
            ("org.mozilla.firefox", "meet"),
        ],
    )
    def test_known_platforms(self, monkeypatch, bundle_id, expected):
        self.patch_processes(monkeypatch, [process(bundle_id, input=True)])
        assert meeting.detect_source() == expected

    def test_bundle_matching_is_case_insensitive(self, monkeypatch):
        self.patch_processes(monkeypatch, [process("US.ZOOM.XOS", input=True)])
        assert meeting.detect_source() == "zoom"

    def test_output_only_playback_is_not_a_meeting(self, monkeypatch):
        """A browser playing a video must not be reported as a call."""
        self.patch_processes(monkeypatch, [process("com.google.Chrome", output=True)])
        assert meeting.detect_source() == "in-person"

    def test_capturing_input_wins_over_an_idle_meeting_app(self, monkeypatch):
        self.patch_processes(monkeypatch, [
            process("us.zoom.xos", output=True),
            process("com.google.Chrome", input=True),
        ])
        assert meeting.detect_source() == "meet"

    def test_nothing_playing_is_in_person(self, monkeypatch):
        self.patch_processes(monkeypatch, [])
        assert meeting.detect_source() == "in-person"

    def test_unrecognised_app_using_the_mic_is_in_person(self, monkeypatch):
        self.patch_processes(monkeypatch, [process("com.example.dictaphone", input=True)])
        assert meeting.detect_source() == "in-person"

    def test_process_without_a_bundle_id_is_ignored(self, monkeypatch):
        """Command line tools have no bundle identifier."""
        self.patch_processes(monkeypatch, [process(None, input=True)])
        assert meeting.detect_source() == "in-person"

    def test_falls_back_to_process_names_when_the_helper_is_absent(self, monkeypatch):
        """Without the helper, detection is cruder but must still answer."""
        monkeypatch.setattr(meeting, "_audio_processes", lambda: None)
        monkeypatch.setattr(meeting, "_process_running", lambda name: name == "zoom.us")
        assert meeting.detect_source() == "zoom"

    def test_fallback_reports_in_person_when_nothing_runs(self, monkeypatch):
        monkeypatch.setattr(meeting, "_audio_processes", lambda: None)
        monkeypatch.setattr(meeting, "_process_running", lambda name: False)
        assert meeting.detect_source() == "in-person"


class TestInferLocation:
    @pytest.mark.parametrize(
        "source,expected",
        [
            ("zoom", "Zoom"),
            ("teams", "Microsoft Teams"),
            ("meet", "Google Meet"),
            ("in-person", ""),
            ("other", ""),
            ("nonsense", ""),
        ],
    )
    def test_display_names(self, source, expected):
        assert meeting.infer_location(source) == expected


class TestAudioProcesses:
    def test_missing_helper_returns_none_not_an_empty_list(self, monkeypatch):
        """None and [] mean different things: unavailable versus nothing playing."""
        import overheard.helper

        monkeypatch.setattr(overheard.helper, "helper_path", lambda: None)
        assert meeting._audio_processes() is None
