"""Track splitting, echo suppression and signal checks.

Keeping the microphone and system audio apart is what lets the pipeline know who
the local speaker is rather than infer it. These tests cover the splitting itself
and the echo suppression that separation made necessary: with both sides recorded
independently, a speaker bleeding into the mic would otherwise be transcribed
twice.
"""

import numpy as np
import pytest
import soundfile as sf

from overheard.transcribe import (
    _check_audio_signal,
    _drop_echo,
    _overlap,
    _prepare_mono_audio,
    _split_tracks,
    _warn_silent_tracks,
)

from ..conftest import segment, silence, tone

DUAL = {"mic_channel": 2, "system_channels": [0, 1]}


class TestSplitTracks:
    def test_dual_track_yields_separate_mono_files(self, dual_track_wav):
        tracks = _split_tracks(str(dual_track_wav), DUAL)
        assert [t["label"] for t in tracks] == ["mic", "system"]

        for track in tracks:
            data, _ = sf.read(track["path"])
            assert data.ndim == 1, "each track must be mono"

        mic, _ = sf.read(tracks[0]["path"])
        system, _ = sf.read(tracks[1]["path"])
        # The fixture puts the mic in the first half and system in the second
        assert np.sqrt(np.mean(mic[: len(mic) // 2] ** 2)) > 0.01
        assert np.sqrt(np.mean(mic[len(mic) // 2:] ** 2)) < 0.01
        assert np.sqrt(np.mean(system[len(system) // 2:] ** 2)) > 0.01

    def test_mono_input_passes_through_untouched(self, mono_wav):
        tracks = _split_tracks(str(mono_wav), None)
        assert len(tracks) == 1
        assert tracks[0]["label"] == "mixed"
        assert tracks[0]["temp"] is False, "no copy needed for mono"
        assert tracks[0]["path"] == str(mono_wav)

    def test_unknown_layout_falls_back_to_one_folded_track(self, dual_track_wav):
        """Multichannel audio with no layout information still transcribes."""
        tracks = _split_tracks(str(dual_track_wav), None)
        assert len(tracks) == 1 and tracks[0]["label"] == "mixed"
        data, _ = sf.read(tracks[0]["path"])
        assert data.ndim == 1

    def test_out_of_range_channels_are_rejected(self, dual_track_wav):
        tracks = _split_tracks(str(dual_track_wav), {"mic_channel": 9, "system_channels": [0]})
        assert tracks[0]["label"] == "mixed", "a bad layout must not crash or mislabel"


class TestPrepareMonoAudio:
    def test_mic_is_not_buried_under_the_system_side(self, tmp_path):
        """ffmpeg's -ac 1 drops channel 2 of a 3-channel file as LFE.

        Folding here instead is what stops the local speaker vanishing, so the
        mic must survive at meaningful level.
        """
        mic = tone(1.0, 220.0, amplitude=0.5)
        system = silence(1.0)
        path = tmp_path / "mic_only.wav"
        sf.write(path, np.stack([system, system, mic], axis=1), 48000)

        folded, is_temp = _prepare_mono_audio(str(path), DUAL)
        assert is_temp
        data, _ = sf.read(folded)
        assert np.sqrt(np.mean(data ** 2)) > 0.1, "microphone was lost in the downmix"

    def test_clipping_is_scaled_back_not_clipped(self, tmp_path):
        loud = tone(0.5, 300.0, amplitude=0.9)
        path = tmp_path / "loud.wav"
        sf.write(path, np.stack([loud, loud, loud], axis=1), 48000)

        folded, _ = _prepare_mono_audio(str(path), DUAL)
        data, _ = sf.read(folded)
        assert np.max(np.abs(data)) <= 1.0 + 1e-6

    def test_mono_is_returned_as_is(self, mono_wav):
        path, is_temp = _prepare_mono_audio(str(mono_wav), None)
        assert path == str(mono_wav) and is_temp is False


class TestDropEcho:
    def test_echo_of_the_remote_side_is_removed(self):
        system = [segment(5.0, 9.0, "The numbers are looking better than we forecast")]
        mic = [
            segment(0.0, 4.0, "Right, let's kick off"),
            segment(5.1, 8.9, "the numbers are looking better than we forecast"),
        ]
        kept = _drop_echo(mic, system)
        assert [s["text"] for s in kept] == ["Right, let's kick off"]

    def test_a_genuine_interruption_survives(self):
        """Overlapping in time is not enough; the words must also match."""
        system = [segment(5.0, 9.0, "The numbers are looking better than we forecast")]
        mic = [segment(6.0, 7.0, "sorry, can I jump in")]
        assert _drop_echo(mic, system) == mic

    def test_matching_words_at_a_different_time_survive(self):
        """Agreeing later is not an echo."""
        system = [segment(0.0, 2.0, "shall we start")]
        mic = [segment(30.0, 32.0, "shall we start")]
        assert _drop_echo(mic, system) == mic

    def test_no_system_segments_keeps_everything(self):
        mic = [segment(0.0, 1.0, "hello")]
        assert _drop_echo(mic, []) == mic

    def test_empty_mic_track(self):
        assert _drop_echo([], [segment(0.0, 1.0, "hi")]) == []

    def test_zero_length_segment_does_not_divide_by_zero(self):
        mic = [segment(1.0, 1.0, "hi")]
        _drop_echo(mic, [segment(0.0, 2.0, "hi")])   # must not raise


class TestOverlap:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ((0.0, 2.0), (1.0, 3.0), 1.0),
            ((0.0, 1.0), (2.0, 3.0), 0.0),
            ((0.0, 5.0), (1.0, 2.0), 1.0),
            ((1.0, 2.0), (1.0, 2.0), 1.0),
            ((0.0, 1.0), (1.0, 2.0), 0.0),
        ],
    )
    def test_interval_overlap(self, a, b, expected):
        assert _overlap(a[0], a[1], b[0], b[1]) == pytest.approx(expected)


class TestSignalChecks:
    def test_silent_file_is_rejected_before_models_load(self, tmp_path):
        path = tmp_path / "silent.wav"
        sf.write(path, silence(1.0), 48000)
        with pytest.raises(RuntimeError, match="silent"):
            _check_audio_signal(str(path))

    def test_file_with_signal_passes(self, mono_wav):
        _check_audio_signal(str(mono_wav))

    def test_a_silent_track_is_reported(self, silent_mic_wav, capsys):
        """A muted mic leaves plenty of overall signal, so only a per-track
        check can catch the user's own voice going missing."""
        tracks = _split_tracks(str(silent_mic_wav), DUAL)
        _warn_silent_tracks(tracks)
        assert "mic track is silent" in capsys.readouterr().err

    def test_no_warning_when_both_tracks_have_signal(self, dual_track_wav, capsys):
        _warn_silent_tracks(_split_tracks(str(dual_track_wav), DUAL))
        assert capsys.readouterr().err == ""

    def test_no_warning_for_a_single_track(self, mono_wav, capsys):
        _warn_silent_tracks(_split_tracks(str(mono_wav), None))
        assert capsys.readouterr().err == ""
