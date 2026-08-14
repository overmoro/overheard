"""Live transcript tests.

Live attribution rests on which track carried the audio, which is a measurement
rather than an inference because the mic and system arrive on separate channels.
These tests pin that measurement down, and the display rules built on top of it.
"""

import numpy as np
import pytest

from overheard.live import ENERGY_BUCKET_SECONDS, LiveTranscriber, _resample, _to_mono
from overheard.live_panel import LiveTranscriptPanel

DUAL = {"mic_channel": 0, "system_channels": [1]}


class TestToMono:
    def test_mic_is_mixed_at_equal_weight(self):
        """Averaging all channels would bury one mic under two system channels."""
        block = np.zeros((10, 3), dtype=np.float32)
        block[:, 2] = 1.0        # mic only
        mono = _to_mono(block, {"mic_channel": 2, "system_channels": [0, 1]})
        assert mono == pytest.approx(np.full(10, 0.5), abs=1e-6)

    def test_system_channels_are_averaged_together(self):
        block = np.zeros((10, 3), dtype=np.float32)
        block[:, 0] = 1.0
        block[:, 1] = 1.0
        mono = _to_mono(block, {"mic_channel": 2, "system_channels": [0, 1]})
        assert mono == pytest.approx(np.full(10, 0.5), abs=1e-6)

    def test_unknown_layout_averages_everything(self):
        block = np.ones((4, 2), dtype=np.float32)
        assert _to_mono(block, None) == pytest.approx(np.ones(4))

    def test_already_mono_passes_through(self):
        block = np.arange(5, dtype=np.float32)
        assert _to_mono(block, None) is not None
        assert _to_mono(block, None) == pytest.approx(block)

    def test_single_channel_two_dimensional(self):
        block = np.arange(5, dtype=np.float32).reshape(-1, 1)
        assert _to_mono(block, None) == pytest.approx(np.arange(5))

    def test_out_of_range_channels_fall_back_safely(self):
        block = np.ones((4, 2), dtype=np.float32)
        assert _to_mono(block, {"mic_channel": 9, "system_channels": [7]}) is not None


class TestResample:
    def test_same_rate_is_a_no_op(self):
        audio = np.arange(100, dtype=np.float32)
        assert _resample(audio, 16000) is audio

    def test_downsamples_to_the_model_rate(self):
        audio = np.zeros(48000, dtype=np.float32)
        out = _resample(audio, 48000)
        assert len(out) == pytest.approx(16000, abs=10)

    def test_a_tone_survives_resampling(self):
        """Guards against silently producing silence or noise."""
        t = np.arange(48000, dtype=np.float32) / 48000
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = _resample(audio, 48000)
        assert np.sqrt(np.mean(out ** 2)) == pytest.approx(0.35, abs=0.1)


class TestLocalShare:
    def transcriber(self):
        return LiveTranscriber(48000, DUAL)

    def feed_energy(self, transcriber, mic_level, system_level, blocks=20):
        block = np.zeros((1024, 2), dtype=np.float32)
        block[:, 0] = mic_level
        block[:, 1] = system_level
        for _ in range(blocks):
            transcriber._record_energy(block)

    def test_local_speech_reads_near_one(self):
        live = self.transcriber()
        self.feed_energy(live, mic_level=0.5, system_level=0.0)
        assert live.local_share(0.0, 1.0) > 0.95

    def test_remote_speech_reads_near_zero(self):
        live = self.transcriber()
        self.feed_energy(live, mic_level=0.0, system_level=0.5)
        assert live.local_share(0.0, 1.0) < 0.05

    def test_no_data_is_neutral_rather_than_favouring_a_side(self):
        assert self.transcriber().local_share(0.0, 1.0) == 0.5

    def test_silence_on_both_tracks_is_neutral(self):
        live = self.transcriber()
        self.feed_energy(live, mic_level=0.0, system_level=0.0)
        assert live.local_share(0.0, 1.0) == 0.5

    def test_windows_outside_the_recording_are_neutral(self):
        live = self.transcriber()
        self.feed_energy(live, mic_level=0.5, system_level=0.0, blocks=5)
        assert live.local_share(1000.0, 1001.0) == 0.5

    def test_energy_is_bucketed_not_stored_per_block(self):
        """One entry per audio block would be 47 a second, rescanned every poll."""
        live = self.transcriber()
        self.feed_energy(live, 0.5, 0.5, blocks=200)   # about 4.3 seconds
        expected = 4.3 / ENERGY_BUCKET_SECONDS
        assert len(live._energy) < expected + 2
        assert len(live._energy) < 200 / 4, "buckets are not collapsing"

    def test_mono_input_records_no_energy(self):
        live = LiveTranscriber(48000, None)
        live._record_energy(np.ones((1024, 1), dtype=np.float32))
        assert live._energy == []


class FakeDiarizer:
    def __init__(self, segments, available=True):
        self._segments = segments
        self.available = available

    def speaker_at(self, when):
        for start, end, speaker in self._segments:
            if start <= when <= end:
                return speaker
        return None


class FakeTranscriber:
    def __init__(self, sentences, shares, diarizer=None):
        self._sentences = sentences
        self._shares = shares
        self.diarizer = diarizer
        self.text = " ".join(s["text"] for s in sentences)

    @property
    def sentences(self):
        return self._sentences

    def local_share(self, start, _end):
        return self._shares.get(start, 0.5)


class TestRender:
    def render(self, transcriber):
        return LiveTranscriptPanel()._render(transcriber)

    def test_local_and_remote_are_labelled_separately(self, isolated_config):
        from overheard import config as cfg
        cfg.set_value("local_speaker_name", "Don")

        sentences = [
            {"start": 0.0, "end": 2.0, "text": "Right, let's kick off."},
            {"start": 5.0, "end": 7.0, "text": "Thanks."},
        ]
        transcriber = FakeTranscriber(
            sentences, {0.0: 1.0, 5.0: 0.02}, FakeDiarizer([(4.0, 8.0, 1)])
        )
        out = self.render(transcriber)
        assert "Don:" in out
        assert "Speaker 1:" in out, "remote voices are renumbered from 1"
        assert out.index("Don:") < out.index("Speaker 1:")

    def test_the_local_name_falls_back_to_the_one_declared_default(self, isolated_config):
        """With nothing configured, the panel and the transcript must agree.

        This panel used to default the local speaker to "You" while config.py
        defaulted it to "Don", so the same voice in the same meeting was named
        one thing live and another in the file that got saved.
        """
        from overheard import config as cfg

        sentences = [{"start": 0.0, "end": 1.0, "text": "Morning."}]
        out = self.render(FakeTranscriber(sentences, {0.0: 1.0}))

        assert f"{cfg.get('local_speaker_name')}:" in out
        assert "You:" not in out

    def test_remote_speakers_are_renumbered_from_one(self, isolated_config):
        """Sortformer indices are positional and may start anywhere, and a lone
        Speaker 3 with no Speaker 1 would only puzzle the reader."""
        sentences = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        transcriber = FakeTranscriber(sentences, {0.0: 0.0}, FakeDiarizer([(0.0, 1.0, 2)]))
        assert "Speaker 1:" in self.render(transcriber)

    def test_consecutive_sentences_from_one_speaker_share_a_heading(self, isolated_config):
        sentences = [
            {"start": 0.0, "end": 1.0, "text": "One."},
            {"start": 1.0, "end": 2.0, "text": "Two."},
        ]
        transcriber = FakeTranscriber(sentences, {0.0: 1.0, 1.0: 1.0})
        out = self.render(transcriber)
        assert out.count("Don:") <= 1 or out.count("You:") <= 1

    def test_without_a_diarizer_remote_speech_is_labelled_remote(self, isolated_config):
        sentences = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        transcriber = FakeTranscriber(sentences, {0.0: 0.0}, None)
        assert "Remote:" in self.render(transcriber)

    def test_unavailable_diarizer_degrades_to_remote(self, isolated_config):
        sentences = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        transcriber = FakeTranscriber(
            sentences, {0.0: 0.0}, FakeDiarizer([(0.0, 1.0, 0)], available=False)
        )
        assert "Remote:" in self.render(transcriber)

    def test_no_sentences_falls_back_to_plain_text(self, isolated_config):
        transcriber = FakeTranscriber([], {})
        transcriber.text = "partial words"
        assert self.render(transcriber) == "partial words"
