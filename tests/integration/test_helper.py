"""Helper binary contract tests.

The Python side parses the helper's JSON, so the shape of that output is an
interface between two languages with nothing checking it at build time. These
tests are that check.

All of these need the built binary and skip without it, so a fresh checkout with
no Swift toolchain still passes the suite.
"""

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from overheard.helper import helper_path

from ..conftest import requires_helper, requires_say

pytestmark = [pytest.mark.integration, requires_helper]


def run(*args, **kwargs):
    return subprocess.run(
        [str(helper_path()), *args], capture_output=True, timeout=600, **kwargs
    )


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory):
    """Two distinguishable voices, 16 kHz mono.

    Real speech is required: the ASR and diarization models cannot do useful
    work on synthetic tones, which is why this tier exists separately.
    """
    import shutil

    if shutil.which("say") is None:
        pytest.skip("needs macOS `say`")

    directory = tmp_path_factory.mktemp("speech")
    parts = []
    for index, (voice, line) in enumerate([
        ("Daniel", "Right, let us kick off. The first item is the budget review."),
        ("Karen", "Thanks. The numbers are looking better than we forecast."),
    ]):
        aiff = directory / f"{index}.aiff"
        wav = directory / f"{index}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), line], check=True, timeout=120)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
            check=True, timeout=120,
        )
        parts.append(sf.read(wav, dtype="float32")[0])

    gap = np.zeros(int(16000 * 0.4), dtype=np.float32)
    combined = np.concatenate([parts[0], gap, parts[1]])
    path = directory / "speech.wav"
    sf.write(path, combined, 16000)
    return path


class TestVersion:
    def test_reports_a_version(self):
        result = run("--version")
        assert result.returncode == 0
        assert result.stdout.decode().strip()

    def test_unknown_subcommand_is_rejected(self):
        assert run("nonsense").returncode != 0

    def test_every_subcommand_has_help(self):
        for command in ("capture", "diarize", "download", "processes", "diarize-stream"):
            result = run(command, "--help")
            assert result.returncode == 0, f"{command} --help failed"
            assert result.stdout.decode().strip(), f"{command} --help was empty"


class TestProcesses:
    def test_emits_a_json_list(self):
        result = run("processes")
        assert result.returncode == 0
        payload = json.loads(result.stdout.decode())
        assert isinstance(payload, list)

    def test_entries_carry_the_fields_detection_relies_on(self):
        """meeting.detect_source reads input, output and bundleID."""
        payload = json.loads(run("processes").stdout.decode())
        for entry in payload:
            assert isinstance(entry.get("input"), bool)
            assert isinstance(entry.get("output"), bool)
            assert entry["input"] or entry["output"], "silent processes should be omitted"


class TestDiarize:
    def test_separates_two_speakers(self, speech_wav):
        result = run("diarize", str(speech_wav), "--speakers", "2")
        assert result.returncode == 0, result.stderr.decode()[-400:]
        payload = json.loads(result.stdout.decode())

        assert payload["speakerCount"] == 2
        assert len(payload["segments"]) >= 2
        for entry in payload["segments"]:
            assert entry["end"] > entry["start"]
            assert isinstance(entry["speaker"], str)

    def test_needs_no_credentials(self, speech_wav, monkeypatch):
        """No Hugging Face token, no account. That is the point of FluidAudio."""
        environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(speech_wav.parent)}
        result = subprocess.run(
            [str(helper_path()), "diarize", str(speech_wav)],
            capture_output=True, timeout=600, env={**environment},
        )
        assert result.returncode == 0, result.stderr.decode()[-400:]

    def test_embeddings_are_opt_in(self, speech_wav):
        """256 floats a segment dwarf the rest, so they are only sent on request."""
        without = json.loads(run("diarize", str(speech_wav)).stdout.decode())
        assert "embedding" not in without["segments"][0]

        with_embeddings = json.loads(
            run("diarize", str(speech_wav), "--embeddings").stdout.decode()
        )
        embedding = with_embeddings["segments"][0]["embedding"]
        assert len(embedding) == 256

    def test_missing_file_fails_cleanly(self):
        result = run("diarize", "/nonexistent/audio.wav")
        assert result.returncode != 0
        assert b"no_input" in result.stderr or b"not found" in result.stderr


class TestDiarizeStream:
    def test_emits_segments_while_consuming_stdin(self, speech_wav):
        """Streaming attribution is what makes live speaker labels possible."""
        audio, rate = sf.read(speech_wav, dtype="float32")
        assert rate == 16000

        result = subprocess.run(
            [str(helper_path()), "diarize-stream"],
            input=audio.tobytes(), capture_output=True, timeout=600,
        )
        assert result.returncode == 0, result.stderr.decode()[-400:]

        segments = [
            json.loads(line)
            for line in result.stdout.decode().splitlines()
            if line.strip()
        ]
        assert segments, "no segments emitted"

        for entry in segments:
            assert entry["type"] == "segment"
            assert entry["end"] > entry["start"]
            assert isinstance(entry["speaker"], int)
            assert isinstance(entry["final"], bool)

        assert len({e["speaker"] for e in segments}) >= 2, "both voices should appear"
        assert any(e["final"] for e in segments), "some segments should confirm"

    def test_empty_input_terminates_cleanly(self):
        result = subprocess.run(
            [str(helper_path()), "diarize-stream"],
            input=b"", capture_output=True, timeout=600,
        )
        assert result.returncode == 0
