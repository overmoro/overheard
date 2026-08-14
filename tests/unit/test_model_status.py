"""Reporting what is actually on disk.

Preferences used to show a fixed "Downloads on first use" whether or not the
models were present. A label that never checks is worse than no label, because
the user cannot tell it apart from a real answer, and it sent Don looking for a
download that had already happened.
"""

import pytest

from overheard import asr


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """An isolated Hugging Face cache, so the developer's real one is not read."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    return tmp_path / "hub"


def _install(hub, repo_id, *, with_files=True):
    snapshot = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    if with_files:
        (snapshot / "config.json").write_text("{}")
    return snapshot


class TestIsModelCached:
    def test_an_empty_cache_reports_missing(self, hf_cache):
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is False

    def test_a_downloaded_model_reports_present(self, hf_cache):
        _install(hf_cache, "mlx-community/parakeet-tdt-0.6b-v3")
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is True

    def test_an_interrupted_download_reports_missing(self, hf_cache):
        """A cancelled download leaves the directories but no weights.

        Treating the folder's existence as proof is how a status check starts
        lying again, one layer down from the label it replaced.
        """
        _install(hf_cache, "mlx-community/parakeet-tdt-0.6b-v3", with_files=False)
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is False

    def test_another_model_being_present_does_not_count(self, hf_cache):
        _install(hf_cache, "meta-llama/Llama-3.1-8B-Instruct")
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is False

    def test_the_whisper_repo_is_the_ctranslate2_conversion(self):
        """WHISPER_MODEL is a size, not a repo, so checking it would always miss."""
        assert "/" in asr.WHISPER_REPO
        assert "/" not in asr.WHISPER_MODEL
