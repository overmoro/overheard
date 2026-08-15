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


def _install(hub, repo_id, *, weights=True):
    """Lay out a cache entry the way huggingface_hub does.

    Completed files appear under snapshots/<rev>/ as they finish, and they
    finish smallest first, so config.json lands long before the weights.
    """
    snapshot = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"\x00" * 32)
    return snapshot


class TestIsModelCached:
    def test_an_empty_cache_reports_missing(self, hf_cache):
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is False

    def test_a_downloaded_model_reports_present(self, hf_cache):
        _install(hf_cache, "mlx-community/parakeet-tdt-0.6b-v3")
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is True

    def test_an_interrupted_download_reports_missing(self, hf_cache):
        """The real shape of a cancelled download: config.json, no weights.

        huggingface_hub writes completed files smallest first, so a download
        stopped after seconds leaves a good config and none of the model.
        Counting any file as proof would put the lie one layer below the label
        this check replaced, and the user would only find out when a meeting
        failed to transcribe.
        """
        _install(hf_cache, "mlx-community/parakeet-tdt-0.6b-v3", weights=False)
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is False

    def test_another_model_being_present_does_not_count(self, hf_cache):
        _install(hf_cache, "meta-llama/Llama-3.1-8B-Instruct")
        assert asr.is_model_cached("mlx-community/parakeet-tdt-0.6b-v3") is False

    def test_the_parakeet_constant_is_a_repo_id(self):
        """A bare model size would never match a cache folder, so it must be a repo.

        The whisper engine's constant was a size rather than a repo id, which is
        why a second WHISPER_REPO existed alongside it. One engine, one constant,
        and this pins it as the kind of name the cache is actually keyed on.
        """
        assert "/" in asr.PARAKEET_MODEL
