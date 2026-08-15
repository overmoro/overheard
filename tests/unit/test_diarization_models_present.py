"""The speaker-model check must require the models, not the folder.

This is the sibling of the ASR check, and it carried the defect that one was
rewritten to remove: "the directory exists and has something in it" counts a
download cancelled after a few seconds as a finished one, because the small
files land first. Preferences then reports "Installed" and the user finds out
at the end of a meeting.

Both halves feed the same label, so the weaker one set the real bar.
"""

import pytest

from overheard import helper


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Point the check at a temporary tree instead of the real machine.

    The old implementation read Application Support directly, so a test written
    against it passed or failed depending on what the developer had downloaded.
    """
    root = tmp_path / "Models"
    root.mkdir()
    monkeypatch.setattr(helper, "FLUIDAUDIO_MODELS", root)
    return root


def _install(root, *, segmentation=True, embedding=True, populated=True):
    """Lay out the compiled bundles the way FluidAudio does.

    They are .mlmodelc directories, not files, so "created but empty" is a real
    intermediate state rather than a contrived one.
    """
    wanted = []
    if segmentation:
        wanted.append("Segmentation.mlmodelc")
    if embedding:
        wanted.append("Embedding.mlmodelc")
    for name in wanted:
        bundle = root / "speaker-diarization" / name
        bundle.mkdir(parents=True)
        if populated:
            (bundle / "coremldata.bin").write_bytes(b"\x00" * 16)


def test_a_complete_install_is_present(models_dir):
    _install(models_dir)
    assert helper.diarization_models_present() is True


def test_nothing_downloaded_is_absent(models_dir):
    assert helper.diarization_models_present() is False


def test_a_config_file_alone_does_not_count(models_dir):
    """The exact interrupted-download shape: small files land first.

    Under the old check this returned True, and Preferences said the speaker
    models were installed with none of them on disk.
    """
    (models_dir / "speaker-diarization").mkdir()
    (models_dir / "speaker-diarization" / "config.json").write_text("{}")
    assert helper.diarization_models_present() is False


def test_an_empty_bundle_directory_does_not_count(models_dir):
    """A .mlmodelc created but never filled is an unfinished download."""
    _install(models_dir, populated=False)
    assert helper.diarization_models_present() is False


@pytest.mark.parametrize("missing", ["segmentation", "embedding"])
def test_one_model_without_the_other_is_absent(models_dir, missing):
    """Diarization needs both, so either one missing means not installed."""
    _install(
        models_dir,
        segmentation=(missing != "segmentation"),
        embedding=(missing != "embedding"),
    )
    assert helper.diarization_models_present() is False


def test_the_check_does_not_read_the_real_machine(models_dir):
    """The path is overridable, which is what makes the cases above hermetic."""
    assert helper.FLUIDAUDIO_MODELS == models_dir
