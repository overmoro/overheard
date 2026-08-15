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


#: FluidAudio's ModelNames.OfflineDiarizer.requiredModels, which is what
#: OfflineDiarizerModels.load() hard-fails without. Named here so a test that
#: omits one is obviously omitting one.
BUNDLES = (
    "Segmentation.mlmodelc",
    "FBank.mlmodelc",
    "Embedding.mlmodelc",
    "PldaRho.mlmodelc",
)
PLDA = "plda-parameters.json"


def _install(root, *, bundles=BUNDLES, plda=True, weights=True,
             subdir="speaker-diarization"):
    """Lay out the compiled models the way FluidAudio does.

    The bundles are .mlmodelc directories holding weights/weight.bin, so both
    "created but empty" and "created with metadata but no weights" are real
    intermediate states of an interrupted download rather than contrived ones.
    """
    parent = root / subdir
    for name in bundles:
        bundle = parent / name
        bundle.mkdir(parents=True, exist_ok=True)
        # Written first by a real download, and on its own it proves nothing.
        (bundle / "metadata.json").write_text("{}")
        (bundle / "analytics").mkdir(exist_ok=True)
        (bundle / "analytics" / "coremldata.bin").write_bytes(b"\x00" * 8)
        if weights:
            (bundle / "weights").mkdir(exist_ok=True)
            (bundle / "weights" / "weight.bin").write_bytes(b"\x00" * 64)
    if plda:
        parent.mkdir(parents=True, exist_ok=True)
        (parent / PLDA).write_text("{}")


def test_a_complete_install_is_present(models_dir):
    _install(models_dir)
    assert helper.diarization_models_present() is True


def test_nothing_downloaded_is_absent(models_dir):
    assert helper.diarization_models_present() is False


def test_a_config_file_alone_does_not_count(models_dir):
    """The exact interrupted-download shape: small files land first.

    Under the original check this returned True, and Preferences said the
    speaker models were installed with none of them on disk.
    """
    (models_dir / "speaker-diarization").mkdir()
    (models_dir / "speaker-diarization" / "config.json").write_text("{}")
    assert helper.diarization_models_present() is False


def test_bundles_without_weights_do_not_count(models_dir):
    """Metadata lands before weights, so a bundle can exist and be useless.

    "The .mlmodelc directory is not empty" is the same lie as "the Models
    directory is not empty", one level down.
    """
    _install(models_dir, weights=False)
    assert helper.diarization_models_present() is False


@pytest.mark.parametrize("missing", BUNDLES)
def test_any_single_missing_bundle_is_absent(models_dir, missing):
    """load() hard-fails without any one of them, so all four are required.

    Parametrised over FluidAudio's own list rather than spot-checking two, so
    adding a model to that tuple automatically demands a case for it.
    """
    _install(models_dir, bundles=[b for b in BUNDLES if b != missing])
    assert helper.diarization_models_present() is False


def test_the_plda_parameters_file_is_required(models_dir):
    """It is a plain file rather than a bundle, and load() needs it too."""
    _install(models_dir, plda=False)
    assert helper.diarization_models_present() is False


def test_an_empty_plda_file_does_not_count(models_dir):
    """A created-but-unwritten file is an unfinished download."""
    _install(models_dir)
    (models_dir / "speaker-diarization" / PLDA).write_text("")
    assert helper.diarization_models_present() is False


@pytest.mark.parametrize(
    "subdir",
    ["speaker-diarization", "speaker-diarization-coreml", "speaker-diarization-offline"],
)
def test_the_folder_name_upstream_probes_for_are_all_accepted(models_dir, subdir):
    """FluidAudio itself tries three names, so it has moved this before.

    Naming one literal would mean a rename leaves the label reading Missing
    while the helper finds everything and returns in under a second: a Download
    button that completes instantly, changes nothing, and never stops asking.
    """
    _install(models_dir, subdir=subdir)
    assert helper.diarization_models_present() is True


def test_the_check_does_not_read_the_real_machine(models_dir):
    """The path is overridable, which is what makes the cases above hermetic."""
    assert helper.FLUIDAUDIO_MODELS == models_dir


def test_a_stale_sibling_cannot_vouch_for_the_live_folder(models_dir):
    """FluidAudio loads from exactly one path, so only that one can answer.

    ModelHub resolves the diarizer to directory/Repo.diarizer.folderName, which
    is speaker-diarization: folderName has no .diarizer case and its default
    strips the "-coreml" that Repo.diarizer.name carries. A machine carrying a
    complete speaker-diarization-coreml/ next to a half-downloaded
    speaker-diarization/ would otherwise report Installed while the folder the
    helper actually loads is incomplete, and the download would surprise the
    user mid-meeting.

    The roles here were reversed until the seventh review round, which made
    this file pass against a folder layout that has never existed on disk.
    """
    _install(models_dir, subdir="speaker-diarization-coreml")            # complete sibling
    _install(models_dir, subdir="speaker-diarization", weights=False)    # live, partial
    assert helper.diarization_models_present() is False


def test_the_live_folder_answers_when_it_is_complete(models_dir):
    """The converse, so the rule is pinned in both directions."""
    _install(models_dir, subdir="speaker-diarization-coreml", weights=False)  # sibling, partial
    _install(models_dir, subdir="speaker-diarization")                       # live, complete
    assert helper.diarization_models_present() is True
