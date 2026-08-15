"""The Preferences panes must report what is true now, not at first open.

The window is built once and reused, so anything computed during ``_build`` is
a reading from the first time Preferences was ever opened. Models get
downloaded, aggregate devices get created and deleted in Audio MIDI Setup, and
voices are added to the library by every transcription, so three labels were
reporting state that had since moved.

Nothing in this file needs a window server. The delegate is a real NSObject and
the labels are stand-ins that record what was written to them, which is enough
to pin who writes what and when.
"""

import pytest

from overheard import preferences


class FakeLabel:
    """Stands in for an NSTextField, recording writes.

    Implements performSelectorOnMainThread_withObject_waitUntilDone_ because
    that is how the production code writes to a label from a worker thread.
    Recording it separately is what lets a test assert the marshalling happened
    rather than assuming it.
    """

    def __init__(self):
        self.value = ""
        self.direct_writes = 0
        self.marshalled_writes = 0
        self.modes = []

    def setStringValue_(self, text):
        self.value = text
        self.direct_writes += 1

    def performSelectorOnMainThread_withObject_waitUntilDone_(self, sel, obj, wait):
        raise AssertionError(
            "the default-mode perform was used. It is delivered only in "
            "NSDefaultRunLoopMode, so progress freezes while a modal or "
            "tracking loop is running, which is when the user clicks again."
        )

    def performSelectorOnMainThread_withObject_waitUntilDone_modes_(
        self, sel, obj, wait, modes
    ):
        assert sel == "setStringValue:", f"unexpected selector {sel!r}"
        self.value = obj
        self.marshalled_writes += 1
        self.modes.append(modes)


@pytest.fixture
def delegate(monkeypatch):
    """A real delegate with stand-in widgets and no filesystem underneath."""
    d = preferences._PreferencesDelegate.alloc().initWithWindow_(None)
    d._deps_status = FakeLabel()
    d._aggregate_status = FakeLabel()
    d._multiout_status = FakeLabel()

    monkeypatch.setattr(preferences, "_model_status", lambda: "✓ Installed")
    monkeypatch.setattr(preferences, "_device_exists", lambda name: f"✓ {name}")
    monkeypatch.setattr(d, "_refresh_speakers", lambda: None)
    return d


class TestRefreshStatus:
    def test_it_recomputes_the_labels_that_can_go_stale(self, delegate):
        """All three read state outside the window and were build-time only."""
        delegate.refresh_status()
        assert delegate._deps_status.value == "✓ Installed"
        assert delegate._aggregate_status.value == "✓ Meeting Capture"
        assert delegate._multiout_status.value == "✓ Meeting Monitor"

    def test_it_reloads_the_known_voices(self, delegate, monkeypatch):
        """The speaker list is written by every transcription, not just by _build."""
        called = []
        monkeypatch.setattr(delegate, "_refresh_speakers", lambda: called.append(True))
        delegate.refresh_status()
        assert called, "the known-voices popup was not reloaded"

    def test_a_second_open_sees_a_model_that_arrived_in_between(self, delegate, monkeypatch):
        """The defect in one test: install a model between two opens."""
        installed = {"yes": False}
        monkeypatch.setattr(
            preferences, "_model_status",
            lambda: "✓ Installed" if installed["yes"] else "Missing: transcription models",
        )
        delegate.refresh_status()
        assert delegate._deps_status.value.startswith("Missing")
        installed["yes"] = True
        delegate.refresh_status()
        assert delegate._deps_status.value == "✓ Installed"

    def test_it_does_not_raise_when_reading_the_cache_fails(self, delegate, monkeypatch):
        """This runs from the gear button, where an escaping exception aborts.

        _model_status walks the Hugging Face cache and _device_exists queries
        Core Audio. Either can raise OSError, and unwinding out of an AppKit
        action callback kills the process with no crash report, which is worse
        than a stale label.
        """
        def boom():
            raise OSError("cache directory unreadable")
        monkeypatch.setattr(preferences, "_model_status", boom)

        delegate.refresh_status()   # must not raise

        assert "unavailable" in delegate._deps_status.value, (
            "a failed refresh must say so on the pane, not leave it blank"
        )


class TestDownloadLatch:
    def test_a_second_click_while_downloading_starts_nothing(self, delegate, monkeypatch):
        """Two clicks used to mean two 30 minute subprocesses on one cache."""
        started = []
        monkeypatch.setattr(
            preferences.threading, "Thread",
            lambda *a, **k: type("T", (), {"start": lambda s: started.append(True)})(),
        )
        delegate.downloadModels_(None)
        delegate.downloadModels_(None)
        assert len(started) == 1, f"expected one download thread, started {len(started)}"

    def test_a_download_in_flight_keeps_its_own_label(self, delegate):
        """refresh_status must not report Missing over a running download.

        Opening Preferences while a download runs used to overwrite the
        progress text with a state read from a cache the download had not
        finished writing, so the UI said nothing was happening and invited a
        second click.
        """
        delegate._downloading = True
        delegate._deps_status.value = "Downloading Parakeet TDT v3..."
        delegate.refresh_status()
        assert delegate._deps_status.value == "Downloading Parakeet TDT v3..."

    def test_the_latch_clears_so_a_failed_download_can_be_retried(self, delegate, monkeypatch):
        """A latch that survives a failure is a button that never works again."""
        def explode():
            raise RuntimeError("network gone")
        monkeypatch.setattr(preferences.subprocess, "run", lambda *a, **k: explode())

        delegate._downloading = True
        delegate._do_download_models()
        assert delegate._downloading is False


class TestThreadSafety:
    def test_a_worker_thread_write_is_marshalled(self, delegate):
        """Mutating an NSTextField off the main thread is undefined behaviour.

        The workers run on daemon threads, and this app has already died once
        from an exception crossing the ObjC boundary with no crash report. This
        runs on a real thread rather than asserting the branch, because the
        branch is the thing under test.
        """
        import threading

        done = threading.Event()

        def worker():
            preferences._set_label(delegate._deps_status, "from a worker")
            done.set()

        threading.Thread(target=worker, daemon=True).start()
        assert done.wait(timeout=5), "the worker never finished"
        assert delegate._deps_status.marshalled_writes == 1
        assert delegate._deps_status.direct_writes == 0

    def test_a_worker_write_is_delivered_in_every_run_loop_mode(self, delegate):
        """The default-mode perform freezes while a modal or tracking loop runs.

        A user who clicks Download and then opens the Browse panel, or holds
        the mouse on the popover header, would see the progress label stop
        updating for the duration. A frozen label on a multi-gigabyte download
        is what makes people click the button again.
        """
        import threading
        from Foundation import NSRunLoopCommonModes

        done = threading.Event()

        def worker():
            preferences._set_label(delegate._deps_status, "progress")
            done.set()

        threading.Thread(target=worker, daemon=True).start()
        assert done.wait(timeout=5)
        assert delegate._deps_status.modes, "no modes recorded"
        assert NSRunLoopCommonModes in delegate._deps_status.modes[0]

    def test_a_main_thread_write_is_immediate(self, delegate):
        """Queueing the main thread's own writes leaves the window blank.

        show() calls refresh_status and then orders the window front. Since
        _build no longer paints these labels, marshalling unconditionally would
        put an empty pane on screen and fill it a run-loop pass later, or
        longer if the click arrived during a tracking loop.
        """
        preferences._set_label(delegate._deps_status, "on the main thread")
        assert delegate._deps_status.direct_writes == 1
        assert delegate._deps_status.marshalled_writes == 0
        assert delegate._deps_status.value == "on the main thread"

    def test_refresh_status_leaves_no_label_empty(self, delegate):
        """The first open must not show three blank labels.

        _build stopped painting them when refresh_status became the single
        writer, so this pins the property that made that safe.
        """
        delegate.refresh_status()
        for name in ("_deps_status", "_aggregate_status", "_multiout_status"):
            assert getattr(delegate, name).value != "", f"{name} was left blank"

    def test_the_worker_can_ask_for_a_refresh_on_the_main_thread(self, delegate):
        """The selector has to exist, or the request is silently a no-op."""
        assert delegate.respondsToSelector_("refreshStatusOnMain:"), (
            "the download worker calls this selector by name when it finishes"
        )


class TestTheCalendarWorker:
    """The fourth worker in this file, which the first pass missed."""

    def test_its_label_writes_are_marshalled(self, delegate, monkeypatch):
        """_set_label exists for exactly this, and this worker bypassed it.

        Three of the four _do_* workers were converted and this one was not, so
        the class carried two contradictory patterns with nothing to say which
        was intended.
        """
        import threading

        delegate._calendar_status = FakeLabel()
        monkeypatch.setattr(
            preferences.subprocess, "run",
            lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )

        done = threading.Event()

        def worker():
            try:
                delegate._do_connect_calendar()
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()
        assert done.wait(timeout=10), "the calendar worker never finished"

        assert delegate._calendar_status.direct_writes == 0, (
            "the calendar worker wrote to an NSTextField from a daemon thread"
        )
        assert delegate._calendar_status.marshalled_writes >= 1


class TestTheLatchGeneration:
    def test_a_finishing_worker_does_not_release_a_later_download(self, delegate):
        """The counter exists for this, and nothing tested it.

        Worker 1 finishes after the user has already started download 2. Its
        release must be a no-op, or a third click starts a download concurrent
        with the second.
        """
        delegate._downloading = True
        delegate._download_generation = 2      # download 2 owns the latch
        delegate.releaseDownload_(1)           # worker 1 finishes late
        assert delegate._downloading is True, (
            "a stale worker released a latch belonging to a later download"
        )

    def test_the_owning_worker_does_release(self, delegate):
        delegate._downloading = True
        delegate._download_generation = 2
        delegate.releaseDownload_(2)
        assert delegate._downloading is False
