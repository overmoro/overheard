"""Structural interfaces shared across Overheard.

The app talks to two interchangeable capture backends: ``capture.TapRecorder``,
which wraps the Core Audio tap helper, and ``audio.Recorder``, which wraps an
input device through PortAudio. They share no ancestry and never will, since one
drives a subprocess and the other drives a stream callback. What they do share
is a calling convention, and until now that convention lived only in a comment.

``AudioSource`` writes it down. It is a ``Protocol`` rather than a base class
precisely because the relationship is structural: neither recorder is a kind of
the other, they simply answer the same messages.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AudioSource(Protocol):
    """What ``app.py`` may rely on from a capture backend.

    ``@runtime_checkable`` allows ``isinstance``, but that check only confirms
    the member names exist; it says nothing about signatures. The value here is
    static: a type checker can see that both backends satisfy every attribute
    the app reaches for, which is what stopped being true when the app started
    reading private names.
    """

    #: Rate the hardware actually opened at, which is not always the requested one.
    sample_rate: int

    #: Channels the backend is capturing.
    channels: int

    @property
    def channels_info(self) -> dict | None:
        """``{"mic_channel": int, "system_channels": [int, ...]}``, or None.

        None means the recording is a single undifferentiated track, so the
        transcription pipeline cannot tell the local speaker from the remote
        one and must fall back to a mixed track.
        """
        ...

    @property
    def is_multichannel(self) -> bool:
        """True when a separate microphone track is present.

        Drives the level meters, which show one bar for a mixed source and two
        when the mic and the system audio arrive separately.
        """
        ...

    def start(self) -> None:
        """Begin capturing. Raises if the backend cannot start."""
        ...

    def stop(self) -> tuple[Any, dict | None]:
        """Stop capturing and return (audio, channels_info)."""
        ...

    def pause(self) -> None:
        """Stop retaining audio without tearing the capture down."""
        ...

    def resume(self) -> None:
        """Resume retaining audio after a pause."""
        ...

    def get_levels(self) -> tuple[float, float]:
        """(mic_rms, system_rms) from the most recent block, for the meters."""
        ...

    def set_tap(self, tap) -> None:
        """Register a callable fed every captured block, or None to clear.

        The tap runs on the capture thread, so it must return quickly. Both
        backends swallow exceptions raised by a tap: live transcription is a
        convenience and must never cost the recording.
        """
        ...
