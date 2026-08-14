"""Live (near real-time) transcription using Parakeet's streaming decoder.

Audio arrives from the recording thread via :meth:`LiveTranscriber.feed`, which
must never block: it only downmixes, resamples and enqueues. A worker thread
owns the model and does the actual inference.

Measured on Apple Silicon, streaming inference runs at roughly 0.4x realtime and
stays flat over time, so a meeting is comfortably kept up with. If the machine
does fall behind, the queue drops the oldest audio rather than growing without
bound: the live view is a convenience, and the authoritative transcript is still
produced from the full recording when the meeting ends.

Text is exposed through the thread-safe :attr:`text` property rather than pushed
into AppKit from the worker thread. The UI polls it from a main-thread timer,
which keeps every AppKit call on the main thread.
"""

import queue
import sys
import threading

import numpy as np

# Parakeet's encoder expects 16 kHz mono
TARGET_RATE = 16000

# Audio is accumulated into blocks of this length before being fed to the model.
# Shorter means lower latency but more per-call overhead.
CHUNK_SECONDS = 1.0

# Attention context (left, right) carried across chunks by the streaming decoder
CONTEXT_SIZE = (256, 256)

# Maximum backlog before the oldest audio is dropped, in chunks
MAX_QUEUE_CHUNKS = 32


def is_available() -> bool:
    """Return True if the streaming backend can be imported."""
    try:
        import parakeet_mlx  # noqa: F401
        return True
    except Exception:
        return False


def _to_mono(block: np.ndarray, channels_info: dict | None = None) -> np.ndarray:
    """Collapse a (frames, channels) block to mono float32.

    Mirrors the offline downmix in transcribe._prepare_mono_audio: when the
    layout is known, the mic is mixed at equal weight against the combined
    system audio so the local speaker is not buried under the remote side.
    """
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)

    n_ch = block.shape[1]
    if n_ch == 1:
        return block[:, 0].astype(np.float32, copy=False)

    mic_ch = (channels_info or {}).get("mic_channel")
    sys_chs = [c for c in ((channels_info or {}).get("system_channels") or [])
               if 0 <= c < n_ch]

    if mic_ch is not None and 0 <= mic_ch < n_ch and sys_chs:
        mono = 0.5 * block[:, sys_chs].mean(axis=1) + 0.5 * block[:, mic_ch]
    else:
        mono = block.mean(axis=1)
    return mono.astype(np.float32, copy=False)


def _resample(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample mono audio to TARGET_RATE.

    Uses polyphase filtering, which applies a proper anti-aliasing filter.
    Naive decimation would alias badly at the 48k to 16k ratio typical here.
    """
    if src_rate == TARGET_RATE:
        return audio
    from fractions import Fraction
    from scipy.signal import resample_poly

    ratio = Fraction(TARGET_RATE, int(src_rate)).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


class LiveTranscriber:
    """Streams recorded audio through Parakeet and exposes a running transcript."""

    def __init__(self, sample_rate: int, channels_info: dict | None = None,
                 model_name: str | None = None):
        from overheard.transcribe import PARAKEET_MODEL

        self.sample_rate = int(sample_rate)
        self.channels_info = channels_info
        self.model_name = model_name or PARAKEET_MODEL

        self._queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_CHUNKS)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._text = ""
        self._status = "Loading model..."
        self._ready = False

        # Partial block held between feed() calls until it reaches CHUNK_SECONDS
        self._pending = np.empty(0, dtype=np.float32)
        self._chunk_frames = int(TARGET_RATE * CHUNK_SECONDS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """The running transcript. Safe to read from any thread."""
        with self._lock:
            return self._text

    @property
    def status(self) -> str:
        """A short human-readable state string. Safe to read from any thread."""
        with self._lock:
            return self._status

    @property
    def ready(self) -> bool:
        """True once the model is loaded and audio is actually being decoded."""
        with self._lock:
            return self._ready

    def start(self) -> None:
        """Start the worker thread. Returns immediately; the model loads async."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, block: np.ndarray) -> None:
        """Accept a block of raw recorded audio. Called from the audio thread.

        Must stay cheap and non-blocking: the recording loop is the only thing
        keeping up with CoreAudio, so it can never wait on inference.
        """
        if self._stop_event.is_set():
            return
        try:
            mono = _resample(_to_mono(block, self.channels_info), self.sample_rate)
            self._pending = np.concatenate([self._pending, mono])

            while len(self._pending) >= self._chunk_frames:
                chunk = self._pending[: self._chunk_frames]
                self._pending = self._pending[self._chunk_frames:]
                self._enqueue(chunk)
        except Exception as e:
            print(f"[overheard] live feed error: {e}", file=sys.stderr)

    def stop(self, timeout: float = 3.0) -> str:
        """Stop the worker and return the final live transcript."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        return self.text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue(self, chunk: np.ndarray) -> None:
        """Queue a chunk, dropping the oldest if the worker has fallen behind."""
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
                print("[overheard] live transcription behind, dropped audio",
                      file=sys.stderr)
            except queue.Empty:
                pass

    def _set(self, *, text: str | None = None, status: str | None = None,
             ready: bool | None = None) -> None:
        with self._lock:
            if text is not None:
                self._text = text
            if status is not None:
                self._status = status
            if ready is not None:
                self._ready = ready

    def _run(self) -> None:
        """Worker thread: own the model, drain the queue, update the transcript."""
        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
        except Exception as e:
            self._set(status=f"Live transcription unavailable: {e}")
            return

        try:
            model = from_pretrained(self.model_name)
        except Exception as e:
            self._set(status=f"Could not load model: {e}")
            return

        self._set(status="Listening...", ready=True)

        try:
            with model.transcribe_stream(context_size=CONTEXT_SIZE, depth=1) as stream:
                while not self._stop_event.is_set():
                    try:
                        chunk = self._queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    stream.add_audio(mx.array(chunk))
                    self._set(text=stream.result.text.strip())

                # Drain whatever is still queued so the tail is not lost
                while True:
                    try:
                        chunk = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    stream.add_audio(mx.array(chunk))
                self._set(text=stream.result.text.strip(), status="Stopped")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set(status=f"Live transcription stopped: {e}")
