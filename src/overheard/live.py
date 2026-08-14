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

import json
import queue
import subprocess
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


class LiveDiarizer:
    """Streaming speaker attribution via the helper's Sortformer pipeline.

    Sortformer assigns speakers as audio arrives rather than clustering the
    whole recording afterwards, which is the only way to attribute speech live.
    Two limits come with that: at most four concurrent speakers, and labels that
    are positional rather than identities. The transcript written at the end
    still comes from the offline diarizer, so nothing here has to be perfect.
    """

    MAX_SPEAKERS = 4

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._segments: list[tuple[float, float, int]] = []
        self._failed = False

    @property
    def available(self) -> bool:
        return not self._failed and self._proc is not None

    def start(self) -> bool:
        """Launch the diarizer. Returns False if it could not start."""
        from overheard.helper import helper_path

        binary = helper_path()
        if binary is None:
            self._failed = True
            return False
        try:
            self._proc = subprocess.Popen(
                [str(binary), "diarize-stream"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )
        except OSError as e:
            print(f"[overheard] live diarizer failed to start: {e}", file=sys.stderr)
            self._failed = True
            return False

        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return True

    def feed(self, mono16k: np.ndarray) -> None:
        """Send 16 kHz mono audio. Never blocks the caller for long."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(mono16k.astype(np.float32).tobytes())
        except (BrokenPipeError, ValueError, OSError):
            self._failed = True
            self._proc = None

    def speaker_at(self, when: float) -> int | None:
        """Speaker index active at a point in time, or None."""
        with self._lock:
            for start, end, speaker in reversed(self._segments):
                if start <= when <= end:
                    return speaker
        return None

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()      # EOF makes the helper finalize cleanly
            proc.wait(timeout=3.0)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
        if self._reader:
            self._reader.join(timeout=2.0)
            self._reader = None

    def _read(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "segment":
                continue
            start, end = float(msg["start"]), float(msg["end"])
            speaker = int(msg["speaker"])
            with self._lock:
                # Tentative segments get revised, so replace any overlapping
                # entry from the same speaker rather than accumulating dupes.
                self._segments = [
                    s for s in self._segments
                    if not (s[2] == speaker and s[0] == start)
                ]
                self._segments.append((start, end, speaker))


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
        self._sentences: list[dict] = []
        self._status = "Loading model..."
        self._ready = False

        # Per-chunk energy on each track, used to tell the local speaker apart
        # from the remote side. With separate tracks this is a direct
        # measurement rather than the inference it would be on a mixed signal.
        self._energy: list[tuple[float, float, float, float]] = []
        self._elapsed = 0.0
        self.diarizer: LiveDiarizer | None = None

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
    def sentences(self) -> list[dict]:
        """Running transcript as {start, end, text, speaker}, newest last."""
        with self._lock:
            return list(self._sentences)

    def local_share(self, start: float, end: float) -> float:
        """Fraction of energy in a window that came from the microphone.

        Near 1.0 means the local speaker, near 0.0 the remote side. Returns 0.5
        when there is nothing to judge by, so neither side is favoured.
        """
        mic_total = system_total = 0.0
        with self._lock:
            for chunk_start, chunk_end, mic, system in self._energy:
                if chunk_end <= start or chunk_start >= end:
                    continue
                mic_total += mic
                system_total += system
        total = mic_total + system_total
        return mic_total / total if total > 1e-12 else 0.5

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
            self._record_energy(block)
            mono = _resample(_to_mono(block, self.channels_info), self.sample_rate)
            self._pending = np.concatenate([self._pending, mono])

            while len(self._pending) >= self._chunk_frames:
                chunk = self._pending[: self._chunk_frames]
                self._pending = self._pending[self._chunk_frames:]
                self._enqueue(chunk)
                if self.diarizer is not None and self.diarizer.available:
                    self.diarizer.feed(chunk)
        except Exception as e:
            print(f"[overheard] live feed error: {e}", file=sys.stderr)

    def _record_energy(self, block: np.ndarray) -> None:
        """Log mic and system energy for this block against the session clock."""
        if block.ndim < 2 or block.shape[1] < 2:
            return
        mic_ch = (self.channels_info or {}).get("mic_channel")
        sys_chs = [c for c in ((self.channels_info or {}).get("system_channels") or [])
                   if 0 <= c < block.shape[1]]
        if mic_ch is None or not (0 <= mic_ch < block.shape[1]) or not sys_chs:
            return

        duration = len(block) / self.sample_rate
        mic = float(np.sum(block[:, mic_ch] ** 2))
        system = float(np.sum(block[:, sys_chs].mean(axis=1) ** 2))
        with self._lock:
            start = self._elapsed
            self._elapsed += duration
            self._energy.append((start, self._elapsed, mic, system))

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

    def _publish(self, result) -> None:
        """Snapshot the decoder's sentences with their timings."""
        sentences = []
        for sentence in getattr(result, "sentences", []) or []:
            text = (sentence.text or "").strip()
            if not text:
                continue
            sentences.append({
                "start": float(sentence.start),
                "end": float(sentence.end),
                "text": text,
            })
        with self._lock:
            self._sentences = sentences
            self._text = (result.text or "").strip()

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
                    self._publish(stream.result)

                # Drain whatever is still queued so the tail is not lost
                while True:
                    try:
                        chunk = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    stream.add_audio(mx.array(chunk))
                self._publish(stream.result)
                self._set(status="Stopped")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set(status=f"Live transcription stopped: {e}")
