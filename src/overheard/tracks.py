"""Turning one recording into the tracks the pipeline transcribes.

Everything here happens before any model loads: check the audio is not silent,
split the mic from the system audio, and afterwards drop the parts of the mic
track that are only the system audio bleeding back in.

Imports of numpy and soundfile stay inside the functions. Both are heavy, and
this module is reached from ``app.py`` at startup, where nothing has been
recorded yet.
"""

import sys


def check_audio_signal(audio_path: str) -> None:
    """Raise if the recording is effectively silent, before loading any models."""
    import numpy as np
    import soundfile as sf

    audio_check, _ = sf.read(audio_path)
    rms = float(np.sqrt(np.mean(audio_check ** 2)))
    if rms < 0.0001:
        raise RuntimeError(
            f"Audio appears silent (RMS={rms:.6f}). Check that Overheard has "
            "microphone access in System Settings, and that the meeting audio "
            "was playing through your selected output device."
        )


def warn_silent_tracks(tracks: list[dict]) -> None:
    """Warn when one track is silent while another has signal.

    The whole-file signal check cannot catch this: with the system track
    carrying a meeting, a muted microphone still leaves plenty of overall
    signal, and the transcript comes out looking fine while missing everything
    the user said. Worth saying loudly, but not worth discarding the half of
    the conversation that did record, so this warns rather than raises.
    """
    import numpy as np
    import soundfile as sf

    if len(tracks) < 2:
        return

    levels = {}
    for track in tracks:
        try:
            data, _ = sf.read(track["path"], dtype="float32")
        except Exception:
            return
        levels[track["label"]] = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0

    for label, rms in levels.items():
        others = [v for k, v in levels.items() if k != label]
        if rms < 0.0001 and any(other > 0.001 for other in others):
            source = ("microphone" if label == "mic" else "system audio")
            print(
                f"[overheard] the {label} track is silent while the other has signal. "
                f"Check that the {source} is not muted; nothing from that side "
                "will appear in this transcript.",
                file=sys.stderr,
            )


def split_tracks(audio_path: str, channels_info: dict | None) -> list[dict]:
    """Split a recording into independent mono tracks.

    When the capture knows which channels are the microphone and which are the
    system, the two are written out separately rather than mixed. Keeping them
    apart matters for three reasons:

    - The local speaker is known by construction. There is no need to infer who
      the user is from relative channel energy.
    - Each side is diarized on its own, so the two halves of a conversation
      never have to be told apart by clustering.
    - Mixing a mic that is picking up the speakers against the system audio it
      is echoing produces a doubled, phase-shifted signal that measurably
      degrades transcription.

    Returns a list of {label, path, temp} dicts: either one "mixed" track or a
    "mic" and a "system" track.
    """
    import numpy as np
    import soundfile as sf
    import tempfile

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    n_ch = data.shape[1]

    mic_ch = (channels_info or {}).get("mic_channel")
    sys_chs = [c for c in ((channels_info or {}).get("system_channels") or [])
               if 0 <= c < n_ch]

    if mic_ch is None or not (0 <= mic_ch < n_ch) or not sys_chs:
        # Unknown layout: fall back to a single folded track.
        path, temp = prepare_mono_audio(audio_path, channels_info)
        return [{"label": "mixed", "path": path, "temp": temp}]

    def _write(samples) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak > 1.0:
            samples = samples / peak
        sf.write(tmp.name, samples.astype(np.float32), int(sr))
        return tmp.name

    return [
        {"label": "mic", "path": _write(data[:, mic_ch]), "temp": True},
        {"label": "system", "path": _write(data[:, sys_chs].mean(axis=1)), "temp": True},
    ]


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Seconds of overlap between two intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def drop_echo(mic_segments: list[dict], system_segments: list[dict]) -> list[dict]:
    """Remove mic segments that are just the speakers bleeding into the mic.

    When the user listens on speakers rather than headphones, the microphone
    re-records the remote side, so the same words arrive on both tracks and
    would be transcribed twice.

    A mic segment is treated as echo when it overlaps system speech for most of
    its length *and* says substantially the same thing. Requiring both keeps
    genuine interruptions, where the user talks over the remote side and the
    words differ.
    """
    from difflib import SequenceMatcher

    if not system_segments:
        return mic_segments

    kept = []
    for segment in mic_segments:
        duration = max(1e-6, segment["end"] - segment["start"])
        covered = 0.0
        best_similarity = 0.0

        for other in system_segments:
            shared = overlap(segment["start"], segment["end"], other["start"], other["end"])
            if shared <= 0:
                continue
            covered += shared
            similarity = SequenceMatcher(
                None, segment["text"].lower(), other["text"].lower()
            ).ratio()
            best_similarity = max(best_similarity, similarity)

        if covered / duration > 0.5 and best_similarity > 0.6:
            continue
        kept.append(segment)

    return kept


def prepare_mono_audio(audio_path: str, channels_info: dict | None) -> tuple[str, bool]:
    """Downmix a multichannel recording to mono, returning (path, is_temporary).

    Both engines ultimately load audio through ffmpeg's ``-ac 1``, and ffmpeg
    reads a 3-channel WAV as a 2.1 layout: channel 2 is treated as LFE and
    dropped entirely from the mono downmix. On the Meeting Capture aggregate
    channel 2 is the microphone, so the local speaker's voice was being
    discarded before transcription ever saw it.

    Doing the downmix here avoids that. When the channel layout is known, the
    microphone is mixed at equal weight against the combined system audio, so
    neither side of the conversation dominates. Otherwise all channels are
    averaged evenly.

    The result is written at the source sample rate; ffmpeg resamples mono
    audio correctly, it is only the channel folding that is wrong.
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        return audio_path, False

    n_ch = data.shape[1]
    mic_ch = (channels_info or {}).get("mic_channel")
    sys_chs = [c for c in ((channels_info or {}).get("system_channels") or [])
               if 0 <= c < n_ch]

    if mic_ch is not None and 0 <= mic_ch < n_ch and sys_chs:
        mono = 0.5 * data[:, sys_chs].mean(axis=1) + 0.5 * data[:, mic_ch]
    else:
        mono = data.mean(axis=1)

    # Mixing can push peaks past full scale; scale back rather than clip
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 1.0:
        mono = mono / peak

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, mono.astype(np.float32), int(sr))
    return tmp.name, True
